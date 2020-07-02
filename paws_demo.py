# Construction of dataset

import os, itertools, time, pickle, sys
import subprocess
from xml.dom import minidom
from collections import Counter, OrderedDict
from operator import itemgetter
import tensorflow as tf
from scipy import spatial
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from math import ceil, exp
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

embeddings, train_sents, val_sents, test_sents = pickle.load(open("embeddings.pkl", "rb"))

flatten = lambda l: [item for sublist in l for item in sublist]

all_sents = flatten([[el[0], el[1]] for el in train_sents + val_sents + test_sents])

def greedy_matching():
    global batch_size, test_data_t, test_data_f, model, optimizer, emb_indexer_inv, gt_mappings, all_metrics
    all_results = OrderedDict()
    with torch.no_grad():
        all_pred = []
        batch_size = min(batch_size, len(test_data_t))
        num_batches = int(ceil(len(test_data_t)/batch_size))
        batch_size_f = int(ceil(len(test_data_f)/num_batches))
        
        print ("F batch size: ", batch_size_f)
        np.random.shuffle(test_data_t)
        np.random.shuffle(test_data_f)

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx+1) * batch_size

            batch_start_f = batch_idx * batch_size_f
            batch_end_f = (batch_idx+1) * batch_size_f
            
            pos_elems = np.array(test_data_t)[batch_start:batch_end]
            neg_elems = np.array(test_data_f)[batch_start_f:batch_end_f]
            optimizer.zero_grad()

            inputs = np.array([generate_data(elem) for elem in list(pos_elems) + list(neg_elems)])
            print ("Inputs len: ", len(inputs))
            targets = np.array([1 for i in range(len(pos_elems))] + [0 for i in range(len(neg_elems))])

            indices = np.random.permutation(inputs.shape[0])
            inputs, targets = inputs[indices], targets[indices]
            inputs = torch.LongTensor(inputs)
            targets = torch.LongTensor(targets)

            outputs = model(inputs, 1).cpu().numpy()
            #outputs = [el.item() for el in outputs]
            
            
            targets = [True if el.item() else False for el in targets]

            for idx, pred_elem in enumerate(outputs):
                ent1 = emb_indexer_inv[inputs.T.numpy()[0][idx]]
                ent2 = emb_indexer_inv[inputs.T.numpy()[1][idx]]
                if (ent1, ent2) in all_results:
                    print ("Error: ", ent1, ent2, "already present")
                all_results[(ent1, ent2)] = (pred_elem, targets[idx])
        
#         all_results = OrderedDict(sorted(all_results.items(), key=lambda x: x[0], reverse=True))
#         filtered_results = dict()
#         entities_to_assign = set([el[0] for el in list(all_results.keys())])
#         for pair in all_results:
#             if pair[0] in entities_to_assign:
#                 filtered_results[pair] = all_results[pair]
#                 entities_to_assign.remove(pair[0])
#         filtered_results = OrderedDict(sorted(filtered_results.items(), key=lambda x: x[1][0], reverse=True))
         
        min_val = np.min([el[0] for el in list(all_results.values())])
        max_val = np.max([el[0] for el in list(all_results.values())])
        normalized_results = {}
        for key,val in all_results.items():
            tmp = (np.array(val[0]) - min_val) / (max_val - min_val)
            normalized_results[key] = (tmp[1], val[1])
        all_results = normalized_results
        
        optimum_metrics, opt_threshold = [-1000 for i in range(5)], -1000
        low_threshold = np.min([el[0] for el in all_results.values()]) - 0.01
        high_threshold = np.max([el[0] for el in all_results.values()]) + 0.01
        print ("Low:", low_threshold, "High:", high_threshold)
        for j,threshold in enumerate(np.arange(low_threshold, high_threshold, 0.01)):
            res = []
            for i,key in enumerate(all_results):
                if all_results[key][0] > threshold:
                    res.append(key)

            fn_list = [key for key in test_data_t if key not in set(res)]
            fp_list = [elem for elem in res if not all_results[elem][1]]
            tp_list = [elem for elem in res if all_results[elem][1]]
            
            tp, fn, fp = len(tp_list), len(fn_list), len(fp_list)
            
            
            try:
                precision = tp/(tp+fp)
                recall = tp/(tp+fn)
                f1score = 2 * precision * recall / (precision + recall)
                f2score = 5 * precision * recall / (4 * precision + recall)
                f0_5score = 1.25 * precision * recall / (0.25 * precision + recall)
            except Exception as e:
                print (e)
                continue
            print ("Threshold: ", threshold, precision, recall, f1score, f2score, f0_5score)

            if f1score > optimum_metrics[2]:
                optimum_metrics = [precision, recall, f1score, f2score, f0_5score]
                opt_threshold = threshold
        
        print ("Precision: {} Recall: {} F1-Score: {} F2-Score: {} F0.5-Score: {}".format(*optimum_metrics))
        all_metrics.append((opt_threshold, optimum_metrics))
        
    print ("Final Results: ", np.mean([el[1] for el in all_metrics], axis=0))
    print ("Best threshold: ", all_metrics[np.argmax([el[1][2] for el in all_metrics])][0])
    return all_results
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def write(elem):
    f = open("Logs", "a+")
    if type(elem) == list or type(elem) == tuple:
        string = str("\n".join([str(s) for s in elem]))
    else:
        string = str(elem)
    f.write("\n"+string)
    f.close()
    

class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()    
        self.name_embedding = nn.Embedding(len(embeddings), 512)
        self.name_embedding.load_state_dict({'weight': torch.from_numpy(np.array(emb_vals))})
        self.name_embedding.require_grad = False
        
        self.dropout = dropout
        self.cosine_sim_layer = nn.CosineSimilarity(dim=1)
        self.layer1 = nn.Bilinear(512, 512, 2)
        self.layer2 = nn.Linear(512, 150)
        self.layer3 = nn.Linear(2048, 2)

    def forward(self, inputs, epoch):
        results = []
        inputs = inputs.T
        for i in range(2):
            x = self.name_embedding(inputs[i])
            #x = self.layer2(x)
            #x = F.dropout(x, p=0.3)
            results.append(x)
        inp = torch.cat((results[0]-results[1], results[0]+results[1], results[0], results[1]), dim=1)
        #x = self.cosine_sim_layer(results[0], results[1])
        x = self.layer3(inp)
        return F.log_softmax(x)
        #return x
emb_indexer = {word: i for i, word in enumerate(list(embeddings.keys()))}
emb_indexer_inv = {i: word for i, word in enumerate(list(embeddings.keys()))}
emb_vals = list(embeddings.values())

def generate_data(elem_tuple):
    return np.array([emb_indexer[elem] for elem in elem_tuple])

def generate_input(elems, target):
    inputs = np.array([generate_data(elem) for elem in list(elems)])
    targets = np.array([target for i in range(len(elems))])
    return inputs, targets

train_data = {tuple(sent[:2]): sent[-1] for sent in train_sents + val_sents}
data_items = train_data.items()
np.random.shuffle(list(data_items))
train_data = OrderedDict(data_items)

test_data = {tuple(sent[:2]): sent[-1] for sent in test_sents}
data_items = test_data.items()
np.random.shuffle(list(data_items))
test_data = OrderedDict(data_items)

print ("Number of entities:", len(train_data))

all_metrics = []
    
torch.set_default_dtype(torch.float64)

train_data_t = [key for key in train_data if train_data[key]]
train_data_f = [key for key in train_data if not train_data[key]]



lr = 0.01
num_epochs = 10
weight_decay = 0.001
batch_size = 10
dropout = 0.3
batch_size = min(batch_size, len(train_data_t))
num_batches = int(ceil(len(train_data_t)/batch_size))
batch_size_f = int(ceil(len(train_data_f)/num_batches))

model = nn.DataParallel(SiameseNetwork()).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

for epoch in range(num_epochs):
    inputs_pos, targets_pos = generate_input(train_data_t, 1)
    inputs_neg, targets_neg = generate_input(train_data_f, 0)

#         indices = np.random.permutation(len(inputs_pos) + len(inputs_neg))

#         inputs = np.array(list(inputs_pos) + list(inputs_neg))[indices]
#         targets = np.array(list(targets_pos) + list(targets_neg))[indices]

    indices_pos = np.random.permutation(len(inputs_pos))
    indices_neg = np.random.permutation(len(inputs_neg))

    inputs_pos, targets_pos = inputs_pos[indices_pos], targets_pos[indices_pos]
    inputs_neg, targets_neg = inputs_neg[indices_neg], targets_neg[indices_neg]
#         inputs = np.array(list(inputs_pos) + list(inputs_neg))[indices]
#         targets = np.array(list(targets_pos) + list(targets_neg))[indices]


#         inputs = np.array(list(inputs_pos) + list(inputs_neg))
#         targets = np.array(list(targets_pos) + list(targets_neg))

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx+1) * batch_size

        batch_start_f = batch_idx * batch_size_f
        batch_end_f = (batch_idx+1) * batch_size_f

        inputs = np.concatenate((inputs_pos[batch_start: batch_end], inputs_neg[batch_start_f: batch_end_f]))
        targets = np.concatenate((targets_pos[batch_start: batch_end], targets_neg[batch_start_f: batch_end_f]))
#             print (inputs.shape)
        inp_elems = torch.LongTensor(inputs)
        targ_elems = torch.LongTensor(targets).to(device)
#             print (targ_elems)
        optimizer.zero_grad()

        outputs = model(inp_elems, epoch)
#             print (outputs)
        loss = F.nll_loss(outputs, targ_elems)
        loss.backward()

        optimizer.step()

        if batch_idx%10 == 0:
#                 print ("Outupts: ", list(zip(outputs.detach().numpy(), targ_elems.detach().numpy())))
            print ("Epoch: {} Idx: {} Loss: {}".format(epoch, batch_idx, loss.item()))

model.eval()

test_data_t = [key for key in test_data if test_data[key]]
test_data_f = [key for key in test_data if not test_data[key]]

res = greedy_matching()

