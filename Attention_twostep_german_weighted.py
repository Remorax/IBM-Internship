import os, itertools, time, pickle, operator, random
import subprocess
from xml.dom import minidom
from collections import Counter, OrderedDict
from operator import itemgetter
from scipy import spatial
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import re, sys
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from math import ceil, exp
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

f = open(sys.argv[1], "rb")
data, emb_indexer, emb_indexer_inv, emb_vals, neighbours_dicts = pickle.load(f)
max_paths = int(sys.argv[2])
max_pathlen = int(sys.argv[3])
max_types = 2
flatten = lambda l: [item for sublist in l for item in sublist]

direct_inputs, direct_targets = [], []

def cos_sim(a,b):
    return 1 - spatial.distance.cosine(a,b)

all_fn, all_fp = [], []

threshold_results = {}

def test():
    global batch_size, test_data_t, test_data_f, model, optimizer, all_metrics, direct_inputs, direct_targets, threshold_results, emb_indexer_inv, emb_vals
    all_results = OrderedDict()    
    direct_inputs, direct_targets = [], []
    with torch.no_grad():
        all_pred = []
        
        np.random.shuffle(test_data_t)
        np.random.shuffle(test_data_f)

        inputs_pos, nodes_pos, targets_pos = generate_input(test_data_t, 1)
        inputs_neg, nodes_neg, targets_neg = generate_input(test_data_f, 0)

        inputs_all = list(inputs_pos) + list(inputs_neg)
        targets_all = list(targets_pos) + list(targets_neg)
        nodes_all = list(nodes_pos) + list(nodes_neg)
        
        all_inp = list(zip(inputs_all, targets_all, nodes_all))
        all_inp_shuffled = random.sample(all_inp, len(all_inp))
        inputs_all, targets_all, nodes_all = list(zip(*all_inp_shuffled))

        batch_size = min(batch_size, len(inputs_all))
        num_batches = int(ceil(len(inputs_all)/batch_size))
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx+1) * batch_size

            inputs = np.array(to_feature(inputs_all[batch_start: batch_end]))
            targets = np.array(targets_all[batch_start: batch_end])
            nodes = np.array(nodes_all[batch_start: batch_end])
            
            inp_elems = torch.LongTensor(inputs).to(device)
            node_elems = torch.LongTensor(nodes).to(device)
            targ_elems = torch.DoubleTensor(targets)

            outputs = model(node_elems, inp_elems)
            outputs = [el.item() for el in outputs]
            targets = [True if el.item() else False for el in targets]

            for idx, pred_elem in enumerate(outputs):
                ent1 = emb_indexer_inv[nodes[idx][0]]
                ent2 = emb_indexer_inv[nodes[idx][1]]
                if (ent1, ent2) in all_results:
                    print ("Error: ", ent1, ent2, "already present")
                all_results[(ent1, ent2)] = (round(pred_elem, 3), targets[idx])
        
        direct_targets = [True if el else False for el in direct_targets]
        
        print ("Len (direct inputs): ", len(direct_inputs))
        for idx, direct_input in enumerate(direct_inputs):
            ent1 = emb_indexer_inv[direct_input[0]]
            ent2 = emb_indexer_inv[direct_input[1]]
            sim = cos_sim(emb_vals[direct_input[0]], emb_vals[direct_input[1]])
            all_results[(ent1, ent2)] = (round(sim, 3), direct_targets[idx])
    return (test_data_t, all_results)

def optimize_threshold(emb_indexer_inv, emb_vals):
    global batch_size, val_data_t, val_data_f, model, optimizer, gt_mappings, all_metrics, direct_inputs, direct_targets, threshold_results
    all_results = OrderedDict()
    direct_inputs, direct_targets = [], []
    with torch.no_grad():
        all_pred = []
        
        np.random.shuffle(val_data_t)
        np.random.shuffle(val_data_f)

        inputs_pos, nodes_pos, targets_pos = generate_input(val_data_t, 1)
        inputs_neg, nodes_neg, targets_neg = generate_input(val_data_f, 0)

        inputs_all = list(inputs_pos) + list(inputs_neg)
        targets_all = list(targets_pos) + list(targets_neg)
        nodes_all = list(nodes_pos) + list(nodes_neg)
        
        all_inp = list(zip(inputs_all, targets_all, nodes_all))
        all_inp_shuffled = random.sample(all_inp, len(all_inp))
        inputs_all, targets_all, nodes_all = list(zip(*all_inp_shuffled))

        batch_size = min(batch_size, len(inputs_all))
        num_batches = int(ceil(len(inputs_all)/batch_size))
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx+1) * batch_size

            inputs = np.array(to_feature(inputs_all[batch_start: batch_end]))
            targets = np.array(targets_all[batch_start: batch_end])
            nodes = np.array(nodes_all[batch_start: batch_end])
            
            inp_elems = torch.LongTensor(inputs).to(device)
            node_elems = torch.LongTensor(nodes).to(device)
            targ_elems = torch.DoubleTensor(targets)

            outputs = model(node_elems, inp_elems)
            outputs = [el.item() for el in outputs]
            targets = [True if el.item() else False for el in targets]

            for idx, pred_elem in enumerate(outputs):
                ent1 = emb_indexer_inv[nodes[idx][0]]
                ent2 = emb_indexer_inv[nodes[idx][1]]
                if (ent1, ent2) in all_results:
                    print ("Error: ", ent1, ent2, "already present")
                all_results[(ent1, ent2)] = (round(pred_elem, 3), targets[idx])
        
        direct_targets = [True if el else False for el in direct_targets]
        
        print ("Len (direct inputs): ", len(direct_inputs))
        for idx, direct_input in enumerate(direct_inputs):
            ent1 = emb_indexer_inv[direct_input[0]]
            ent2 = emb_indexer_inv[direct_input[1]]
            sim = cos_sim(emb_vals[direct_input[0]], emb_vals[direct_input[1]])
            all_results[(ent1, ent2)] = (round(sim, 3), direct_targets[idx])
        
        low_threshold = round(np.min([el[0] for el in all_results.values()]) - 0.02, 3)
        high_threshold = round(np.max([el[0] for el in all_results.values()]) + 0.02, 3)
        threshold = low_threshold
        step = 0.001
        print (low_threshold, high_threshold)
        while threshold < high_threshold:
            threshold = round(threshold, 3)
#             print (threshold)
            res = []
            for i,key in enumerate(all_results):
                if all_results[key][0] > threshold:
                    res.append(key)
            fn_list = [(key, all_results[key][0]) for key in val_data_t if key not in set(res)]
            fp_list = [(elem, all_results[elem][0]) for elem in res if not all_results[elem][1]]
            tp_list = [(elem, all_results[elem][0]) for elem in res if all_results[elem][1]]
            
            tp, fn, fp = len(tp_list), len(fn_list), len(fp_list)
            exception = False
            
            try:
                precision = tp/(tp+fp)
                recall = tp/(tp+fn)
                f1score = 2 * precision * recall / (precision + recall)
                f2score = 5 * precision * recall / (4 * precision + recall)
                f0_5score = 1.25 * precision * recall / (0.25 * precision + recall)
            except Exception as e:
                print (e)
                exception = True
                step = 0.001
                threshold += step
                continue
            # print ("Threshold: ", threshold, precision, recall, f1score, f2score, f0_5score)
            if threshold in threshold_results:
                #print (threshold, len(threshold_results[threshold]))
                threshold_results[threshold].append([precision, recall, f1score, f2score, f0_5score])
            else:
                threshold_results[threshold] = [[precision, recall, f1score, f2score, f0_5score]]
            threshold += step
        
def calculate_performance():
    global final_results
    all_metrics, all_fn, all_fp = [], [], []
    for (test_data_t, all_results) in final_results:
        res = []
        for i,key in enumerate(all_results):
            if all_results[key][0] > threshold:
                res.append(key)
        s = set(res)
        fn_list = [(key, all_results[key][0]) for key in test_data_t if key not in s]
        fp_list = [(elem, all_results[elem][0]) for elem in res if not all_results[elem][1]]
        tp_list = [(elem, all_results[elem][0]) for elem in res if all_results[elem][1]]
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
        all_fn.extend(fn_list)
        all_fp.extend(fp_list)
        all_metrics.append((precision, recall, f1score, f2score, f0_5score))
    return all_metrics, all_fn, all_fp

def masked_softmax(inp):
    inp = inp.double()
    mask = ((inp != 0).double() - 1) * 9999  # for -inf
    return (inp + mask).softmax(dim=-1)

class SiameseNetwork(nn.Module):
    def __init__(self, emb_vals, threshold=0.9):
        super().__init__() 
        
        self.n_neighbours = max_types
        self.max_paths = max_paths
        self.max_pathlen = max_pathlen
        self.embedding_dim = np.array(emb_vals).shape[1]
        
        self.name_embedding = nn.Embedding(len(emb_vals), self.embedding_dim)
        self.name_embedding.load_state_dict({'weight': torch.from_numpy(np.array(emb_vals))})
        self.name_embedding.weight.requires_grad = False
        
        self.threshold = nn.Parameter(torch.DoubleTensor([threshold]))
        self.threshold.requires_grad = False
        
        self.cosine_sim_layer = nn.CosineSimilarity(dim=1)
        self.output = nn.Linear(2*self.embedding_dim, 300)
        
        self.v = nn.Parameter(torch.DoubleTensor([1/(self.max_pathlen) for i in range(self.max_pathlen)]))
        self.w_rootpath = nn.Parameter(torch.DoubleTensor([0.5]))
 
    def forward(self, nodes, features):
        '''
        Arguments:
            - nodes: batch_size * 2
            - features: batch_size * 2 * 4 * max_paths * max_pathlen
        '''
        results = []
        nodes = nodes.permute(1,0) # 2 * batch_size
        features = features.permute(1,0,2,3,4) # 2 * batch_size * 4 * max_paths * max_pathlen
        for i in range(2):
            node_emb = self.name_embedding(nodes[i]) # batch_size * 512
            feature_emb = self.name_embedding(features[i]) #  batch_size * 4 * max_paths * max_pathlen * 512
            
            feature_emb_reshaped = feature_emb.permute(0,4,1,2,3).reshape(-1, self.embedding_dim, self.n_neighbours * self.max_paths * self.max_pathlen)
            path_weights = torch.bmm(node_emb[:, None, :], feature_emb_reshaped)
            path_weights = path_weights.squeeze(1).reshape(-1, self.n_neighbours, self.max_paths, self.max_pathlen)
            path_weights = masked_softmax(torch.sum(path_weights, dim=-1))

            feature_emb_reshaped = feature_emb.reshape(-1, self.max_paths, self.max_pathlen * self.embedding_dim)
            best_path = torch.bmm(path_weights.reshape(-1, 1, self.max_paths), feature_emb_reshaped)
            best_path = best_path.squeeze(1).reshape(-1, self.n_neighbours, self.max_pathlen, self.embedding_dim) # batch_size * 4 * max_pathlen * 512

            best_path_reshaped = best_path.permute(0,3,1,2).reshape(-1, self.embedding_dim, self.n_neighbours * self.max_pathlen)
            node_weights = torch.bmm(node_emb.unsqueeze(1), best_path_reshaped) # batch_size * 4 * max_pathlen
            node_weights = masked_softmax(node_weights.squeeze(1).reshape(-1, self.n_neighbours, self.max_pathlen)) # batch_size * 4 * max_pathlen
            attended_path = node_weights.unsqueeze(-1) * best_path # batch_size * 4 * max_pathlen * 512

            distance_weighted_path = torch.sum((self.v[None,None,:,None] * attended_path), dim=2) # batch_size * 4 * 512

            self.w_children = (1-self.w_rootpath)
            context_emb = self.w_rootpath * distance_weighted_path[:,0,:] \
                        + self.w_children * distance_weighted_path[:,1,:]

            contextual_node_emb = torch.cat((node_emb, context_emb), dim=1)
            output_node_emb = self.output(contextual_node_emb)
            results.append(output_node_emb)
        sim = self.cosine_sim_layer(results[0], results[1])
        return sim

def is_valid(test_onto, key):
    return tuple([el.split("#")[0] for el in key]) not in test_onto

def generate_data_neighbourless(elem_tuple):
    return [emb_indexer[elem] for elem in elem_tuple]

def embedify(seq):
    for item in seq:
        if isinstance(item, list):
            yield list(embedify(item))
        else:
            yield emb_indexer[item]

def generate_data(elem_tuple):
    return list(embedify([neighbours_dicts[elem] for elem in elem_tuple]))

def to_feature(inputs):
    inputs_lenpadded = [[[[path[:max_pathlen] + [0 for i in range(max_pathlen -len(path[:max_pathlen]))]
                                    for path in nbr_type[:max_paths]]
                                for nbr_type in ent[:max_types]]
                            for ent in elem]
                        for elem in inputs]
    inputs_pathpadded = [[[nbr_type + [[0 for j in range(max_pathlen)]
                             for i in range(max_paths - len(nbr_type))]
                            for nbr_type in ent] for ent in elem]
                        for elem in inputs_lenpadded]
    return inputs_pathpadded

def generate_input(elems, target):
    inputs, targets, nodes = [], [], []
    global direct_inputs, direct_targets
    for elem in list(elems):
        try:
            inputs.append(generate_data(elem))
            nodes.append(generate_data_neighbourless(elem))
            targets.append(target)
        except:
            direct_inputs.append(generate_data_neighbourless(elem))
            direct_targets.append(target)
    return inputs, nodes, targets

print("Number of neighbours: " + str(sys.argv[1]))

def count_non_unk(elem):
    return len([l for l in elem if l!="<UNK>"])

torch.set_default_dtype(torch.float64)

print ("Number of entities:", len(data))

all_metrics = []
final_results = []

for i in range(5):
    data_t = {elem: data[elem] for elem in data if data[elem]}
    data_f = {elem: data[elem] for elem in data if not data[elem]}

    data_t_items = list(data_t.keys())
    data_f_items = list(data_f.keys())

    test_data_t = data_t_items[int((0.2*i)*len(data_t)):int((0.2*i + 0.2)*len(data_t))]
    test_data_f = data_f_items[int((0.2*i)*len(data_f)):int((0.2*i + 0.2)*len(data_f))]

    train_data_tt = data_t_items[:int(0.2*i*len(data_t))] + data_t_items[int(0.2*(i+1)*len(data_t)):]
    train_data_ff = data_f_items[:int(0.2*i*len(data_f))] + data_f_items[int(0.2*(i+1)*len(data_f)):]

    val_data_t = train_data_tt[:int(0.1*len(data_t))]
    val_data_f = train_data_ff[:int(0.1*len(data_f))]

    train_data_t = train_data_tt[int(0.1*len(data_t)):]
    train_data_f = train_data_ff[int(0.1*len(data_f)):]
    np.random.shuffle(train_data_f)
    train_data_f = train_data_f[:150000]

    print ("Training size:", len(train_data_t) + len(train_data_f) , "Val size:", len(val_data_t) + len(val_data_f))

    train_data_t = np.repeat(train_data_t, ceil(len(train_data_f)/len(train_data_t)), axis=0)
    train_data_t = train_data_t[:len(train_data_f)].tolist()
    
    lr = 0.001
    num_epochs = 50
    weight_decay = 0.001
    batch_size = 32
    dropout = 0.3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = SiameseNetwork(emb_vals).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        inputs_pos, nodes_pos, targets_pos = generate_input(train_data_t, 1)
        inputs_neg, nodes_neg, targets_neg = generate_input(train_data_f, 0)
        inputs_all = list(inputs_pos) + list(inputs_neg)
        targets_all = list(targets_pos) + list(targets_neg)
        nodes_all = list(nodes_pos) + list(nodes_neg)
        
        all_inp = list(zip(inputs_all, targets_all, nodes_all))
        all_inp_shuffled = random.sample(all_inp, len(all_inp))
        inputs_all, targets_all, nodes_all = list(zip(*all_inp_shuffled))

        batch_size = min(batch_size, len(inputs_all))
        num_batches = int(ceil(len(inputs_all)/batch_size))

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx+1) * batch_size
            
            inputs = np.array(to_feature(inputs_all[batch_start: batch_end]))
            targets = np.array(targets_all[batch_start: batch_end])
            nodes = np.array(nodes_all[batch_start: batch_end])
            
            inp_elems = torch.LongTensor(inputs).to(device)
            targ_elems = torch.DoubleTensor(targets).to(device)
            node_elems = torch.LongTensor(nodes).to(device)

            optimizer.zero_grad()
            outputs = model(node_elems, inp_elems)
            loss = F.mse_loss(outputs, targ_elems)
            loss.backward()
            optimizer.step()

            if batch_idx%5000 == 0:
                print ("Epoch: {} Idx: {} Loss: {}".format(epoch, batch_idx, loss.item()))

    model.eval()

    np.random.shuffle(val_data_f)
    fval_len = len(val_data_f)
    val_data_f = val_data_f[:int(0.3*fval_len)]
    
    optimize_threshold(emb_indexer_inv, emb_vals)

    sys.stdout.flush()

    final_results.append(test())

threshold_results_mean = {el: np.mean(threshold_results[el], axis=0) for el in threshold_results}    
threshold = max(threshold_results_mean.keys(), key=(lambda key: threshold_results_mean[key][2]))

all_metrics, all_fn, all_fp = calculate_performance()

print ("Final Results: " + str(np.mean(all_metrics, axis=0)))
print ("Threshold: ", threshold)
