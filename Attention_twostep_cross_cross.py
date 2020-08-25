import os, itertools, time, pickle, operator
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

f = open(sys.argv[2], "rb")
data, emb_indexer, emb_indexer_inv, emb_vals, gt_mappings, features_dict, ontologies_in_alignment = pickle.load(f)
ontologies_in_alignment = [tuple(pair) for pair in ontologies_in_alignment]
flatten = lambda l: [item for sublist in l for item in sublist]

direct_inputs, direct_targets = [], []

def cos_sim(a,b):
    return 1 - spatial.distance.cosine(a,b)

all_fn, all_fp = [], []

threshold_results = {}

def test():
    global batch_size, test_data_t, test_data_f, model, optimizer, emb_indexer_inv, gt_mappings, all_metrics, direct_inputs, direct_targets, threshold_results
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
        
        indices_all = np.random.permutation(len(inputs_all))
        inputs_all = np.array(inputs_all)[indices_all]
        targets_all = np.array(targets_all)[indices_all]
        nodes_all = np.array(nodes_all)[indices_all]

        batch_size = min(batch_size, len(inputs_all))
        num_batches = int(ceil(len(inputs_all)/batch_size))
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx+1) * batch_size

            inputs = inputs_all[batch_start: batch_end]
            targets = targets_all[batch_start: batch_end]
            nodes = nodes_all[batch_start: batch_end]
            
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
                all_results[(ent1, ent2)] = (pred_elem, targets[idx])
        
        direct_targets = [True if el else False for el in direct_targets]
        
        print ("Len (direct inputs): ", len(direct_inputs))
        for idx, direct_input in enumerate(direct_inputs):
            ent1 = emb_indexer_inv[direct_input[0]]
            ent2 = emb_indexer_inv[direct_input[1]]
            sim = cos_sim(emb_vals[direct_input[0]], emb_vals[direct_input[1]])
            all_results[(ent1, ent2)] = (sim, direct_targets[idx])
    return (test_onto, all_results)

def optimize_threshold():
    global batch_size, val_data_t, val_data_f, model, optimizer, emb_indexer_inv, gt_mappings, all_metrics, direct_inputs, direct_targets, threshold_results
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
        
        indices_all = np.random.permutation(len(inputs_all))
        inputs_all = np.array(inputs_all)[indices_all]
        nodes_all = np.array(nodes_all)[indices_all]
        targets_all = np.array(targets_all)[indices_all]

        batch_size = min(batch_size, len(inputs_all))
        num_batches = int(ceil(len(inputs_all)/batch_size))
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx+1) * batch_size

            inputs = inputs_all[batch_start: batch_end]
            nodes = nodes_all[batch_start: batch_end]
            targets = targets_all[batch_start: batch_end]
            
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
                all_results[(ent1, ent2)] = (pred_elem, targets[idx])
        
        direct_targets = [True if el else False for el in direct_targets]
        
        print ("Len (direct inputs): ", len(direct_inputs))
        for idx, direct_input in enumerate(direct_inputs):
            ent1 = emb_indexer_inv[direct_input[0]]
            ent2 = emb_indexer_inv[direct_input[1]]
            sim = cos_sim(emb_vals[direct_input[0]], emb_vals[direct_input[1]])
            all_results[(ent1, ent2)] = (sim, direct_targets[idx])
        
        low_threshold = np.min([el[0] for el in all_results.values()]) - 0.02
        high_threshold = np.max([el[0] for el in all_results.values()]) + 0.02
        threshold = low_threshold
        step = 0.001
        while threshold < high_threshold:
            res = []
            for i,key in enumerate(all_results):
                if all_results[key][0] > threshold:
                    res.append(key)
            fn_list = [(key, all_results[key][0]) for key in val_data_t if key not in set(res) and not is_valid(val_onto, key)]
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
                threshold_results[threshold].append([precision, recall, f1score, f2score, f0_5score])
            else:
                threshold_results[threshold] = [[precision, recall, f1score, f2score, f0_5score]]

            if threshold > 0.98 and not exception:
                step = 0.0001
            else:
                step = 0.001
            threshold += step 

def calculate_performance():
    global final_results
    all_metrics, all_fn, all_fp = [], [], []
    for (test_onto, all_results) in final_results:
        res = []
        for i,key in enumerate(all_results):
            if all_results[key][0] > threshold:
                res.append(key)
        fn_list = [(key, all_results[key][0]) for key in gt_mappings if key not in set(res) and not is_valid(test_onto, key)]
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
        print ("Performance for", test_onto, "is :", (precision, recall, f1score, f2score, f0_5score))
        all_fn.extend(fn_list)
        all_fp.extend(fp_list)
        all_metrics.append((precision, recall, f1score, f2score, f0_5score))
    return all_metrics, all_fn, all_fp


def masked_softmax(inp):
    inp = inp.double()
    mask = ((inp != 0).double() - 1) * 9999  # for -inf
    return (inp + mask).softmax(dim=-1)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__() 
        
        self.features_arr = np.array(list(features_dict.values()))
        self.n_neighbours = self.features_arr.shape[1]
        self.max_paths = self.features_arr.shape[2]
        self.max_pathlen = self.features_arr.shape[3]
        self.embedding_dim = np.array(emb_vals).shape[1]
        
        self.name_embedding = nn.Embedding(len(emb_vals), self.embedding_dim)
        self.name_embedding.load_state_dict({'weight': torch.from_numpy(np.array(emb_vals))})
        self.name_embedding.weight.requires_grad = False

        self.dropout = dropout
        
        self.cosine_sim_layer = nn.CosineSimilarity(dim=1)
        self.output = nn.Linear(2*self.embedding_dim, 300)
        
        self.v = nn.Parameter(torch.DoubleTensor([1/(self.max_pathlen) for i in range(self.max_pathlen)]))
        self.w_rootpath = nn.Parameter(torch.DoubleTensor([0.25]))
        self.w_children = nn.Parameter(torch.DoubleTensor([0.25]))
        self.w_obj_neighbours = nn.Parameter(torch.DoubleTensor([0.25]))
 
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
            node_emb_x = self.name_embedding(nodes[i]) # batch_size * 512
            feature_emb_x = self.name_embedding(features[i]) #  batch_size * 4 * max_paths * max_pathlen * 512

            node_emb_y = self.name_embedding(nodes[1-i]) # batch_size * 512
            feature_emb_y = self.name_embedding(features[1-i]) #  batch_size * 4 * max_paths * max_pathlen * 512
            
            attended_path = feature_emb_x * torch.sum(feature_emb_y, dim=2)
            attended_path = torch.bmm(feature_emb_x.reshape(-1, 1, self.embedding_dim), 
                                         feature_emb_y.reshape(-1, self.embedding_dim, 1))
            attended_path = attended_path.reshape(-1, self.n_neighbours, self.max_paths, self.max_pathlen)
            path_weights = masked_softmax(torch.sum(attended_path, dim=-1))

            feature_emb_x_reshaped = feature_emb_x.reshape(-1, self.max_paths, self.max_pathlen * self.embedding_dim)
            best_path = torch.bmm(path_weights.reshape(-1, 1, self.max_paths), feature_emb_x_reshaped)
            best_path = best_path.squeeze(1).reshape(-1, self.n_neighbours, self.max_pathlen, self.embedding_dim) # batch_size * 4 * max_pathlen * 512

            best_path_reshaped = best_path.permute(0,3,1,2).reshape(-1, self.embedding_dim, self.n_neighbours * self.max_pathlen)
            node_weights = torch.bmm(feature_emb_y.reshape(-1, 1, self.embedding_dim), best_path_reshaped) # batch_size * 4 * max_pathlen
            node_weights = masked_softmax(node_weights.squeeze(1).reshape(-1, self.n_neighbours, self.max_pathlen)) # batch_size * 4 * max_pathlen
            attended_path = node_weights.unsqueeze(-1) * best_path # batch_size * 4 * max_pathlen * 512

            distance_weighted_path = torch.sum((self.v[None,None,:,None] * attended_path), dim=2) # batch_size * 4 * 512

            self.w_data_neighbours = (1-self.w_rootpath-self.w_children-self.w_obj_neighbours)
            context_emb = self.w_rootpath * distance_weighted_path[:,0,:] \
                        + self.w_children * distance_weighted_path[:,1,:] \
                        + self.w_obj_neighbours * distance_weighted_path[:,2,:] \
                        + self.w_data_neighbours * distance_weighted_path[:,3,:]

            contextual_node_emb = torch.cat((node_emb, context_emb), dim=1)
            output_node_emb = self.output(contextual_node_emb)
            results.append(output_node_emb)
        sim = self.cosine_sim_layer(results[0], results[1])
        return sim

def is_valid(test_onto, key):
    return tuple([el.split("#")[0] for el in key]) not in test_onto

def generate_data_neighbourless(elem_tuple):
    return np.vectorize(embedify)(elem_tuple)

def embedify(elem):
    return emb_indexer[elem]

def generate_data(elem_tuple):
    return np.vectorize(embedify)([features_dict[elem] for elem in elem_tuple])

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
    return np.array(inputs), np.array(nodes), np.array(targets)

print("Max number of nodes in a path: " + str(sys.argv[1]))

def count_non_unk(elem):
    return len([l for l in elem if l!="<UNK>"])

features_dict = {elem: features_dict[elem][:,:,:int(sys.argv[1])] for elem in features_dict}

data_items = data.items()
np.random.shuffle(list(data_items))
data = OrderedDict(data_items)

print ("Number of entities:", len(data))

all_metrics = []
final_results = []
for i in list(range(0, len(ontologies_in_alignment), 3)):
    
    test_onto = ontologies_in_alignment[i:i+3]
    
    train_data = {elem: data[elem] for elem in data if tuple([el.split("#")[0] for el in elem]) not in test_onto}

    val_onto = test_onto[:2]
    test_onto = test_onto[2:]
    val_data = {elem: data[elem] for elem in data if tuple([el.split("#")[0] for el in elem]) in val_onto}
    test_data = {elem: data[elem] for elem in data if tuple([el.split("#")[0] for el in elem]) in test_onto}

    print ("Training size:", len(train_data), "Testing size:", len(test_data))
    torch.set_default_dtype(torch.float64)
    
    train_test_split = 0.9

    train_data_t = [key for key in train_data if train_data[key]]
    train_data_f = [key for key in train_data if not train_data[key]]
    train_data_t = np.repeat(train_data_t, ceil(len(train_data_f)/len(train_data_t)), axis=0)
    train_data_t = train_data_t[:len(train_data_f)].tolist()
    #train_data_f = train_data_f[:int(len(train_data_t))]
#     [:int(0.1*(len(train_data) - len(train_data_t)) )]
    np.random.shuffle(train_data_f)
    
    lr = 0.001
    num_epochs = 50
    weight_decay = 0.001
    batch_size = 32
    dropout = 0.3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = SiameseNetwork().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        inputs_pos, nodes_pos, targets_pos = generate_input(train_data_t, 1)
        inputs_neg, nodes_neg, targets_neg = generate_input(train_data_f, 0)
        inputs_all = list(inputs_pos) + list(inputs_neg)
        targets_all = list(targets_pos) + list(targets_neg)
        nodes_all = list(nodes_pos) + list(nodes_neg)
        
        indices_all = np.random.permutation(len(inputs_all))
        inputs_all = np.array(inputs_all)[indices_all]
        targets_all = np.array(targets_all)[indices_all]
        nodes_all = np.array(nodes_all)[indices_all]

        batch_size = min(batch_size, len(inputs_all))
        num_batches = int(ceil(len(inputs_all)/batch_size))

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx+1) * batch_size
            
            inputs = inputs_all[batch_start: batch_end]
            targets = targets_all[batch_start: batch_end]
            nodes = nodes_all[batch_start: batch_end]
            
            inp_elems = torch.LongTensor(inputs).to(device)
            node_elems = torch.LongTensor(nodes).to(device)
            targ_elems = torch.DoubleTensor(targets).to(device)

            optimizer.zero_grad()
            outputs = model(node_elems, inp_elems)

            loss = F.mse_loss(outputs, targ_elems)
            loss.backward()
            optimizer.step()

            if batch_idx%5000 == 0:
                print ("Epoch: {} Idx: {} Loss: {}".format(epoch, batch_idx, loss.item()))

    model.eval()
    
    val_data_t = [key for key in val_data if val_data[key]]
    val_data_f = [key for key in val_data if not val_data[key]]
    
    optimize_threshold()

    test_data_t = [key for key in test_data if test_data[key]]
    test_data_f = [key for key in test_data if not test_data[key]]

    final_results.append(test())

    sys.stdout.flush()

threshold_results_mean = {el: np.mean(threshold_results[el], axis=0) for el in threshold_results}    
threshold = max(threshold_results_mean.keys(), key=(lambda key: threshold_results_mean[key][2]))

all_metrics, all_fn, all_fp = calculate_performance()

print ("Final Results: " + str(np.mean(all_metrics, axis=0)))
print ("Threshold: ", threshold)

f1 = open(sys.argv[-1], "wb")
pickle.dump([all_fn, all_fp], f1)