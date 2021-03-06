import os, itertools, time, pickle, operator, random
import subprocess
from xml.dom import minidom
from collections import Counter, OrderedDict
from operator import itemgetter
from scipy import spatial
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import re, sys, glob
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from math import ceil, exp
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

f = open(sys.argv[1], "rb")
data_ent, data_prop, aml_data_ent, aml_data_prop, emb_indexer, emb_indexer_inv, emb_vals, neighbours_dicts, neighbours_dicts_prop, ontologies_in_alignment = pickle.load(f)
max_types = 4
max_paths = int(sys.argv[2])
max_pathlen = int(sys.argv[3])
threshold = float(sys.argv[4])
fn_percentage1 = float(sys.argv[5])
fn_percentage2 = float(sys.argv[6])

aml_data_ent = {key: float(aml_data_ent[key])>=threshold for key in aml_data_ent}
aml_data_prop = {key: float(aml_data_prop[key])>=threshold for key in aml_data_prop}
data_ent = {key: float(data_ent[key])>0 for key in data_ent}
data_prop = {key: float(data_prop[key])>0 for key in data_prop}

flatten = lambda l: [item for sublist in l for item in sublist]
ontologies_in_alignment = [tuple(pair) for pair in ontologies_in_alignment]

direct_inputs, direct_targets = [], []

def cos_sim(a,b):
    return 1 - spatial.distance.cosine(a,b)

all_fn, all_fp = [], []

threshold_results = {}

def test():
    global batch_size, test_data_t, test_data_f, model, optimizer, emb_indexer_inv, all_metrics, direct_inputs, direct_targets, threshold_results
    all_results = OrderedDict()    
    direct_inputs, direct_targets = [], []
    with torch.no_grad():
        all_pred = []
        
        np.random.shuffle(test_data_t)
        np.random.shuffle(test_data_f)

        inputs_pos, nodes_pos, targets_pos = generate_input(test_data_t, 1, neighbours_dicts)
        inputs_neg, nodes_neg, targets_neg = generate_input(test_data_f, 0, neighbours_dicts)

        inputs_pos_prop, nodes_pos_prop, targets_pos_prop = generate_input(test_data_t_prop, 1, neighbours_dicts_prop)
        inputs_neg_prop, nodes_neg_prop, targets_neg_prop = generate_input(test_data_f_prop, 0, neighbours_dicts_prop)
        
        inputs_all = list(inputs_pos) + list(inputs_neg)
        targets_all = list(targets_pos) + list(targets_neg)
        nodes_all = list(nodes_pos) + list(nodes_neg)
        
        all_inp = list(zip(inputs_all, targets_all, nodes_all))
        all_inp_shuffled = random.sample(all_inp, len(all_inp))
        inputs_all, targets_all, nodes_all = list(zip(*all_inp_shuffled))

        inputs_all_prop = list(inputs_pos_prop) + list(inputs_neg_prop)
        targets_all_prop = list(targets_pos_prop) + list(targets_neg_prop)
        nodes_all_prop = list(nodes_pos_prop) + list(nodes_neg_prop)

        all_inp_prop = list(zip(inputs_all_prop, targets_all_prop, nodes_all_prop))
        all_inp_shuffled_prop = random.sample(all_inp_prop, len(all_inp_prop))
        inputs_all_prop, targets_all_prop, nodes_all_prop = list(zip(*all_inp_shuffled_prop))

        batch_size = min(batch_size, len(inputs_all))
        num_batches = int(ceil(len(inputs_all)/batch_size))
        batch_size_prop = int(ceil(len(inputs_all_prop)/num_batches))

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx+1) * batch_size
            batch_start_prop = batch_idx * batch_size_prop
            batch_end_prop = (batch_idx+1) * batch_size_prop

            inputs = np.array(to_feature(inputs_all[batch_start: batch_end]))
            targets = np.array(targets_all[batch_start: batch_end])
            nodes = np.array(nodes_all[batch_start: batch_end])
            
            inputs_prop = np.array(pad_prop(inputs_all_prop[batch_start_prop: batch_end_prop]))
            targets_prop = np.array(targets_all_prop[batch_start_prop: batch_end_prop])
            nodes_prop = np.array(nodes_all_prop[batch_start_prop: batch_end_prop])
            
            targets = np.concatenate((targets, targets_prop), axis=0)

            inp_elems = torch.LongTensor(inputs).to(device)
            node_elems = torch.LongTensor(nodes).to(device)
            targ_elems = torch.DoubleTensor(targets)

            inp_props = torch.LongTensor(inputs_prop).to(device)
            node_props = torch.LongTensor(nodes_prop).to(device)

            outputs = model(node_elems, inp_elems, node_props, inp_props)
            outputs = [el.item() for el in outputs]
            targets = [True if el.item() else False for el in targets]

            for idx, pred_elem in enumerate(outputs):
                if idx < len(nodes):
                    ent1 = emb_indexer_inv[nodes[idx][0]]
                    ent2 = emb_indexer_inv[nodes[idx][1]]
                else:
                    ent1 = emb_indexer_inv[nodes_prop[idx-len(nodes)][0]]
                    ent2 = emb_indexer_inv[nodes_prop[idx-len(nodes)][1]]
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
    global batch_size, val_data_t_ent, val_data_f_ent, model, optimizer, emb_indexer_inv, all_metrics, direct_inputs, direct_targets, threshold_results
    all_results = OrderedDict()
    direct_inputs, direct_targets = [], []
    with torch.no_grad():
        all_pred = []
        
        np.random.shuffle(val_data_t_ent)
        np.random.shuffle(val_data_f_ent)

        inputs_pos, nodes_pos, targets_pos = generate_input(val_data_t_ent, 1, neighbours_dicts)
        inputs_neg, nodes_neg, targets_neg = generate_input(val_data_f_ent, 0, neighbours_dicts)
        inputs_pos_prop, nodes_pos_prop, targets_pos_prop = generate_input(val_data_t_prop, 1, neighbours_dicts_prop)
        inputs_neg_prop, nodes_neg_prop, targets_neg_prop = generate_input(val_data_f_prop, 0, neighbours_dicts_prop)

        inputs_all = list(inputs_pos) + list(inputs_neg)
        targets_all = list(targets_pos) + list(targets_neg)
        nodes_all = list(nodes_pos) + list(nodes_neg)
        
        all_inp = list(zip(inputs_all, targets_all, nodes_all))
        all_inp_shuffled = random.sample(all_inp, len(all_inp))
        inputs_all, targets_all, nodes_all = list(zip(*all_inp_shuffled))

        inputs_all_prop = list(inputs_pos_prop) + list(inputs_neg_prop)
        targets_all_prop = list(targets_pos_prop) + list(targets_neg_prop)
        nodes_all_prop = list(nodes_pos_prop) + list(nodes_neg_prop)
        
        all_inp_prop = list(zip(inputs_all_prop, targets_all_prop, nodes_all_prop))
        all_inp_shuffled_prop = random.sample(all_inp_prop, len(all_inp_prop))
        inputs_all_prop, targets_all_prop, nodes_all_prop = list(zip(*all_inp_shuffled_prop))

        batch_size = min(batch_size, len(inputs_all))
        num_batches = int(ceil(len(inputs_all)/batch_size))
        batch_size_prop = int(ceil(len(inputs_all_prop)/num_batches))

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx+1) * batch_size
            batch_start_prop = batch_idx * batch_size_prop
            batch_end_prop = (batch_idx+1) * batch_size_prop

            inputs = np.array(to_feature(inputs_all[batch_start: batch_end]))
            targets = np.array(targets_all[batch_start: batch_end])
            nodes = np.array(nodes_all[batch_start: batch_end])

            inputs_prop = np.array(pad_prop(inputs_all_prop[batch_start_prop: batch_end_prop]))
            targets_prop = np.array(targets_all_prop[batch_start_prop: batch_end_prop])
            nodes_prop = np.array(nodes_all_prop[batch_start_prop: batch_end_prop])
            
            targets = np.concatenate((targets, targets_prop), axis=0)

            inp_elems = torch.LongTensor(inputs).to(device)
            node_elems = torch.LongTensor(nodes).to(device)
            targ_elems = torch.DoubleTensor(targets).to(device)

            inp_props = torch.LongTensor(inputs_prop).to(device)
            node_props = torch.LongTensor(nodes_prop).to(device)

            outputs = model(node_elems, inp_elems, node_props, inp_props)
            outputs = [el.item() for el in outputs]
            targets = [True if el.item() else False for el in targets]

            for idx, pred_elem in enumerate(outputs):
                if idx < len(nodes):
                    ent1 = emb_indexer_inv[nodes[idx][0]]
                    ent2 = emb_indexer_inv[nodes[idx][1]]
                else:
                    ent1 = emb_indexer_inv[nodes_prop[idx-len(nodes)][0]]
                    ent2 = emb_indexer_inv[nodes_prop[idx-len(nodes)][1]]
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
        val_data_t_tot = [tuple(pair) for pair in np.concatenate((val_data_t_ent, val_data_t_prop), axis=0)]
        while threshold < high_threshold:
            res = []
            for i,key in enumerate(all_results):
                if all_results[key][0] > threshold:
                    res.append(key)
            s = set(res)
            fn_list = [(key, all_results[key][0]) for key in val_data_t_tot if key not in s and not is_valid(val_onto, key)]
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

            threshold += step 

def calculate_performance():
    global final_results
    all_metrics, all_fn, all_fp = [], [], []
    for (test_onto, all_results) in final_results:
        res = []
        for i,key in enumerate(all_results):
            if all_results[key][0] > threshold:
                res.append(key)
        s = set(res)
        fn_list = [(key, all_results[key][0]) for key in test_data_t if key not in s and not is_valid(test_onto, key)]
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
    def __init__(self, emb_vals, threshold=0.9):
        super().__init__() 
        
        self.n_neighbours = max_types
        self.max_paths = max_paths
        self.max_pathlen = max_pathlen
        self.embedding_dim = np.array(emb_vals).shape[1]
        
        self.threshold = threshold

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

        self.prop_weight = nn.Parameter(torch.DoubleTensor([0.33]))
        self.domain_weight = nn.Parameter(torch.DoubleTensor([0.33]))
 
    def forward(self, nodes, features, prop_nodes, prop_features):
        '''
        Arguments:
            - nodes: batch_size * 2
            - features: batch_size * 2 * 4 * max_paths * max_pathlen
            - prop_nodes: batch_size * 2
            - prop_features: batch_size * 2 * 3 * max_prop_len
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
            path_weights = torch.sum(path_weights, dim=-1)
            best_path_indices = torch.max(path_weights, dim=-1)[1][(..., ) + (None, ) * 3]
            best_path_indices = best_path_indices.expand(-1, -1, -1, self.max_pathlen,  self.embedding_dim)
            best_path = torch.gather(feature_emb, 2, best_path_indices).squeeze(2) # batch_size * 4 * max_pathlen * 512
            # Another way: 
            # path_weights = masked_softmax(path_weights)
            # best_path = torch.sum(path_weights[:, :, :, None, None] * feature_emb, dim=2)

            best_path_reshaped = best_path.permute(0,3,1,2).reshape(-1, self.embedding_dim, self.n_neighbours * self.max_pathlen)
            node_weights = torch.bmm(node_emb.unsqueeze(1), best_path_reshaped) # batch_size * 4 * max_pathlen
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
        sim_ent = self.cosine_sim_layer(results[0], results[1])
        if prop_nodes.nelement() != 0:
            # Calculate prop sum
            aggregated_prop_sum = torch.sum(self.name_embedding(prop_features), dim=-2)
            sim_prop = self.prop_weight * self.cosine_sim_layer(aggregated_prop_sum[:,0,0], aggregated_prop_sum[:,1,0])
            sim_prop += self.domain_weight * self.cosine_sim_layer(aggregated_prop_sum[:,0,1], aggregated_prop_sum[:,1,1])
            sim_prop += (1-self.prop_weight-self.domain_weight) * self.cosine_sim_layer(aggregated_prop_sum[:,0,2], aggregated_prop_sum[:,1,2])

            return torch.cat((sim_ent, sim_prop))

        return sim_ent

def is_valid(test_onto, key):
    return tuple([el.split("#")[0] for el in key]) not in test_onto

def generate_data_neighbourless(elem_tuple):
    return [emb_indexer[elem] for elem in elem_tuple]

def embedify(seq, emb_indexer):
    for item in seq:
        if isinstance(item, list):
            yield list(embedify(item, emb_indexer))
        else:
            yield emb_indexer[item]

def generate_data(elem_tuple, neighbours_dicts):
    return list(embedify([neighbours_dicts[elem] for elem in elem_tuple], emb_indexer))

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

def pad_prop(inputs):
    inputs_padded = [[[elem + [0 for i in range(max_prop_len - len(elem))]
                         for elem in prop]
                    for prop in elem_pair]
                for elem_pair in inputs]
    return inputs_padded

def generate_input(elems, target, neighbours_dicts):
    inputs, targets, nodes = [], [], []
    global direct_inputs, direct_targets
    for elem in list(elems):
        try:
            inputs.append(generate_data(elem, neighbours_dicts))
            nodes.append(generate_data_neighbourless(elem))
            targets.append(target)
        except KeyError as e:
            direct_inputs.append(generate_data_neighbourless(elem))
            direct_targets.append(target)
        except Exception as e:
            print (e)
            raise
    return inputs, nodes, targets

print("Max number of nodes in a path: " + str(sys.argv[1]))

def count_non_unk(elem):
    return len([l for l in elem if l!="<UNK>"])


torch.set_default_dtype(torch.float64)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

data_items = aml_data_ent.items()
np.random.shuffle(list(data_items))
aml_data_ent = OrderedDict(data_items)

print ("Number of entities:", len(aml_data_ent))
lr = 0.001
num_epochs = 50
weight_decay = 0.001
batch_size = 32
dropout = 0.3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
all_metrics = []
final_results = []

model = SiameseNetwork(emb_vals).to(device)
print (model.threshold)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

train_data_t = [key for key in aml_data_ent if aml_data_ent[key]]
train_data_f = [key for key in aml_data_ent if not aml_data_ent[key]]
fn_len = int(fn_percentage1*len(train_data_f))
train_data_f = train_data_f[:fn_len]
train_data_t = np.repeat(train_data_t, ceil(len(train_data_f)/len(train_data_t)), axis=0)
train_data_t = train_data_t[:len(train_data_f)].tolist()
np.random.shuffle(train_data_f)

train_data_t_prop = [key for key in aml_data_prop if aml_data_prop[key]]
train_data_f_prop = [key for key in aml_data_prop if not aml_data_prop[key]]
fn_len = int(fn_percentage2*len(train_data_f_prop))
train_data_f_prop = train_data_f_prop[:fn_len]
train_data_t_prop = np.repeat(train_data_t_prop, ceil(len(train_data_f_prop)/len(train_data_t_prop)), axis=0)
train_data_t_prop = train_data_t_prop[:len(train_data_f_prop)].tolist()
np.random.shuffle(train_data_f_prop)

for epoch in range(num_epochs):
    inputs_pos, nodes_pos, targets_pos = generate_input(train_data_t, 1, neighbours_dicts)
    inputs_neg, nodes_neg, targets_neg = generate_input(train_data_f, 0, neighbours_dicts)
    inputs_pos_prop, nodes_pos_prop, targets_pos_prop = generate_input(train_data_t_prop, 1, neighbours_dicts_prop)
    inputs_neg_prop, nodes_neg_prop, targets_neg_prop = generate_input(train_data_f_prop, 0, neighbours_dicts_prop)
    
    inputs_all = list(inputs_pos) + list(inputs_neg)
    targets_all = list(targets_pos) + list(targets_neg)
    nodes_all = list(nodes_pos) + list(nodes_neg)
    
    all_inp = list(zip(inputs_all, targets_all, nodes_all))
    all_inp_shuffled = random.sample(all_inp, len(all_inp))
    inputs_all, targets_all, nodes_all = list(zip(*all_inp_shuffled))

    inputs_all_prop = list(inputs_pos_prop) + list(inputs_neg_prop)
    targets_all_prop = list(targets_pos_prop) + list(targets_neg_prop)
    nodes_all_prop = list(nodes_pos_prop) + list(nodes_neg_prop)

    max_prop_len = np.max([[[len(elem) for elem in prop] for prop in elem_pair] 
        for elem_pair in inputs_all_prop])
    
    all_inp_prop = list(zip(inputs_all_prop, targets_all_prop, nodes_all_prop))
    all_inp_shuffled_prop = random.sample(all_inp_prop, len(all_inp_prop))
    inputs_all_prop, targets_all_prop, nodes_all_prop = list(zip(*all_inp_shuffled_prop))
    
    print ("Inputs prop length: ", len(inputs_all_prop))
    batch_size = min(batch_size, len(inputs_all))
    num_batches = int(ceil(len(inputs_all)/batch_size))
    batch_size_prop = int(ceil(len(inputs_all_prop)/num_batches))

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx+1) * batch_size
        batch_start_prop = batch_idx * batch_size_prop
        batch_end_prop = (batch_idx+1) * batch_size_prop

        inputs = np.array(to_feature(inputs_all[batch_start: batch_end]))
        targets = np.array(targets_all[batch_start: batch_end])
        nodes = np.array(nodes_all[batch_start: batch_end])

        inputs_prop = np.array(pad_prop(inputs_all_prop[batch_start_prop: batch_end_prop]))
        targets_prop = np.array(targets_all_prop[batch_start_prop: batch_end_prop])
        nodes_prop = np.array(nodes_all_prop[batch_start_prop: batch_end_prop])
        
        targets = np.concatenate((targets, targets_prop), axis=0)

        inp_elems = torch.LongTensor(inputs).to(device)
        node_elems = torch.LongTensor(nodes).to(device)
        targ_elems = torch.DoubleTensor(targets).to(device)

        inp_props = torch.LongTensor(inputs_prop).to(device)
        node_props = torch.LongTensor(nodes_prop).to(device)

        optimizer.zero_grad()
        outputs = model(node_elems, inp_elems, node_props, inp_props)

        loss = F.mse_loss(outputs, targ_elems)
        loss.backward()
        optimizer.step()

        if batch_idx%50 == 0:
            print ("Epoch: {} Idx: {} Loss: {}".format(epoch, batch_idx, loss.item()))


val_data_t_ent = [key for key in aml_data_ent if aml_data_ent[key]]
val_data_f_ent = [key for key in aml_data_ent if not aml_data_ent[key]]
val_data_t_prop = [key for key in aml_data_prop if aml_data_prop[key]]
val_data_f_prop = [key for key in aml_data_prop if not aml_data_prop[key]]

val_onto = ontologies_in_alignment

optimize_threshold()

threshold_results_mean = {el: np.mean(threshold_results[el], axis=0) for el in threshold_results}    
threshold = max(threshold_results_mean.keys(), key=(lambda key: threshold_results_mean[key][2]))

model.threshold = threshold

# def check_best_performance():
#     output_file = "Results/Output_att*" + "_".join(sys.argv[6].split("/")[1].split("_")[:4]) + ".txt"
#     results_lines = [[l for l in open(file).read().split("\n") if "Final Results:" in l] for file in glob.glob(output_file)]
#     results_lines = [line[0] for line in results_lines if line]
#     results_lines = [line.split("[")[1].split("]")[0].split(" ") for line in results_lines]
#     results_lines = [float([value for value in line if value][2]) for line in results_lines]
#     return max(results_lines)
model.eval()

test_onto = ontologies_in_alignment
test_data_ent = {elem: data_ent[elem] for elem in data_ent if tuple([el.split("#")[0] for el in elem]) in test_onto}
test_data_prop = {elem: data_prop[elem] for elem in data_prop if tuple([el.split("#")[0] for el in elem]) in test_onto}

test_data_t = [key for key in test_data_ent if test_data_ent[key]]
test_data_f = [key for key in test_data_ent if not test_data_ent[key]]

test_data_t_prop = [key for key in test_data_prop if test_data_prop[key]]
test_data_f_prop = [key for key in test_data_prop if not test_data_prop[key]]


final_results.append(test())

test_data_t += test_data_t_prop
test_data_f += test_data_f_prop

all_metrics, all_fn, all_fp = calculate_performance()
final_results = np.mean(all_metrics, axis=0)

# if float(final_results[2]) > check_best_performance():
# Remove unneccessary models
# _ = [os.remove(file) for file in glob.glob("_".join(sys.argv[6].split("_")[:4]) + "*.pt")]
# Remove unneccessary error files
# _ = [os.remove(file) for file in glob.glob("_".join(sys.argv[5].split("_")[:5]) + "*.pkl")]
# Save model
torch.save(model.state_dict(), sys.argv[7])

print ("Final Results: ", final_results)
print ("Threshold: ", threshold)

