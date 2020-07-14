import os, itertools, time, pickle
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

f = open("data.pkl", "rb")
data, emb_indexer, emb_indexer_inv, emb_vals, gt_mappings  = pickle.load(f)

ontologies_in_alignment = [l.split(".")[0].split("-") for l in os.listdir("reference-alignment/")]
flatten = lambda l: [item for sublist in l for item in sublist]

ind_test, inp_test1, inp_test2 = None, None, None

def write(statement):
    op_file = open("Logs", "a+")
    op_file.write("\n" + str(statement) + "\n")
    op_file.close()

class Ontology():
    def __init__(self, ontology):
        self.ontology = ontology
        self.ontology_obj = minidom.parse(ontology)
        self.root = self.ontology_obj.documentElement
        self.subclasses = self.parse_subclasses()
        self.object_properties = self.parse_object_properties()
        self.data_properties = self.parse_data_properties()
        self.triples = self.parse_triples()
        self.classes = self.parse_classes()
    
    def get_child_node(self, element, tag):
        return [e for e in element._get_childNodes() if type(e)==minidom.Element and e._get_tagName() == tag]
        
    def has_attribute_value(self, element, attribute, value):
        return True if element.getAttribute(attribute).split("#")[-1] == value else False
    
    def get_subclass_triples(self):
        return [(b,a,"subclass_of") for (a,b) in self.get_subclasses()]
    
    def parse_triples(self, union_flag=0, subclass_of=True):
        obj_props = self.object_properties
        data_props = self.data_properties
        props = obj_props + data_props
        all_triples = []
        for prop in props:
            domain_children = self.get_child_node(prop, "rdfs:domain")
            range_children = self.get_child_node(prop, "rdfs:range")
            domain_prop = self.filter_null([self.extract_ID(el) for el in domain_children])
            range_prop = self.filter_null([self.extract_ID(el) for el in range_children])
            if not domain_children or not range_children:
                continue
            if not domain_prop:
                domain_prop = self.filter_null([self.extract_ID(el) for el in domain_children[0].getElementsByTagName("owl:Class")])
            if not range_prop:
                range_prop = self.filter_null([self.extract_ID(el) for el in range_children[0].getElementsByTagName("owl:Class")])
            if domain_prop and range_prop:
                if union_flag == 0:
                    all_triples.extend([(el[0], el[1], self.extract_ID(prop)) for el in list(itertools.product(domain_prop, range_prop))])
                else:
                    all_triples.append(("###".join(domain_prop), "###".join(range_prop), self.extract_ID(prop)))
        if subclass_of:
            all_triples.extend(self.get_subclass_triples())
        return list(set(all_triples))
    
    def get_triples(self, union_flag=0, subclass_of=True, include_inv=True):
        return self.parse_triples(union_flag, subclass_of)

    def parse_subclasses(self, union_flag=0):
        subclasses = self.root.getElementsByTagName("rdfs:subClassOf")
        subclass_pairs = []
        for el in subclasses:
            inline_subclasses = self.extract_ID(el)
            if inline_subclasses:
                subclass_pairs.append((el, el.parentNode))
            else:
                level1_class = self.get_child_node(el, "owl:Class")
                if not level1_class:
                    continue
                if self.extract_ID(level1_class[0]):
                    subclass_pairs.append((level1_class[0], el.parentNode))
                else:
                    level2classes = level1_class[0].getElementsByTagName("owl:Class")
                    
                    subclass_pairs.extend([(elem, el.parentNode) for elem in level2classes if self.extract_ID(elem)])
        return subclass_pairs
        
    def get_subclasses(self):
        return [(self.extract_ID(a), self.extract_ID(b)) for (a,b) in self.subclasses]
    
    def filter_null(self, data):
        return [el for el in data if el]
    
    def extract_ID(self, element):
        element_id = element.getAttribute("rdf:ID") or element.getAttribute("rdf:resource") or element.getAttribute("rdf:about")
        return element_id.split("#")[-1]
    
    def parse_classes(self):
        class_elems = [self.extract_ID(el) for el in self.root.getElementsByTagName("owl:Class")]
        subclass_classes = list(set(flatten([el[:-1] for el in self.triples])))
        return list(set(self.filter_null(class_elems + subclass_classes)))
    
    def get_classes(self):
        return self.classes
    
    def get_entities(self):
        entities = [self.extract_ID(el) for el in self.root.getElementsByTagName("owl:Class")]
        return list(set(self.filter_null(entities)))

    def parse_data_properties(self):
        data_properties = [el for el in self.get_child_node(self.root, 'owl:DatatypeProperty')]
        fn_data_properties = [el for el in self.get_child_node(self.root, 'owl:FunctionalProperty') if el]
        fn_data_properties = [el for el in fn_data_properties if type(el)==minidom.Element and 
            [el for el in self.get_child_node(el, "rdf:type") if 
             self.has_attribute_value(el, "rdf:resource", "DatatypeProperty")]]
        inv_fn_data_properties = [el for el in self.get_child_node(self.root, 'owl:InverseFunctionalProperty') if el]
        inv_fn_data_properties = [el for el in inv_fn_data_properties if type(el)==minidom.Element and 
            [el for el in self.get_child_node(el, "rdf:type") if 
             self.has_attribute_value(el, "rdf:resource", "DatatypeProperty")]]
        return data_properties + fn_data_properties + inv_fn_data_properties
        
    def parse_object_properties(self):
        obj_properties = [el for el in self.get_child_node(self.root, 'owl:ObjectProperty')]
        fn_obj_properties = [el for el in self.get_child_node(self.root, 'owl:FunctionalProperty') if el]
        fn_obj_properties = [el for el in fn_obj_properties if type(el)==minidom.Element and 
            [el for el in self.get_child_node(el, "rdf:type") if 
             self.has_attribute_value(el, "rdf:resource", "ObjectProperty")]]
        inv_fn_obj_properties = [el for el in self.get_child_node(self.root, 'owl:InverseFunctionalProperty') if el]
        inv_fn_obj_properties = [el for el in inv_fn_obj_properties if type(el)==minidom.Element and 
            [el for el in self.get_child_node(el, "rdf:type") if 
             self.has_attribute_value(el, "rdf:resource", "ObjectProperty")]]
        return obj_properties + fn_obj_properties + inv_fn_obj_properties
    
    def get_object_properties(self):
        obj_props = [self.extract_ID(el) for el in self.object_properties]
        return list(set(self.filter_null(obj_props)))
    
    def get_data_properties(self):
        data_props = [self.extract_ID(el) for el in self.data_properties]
        return list(set(self.filter_null(data_props)))

gt_mappings_filt, direct_inputs, direct_targets = [], [], []

def cos_sim(a,b):
    return 1 - spatial.distance.cosine(a,b)

def greedy_matching():
    global batch_size, test_data_t, test_data_f, model, optimizer, emb_indexer_inv, gt_mappings, all_metrics, gt_mappings_filt, direct_inputs, direct_targets
    all_results = OrderedDict()
    gt_mappings_filt, direct_inputs, direct_targets = [], [], []
    with torch.no_grad():
        all_pred = []
        batch_size = min(batch_size, len(test_data_t))
        num_batches = int(ceil(len(test_data_t)/batch_size))
        batch_size_f = int(ceil(len(test_data_f)/num_batches))
        
        np.random.shuffle(test_data_t)
        np.random.shuffle(test_data_f)

        inputs_pos, targets_pos = generate_input(test_data_t, 1)
        inputs_neg, targets_neg = generate_input(test_data_f, 0)

        indices_pos = np.random.permutation(len(inputs_pos))
        indices_neg = np.random.permutation(len(inputs_neg))

        inputs_pos, targets_pos = inputs_pos[indices_pos], targets_pos[indices_pos]
        inputs_neg, targets_neg = inputs_neg[indices_neg], targets_neg[indices_neg]

        # gt_mappings_filt = [el for el in gt_mappings if el in test_data_t]
        print ("len(gt_mappings_filt)", len(gt_mappings_filt), "len(gt_mappings)", len([el for el in gt_mappings if el in test_data_t]))
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx+1) * batch_size

            batch_start_f = batch_idx * batch_size_f
            batch_end_f = (batch_idx+1) * batch_size_f

            inputs = np.concatenate((inputs_pos[batch_start: batch_end], inputs_neg[batch_start_f: batch_end_f]))
            targets = np.concatenate((targets_pos[batch_start: batch_end], targets_neg[batch_start_f: batch_end_f]))
            
            inp = inputs.transpose(1,0,2)
            
            nonzero_elems = np.count_nonzero(inp, axis=-1) - 1

            inp_elems = torch.LongTensor(inputs).to(device)
            seq_lens = torch.LongTensor(nonzero_elems.T).to(device)
            targ_elems = torch.DoubleTensor(targets)

            outputs = model(inp_elems, seq_lens)
            outputs = [el.item() for el in outputs]
            #outputs /= torch.sum(outputs, dim=1).view(-1, 1)
            #outputs = [(1-el[1].item()) for el in outputs]
            

            targets = [True if el.item() else False for el in targets]
#             print (inputs)
            for idx, pred_elem in enumerate(outputs):
                ent1 = emb_indexer_inv[inp[0][idx][0]]
                ent2 = emb_indexer_inv[inp[1][idx][0]]
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
        #all_results = OrderedDict(sorted(all_results.items(), key=lambda x: x[0], reverse=True))
        #filtered_results = dict()
        
        #entities_to_assign = set([el[0] for el in list(all_results.keys())])
        #for pair in all_results:
        #    if pair[0] in entities_to_assign:
        #        filtered_results[pair] = all_results[pair]
        #        entities_to_assign.remove(pair[0])
                
        #entities_to_assign = set([el[1] for el in list(all_results.keys())])
        #for pair in all_results:
        #    if pair[1] in entities_to_assign:
        #        filtered_results[pair] = all_results[pair]
        #        entities_to_assign.remove(pair[1])        

        #filtered_results = OrderedDict(sorted(filtered_results.items(), key=lambda x: x[1][0], reverse=True))
        
        optimum_metrics, opt_threshold = [-1000 for i in range(5)], -1000
        low_threshold = np.min([el[0] for el in all_results.values()]) - 0.02
        high_threshold = np.max([el[0] for el in all_results.values()]) + 0.02
        #low_threshold, high_threshold = 0.9, 1.02
        threshold = low_threshold
        step = 0.001
        while threshold < high_threshold:
            res = []
            for i,key in enumerate(all_results):
                if all_results[key][0] > threshold:
                    res.append(key)
            fn_list = [key for key in gt_mappings if key not in set(res) and not is_valid(test_onto, key)]
            fp_list = [elem for elem in res if not all_results[elem][1]]
            tp_list = [elem for elem in res if all_results[elem][1]]
            
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
            print ("Threshold: ", threshold, precision, recall, f1score, f2score, f0_5score)

            if f1score > optimum_metrics[2]:
                optimum_metrics = [precision, recall, f1score, f2score, f0_5score]
                opt_threshold = threshold
            
            if threshold > 0.98 and not exception:
                step = 0.0001
            else:
                step = 0.001
            print (step, threshold, exception)
            threshold += step 
        print ("Precision: {} Recall: {} F1-Score: {} F2-Score: {} F0.5-Score: {}".format(*optimum_metrics))
        if optimum_metrics[2] != -1000:
            all_metrics.append((opt_threshold, optimum_metrics))
    return all_results

def write(elem):
    f = open("Logs", "a+")
    if type(elem) == list or type(elem) == tuple:
        string = str("\n".join([str(s) for s in elem]))
    else:
        string = str(elem)
    f.write("\n"+string)
    f.close()
    
inputs3, results3 = None, None

def masked_softmax(inp):
    inp = inp.double()
    mask = ((inp != 0).double() - 1) * 9999  # for -inf
    return (inp + mask).softmax(dim=-1)

def normalize(inp):
    return inp/torch.norm(inp, dim=-1)[:, None]

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers):
        super().__init__() 
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2
        
        self.name_embedding = nn.Embedding(len(emb_vals), self.embedding_dim)
        self.name_embedding.load_state_dict({'weight': torch.from_numpy(np.array(emb_vals))})
        self.name_embedding.weight.requires_grad = False

        self.dropout = dropout
        
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.num_layers, bidirectional=True, batch_first=True)
        self.cosine_sim_layer = nn.CosineSimilarity(dim=1)
        self.output = nn.Linear(1024, 300)
        self.bilinear = nn.Bilinear(self.hidden_dim, self.hidden_dim, 1)

    def forward(self, inputs, seq_lens):
        results = []
        inputs = inputs.permute(1,0,2)
        seq_lens = seq_lens.T
        #print ("input len: {} seq len: {}, rev len: {}".format(inputs.shape, seq_lens.shape, rev_indices.shape))
        for i in range(2):
            x = self.name_embedding(inputs[i])
            
            node = x.permute(1,0,2)[:1].permute(1,0,2) # 3993 * 1 * 512
            neighbours = x.permute(1,0,2)[1:].permute(1,0,2) # 3993 * 9 * 512
            
            att_weights = torch.bmm(neighbours, node.permute(0, 2, 1)).squeeze()
            att_weights = masked_softmax(att_weights).unsqueeze(-1)
            context = torch.mean(att_weights * neighbours, dim=1)
            
            x = torch.cat((node.reshape(-1, 512), context.reshape(-1, 512)), dim=1)
            x = self.output(x)
            results.append(x)
        #global inputs3, results3
        #results3 = results
        #inputs3 = inputs
        #x = self.layer1(results[0], results[1])
        #x = F.log_softmax(x)
        x = self.cosine_sim_layer(results[0], results[1])
        return x


def get_one_hop_neighbours(ont, K=1):
    ont_obj = Ontology("conference_ontologies/" + ont + ".owl")
    triples = ont_obj.get_triples()
    entities = [(a,b) for (a,b,c) in triples]
    neighbours_dict = {elem: [elem] for elem in list(set(flatten(entities)))}
    for e1, e2 in entities:
        neighbours_dict[e1].append(e2)
        neighbours_dict[e2].append(e1)
    
    prop_triples = ont_obj.get_triples(subclass_of=False)
    neighbours_dict_props = {c: [c] for a,b,c in prop_triples}
    for e1, e2, p in prop_triples:
        neighbours_dict_props[p].extend([e1, e2])

    #neighbours_dict = {**neighbours_dict, **neighbours_dict_props}
    
    # for elem in ont_obj.get_entities() + ont_obj.get_object_properties() + ont_obj.get_data_properties():
    #     if elem not in neighbours_dict:
    #         neighbours_dict[elem] = [elem]

    neighbours_dict = {el: neighbours_dict[el][:1] + sorted(list(set(neighbours_dict[el][1:])))
                       for el in neighbours_dict}
    neighbours_dict = {el: neighbours_dict[el][:int(sys.argv[1])] for el in neighbours_dict if len( neighbours_dict[el]) > int(sys.argv[2])}
    neighbours_dict = {ont + "#" + el: [ont + "#" + e for e in neighbours_dict[el]] for el in neighbours_dict}
    return neighbours_dict

def is_valid(test_onto, key):
    return tuple([el.split("#")[0] for el in key]) not in test_onto

def generate_data_neighbourless(elem_tuple):
    op = np.array([emb_indexer[elem] for elem in elem_tuple])
    return op

def generate_data(elem_tuple):
    op = np.array([[emb_indexer[el] for el in neighbours_dicts[elem.split("#")[0]][elem]] for elem in elem_tuple])
    return op

def generate_input(elems, target):
    inputs, targets = [], []
    global gt_mappings_filt, direct_inputs, direct_targets
    for elem in list(elems):
        try:
            inputs.append(generate_data(elem))
            targets.append(target)
            if target==1:
                gt_mappings_filt.append(elem)
        except:
            direct_inputs.append(generate_data_neighbourless(elem))
            direct_targets.append(target)
    print ("Filtered len: ", len(inputs), "Original len:", len(elems))
    return np.array(inputs), np.array(targets)

neighbours_dicts = {ont: get_one_hop_neighbours(ont) for ont in list(set(flatten(ontologies_in_alignment)))}
max_neighbours = np.max(flatten([[len(el[e]) for e in el] for el in neighbours_dicts.values()]))
neighbours_lens = {ont: {key: len(neighbours_dicts[ont][key]) for key in neighbours_dicts[ont]}
                   for ont in neighbours_dicts}
neighbours_dicts = {ont: {key: neighbours_dicts[ont][key] + ["<UNK>" for i in range(max_neighbours -len(neighbours_dicts[ont][key]))]
              for key in neighbours_dicts[ont]} for ont in neighbours_dicts}

print("Number of neighbours: " + sys.argv[1])

data_items = data.items()
np.random.shuffle(list(data_items))
data = OrderedDict(data_items)

print ("Number of entities:", len(data))
all_ont_pairs = list(set([tuple([el.split("#")[0] for el in l]) for l in data.keys()]))

all_metrics = []

for i in list(range(0, len(all_ont_pairs), 3)):
    
    test_onto = all_ont_pairs[i:i+3]
    
    train_data = {elem: data[elem] for elem in data if tuple([el.split("#")[0] for el in elem]) not in test_onto}
    test_data = {elem: data[elem] for elem in data if tuple([el.split("#")[0] for el in elem]) in test_onto}

    torch.set_default_dtype(torch.float64)
    
    train_test_split = 0.9

    train_data_t = [key for key in train_data if data[key]]
    train_data_f = [key for key in train_data if not data[key]]
    #train_data_f = train_data_f[:int(len(train_data_t))]
#     [:int(0.1*(len(train_data) - len(train_data_t)) )]
#     np.random.shuffle(train_data_f)
    
    lr = 0.001
    num_epochs = 50
    weight_decay = 0.001
    batch_size = 8
    dropout = 0.3
    batch_size = min(batch_size, len(train_data_t))
    num_batches = int(ceil(len(train_data_t)/batch_size))
    batch_size_f = int(ceil(len(train_data_f)/num_batches))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = SiameseNetwork(512, 250, 1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        inputs_pos, targets_pos = generate_input(train_data_t, 1)
        inputs_neg, targets_neg = generate_input(train_data_f, 0)
        indices_pos = np.random.permutation(len(inputs_pos))
        indices_neg = np.random.permutation(len(inputs_neg))

        inputs_pos, targets_pos = inputs_pos[indices_pos], targets_pos[indices_pos]
        inputs_neg, targets_neg = inputs_neg[indices_neg], targets_neg[indices_neg]

#        indices = np.random.permutation(len(inputs_pos) + len(inputs_neg))
        
#        inputs = np.array(list(inputs_pos) + list(inputs_neg))[indices]
#        targets = np.array(list(targets_pos) + list(targets_neg))[indices]

#         inputs = np.array(list(inputs_pos) + list(inputs_neg))
#         targets = np.array(list(targets_pos) + list(targets_neg))

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx+1) * batch_size
            
            batch_start_f = batch_idx * batch_size_f
            batch_end_f = (batch_idx+1) * batch_size_f

            
            inputs = np.concatenate((inputs_pos[batch_start: batch_end], inputs_neg[batch_start_f: batch_end_f]))
            targets = np.concatenate((targets_pos[batch_start: batch_end], targets_neg[batch_start_f: batch_end_f]))
            
            inp = inputs.transpose(1,0,2)
            nonzero_elems = np.count_nonzero(inp, axis=-1) - 1
            
            inp_elems = torch.LongTensor(inputs).to(device)
            targ_elems = torch.DoubleTensor(targets).to(device)
            optimizer.zero_grad()

            seq_lens = torch.LongTensor(nonzero_elems.T).to(device)
            outputs = model(inp_elems, seq_lens)
            loss = F.mse_loss(outputs, targ_elems)
            loss.backward()
#             break
            optimizer.step()

            if batch_idx%10 == 0:
                print ("Epoch: {} Idx: {} Loss: {}".format(epoch, batch_idx, loss.item()))

    model.eval()
    torch.save(model.state_dict(), "/u/vivek98/attention.pt")
    
    test_data_t = [key for key in test_data if data[key]]
    test_data_f = [key for key in test_data if not data[key]]
    
    res = greedy_matching()
    f1 = open("test_results.pkl", "wb")
    pickle.dump(res, f1)
print ("Final Results: " + str(np.mean([el[1] for el in all_metrics], axis=0)))
print ("Best threshold: " + str(all_metrics[np.argmax([el[1][2] for el in all_metrics])][0]))
