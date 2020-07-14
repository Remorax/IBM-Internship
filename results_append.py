import os
import operator

mapping_dict = {
    "ent_prop": ["(avg, ent+prop)", ", Dot product of node with neighbours, softmax, weighted average, both entities and props"],
    "sum": ["(sum)", ", Dot product of node with neighbours, softmax, weighted sum"],
    "unsoftmax": ["(avg, no softmax)", ", Dot product of node with neighbours, weighted average"],
    "context": ["(only context)", ", Dot product of node with neighbours, softmax, weighted average, context is directly output"],
    "normalize": ["(avg, normalized)", ", Dot product of node with neighbours, softmax, weighted average, normalized"],
    "v": ["(trainable param)", ", Dot product of node with neighbours, softmax, dot with trainable param"],
    "default": ["", ", Dot product of node with neighbours, softmax, weighted average"]
}

final = []
for file in os.listdir("."):
    if file.startswith("Output_att"):
        neighbours = ''.join(filter(str.isdigit, file))
        intent = "USE + dot Attn + " + neighbours + " neighbours + Cos Sim "
        try:
            threshold = str(round(float([l.split()[-1] for l in open(file).read().split("\n") if "Best threshold:" in l][0]), 3))
        except:
            print (file)
            continue
        if "ent_prop" in file:
            key = mapping_dict["ent_prop"][0]
            desc = "Optimum threshold " + threshold + mapping_dict["ent_prop"][1]
        elif "sum" in file:
            key = mapping_dict["sum"][0]
            desc = "Optimum threshold " + threshold + mapping_dict["ent_prop"][1]
        elif "unsoftmax" in file:
            key = mapping_dict["unsoftmax"][0]
            desc = "Optimum threshold " + threshold + mapping_dict["unsoftmax"][1]
        elif "context" in file:
            key = mapping_dict["context"][0]
            desc = "Optimum threshold " + threshold + mapping_dict["context"][1]
        elif "normalize" in file:
            key = mapping_dict["normalize"][0]
            desc = "Optimum threshold " + threshold + mapping_dict["normalize"][1]
        elif "v" in file:
            key = mapping_dict["v"][0]
            desc = "Optimum threshold " + threshold + mapping_dict["v"][1]
        else:
            key = mapping_dict["default"][0]
            desc = "Optimum threshold " + threshold + mapping_dict["default"][1]
        intent += key
        results = [l for l in open(file).read().split("\n") if "Final Results:" in l][0]  
        results = results.split("[")[1].split("]")[0].strip().split()
        line = "\t".join([intent] + results + [desc])
        final.append((line, key, int(neighbours)))
final = sorted(final, key=operator.itemgetter(1, 2))
final = [l[0] for l in final]
open("results_append.tsv", "w+").write("\n".join(final))
