import os

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
            key = "(avg, ent+prop)"
            desc = "Optimum threshold " + threshold + ", Dot product of node with neighbours, softmax, weighted average, concat with entity, both entities and props"
        elif "sum" in file:
            key = "(sum)"
            desc = "Optimum threshold " + threshold + ", Dot product of node with neighbours, softmax, weighted sum, concat with entity"
        elif "unsoftmax" in file:
            key = "(avg, no softmax)"
            desc = "Optimum threshold " + threshold + ", Dot product of node with neighbours, weighted average, concat with entity"
        elif "context" in file:
            key = "(only context)"
            desc = "Optimum threshold " + threshold + ", Dot product of node with neighbours, softmax, weighted average, context is directly output"
        elif "normalize" in file:
            key = "(avg, normalized)"
            desc = "Optimum threshold " + threshold + ", Dot product of node with neighbours, softmax, weighted average, normalized, concat with entity"
        else:
            key = ""
            desc = "Optimum threshold " + threshold + ", Dot product of node with neighbours, softmax, weighted average, concat with entity"
        intent += key
        results = [l for l in open(file).read().split("\n") if "Final Results:" in l][0]  
        results = results.split("[")[1].split("]")[0].strip().split()
        line = "\t".join([intent] + results + [desc])
        final.append((line, key + " " + str(neighbours)))
final = sorted(final, key=lambda x:x[1])
final = [l[0] for l in final]
open("results_append.tsv", "w+").write("\n".join(final))
