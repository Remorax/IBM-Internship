import os
import operator

mapping_dict = {
    "ent_prop": ["(avg, ent+prop)", ", Dot product of node with neighbours, softmax, weighted average, both entities and props"],
    "sum": ["(sum)", ", Dot product of node with neighbours, softmax, weighted sum"],
    "unsoftmax": ["(avg, no softmax)", ", Dot product of node with neighbours, weighted average"],
    "context": ["(only context)", ", Dot product of node with neighbours, softmax, weighted average, context is directly output"],
    "normalize": ["(avg, normalized)", ", Dot product of node with neighbours, softmax, weighted average, normalized"],
    "resolved": ["(self + v + min + unabbreviated)", ", Min neighbour filter, oversampled, Dot product of node with neighbours, softmax, dot with trainable param"],
    "unhas": ["(self + v + min + removes has from property)", ", Min neighbour filter, oversampled,  Dot product of node with neighbours, softmax, dot with trainable param"],
    "spellchecked": ["(self + v + min + spellchecked)", ", Min neighbour filter, oversampled, Dot product of node with neighbours, softmax, dot with trainable param"],
    "rootpath": ["(includes parent path from root)", ", spellchecked, abbreviation resolved, has removed, oversampled, Dot product of node with neighbours, softmax, dot with trainable param"],
    "ae": ["(auto encoder)", ", Dot product of node with neighbours, softmax, weighted average, auto encoder"],
    "cross": ["(cross attention)", ", Dot product of neighbours with other entity's neighbours, softmax, weighted average"],
    "min": ["(min neighbours)", ", Dot product of node with neighbours, softmax, weighted average"],
    "hybrid_self": ["(self + v + min)", ", Min neighbour filter, Dot product of node with neighbours, softmax, dot with trainable param"],
    "hybrid": ["(cross + v + min)", ", Min neighbour filter, Dot product of neighbours with other entity's neighbours, softmax, dot with trainable param"],
    "self_cross": ["(self+cross)", ", Min neighbour filter, Dot product of neighbours with entity AND other entity's neighbours, softmax, dot with trainable param"],
    "v": ["(trainable param)", ", Dot product of node with neighbours, softmax, dot with trainable param"],
    "default": ["", ", Dot product of node with neighbours, softmax, weighted average"]
}

final = []
for file in os.listdir("."):
    if file.startswith("Output_att"):
        neighbours = [''.join(filter(str.isdigit, num)) for num in file.split("_")]
        neighbours = [el for el in neighbours if el]
        intent = "USE + dot Attn + " + ",".join(neighbours) + " neighbours + Cos Sim "
        try:
            threshold = str(round(float([l.split()[-1] for l in open(file).read().split("\n") if "Best threshold:" in l][0]), 3))
        except:
            print (file)
            continue
        for elem in mapping_dict:
            if elem in file or elem == "default":
                key = mapping_dict[elem][0]
                desc = "Optimum threshold " + threshold + mapping_dict[elem][1]
                break
        intent += key
        results = [l for l in open(file).read().split("\n") if "Final Results:" in l][0]  
        results = results.split("[")[1].split("]")[0].strip().split()
        line = "\t".join([intent] + results + [desc])
        final.append((line, key, tuple([int(el) for el in neighbours])))
final = sorted(final, key=operator.itemgetter(1, 2))
final = [l[0] for l in final]
open("results_append.tsv", "w+").write("\n".join(final))
