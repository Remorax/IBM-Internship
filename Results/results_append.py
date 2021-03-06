import os, re
import operator

mapping_dict = {
    "twostep_wtpath_uniqpath": ["(self+self unique weighted two-step)", ", Self attention of path on entity followed by node on entity, with best path chosen as weighted sum. Paths of any entity are unique within a type"],
    "twostep_wtpath": ["(self+self weighted two-step)", ", Self attention of path on entity followed by node on entity, with best path chosen as weighted sum."],
    "twostep_uniqpath": ["(self+self unique two-step)", ", Self attention of path on entity followed by node on entity, with best path chosen as max of all path weights."],
    "twostep_bagofnbrs_wtpath": ["(self+self weighted two-step with bagged one-hop)", ", Self attention of path on entity followed by node on entity, with best path chosen as max of all path weights. One-hop neighbours are considered as a bag instead of separate paths."],
    "twostep_bagofnbrs": ["(self+self two-step with bagged one-hop)", ", Self attention of path on entity followed by node on entity, with best path chosen as weighted sum. One-hop neighbours are considered as a bag instead of separate paths."],
    "twostep": ["(self+self two-step)", ", Self attention of path on entity followed by node on entity, with best path chosen as weighted sum"],
    "anatomy_aml_bagofnbrs_wtpath": ["(self+self weighted two-step with bagged one-hop)", ", Self attention of path on entity followed by node on entity, with best path chosen as weighted sum. One-hop neighbours are considered as a bag instead of separate paths."],
    "anatomy_aml_bagofnbrs": ["(self+self two-step with bagged one-hop)", ", Self attention of path on entity followed by node on entity, with best path chosen as max of all path weights. One-hop neighbours are considered as a bag instead of separate paths."],
    "ent_prop": ["(avg, ent+prop)", ", Dot product of node with neighbours, softmax, weighted average, both entities and props"],
    "val": ["(val size = 2)", ", Optimized entity model"],
    "unsoftmax": ["(avg, no softmax)", ", Dot product of node with neighbours, weighted average"],
    "context": ["(only context)", ", Dot product of node with neighbours, softmax, weighted average, context is directly output"],
    "resolved": ["(self + v + min + unabbreviated)", ", Min neighbour filter, oversampled, Dot product of node with neighbours, softmax, dot with trainable param"],
    "unhas": ["(self + v + min + removes has from property)", ", Min neighbour filter, oversampled,  Dot product of node with neighbours, softmax, dot with trainable param"],
    "spellchecked": ["(self + v + min + spellchecked)", ", Min neighbour filter, oversampled, Dot product of node with neighbours, softmax, dot with trainable param"],
    "rootpath_multi": ["(includes multiple parent paths from root)", ", spellchecked, abbreviation resolved, has removed, oversampled, Dot product of node with neighbours, softmax, dot with trainable param"],
    "prop_concat_sparse": ["(property matching)", ", optimized entity model + property matching by passing domain and range through siamese followed by weighted sum, validation size=2"],
    "prop_concat": ["(concat range, domain, prop name)", ", optimized entity model + property matching by passing domain and range through siamese followed by concatenation, validation size=2"],
    "german_phrase": ["(full phrase embedding)", ", optimized entity model, validation size=5%, test size = 10%"],
    "german": ["(optimized)", ", optimized entity model, validation size=5%, test size = 10%"],
    "att_sumnorm_dtpath\d+\_\d+\_prop": ["(property matching + sum normalized + datatype props)", ", optimized entity model + property matching (including datatype props) by passing domain and range through siamese followed by weighted sum with only 2 trainable params (third one is 1-sum), validation size=2"],
    "att_dtpath\d+\_\d+\_prop": ["(property matching + datatype props)", ", optimized entity model + property matching (including datatype props) by passing domain and range through siamese followed by weighted sum, validation size=2"],
    "att_sumnorm\d+\_\d+\_prop": ["(property matching + sum normalized)", ", optimized entity model + property matching (including datatype props) by passing domain and range through siamese followed by weighted sum with only 2 trainable params (third one is 1-sum), validation size=2"],
    "att\d+\_\d+\_prop": ["(property matching)", ", optimized entity model + property matching by passing domain and range through siamese followed by weighted sum, validation size=2"],
    "rootpath": ["(includes parent path from root)", ", spellchecked, abbreviation resolved, has removed, oversampled, Dot product of node with neighbours, softmax, dot with trainable param"],
    "ae": ["(auto encoder)", ", Dot product of node with neighbours, softmax, weighted average, auto encoder"],
    "cross": ["(cross attention)", ", Dot product of neighbours with other entity's neighbours, softmax, weighted average"],
    "min": ["(min neighbours)", ", Dot product of node with neighbours, softmax, weighted average"],
    "hybrid_self": ["(self + v + min)", ", Min neighbour filter, Dot product of node with neighbours, softmax, dot with trainable param"],
    "hybrid": ["(cross + v + min)", ", Min neighbour filter, Dot product of neighbours with other entity's neighbours, softmax, dot with trainable param"],
    "self_cross": ["(self+cross)", ", Min neighbour filter, Dot product of neighbours with entity AND other entity's neighbours, softmax, dot with trainable param"],
    "v": ["(trainable param)", ", Dot product of node with neighbours, softmax, dot with trainable param"],
    "sum": ["(sum)", ", Dot product of node with neighbours, softmax, weighted sum"],
    "normalize": ["(avg, normalized)", ", Dot product of node with neighbours, softmax, weighted average, normalized"],
    "default": ["", ", Dot product of node with neighbours, softmax, weighted average"]
}

final = []
for file in os.listdir("."):
    if file.startswith("Output_att"):
        neighbours = [''.join(filter(str.isdigit, num)) for num in file.split("_")]
        neighbours = [el for el in neighbours if el]
        max_path, max_pathlen = neighbours[0], neighbours[1]
        if len(neighbours) >= 3:
            thresh = str(neighbours[2][0]) + "." + str(neighbours[2][1:])
        else:
            thresh = "N/A"
        intent = "VeeAlign (max_path={}, max_pathlen={}, threshold={})".format(max_path, max_pathlen, thresh)
        try:
            threshold = str(round(float([l.split()[-1] for l in open(file).read().split("\n") if "Threshold:" in l][-1]), 3))
        except Exception as e:
            print (e, file)
            continue
        for elem in mapping_dict:
            if re.search(elem, file) or elem == "default":

                key = mapping_dict[elem][0]
                desc = "Optimum threshold " + threshold + mapping_dict[elem][1]
                # print (file, key)
                break
        intent += key
        try:
            results = [l for l in open(file).read().split("\n") if "Final Results:" in l][0]  
        except:
            print ("no results", file)
            continue
        results = results.split("[")[1].split("]")[0].strip().split()
        line = "\t".join([intent] + results + [desc])
        final.append((line, key, tuple([int(el) for el in neighbours])))
final = sorted(final, key=operator.itemgetter(1, 2))
final = [l[0] for l in final]
open("results_append.tsv", "w+").write("\n".join(final))
