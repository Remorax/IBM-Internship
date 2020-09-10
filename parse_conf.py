import glob
import numpy as np
import pickle

def calculate_performance():
    global final_results
    all_metrics, all_fn, all_fp = [], [], []
    for (test_onto, test_data_t, all_results) in final_results:
        res = []
        for i,key in enumerate(all_results):
            if all_results[key][0] > threshold:
                res.append(key)
        fn_list = [(key, all_results[key][0]) for key in test_data_t if key not in set(res)]
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

for file_pair in list(zip(sorted(glob.glob("../temp-dir2/*part1*")), sorted(glob.glob("../temp-dir/*part2*")))):
    final_results1, threshold_results1 = pickle.load(open(file_pair[0], "rb"))
    final_results2, threshold_results2 = pickle.load(open(file_pair[1], "rb"))
    final_results = final_results1 + final_results2 
    threshold_results = {}
    for thresh in threshold_results1:
        threshold_results[thresh] = threshold_results1[thresh]
    for thresh in threshold_results2:
        if thresh in threshold_results:
            threshold_results[thresh].extend(threshold_results2[thresh])
        else:
            threshold_results[thresh] = threshold_results2[thresh]
    threshold_results_mean = {el: np.mean(threshold_results[el], axis=0) for el in threshold_results}    
    threshold = max(threshold_results_mean.keys(), key=(lambda key: threshold_results_mean[key][2]))

    all_metrics, all_fn, all_fp = calculate_performance()

    output_file = "Output/21step" + "_".join(file_pair[0].split("/")[-1].split("_")[:-1]) + ".pkl"
    f = open(output_file, "wb")
    pickle.dump([all_fn, all_fp], f)
    f.close()

    results_file = "Results/Output_att21step" + "_".join(file_pair[0].split("/")[-1].split("_")[:-1]) + ".txt"
    f = open(results_file, "w+")
    f.write("Final Results: " + str(np.mean(all_metrics, axis=0)))
    f.write("\nThreshold: " + str(threshold) + "\n")
    f.close()
