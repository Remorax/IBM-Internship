#!/bin/bash

max_neighbours=(2 3 4 5 6 7)
min_neighbours=(1 2)

for max_limit in "${max_neighbours[@]}";
do
	for min_limit in "${min_neighbours[@]}";
	do
	        jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Output_att"$max_limit"_"$min_limit"_lstm_path.txt" python LSTM_path.py $max_limit $min_limit data_path.pkl "test_lstm"$max_limit"_"$min_limit".pkl"
	        jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Output_att"$max_limit"_"$min_limit"_bilstm_path.txt" python BiLSTM_path.py $max_limit $min_limit data_path.pkl "test_bilstm"$max_limit"_"$min_limit".pkl"
	done
done
