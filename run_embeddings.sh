c=('data_elmo.pkl' 'data_elmo.pkl' 'data_elmo.pkl' 'data_elmo.pkl' 'data_elmo.pkl')
n=(1600 1024 800 512 300)

for i in $(seq 0 $((${#c[*]}-1)))
do
    jbsub -q x86_24h -mem 16g -require k80 -cores 1x1+1 -out "Output_att_emb"${c[i]}"_"${n[i]}".txt" python Attention_optimum.py ${c[i]} ${n[i]}
done
