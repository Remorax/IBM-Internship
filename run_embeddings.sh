c=('data_bert.pkl' 'data_roberta.pkl' 'data_bert_nli.pkl' 'data_roberta_nli.pkl' 
  'data_distilbert.pkl' 'data_distilbert_nli.pkl' 'data_distilbert.pkl' 'data_distilbert_nli.pkl'
  'data_bert.pkl' 'data_roberta.pkl' 'data_bert_nli.pkl' 'data_roberta_nli.pkl' 
  'data_distilbert.pkl' 'data_distilbert_nli.pkl' 'data_distilbert.pkl' 'data_distilbert_nli.pkl'
  'data_bert.pkl' 'data_roberta.pkl' 'data_bert_nli.pkl' 'data_roberta_nli.pkl' )
n=(1024 1024 1024 1024 1024 1024 800 800 600 600 600 600 600 600 300 300 300 300 300 300)

for i in $(seq 0 $((${#c[*]}-1)))
do
    jbsub -q x86_24h -mem 16g -require k80 -cores 1x1+1 -out "Output_att_emb"$1"_"$2".txt" python Attention_optimum.py $1 $2
done
