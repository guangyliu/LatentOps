data_type='humor' # humor, politeness, formality
gpt_size='base'
dataset='yelp'
weight_energy=3.0
#bash train_classifier_latent.sh aug $data_type $gpt_size $dataset
echo "Start Sampling"
bash lace_sampling_multiple.sh aug $data_type $gpt_size $dataset $weight_energy
echo "Start Evaluating"
cd ..
python eval_sampling.py --path $dataset --data_type $data_type --data $gpt_size
