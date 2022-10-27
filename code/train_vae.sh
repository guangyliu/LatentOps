
export PYTHONPATH="${PYTHONPATH}:/workspace/code"
#source ../bashrc

data=yelp
dataset=$data
#
#export TRAIN_FILE=../data/datasets/news_data/train.shuf.lower.merge
#export TEST_FILE=../data/datasets/news_data/valid.lower.merge
#export TRAIN_FILE=../data/datasets/news_data/train.shuf.lower.merge
#export TEST_FILE=../data/datasets/news_data/valid.lower.merge

#export TRAIN_FILE=../data/datasets/political_data/train.merge
#export TEST_FILE=../data/datasets/political_data/dev.merge
#
#export TRAIN_FILE=../data/datasets/amazon_data/train.shuf.merge
#export TEST_FILE=../data/datasets/amazon_data/test.merge
TRAIN_FILE=../data/datasets/yelp_data/train.shuf.merge
TEST_FILE=../data/datasets/yelp_data/test.merge
#TRAIN_FILE=../data/datasets/snli_data/train.shuf.merge
#TEST_FILE=../data/datasets/snli_data/test.txt


#TRAIN_FILE=../data/datasets/commongen_data/commongen/train.shuf.merge
#TEST_FILE=../data/datasets/commongen_data/commongen/test.merge

#TRAIN_FILE=../data/datasets/yelp_data/train.debug.merge
#TEST_FILE=../data/datasets/yelp_data/test.debug.merge
#TRAIN_FILE=../data/datasets/gyafc/train.merge
#TEST_FILE=../data/datasets/gyafc/valid.merge
 # 0. train all
 # 1. train linear
 # 2. train linear & BERT
 # 3. train extra layers in BERT and GPT2
 # 4. train extra layers in BERT and GPT2, and pooler
 # 5. fix both, extra layer in BERT, and train pooler
 # 6. train BERT & extra of GPT2 & linear
 # 7. train linear & BERT
 # 8. bert, extra layer of GPT2 & wte
 #
apex_opt=O2
fix_model=84
gpt_size='base'
batch=64 #64
accumulation_steps=1
beta=0.9

args=" --disable_bar $1" #--disable_bar --no_save --disable_bar --disable_bar

latent_size=64

eval_batch=64 #32
max_steps=110000 # 30000
epoch=100 #50
ratio_zero=0.5
learning_rate=5e-5
dim_target_kl=0.9
model='bertu'

#for beta in '1.0' '0.0'
#do


#TEST_FILE=../data/datasets/AEGS_data/yelp/from.0.merge


if [ $model == 'roberta' ]; then
  model_path='../output/roberta-small'
elif [ $model == 'albert' ]; then
  model_path='../output/albert-xlarge-v2'
elif [ $model == 'deberta' ]; then
  model_path='../output/deberta-v3-base'
elif [ $model == 't5' ]; then
  model_path='../output/t5-large'
elif [ $model == 'bertu' ]; then
  model_path='prajjwal1/bert-small' #'../output/bert-base-uncased' #bert-small' bert-base-uncased
else
  model_path='../output/bert-cased' #'../output/bert-cased'
fi

if [ $beta == '0.0' ]; then
  prefix='AE'
else
  prefix='VAE'$beta
fi

if [ $gpt_size == 'xl' ]; then
  gpt_path=gpt2-xl
elif [ $gpt_size == 'm' ]; then
  gpt_path=gpt2-medium
elif [ $gpt_size == 'large' ]; then
  gpt_path=gpt2-large
elif [ $gpt_size == 'base' ]; then
  gpt_path=gpt2
else
  echo "False"
fi
##
# ../output/gpt2
#/home/yiwenhu/data_med4/data4/Optimus/output/gpt2-xl
 # ../output/gpt2 #
name='v8_'$model's_'$gpt_size'_'$prefix'_fx'$fix_model'_'$latent_size'_'b$batch'_'e$epoch'_d'$dim_target_kl #'_d'$dim_target_kl'_lr'$learning_rate
CUDA_VISIBLE_DEVICES=1 python examples/big_ae/run_lm_vae_training.py \
    --output_dir=../output_home/LM/$data/$name  \
    --dataset $dataset \
    --encoder_model_type=$model \
    --encoder_model_name_or_path=$model_path \
    --decoder_model_type=gpt2 \
    --decoder_model_name_or_path=$gpt_path \
    --beta $beta \
    --do_train \
    --do_eval \
    --train_data_file=$TRAIN_FILE \
    --eval_data_file=$TEST_FILE \
    --num_train_epochs $epoch \
    --overwrite_output_dir \
    --per_gpu_train_batch_size=$batch \
    --per_gpu_eval_batch_size=$eval_batch \
    --block_size 30 \
    --ratio_zero $ratio_zero\
    --length_weighted_loss \
    --latent_size $latent_size \
    --evaluate_during_training \
    --latent_as_gpt_memory 1 --latent_as_gpt_emb 1 \
    --gradient_accumulation_steps $accumulation_steps --max_steps $max_steps \
    --dim_target_kl $dim_target_kl  --learning_rate $learning_rate --fix_model $fix_model --fp16  --fp16_opt_level $apex_opt $args \
    # --use_pretrained_vae  --use_pretrained_model \
    # --gloabl_step_eval 1 \
    # --checkpoint_dir ../output_home/LM/base/snli #v2_bert_l_VAE1.0_fx8_64_b64_e50_d1.0 #bert_l_VAE1.0_fx8_64_b64_e50_d1.0
#done

