#!/bin/bash
#SBATCH -J XLyelp
#SBATCH -p p-V100
#SBATCH -o log/%j.out
#SBATCH -e log/%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1

export PYTHONPATH="${PYTHONPATH}:/workspace/code"
source ../bashrc

data=yelp_poly3
dataset=$data
#export TRAIN_FILE=../data/datasets/yelpshort_data/train.merge
#export TEST_FILE=../data/datasets/yelpshort_data/test.merge #test_tmp.merge
#export TRAIN_FILE=../data/datasets/c4_data/train_1m.merge
#export TEST_FILE=../data/datasets/c4_data/test.merge
#export TRAIN_FILE=../data/datasets/sst2_data/train.merge
#export TEST_FILE=../data/datasets/sst2_data/valid.merge
#
#export TRAIN_FILE=../data/datasets/news_data/train.merge
#export TEST_FILE=../data/datasets/news_data/valid.merge
#export TRAIN_FILE=../data/datasets/owt_data/train.merge
#export TEST_FILE=../data/datasets/owt_data/test.merge

#export TRAIN_FILE=../data/datasets/wiki_full/train_1m.merge
#export TEST_FILE=../data/datasets/wiki_full/test.merge
#export TRAIN_FILE=../data/datasets/amazon_data/train.merge
#export TEST_FILE=../data/datasets/amazon_data/test.merge

#TRAIN_FILE=../data/datasets/snli_data/train.txt
#TEST_FILE=../data/datasets/snli_data/test.txt

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
fix_model=8
gpt_size='b'
batch=64 #64
accumulation_steps=1
beta=0.5

args="--disable_bar $1" #

latent_size=64

eval_batch=64 #32
epoch=50 #50
ratio_zero=0.5
learning_rate=5e-5
dim_target_kl=1.0
model='bert'
#for beta in '1.0' '0.0'
#do

TRAIN_FILE=../data/datasets/yelpshort'_'data/train.merge
TEST_FILE=../data/datasets/yelpshort'_'data/test.merge


if [ $model == 'roberta' ]; then
  model_path='../output/roberta-base'
elif [ $model == 'deberta' ]; then
  model_path='../output/deberta-base'
elif [ $model == 't5' ]; then
  model_path='../output/t5-large'
else
  model_path='../output/bert-cased' #'../output/bert-cased'
fi

if [ $beta == '0.0' ]; then
  prefix='AE'
else
  prefix='VAE'$beta
fi

if [ $gpt_size == 'xl' ]; then
  gpt_path=/home/yiwenhu/data_med4/data4/Optimus/output/gpt2-xl
elif [ $gpt_size == 'm' ]; then
  gpt_path=../output/gpt2-medium
elif [ $gpt_size == 'l' ]; then
  gpt_path=/home/yiwenhu/data_med4/data4/Optimus/output/gpt2-large
elif [ $gpt_size == 'b' ]; then
  gpt_path=../output/gpt2
else
  echo "False"
fi
##
# ../output/gpt2
#/home/yiwenhu/data_med4/data4/Optimus/output/gpt2-xl
 # ../output/gpt2 #
name=$model'_'$gpt_size'_'$prefix'_fx'$fix_model'_'$latent_size'_'b$batch'_'e$epoch'_d'$dim_target_kl #'_d'$dim_target_kl'_lr'$learning_rate
CUDA_VISIBLE_DEVICES=0 python examples/big_ae/run_lm_vae_training.py \
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
    --length_weighted_loss \
    --latent_size $latent_size \
    --evaluate_during_training \
    --latent_as_gpt_memory 1 --latent_as_gpt_emb 1 \
    --gradient_accumulation_steps $accumulation_steps \
    --dim_target_kl $dim_target_kl  --learning_rate $learning_rate --fix_model $fix_model --fp16  $args
#    --use_pretrained_vae  --use_pretrained_model \
#    --gloabl_step_eval 1 \
#    --checkpoint_dir ../output_home/LM/yelpshort/bert_VAE0.5_fx8_64_b64_e50_d1.0_nopre
#done

