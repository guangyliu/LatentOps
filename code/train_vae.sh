#!/bin/bash
#SBATCH -J med_yelp
#SBATCH -p p-V100
#SBATCH -o log/%j.out
#SBATCH -e log/%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1

export PYTHONPATH="${PYTHONPATH}:/workspace/code"
data=yelpnew
dataset=$data
export TRAIN_FILE=../data/datasets/yelpshort_data/train.shuf.merge
export TEST_FILE=../data/datasets/yelpshort_data/test.merge
#export TRAIN_FILE=../data/datasets/sst2_data/train.merge
#export TEST_FILE=../data/datasets/sst2_data/valid.merge

#export TRAIN_FILE=../data/datasets/news_data/train.merge
#export TEST_FILE=../data/datasets/news_data/valid.merge

#export TRAIN_FILE=../data/datasets/wiki_data/wikilarge/train.merge
#export TEST_FILE=../data/datasets/wiki_data/wikilarge/dev.merge
#export TRAIN_FILE=../data/datasets/amazon'_'data/train.merge
#export TEST_FILE=../data/datasets/amazon'_'data/test.merge
#export TRAIN_FILE=../data/datasets/snli_data/train.txt
#export TEST_FILE=../data/datasets/snli_data/test.txt
fix_model=2 # 0 nofix, 1 fix both, 2 fix gpt2
gpt_memory=1
gpt_emb=1
latent_size=64
batch=64
eval_batch=32
epoch=100
learning_rate=5e-5 #5e-5

beta=1.0
if [ $beta == '1.0' ]; then
  prefix='VAE'
else
  prefix='AE'
fi
name=DEBUG_$prefix'_fx'$fix_model'_m'$gpt_memory'e'$gpt_emb'_'$latent_size'_'b$batch'_'e$epoch #'_d'$dim_target_kl'_lr'$learning_rate
CUDA_VISIBLE_DEVICES=0 python examples/big_ae/run_lm_vae_training.py \
    --output_dir=../output_med/LM/$data/$name  \
    --dataset $dataset \
    --encoder_model_type=bert \
    --encoder_model_name_or_path=../output/bert-cased \
    --decoder_model_type=gpt2 \
    --decoder_model_name_or_path=../output/gpt2 \
    --beta $beta --ratio_zero 0.5\
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
    --latent_as_gpt_memory $gpt_memory --latent_as_gpt_emb $gpt_emb \
    --dim_target_kl 1.0  --learning_rate $learning_rate --fix_model $fix_model \
    --use_pretrained_model \
    --use_pretrained_vae \
    --gloabl_step_eval 1 \
    --checkpoint_dir ../output_med/LM/yelpnew/AE_fx2_m1e1_64_b64_e50
#    --checkpoint_dir ../output/LM/snli/snli_64_b10_e20_d1.0_lr5e-5
#    --checkpoint_dir ../output/LM/yelpnew/yelpnew_64_b128_e30_d1.0_lr5e-5 --gloabl_step_eval 1
#    --checkpoint_dir ../output/LM/snli/snli_64_b64_e10_d1.0_lr1e-5 --gloabl_step_eval 1
#    --checkpoint_dir ../output/LM/yelpnew/final_yelp_64_b10_e50 --gloabl_step_eval 1
#
