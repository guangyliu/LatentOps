#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/workspace/code"

train_cls_gan='cls'
cuda=0

data_type='sentiment'
gpt_size='base'
name='yelp'

data=$gpt_size
bert_model='prajjwal1/bert-small'
bert_type='bertu'
fix_model=84
latent_size=64

epoch=5
batch=64

n_classes=2
logging_steps=1

cls_step=1
export TRAIN_FILE=../data/datasets/yelp_data/train.shuf.txt #test_ref.shuf.txt #yelpshort_data/train_cls_1000.txt #train_pos.txt #train_cls_1000.txt # #batch=5 epoch 50
export TEST_FILE=../data/datasets/yelp_data/test.txt #test.txt

if [ $gpt_size == 'large' ]; then
  gpt_path=gpt2-large
elif [ $gpt_size == 'base' ]; then
  gpt_path=gpt2
fi

eval_batch=$batch


CUDA_VISIBLE_DEVICES=$cuda python examples/big_ae/train_ddpm_latent.py \
   --do_train \
   --do_eval \
   --num_train_epochs $epoch \
   --per_gpu_train_batch_size=$batch \
   --per_gpu_eval_batch_size=$eval_batch \
   --latent_size $latent_size \
    --learning_rate 5e-4 \
   --checkpoint_dir ../ckpts/base_yelp --gloabl_step_eval 1 \
   --n_classes $n_classes --train_cls_gan $train_cls_gan --n_cyc 8  --save_step $cls_step  --fix_model $fix_model   --nt 2000