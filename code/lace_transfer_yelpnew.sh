#!/bin/bash
#SBATCH -J cls_yelp
#SBATCH -p p-V100
#SBATCH -o log/%j.out
#SBATCH -e log/%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1

export PYTHONPATH="${PYTHONPATH}:/workspace/code"

logits=0.0
weight_energy=1.0
batch=250
fix_model=84
name='yelp' #'amazon'
gpt_size='large'
data=$gpt_size
bert_model='prajjwal1/bert-small'
bert_type='bertu'
latent_size=64

echo "model_size $gpt_size"




eval_batch=100
data_type='sentiment' #'amazon'
reg_z=0.0


latent_size=64

dataset=Yelp_cls
if [ $data_type == 'sentiment' ]
then
#  export TRAIN_FILE=../data/datasets/yelpnew_data/sentiment/train100.txt
#  export TEST_FILE=../data/datasets/yelpshort_data/test_500neg.txt
  export TEST_FILE=../data/datasets/yelp_data/test_ref.txt
  # export TEST_FILE=../data/datasets/yelp_data/test_neg_500.txt
  if [ $gpt_size == 'large' ];then
    weight_energy=1.0
  elif [ $gpt_size == 'base' ];then
    weight_energy=3
  fi
  cls_step=1
elif [ $data_type == 'amazon' ]
then
  cls_step=2
  export TEST_FILE=../data/datasets/amazon_data/test_ref.txt
  export TEST_FILE=../data/datasets/amazon_data/test_ref.txt
elif [ $data_type == 'dog' ]
then
  cls_step=3
#  export TRAIN_FILE=../data/datasets/amazon_data/dog_test.txt
#  export TEST_FILE=../data/datasets/amazon_data/dog_test.txt
  export TRAIN_FILE=../data/datasets/amazon_data/test_1000.txt
  export TEST_FILE=../data/datasets/amazon_data/test_1000.txt
#  export TEST_FILE=../data/datasets/AEGS_data/yelp/test_ref.txt
#  export TEST_FILE=../data/datasets/AEGS_data/yelp/test_ref.txt
elif [ $data_type == 'so' ]
then
  cls_step=4
  export TEST_FILE=../data/datasets/AEGS_data/yelp/test_ref.txt
  export TEST_FILE=../data/datasets/AEGS_data/yelp/test_ref.txt
elif [ $data_type == 'sequential' ]
then
#  export TRAIN_FILE=../data/datasets/yelpnew_data/sentiment/train100.txt
#  export TEST_FILE=../data/datasets/yelpshort_data/test_500neg.txt
  export TRAIN_FILE=../data/datasets/yelpshort_data/s1t0f1_label.txt  #test_neg_500.txt
  export TEST_FILE=../data/datasets/yelpshort_data/s1t0f1_label.txt  #test_neg_500.txt
  if [ $gpt_size == 'large' ];then
    weight_energy=0.5
  elif [ $gpt_size == 'base' ];then
    weight_energy=3
  fi
  cls_step=1
else
  echo 'Wrong data_type, EXIT'
  return 1
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
#export TEST_FILE=../data/datasets/$data'_'data/tense/test.txt

#export TRAIN_FILE=../data/datasets/$data_type'_data/train_5000.txt'
#export TEST_FILE=../data/datasets/$data_type'_data/test_200.txt'

#export TRAIN_FILE=/home/guangyiliu/Optimus-VAE/data/datasets/semeval_data/train.txt # ../data/datasets/semeval_data/train.merge
#export TEST_FILE=/home/guangyiliu/Optimus-VAE/data/datasets/semeval_data/test.txt #../data/datasets/semeval_data/test.merge

#

epoch=250
save_step=8980
dim_target_kl=1.0
# beta=0.5s
ratio_zero=0.5
beta="1.0"
ratio_increase="0.25"
cls_step=1 #33 #1,4,33
att_list=1 # $1
weight_energy=1
reg_z=0.0
cuda=0 #$2
echo "Energy $weight_energy"
repa_num=20

    CUDA_VISIBLE_DEVICES=$cuda python examples/big_ae/lace_tst_my.py \
        --output_dir=../ckpts/large_yelp  \
        --dataset $dataset \
        --encoder_model_type=$bert_type \
        --encoder_model_name_or_path=$bert_model \
        --decoder_model_type=gpt2 \
        --decoder_model_name_or_path=$gpt_path \
        --beta $beta \
        --ratio_zero $ratio_zero \
        --ratio_increase $ratio_increase \
        --do_train \
        --do_eval \
        --fb_mode 1 \
        --train_data_file=$TRAIN_FILE \
        --eval_data_file=$TEST_FILE \
        --num_train_epochs $epoch \
        --save_steps $save_step \
        --logging_steps 898 \
        --overwrite_output_dir \
        --per_gpu_train_batch_size=$batch \
        --per_gpu_eval_batch_size=$eval_batch \
        --block_size 50 \
        --length_weighted_loss \
        --latent_size $latent_size \
        --evaluate_during_training \
        --dim_target_kl $dim_target_kl  --learning_rate 1e-5 \
        --use_pretrained_model \
        --use_pretrained_vae \
        --checkpoint_dir  ../ckpts/large_yelp --gloabl_step_eval 1 \
        --cls_dir ../ckpts/large_yelp/checkpoint-cls- --n_classes 2 \
        --repa_num $repa_num \
        --reg_z $reg_z --reg_logits $logits --cls_step $cls_step --data_type $data_type --weight_energy $weight_energy --fix_model $fix_model --att_list $att_list