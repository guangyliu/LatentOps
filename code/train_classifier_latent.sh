#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/workspace/code"

cuda=0
train_cls_gan='cls' # 'gan' or 'cls'
data_type='sentiment' 
gpt_size='base' # base large 
ckpt_path=../ckpts/base_yelp
name='yelp'

bert_model='prajjwal1/bert-small'
bert_type='bertu'
fix_model=84
latent_size=64
if [ $train_cls_gan == 'gan' ]
then
  epoch=5 # 10
  batch=8 #32
  TRAIN_FILE=../data/datasets/yelp_data/train_gan.txt #test_ref.shuf.txt #yelpshort_data/train_cls_1000.txt #train_pos.txt #train_cls_1000.txt # #batch=5 epoch 50
  TEST_FILE=../data/datasets/yelp_data/test.txt #test.txt
elif [ $train_cls_gan == 'cls' ]; then
  epoch=50 # 10
  batch=8 #6 #32
fi


n_classes=2
logging_steps=1

if [ $data_type == 'sentiment' ]
then
cls_step=1
TRAIN_FILE=../data/datasets/yelp_data/train_sentiment.txt #test_ref.shuf.txt #yelpshort_data/train_cls_1000.txt #train_pos.txt #train_cls_1000.txt # #batch=5 epoch 50
TEST_FILE=../data/datasets/yelp_data/test_sentiment.txt #test.txt
elif [ $data_type == 'tense' ]
then
  cls_step=4
  n_classes=3
TRAIN_FILE=../data/datasets/yelpshort_data/train_tense.txt  #train_gan.txt #test_ref.shuf.txt #yelpshort_data/train_cls_1000.txt #train_pos.txt #train_cls_1000.txt # #batch=5 epoch 50
TEST_FILE=../data/datasets/yelpshort_data/test_tense.txt  #test.txt
elif [ $data_type == 'formality' ]
then
  cls_step=33
  n_classes=2
TRAIN_FILE=../data/datasets/gyafc/train_formal.txt  #train_gan.txt #test_ref.shuf.txt #yelpshort_data/train_cls_1000.txt #train_pos.txt #train_cls_1000.txt # #batch=5 epoch 50
TEST_FILE=../data/datasets/gyafc/test_formal.txt  #test.txt
elif [ $data_type == 'amazon' ]
then
  cls_step=2
#  data=amazon
TRAIN_FILE=../data/datasets/amazon_data/train_cls_200.txt  #train_cls_1000.txt
TEST_FILE=../data/datasets/amazon_data/test_cls_200.txt #test_cls_200.txt
else
  echo 'Wrong data_type, EXIT'
  return 1
fi
echo "cls_step is $cls_step "
echo "data_type is $data_type "
echo "n_classes is $n_classes"
echo "train" $train_cls_gan
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

eval_batch=$batch
dim_target_kl=1.0

CUDA_VISIBLE_DEVICES=$cuda python examples/big_ae/train_cls_latent.py \
   --output_dir=$ckpt_path \
   --dataset Yelp_cls \
   --encoder_model_type=$bert_type \
   --encoder_model_name_or_path=$bert_model \
   --decoder_model_type=gpt2 \
   --decoder_model_name_or_path=$gpt_path \
   --beta 0.5 \
   --ratio_zero 0.5 \
   --ratio_increase 0.25 \
   --do_train \
   --do_eval \
   --fb_mode 1 \
   --train_data_file=$TRAIN_FILE \
   --eval_data_file=$TEST_FILE \
   --num_train_epochs $epoch \
   --save_steps 898 \
   --logging_steps $logging_steps \
   --overwrite_output_dir \
   --per_gpu_train_batch_size=$batch \
   --per_gpu_eval_batch_size=$eval_batch \
   --block_size 30 \
   --length_weighted_loss \
   --latent_size $latent_size \
   --evaluate_during_training \
   --dim_target_kl $dim_target_kl  --learning_rate 5e-4 \
   --use_pretrained_model \
   --use_pretrained_vae \
   --checkpoint_dir $ckpt_path --gloabl_step_eval 1  \
   --n_classes $n_classes --train_cls_gan $train_cls_gan --n_cyc 8  --save_step $cls_step  --fix_model $fix_model  #--fp16 # --is_tense
# save_step 1: sentiment 2: amazon 3: imdb
# bash lace_sampling_length.sh $name
# echo 'lgylgy_sampling'
#bash lace_transfer_yelpnew.sh $name $data_type
#echo 'lgylgy_transfer'
#bash conditional_generation.sh $name $latent_size $data_type
