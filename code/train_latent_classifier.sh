#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/workspace/code"


train_cls_gan='cls'
 #'yelp_debug' # .9 #_fb1_bert_l_VAE1.0_fx8_64_b64_e50_d0.9 #bert_l_VAE1.0_fx8_64_b64_e50_d1.0 #bert_l_VAE1.0_fx882_64_b64_e50_d1.0 #yelp_weak_optimus #f_pre_amazon_64_b32_e20_beta_1.0_d1.0_3 #$1 #integrated_32_d1.0 #pre_f_sst2_32_b5_e50_d1.0_lr1e-5 #f_pre_integrated_32_b32_e100_beta_1.0_d1.0
cuda=0
if [ $1 == 'aug' ]
then
  data_type=$2
  gpt_size=$3
  name=$4
else
  data_type='sentiment' #'tweetsentiment' #sentiment #$2
   gpt_size='base'
  name='yelp'
fi
data=$gpt_size
bert_model='prajjwal1/bert-small'
bert_type='bertu'
fix_model=84
latent_size=64
if [ $train_cls_gan == 'gan' ]
then
  epoch=10 # 10
  batch=32 #32
  train_data=train_gan.txt
elif [ $train_cls_gan == 'cls' ]; then
  epoch=50 # 10
  batch=8 #6 #32
  train_data=train_cls_200.txt
fi


n_classes=2
logging_steps=1


if [ $data_type == 'sentiment' ]
then
  cls_step=1
  export TRAIN_FILE=../data/datasets/yelp_data/train_gan.txt #test_ref.shuf.txt #yelpshort_data/train_cls_1000.txt #train_pos.txt #train_cls_1000.txt # #batch=5 epoch 50
  export TEST_FILE=../data/datasets/yelp_data/test.txt #test.txt
elif [ $data_type == 'tense' ]
then
  cls_step=4
  n_classes=3
  export TRAIN_FILE=../data/datasets/yelp_data/train_tense.txt  #train_gan.txt #test_ref.shuf.txt #yelpshort_data/train_cls_1000.txt #train_pos.txt #train_cls_1000.txt # #batch=5 epoch 50
  export TEST_FILE=../data/datasets/yelp_data/test_tense.txt  #test.txt
elif [ $data_type == 'formality' ]
then
  cls_step=33
  n_classes=2
  export TRAIN_FILE=../data/datasets/gyafc/train_formal.txt  #train_gan.txt #test_ref.shuf.txt #yelpshort_data/train_cls_1000.txt #train_pos.txt #train_cls_1000.txt # #batch=5 epoch 50
  export TEST_FILE=../data/datasets/gyafc/test_formal.txt  #test.txt
else
  echo 'Wrong data_type, EXIT'
  return 1
fi
echo "cls_step is $cls_step "
echo "data_type is $data_type "
echo "n_classes is $n_classes"
echo "train" $train_cls_gan
if [ $gpt_size == 'xl' ]; then
  gpt_path=/home/yiwenhu/data_med4/data4/Optimus/output/gpt2-xl
elif [ $gpt_size == 'm' ]; then
  gpt_path=../output/gpt2-medium
elif [ $gpt_size == 'large' ]; then
  gpt_path=gpt2-large
elif [ $gpt_size == 'base' ]; then
  gpt_path=gpt2
else
  echo "False"
fi

eval_batch=$batch
dim_target_kl=1.0

CUDA_VISIBLE_DEVICES=$cuda python examples/big_ae/train_ddpm_latent.py \
   --output_dir=../output_home/LM/$data/$name  \
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
   --checkpoint_dir ../output_home/LM/$data/$name --gloabl_step_eval 1 \
   --n_classes $n_classes --train_cls_gan $train_cls_gan --n_cyc 8  --save_step $cls_step  --fix_model $fix_model  #--fp16 # --is_tense
# save_step 1: sentiment 2: amazon 3: imdb
# bash lace_sampling_length.sh $name
# echo 'lgylgy_sampling'
#bash lace_transfer_yelpnew.sh $name $data_type
#echo 'lgylgy_transfer'
#bash lace_sampling_multiple.sh $name $latent_size $data_type
