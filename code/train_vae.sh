
export PYTHONPATH="${PYTHONPATH}:/workspace/code"

dataset=yelp
cuda=0
#export TRAIN_FILE=../data/datasets/amazon_data/train.shuf.merge
#export TEST_FILE=../data/datasets/amazon_data/test.merge
TRAIN_FILE=../data/datasets/yelp_data/train.shuf.merge
TEST_FILE=../data/datasets/yelp_data/test.merge


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
fix_model=84  # fix 
gpt_size='base' # xl m large base
batch=64 #64
accumulation_steps=1
beta=0.9

args=" $1" #--disable_bar --no_save --disable_bar --disable_bar

latent_size=64

eval_batch=64 #32
epoch=100 #50
ratio_zero=0.5
learning_rate=5e-5
dim_target_kl=0.9
model='bertu'


if [ $model == 'bertu' ]; then
  model_path='prajjwal1/bert-small' 
else
  model_path='bert-base-cased' 
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

name='v8_'$model's_'$gpt_size'_'$prefix'_fx'$fix_model'_'$latent_size'_'b$batch'_'e$epoch'_d'$dim_target_kl #'_d'$dim_target_kl'_lr'$learning_rate
CUDA_VISIBLE_DEVICES=$cuda python examples/big_ae/run_lm_vae_training.py \
    --output_dir=../ckpts/LM/$dataset/$name  \
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
    --gradient_accumulation_steps $accumulation_steps --max_steps -1 \
    --dim_target_kl $dim_target_kl  --learning_rate $learning_rate --fix_model $fix_model --fp16  --fp16_opt_level $apex_opt $args \
    --gloabl_step_eval 1 \
    # --use_pretrained_vae  --use_pretrained_model \
    # --checkpoint_dir ../output_home/LM/base/snli #v2_bert_l_VAE1.0_fx8_64_b64_e50_d1.0 #bert_l_VAE1.0_fx8_64_b64_e50_d1.0
