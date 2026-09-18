# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os

import torch
from my_transformers import *
from tqdm import tqdm
from transformers import AutoTokenizer

from modules import ConditionalSampling
from modules import GAN  # GANVAE as GAN
from modules import VAE, DenseEmbedder, CCF

logger = logging.getLogger(__name__)

from run_lm_vae_training import MODEL_CLASSES
from lace_sampling_my import sample_q_ode, set_seed, STEP_CLASSES
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--checkpoint_dir", default=None, type=str,
                        help="The directory where checkpoints are saved.")
    parser.add_argument("--cls_dir", default=None, type=str,
                        help="The directory where cls checkpoints are saved.")
    parser.add_argument("--gan_dir", default=None, type=str,
                        help="The directory where cls checkpoints are saved.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--dataset", default=None, type=str, help="The dataset.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--ExpName", default="", type=str,
                        help="The experiment name used in Azure Table.")
    parser.add_argument("--save_bert_gpt_init", action='store_true',
                        help="Use Philly for computing.")
    parser.add_argument("--length_weighted_loss", action='store_true',
                        help="Use sentence length re-weight the reconstruction loss.")

    ## Encoder options
    parser.add_argument("--encoder_model_type", default="bert", type=str,
                        help="The encoder model architecture to be fine-tuned.")
    parser.add_argument("--encoder_model_name_or_path", default="bert-base-cased", type=str,
                        help="The encoder model checkpoint for weights initialization.")
    parser.add_argument("--encoder_config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--encoder_tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    ## Decoder options
    parser.add_argument("--decoder_model_type", default="gpt2", type=str,
                        help="The decoder model architecture to be fine-tuned.")
    parser.add_argument("--decoder_model_name_or_path", default="bert-base-cased", type=str,
                        help="The decoder model checkpoint for weights initialization.")
    parser.add_argument("--decoder_config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--decoder_tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    ## Variational auto-encoder
    parser.add_argument("--latent_size", default=32, type=int, help="Latent space dimension.")
    parser.add_argument("--use_deterministic_connect", action='store_true',
                        help="Use deterministic inference to generate latent codes, i.e., standard auto-encoders.")
    parser.add_argument("--use_pretrained_model", action='store_true',
                        help="Use pre-trained auto-encoder models as the initialization")
    parser.add_argument("--latent_as_gpt_memory", default=1, type=int,
                        help="Latent vector as memery for GPT2 to attend.")
    parser.add_argument("--latent_as_gpt_emb", default=1, type=int, help="Latent vector as embeddings for GPT2.")

    ## Objective functions
    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="The weighting hyper-parameter of the KL term in VAE")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="Optional input sequence length before tokenization. The sequence will be dropped if it is longer the max_seq_length")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval_rec", action='store_true',
                        help="Whether to run eval reconstruction on a set of models.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    # Training Schedules
    parser.add_argument("--ratio_increase", default=0.25, type=float,
                        help="Learning schedule, the percentage for the annealing stage.")
    parser.add_argument("--ratio_zero", default=0.25, type=float,
                        help="Learning schedule, the percentage for the pure auto-encoding stage.")
    parser.add_argument("--fb_mode", default=0, type=int,
                        help="free bit training mode.")
    parser.add_argument("--dim_target_kl", default=3.0, type=float,
                        help="dim_target_kl free bit training mode.")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--use_philly", action='store_true',
                        help="Use Philly for computing.")
    parser.add_argument("--use_pretrained_vae", action='store_true',
                        help="Use use_pretrained_vae as initialization, where beta value is specified in the folder")
    parser.add_argument("--use_random_weight", action='store_true',
                        help="Use random weights as initialization")

    ## IO: Logging and Saving
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gloabl_step_eval', type=int, default=661,
                        help="Evaluate the results at the given global step")

    # Precision & Distributed Training
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--cls_step', type=str, default='1,4,5')
    parser.add_argument('--att_val_list', type=str, default='1')
    parser.add_argument('--sampling_num', type=int, default=200)
    parser.add_argument('--data_type', type=str, default='sentiment')
    parser.add_argument('--weight_energy', type=float, default=1.0)
    parser.add_argument('--fix_model', type=int, default=8, )
    parser.add_argument('--reg_logits', type=float, default='0.01')
    args = parser.parse_args()
    if args.fix_model == 84:
        MODEL_CLASSES['bertu'] = BertForLatentConnectorAVG
        if 'large' in args.decoder_model_name_or_path:
            MODEL_CLASSES['gpt2'] = GPT2ForLatentConnectorNew
        else:
            MODEL_CLASSES['gpt2'] = GPT2ForLatentConnectorNew2

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    set_seed(args)

    gan_dir = os.path.join(args.gan_dir, 'training_gan.bin')
    gan_checkpoint = torch.load(gan_dir)
    global_step = args.gloabl_step_eval

    output_full_dir = os.path.join(args.checkpoint_dir, 'checkpoint-full-{}'.format(global_step))
    checkpoint = torch.load(os.path.join(output_full_dir, 'training.bin'))

    ## Encoder
    encoder_model_class = MODEL_CLASSES[args.encoder_model_type]
    tokenizer_encoder = AutoTokenizer.from_pretrained(
        args.encoder_tokenizer_name if args.encoder_tokenizer_name else args.encoder_model_name_or_path,
        do_lower_case=args.do_lower_case)

    if args.block_size <= 0:
        args.block_size = tokenizer_encoder.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer_encoder.max_len_single_sentence)

    model_encoder = encoder_model_class.from_pretrained(args.encoder_model_name_or_path, latent_size=args.latent_size,
                                                        pad_id=tokenizer_encoder.pad_token_id)

    ## Decoder
    decoder_model_class = MODEL_CLASSES[args.decoder_model_type]
    tokenizer_decoder = AutoTokenizer.from_pretrained(
        args.decoder_tokenizer_name if args.decoder_tokenizer_name else args.decoder_model_name_or_path,
        do_lower_case=args.do_lower_case)
    if args.block_size <= 0:
        args.block_size = tokenizer_decoder.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer_decoder.max_len_single_sentence)

    if args.latent_as_gpt_emb + args.latent_as_gpt_memory == 0:
        return  # latent vector should pass into GPT to decode
    else:
        latent_as_gpt_emb = True if args.latent_as_gpt_emb == 1 else False
        latent_as_gpt_memory = True if args.latent_as_gpt_memory == 1 else False

    model_decoder = decoder_model_class.from_pretrained(args.decoder_model_name_or_path, latent_size=args.latent_size,
                                                        latent_as_gpt_emb=latent_as_gpt_emb,
                                                        latent_as_gpt_memory=latent_as_gpt_memory)
    decoder_n_layer = model_decoder.transformer.config.n_layer
    if args.fix_model == 8:
        print("Initialize the Extra Layer.")
        model_decoder.transformer.h[decoder_n_layer].load_state_dict(model_decoder.transformer.h[0].state_dict())
        model_decoder.transformer.change_order()
    elif args.fix_model == 84:
        print("Initialize the Extra Layer.")
        model_decoder.transformer.change_order()
    # Chunyuan: Add Padding token to GPT2
    special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>', }
    num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)
    print('We have added', num_added_toks, 'tokens to GPT2')
    model_decoder.resize_token_embeddings(len(
        tokenizer_decoder))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.

    assert tokenizer_decoder.pad_token == '<PAD>'

    model_vae = VAE(model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, args)
    gan = GAN(args)
    model_vae.load_state_dict(checkpoint['model_state_dict'], strict=False)
    gan.load_state_dict(gan_checkpoint['model_state_dict'], strict=True)
    logger.info("Pre-trained Optimus is successfully loaded")
    model_vae.to(args.device)  #
    gan.to(args.device)
    model_kwargs = {
        'model': model_vae,
        'enc_tokenizer': tokenizer_encoder,
        'dec_tokenizer': tokenizer_decoder,
        'sampling_num': args.sampling_num,
        'args': args
    }
    sde_kwargs = {}
    ode_kwargs = {'atol': 1e-3, 'rtol': 1e-3, 'method': 'dopri5', 'use_adjoint': True, 'latent_dim': args.latent_size}
    ld_kwargs = {'batch_size': args.per_gpu_eval_batch_size, 'sgld_lr': 1,
                 'sgld_std': 1e-2, 'n_steps': 20}
    sample_q = sample_q_ode  # sample_q_ode
    latent_dim = args.latent_size
    n_classes = args.n_classes
    save_path = os.path.join(args.output_dir, 'sample')
    ## find weight
    wnl = WordNetLemmatizer()

    ## cls
    cls_list_all = args.cls_step.split(';')
    weight_dict = {}
    file_path_dict = {}
    done_cnt = 0
    for word in cls_list_all:
        weight_dict[word] = [0, False, 0]  # weight, done, acc
        path = '/home/guangyi/Optimus-ODE/output_home/LM/large/yelp/sample'
        file_path_dict[word] = path + '/sampling_word_' + word + '_1.0.txt'
    weight_all_list =list(range(3,10,2))+ list(range(10,100,8)) #base: list(range(1,10)) + list(range(10,50,4)) #list(range(46,70,4)) #
    all_num = len(cls_list_all)
    pbar = tqdm(weight_all_list)
    acc = 0
    for weight in pbar:
        if done_cnt == all_num:
            print("Finished",all_num)
            break
        args.weight_energy = weight
        pbar1 = tqdm(enumerate(cls_list_all), disable=True)
        for jj, cls_list in pbar1:
            if weight_dict[cls_list][1]:
                continue
            cls_step_list = cls_list.split(',')
            classifier_list = []
            for ii, cls_ in enumerate(cls_step_list):
                if cls_.isdigit():
                    num_classes = STEP_CLASSES[cls_]
                else:
                    # print('Target Word Setting:', cls_)
                    pbar.set_description('%s' % (str(jj)+'/'+str(all_num)+'-'+cls_+'-'+str(acc) +' Weight:'+str(weight)+ ' Done ' +str(done_cnt)))
                    num_classes = 2
                args.cls_ = cls_
                classifier_ = DenseEmbedder(args.latent_size, up_dim=2, depth=4, num_classes=num_classes)
                cls_dir = os.path.join(args.cls_dir + cls_, 'training_cls.bin')
                cls_checkpoint = torch.load(cls_dir)
                classifier_.load_state_dict(cls_checkpoint['model_state_dict'], strict=False)
                classifier_.set_energy_weight(weight)
                if ii == 1:
                    classifier_.set_energy_weight(weight)
                classifier_list.append(classifier_)
            classifier = CCF(classifier_list)
            classifier.to(args.device)
            # if first_flag:

            condSampling = ConditionalSampling(sample_q, args.per_gpu_eval_batch_size, latent_dim, n_classes,
                                               classifier,
                                               device, save_path, ode_kwargs, ld_kwargs, sde_kwargs,
                                               every_n_plot=5, model_kwargs=model_kwargs, gan=gan, disable_bar=True)

            test = condSampling.get_samples_multiple(args.att_val_list.split(';'), mode=[1, 0, 0], out_sentences=True)

            cls = [0, 0]
            for line in test:
                success_flag = False
                tokens = word_tokenize(line)
                for token in tokens:
                    if wnl.lemmatize(token, 'n') == cls_list or wnl.lemmatize(token, 'v') == cls_list:
                        cls[1] += 1
                        success_flag = True
                        break
                    # elif cls_list == 'bag' and cls_list in line:
                    #     cls[1] +=1
                    #     success_flag = True
                    #     break
                if success_flag:
                    continue
            acc = float(cls[1]) / len(test)
            weight_dict[cls_list][2] = round(acc, 2)
            if acc >= 0.55:
                weight_dict[cls_list][1] = True
                weight_dict[cls_list][0] = weight
                done_cnt += 1
                # print('Done\t' + str(done_cnt))
                pbar.set_description('%s' % (str(jj)+'/'+str(all_num)+'-'+cls_+'-'+str(acc) +' Weight:'+str(weight)+ ' Done ' +str(done_cnt)))
            if weight == weight_all_list[-1] and not weight_dict[cls_list][1]: #last one
                weight_dict[cls_list][0] = weight
    json_str = json.dumps(weight_dict, indent=4)
    with open(path + '/weight_'+str(all_num)+'.json', 'w') as json_file:
        json_file.write(json_str)


if __name__ == "__main__":
    main()
