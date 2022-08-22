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
from my_transformers import *
import argparse
import logging
import os
import random
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn.init as init
# from run_latent_generation import sample_sequence_conditional
from nltk.translate.bleu_score import corpus_bleu
from transformers import AdamW  # ,OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from transformers import AutoTokenizer

from modules import VAE
from utils import (calc_iwnll, calc_mi, calc_au, BucketingDataLoader, TextDataset_Split,
                   TextDataset_2Tokenizers, frange_cycle_zero_linear)

# BertConfig, BertForLatentConnectorNew,
# GPT2Config, GPT2ForLatentConnectorNew,
#
# RobertaConfig)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': GPT2ForLatentConnector,
    # 'openai-gpt': (None, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bertu': BertForLatentConnector,
    'bert': BertForLatentConnector,
    'roberta': RobertaForLatentConnector,
    'deberta': DebertaForLatentConnector,
    't5': T5EncoderForLatentConnector,
    'albert':AlbertForLatentConnector,
}

parameter_name = []


def load_and_cache_examples(args, tokenizer, evaluate=False):
    if isinstance(tokenizer, list):
        dataset = TextDataset_2Tokenizers(tokenizer, args,
                                          file_path=args.eval_data_file if evaluate else args.train_data_file,
                                          block_size=args.block_size)
    else:
        dataset = TextDataset_Split(tokenizer, args,
                                    file_path=args.eval_data_file if evaluate else args.train_data_file,
                                    block_size=args.block_size)
    return dataset


def build_dataload_and_cache_examples(args, tokenizer, evaluate=False):
    if isinstance(tokenizer, list):
        if not evaluate:
            args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
            file_path = args.train_data_file
        else:
            args.batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
            file_path = args.eval_data_file
        dataloader = BucketingDataLoader(file_path, args.batch_size, args.max_seq_length, tokenizer, args, bucket=100,
                                         shuffle=False)
    else:
        pass
    return dataloader


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def mask_tokens(inputs, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

    masked_indices = torch.bernoulli(torch.full(labels.shape, args.mlm_probability)).to(torch.uint8)
    labels[masked_indices == 1] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).to(torch.uint8) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).to(torch.uint8) & masked_indices & ~indices_replaced
    indices_random = indices_random
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def weights_init_rondom(model):
    model = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_state_dict = model.state_dict()
    for key in model_state_dict:
        #         pdb.set_trace()
        if 'encoder' in key:
            init.normal_(model_state_dict[key].data)
            # weight_init(item)


def save_checkpoint(model_vae, optimizer, global_step, args):
    # Create output directory if needed
    # Save model checkpoint
    save_last = 1
    model_to_save = model_vae.module if hasattr(model_vae,
                                                'module') else model_vae  # Take care of distributed/parallel training
    state_dict_new = {}
    state_dict = model_to_save.state_dict()
    for key in parameter_name:
        state_dict_new[key] = state_dict[key]
    checkpoint = {
        'iter': global_step,
        'model_state_dict': state_dict_new,  # model_to_save.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        'beta': model_to_save.args.beta,
        'args': args
    }

    output_full_dir = os.path.join(args.output_dir, 'checkpoint-full-{}'.format(save_last))
    if not os.path.exists(output_full_dir) and args.local_rank in [-1, 0]:
        os.makedirs(output_full_dir)

    logger.info("Start saving full model checkpoint to %s", output_full_dir)
    if args.use_philly:
        save_solid = False
        n_save_attempts = 0
        while not save_solid:
            try:
                n_save_attempts += 1
                logger.info(f"Saving full checkpoint: {n_save_attempts} attempts made")
                torch.save(checkpoint, os.path.join(output_full_dir, 'training.bin'))
                logger.info("Saving full checkpoint to %s,", output_full_dir)
                save_solid = True
            except:
                pass
    else:
        torch.save(checkpoint, os.path.join(output_full_dir, 'training.bin'))
        logger.info("Saving full checkpoint to %s", output_full_dir)


def train(args, train_dataloader, model_vae, encoder_tokenizer, decoder_tokenizer, table_name, checkpoint=None):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter('./runs/' + args.output_dir.split('/')[-2] + '/' + args.output_dir.split('/')[-1])

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)

    # model_encoder, model_decoder, model_connector = model_vae.encoder,  model_vae.decoder, model_vae.linear
    no_decay = ['bias', 'LayerNorm.weight']
    if args.fix_model == 0:  # no fix o
        print('\nNo Fixed\n')
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_vae.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model_vae.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        parameter_name.extend([n for n, p in model_vae.named_parameters()])
    elif args.fix_model == 1:  # fix both bert & gpt
        print('\nFix BERT & GPT2\n')
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_vae.named_parameters() if ('linear' in n or 'pooler' in n)],
             'weight_decay': 0.0}
        ]
        parameter_name.extend([n for n, p in model_vae.named_parameters() if ('linear' in n or 'pooler' in n)])

    elif args.fix_model == 2 or args.fix_model == 7:  # fix gpt
        print('\nOnly Fix GPT2\n')
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_vae.encoder.named_parameters()], 'weight_decay': 0.0},
            {'params': [p for n, p in model_vae.decoder.named_parameters() if 'linear' in n],
             'weight_decay': 0.0}
        ]
        parameter_name.extend([n for n, p in model_vae.named_parameters() if ('linear' in n or 'encoder' in n)])
    elif args.fix_model == 3:
        print('\nFix BERT & GPT2, train extra layers\n')
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_vae.named_parameters() if
                        ('linear' in n or 'decoder.transformer.h.0' in n)],
             'weight_decay': 0.0}
        ]
        parameter_name.extend([n for n, p in model_vae.named_parameters() if
                               ('linear' in n or 'decoder.transformer.h.0' in n)])
    elif args.fix_model == 4:
        print('\nFix BERT & GPT2, train extra layers and the BERT Pooler\n')
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_vae.named_parameters() if
                        ('linear' in n or 'decoder.transformer.h.0' in n or 'pooler' in n or 'wte' in n )],
             'weight_decay': 0.0}
        ]
        parameter_name.extend([n for n, p in model_vae.named_parameters() if
                               ('linear' in n or 'decoder.transformer.h.0' in n or 'pooler' in n or 'wte' in n )])
    elif args.fix_model == 5:
        print('\nFix BERT, train extra BERT layer and the BERT Pooler\n')
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_vae.named_parameters() if
                        ('linear' in n or 'pooler' in n)],
             'weight_decay': 0.0}
        ]
        parameter_name.extend([n for n, p in model_vae.named_parameters() if
                               ('linear' in n or 'pooler' in n)])
    elif args.fix_model == 6:
        print('\nFix GPT2, train BERT and extra layer in GPT2.\n')
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_vae.named_parameters() if
                        ('encoder' in n or 'decoder.transformer.h.0' in n or 'linear' in n)],
             'weight_decay': 0.0}
        ]
        parameter_name.extend([n for n, p in model_vae.named_parameters() if
                               ('encoder' in n or 'decoder.transformer.h.0' in n or 'linear' in n)])
    elif args.fix_model == 8 or args.fix_model == 84 or args.fix_model == 85:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_vae.named_parameters() if
                        ('linear' in n or 'wte' in n or 'decoder.transformer.h.0' in n or 'encoder' in n)],
             'weight_decay': 0.0}
        ]
        parameter_name.extend([n for n, p in model_vae.named_parameters() if
                               ('linear' in n or 'wte' in n or 'decoder.transformer.h.0' in n or 'encoder' in n)])
    elif args.fix_model == 81:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_vae.named_parameters() if
                        ('linear' in n or 'wte' in n or 'decoder.transformer.h.0' in n or 'encoder' in n or 'lm_head' in n)],
             'weight_decay': 0.0}
        ]
        parameter_name.extend([n for n, p in model_vae.named_parameters() if
                               ('linear' in n or 'wte' in n or 'decoder.transformer.h.0' in n or 'encoder' in n or 'lm_head' in n)])
    elif args.fix_model == 9:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_vae.named_parameters() if
                        ('linear' in n or 'wte' in n or 'encoder' in n)],
             'weight_decay': 0.0}
        ]
        parameter_name.extend([n for n, p in model_vae.named_parameters() if
                               ('linear' in n or 'wte' in n or 'encoder' in n)])
    elif args.fix_model == 10:

        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_vae.named_parameters() if
                        ('linear' in n or 'wte' in n or 'encoder' in n or 'adapter' in n or 'output.LayerNorm' in n)],
             'weight_decay': 0.0}
        ]
        parameter_name.extend([n for n, p in model_vae.named_parameters() if
                               ('linear' in n or 'wte' in n or 'encoder' in n or 'adapter' in n or 'output.LayerNorm' in n)])
    elif args.fix_model == 11 or args.fix_model == 12:

        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_vae.named_parameters() if
                        ('linear' in n or 'encoder' in n or 'adapter' in n or 'output.LayerNorm' in n)],
             'weight_decay': 0.0}
        ]
        parameter_name.extend([n for n, p in model_vae.named_parameters() if
                               ('linear' in n or 'encoder' in n or 'adapter' in n or 'output.LayerNorm' in n)])
    elif args.fix_model == 13:
        # print('\nFix BERT & GPT2, train extra layers and the BERT Pooler\n')
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_vae.named_parameters() if
                        ('linear' in n or 'decoder.transformer.h.0' in n or 'decoder.transformer.h.13' in n or 'encoder' in n)],
             'weight_decay': 0.0}
        ]
        parameter_name.extend([n for n, p in model_vae.named_parameters() if
                               ('linear' in n or 'decoder.transformer.h.0' in n or 'decoder.transformer.h.13' in n or 'encoder' in n)])
    elif args.fix_model == 14:
        # print('\nFix BERT & GPT2, train extra layers and the BERT Pooler\n')
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_vae.named_parameters() if
                        ('linear' in n or 'decoder.transformer.h.0' in n or 'decoder.transformer.h.13' in n or 'encoder' in n or 'wte' in n)],
             'weight_decay': 0.0}
        ]
        parameter_name.extend([n for n, p in model_vae.named_parameters() if
                               ('linear' in n or 'decoder.transformer.h.0' in n or 'decoder.transformer.h.13' in n or 'encoder' in n or 'wte' in n)])
    elif args.fix_model == 82:
        # print('\nFix BERT & GPT2, train extra layers and the BERT Pooler\n')
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_vae.named_parameters() if
                        ('linear' in n or 'decoder.transformer.h.0' in n or 'decoder.transformer.h.13' in n or 'encoder' in n or 'wte' in n or 'lm_head' in n or 'embeddings.LayerNorm' in n)],
             'weight_decay': 0.0}
        ]
        parameter_name.extend([n for n, p in model_vae.named_parameters() if
                               ('linear' in n or 'decoder.transformer.h.0' in n or 'decoder.transformer.h.13' in n or 'encoder' in n or 'wte' in n or 'lm_head' in n)])
    elif args.fix_model == 83:
        # print('\nFix BERT & GPT2, train extra layers and the BERT Pooler\n')
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_vae.named_parameters() if
                        (
                                    'linear' in n  or 'decoder.transformer.h.12' in n or 'encoder' in n or 'lm_head' in n)],
             'weight_decay': 0.0}
        ]
        parameter_name.extend([n for n, p in model_vae.named_parameters() if
                               ('linear' in n or 'decoder.transformer.h.12' in n or 'encoder' in n or 'lm_head' in n)])
    elif args.fix_model == 881: # wte, lm_head, embd LN
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_vae.named_parameters() if
                        ('linear' in n or 'wte' in n or 'lm_head' in n or 'decoder.transformer.h.0' in n or 'encoder' in n or 'embeddings.LayerNorm' in n)],
             'weight_decay': 0.0}
        ]
        parameter_name.extend([n for n, p in model_vae.named_parameters() if
                               ('linear' in n or 'wte' in n or 'decoder.transformer.h.0' in n or 'encoder' in n or 'embeddings.LayerNorm' in n)])
    elif args.fix_model == 882: # wte, embd LN
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_vae.named_parameters() if
                        ('linear' in n or 'wte' in n or 'decoder.transformer.h.0' in n or 'encoder' in n or 'embeddings.LayerNorm' in n)],
             'weight_decay': 0.0}
        ]
        parameter_name.extend([n for n, p in model_vae.named_parameters() if
                               ('linear' in n or 'wte' in n or 'decoder.transformer.h.0' in n or 'encoder' in n or 'embeddings.LayerNorm' in n)])

    elif args.fix_model == 883:  # wte, lm_head, embd LN, adapter, LM
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_vae.named_parameters() if
                        ('linear' in n or 'wte' in n or 'lm_head' in n or 'decoder.transformer.h.0' in n or 'encoder' in n or 'LayerNorm' in n or 'adapter' in n)],
             'weight_decay': 0.0}
        ]
        parameter_name.extend([n for n, p in model_vae.named_parameters() if
                               ('linear' in n or 'wte' in n or 'decoder.transformer.h.0' in n or 'encoder' in n or 'LayerNorm' in n or 'adapter' in n)])

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # optimizer = AdamW(optimizer_grouped_parameters,
    # lr=checkpoint['optimizer_state_dict']['param_groups'][0]['lr'], eps=args.adam_epsilon)
    # extra_steps = t_total // args.num_train_epochs
    # args.warmup_steps = extra_steps // 5
    from transformers import get_polynomial_decay_schedule_with_warmup,get_cosine_schedule_with_warmup
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, args.warmup_steps, num_training_steps=t_total, lr_end=5e-7, power=3.0)
    # news2 = get_polynomial_decay_schedule_with_warmup(optimizer, args.warmup_steps, num_training_steps=t_total,
    #                                                   lr_end=1e-7, power=2.0)
    # newscos5 =get_cosine_schedule_with_warmup(optimizer,0,t_total,0.5)
    # newscos9 = get_cosine_schedule_with_warmup(optimizer, 0, t_total, 0.9)
    # def print_lr(news):
    #     tmp = [str(news.lr_lambdas[0](i*1e4)) for i in range(11)]
    #     print('\t'.join(tmp))

    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model_vae, optimizer = amp.initialize(model_vae, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model_vae = torch.nn.DataParallel(model_vae, device_ids=range(args.n_gpu)).to(args.device)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model_vae = torch.nn.parallel.DistributedDataParallel(model_vae, device_ids=[args.local_rank],
                                                              output_device=args.local_rank,
                                                              find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_dataloader.num_examples)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    train_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model_vae.zero_grad()

    model_vae = model_vae.module if hasattr(model_vae,
                                            'module') else model_vae  # Take care of distributed/parallel training

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.disable_bar)

    n_iter = int(args.num_train_epochs) * len(train_dataloader)
    beta_t_list = frange_cycle_zero_linear(n_iter, start=0.0, stop=args.beta, n_cycle=int(args.num_train_epochs),
                                           ratio_increase=args.ratio_increase, ratio_zero=args.ratio_zero)
    tmp_list = []
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    with torch.no_grad():
        result_new = calc_rec_lgy(model_vae, encoder_tokenizer, decoder_tokenizer, args, ns=100)
        result_new.update(evaluate(args, model_vae, encoder_tokenizer, decoder_tokenizer, table_name))
        for key, value in result_new.items():
            logger.info('eval_%s:%f',key,value)
            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
    logger.info('\nBLEU is %f\n"', result_new['bleu'])
    args.logging_steps = len(train_dataloader) // args.gradient_accumulation_steps
    args.save_steps = args.logging_steps
    best_bleu = 0
    final_beta = args.beta
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=args.disable_bar)  # , disable=args.local_rank not in [-1, 0] args.logging_steps//50
        for step, batch in enumerate(epoch_iterator):

            tokenized_text0, tokenized_text1, tokenized_text_lengths = batch

            inputs, labels = mask_tokens(tokenized_text0, encoder_tokenizer, args) if args.mlm else (
                tokenized_text0, tokenized_text1)
            labels = tokenized_text1

            tokenized_text1 = tokenized_text1.to(args.device)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            model_vae.train()
            beta_t = beta_t_list[step + epoch * len(epoch_iterator)]
            # model_vae.args.fb_mode = 1
            if args.n_gpu == 1:
                model_vae.args.beta = beta_t
                if beta_t == 0.0:
                    model_vae.args.fb_mode = 3
                else:
                    model_vae.args.fb_mode = 1

                if args.use_deterministic_connect:
                    model_vae.args.fb_mode = 2
                if final_beta == 0.0:
                    model_vae.args.fb_mode = 3
            else:
                model_vae.module.args.beta = beta_t

                if beta_t == 0.0:
                    model_vae.module.args.fb_mode = 0
                else:
                    model_vae.module.args.fb_mode = 1

                if args.use_deterministic_connect:
                    model_vae.module.args.fb_mode = 2

            loss_rec, loss_kl, loss, mu, std = model_vae(inputs, labels, std=True)

            if train_step % 100 == 0:
                tb_writer.add_scalar('loss_rec_train', loss_rec.mean().item(), train_step)
                tb_writer.add_scalar('loss_kl_train', loss_kl.mean().item(), train_step)
                tb_writer.add_scalar('loss_train', loss.mean().item(), train_step)
                tb_writer.add_scalar('beta_train', beta_t, train_step)
                tb_writer.add_scalar('lr_train', scheduler.get_last_lr()[0], train_step)
                tb_writer.add_scalar('std', std.mean().item(), train_step)
                tb_writer.add_scalar('mean', mu.mean().item(), train_step)
            train_step += 1

            loss_rec = loss_rec.mean()  # mean() to average on multi-gpu parallel training
            loss_kl = loss_kl.mean()
            loss = loss.mean()

            if args.n_gpu == 1:
                beta_ = model_vae.args.beta
            else:
                beta_ = model_vae.module.args.beta
            if args.use_philly:
                epoch_iterator.set_description(
                    (
                        f'progress: {round(100 * (step + epoch * len(epoch_iterator)) / (int(args.num_train_epochs) * len(epoch_iterator)), 4)};'
                        f'iter: {step + epoch * len(epoch_iterator)}; loss: {loss.item():.3f}; '
                        f'loss_rec: {loss_rec.item():.3f}; loss_kl: {loss_kl.item():.3f}; '
                        f'beta: {beta_:.3f}'
                    )
                )
            else:
                if train_step % (args.logging_steps // 50) == 0:
                    epoch_iterator.set_description(
                        (
                            f'iter: {step + epoch * len(epoch_iterator)}; loss: {loss.item():.3f}; '
                            f'loss_rec: {loss_rec.item():.3f}; loss_kl: {loss_kl.item():.3f}; '
                            f'beta: {beta_:.3f}'
                        )
                    )

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model_vae.parameters(), args.max_grad_norm)

                optimizer.step()

                scheduler.step()  # Update learning rate schedule

                model_vae.zero_grad()

                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        #                         args.per_gpu_eval_batch_size = args.per_gpu_eval_batch_size // 2
                        results = evaluate(args, model_vae, encoder_tokenizer, decoder_tokenizer, table_name)
                        #                         args.per_gpu_eval_batch_size = args.per_gpu_eval_batch_size * 2
                        results.update(calc_rec_lgy(model_vae, encoder_tokenizer, decoder_tokenizer, args, ns=100))
                        #                         results['ppl_sample'] = sampling_lgy(model_vae, decoder_tokenizer, args, LM_model, LM_tokenizer)['ppl']
                        #                         results['bleu'] = result_new['bleu']
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                    # if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    if results['bleu'] >= best_bleu:
                        best_bleu = results['bleu']
                        if not args.no_save:
                            save_checkpoint(model_vae, optimizer, global_step, args)


            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    results = calc_rec_lgy(model_vae, encoder_tokenizer, decoder_tokenizer, args, ns=100)
    #     results['ppl_sample'] = sampling_lgy(model_vae, decoder_tokenizer, args, LM_model, LM_tokenizer, cnt=1000)['ppl']
    for key, value in results.items():
        tb_writer.add_scalar('final_{}'.format(key), value, global_step)
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, optimizer


def evaluate(args, model_vae, encoder_tokenizer, decoder_tokenizer, table_name, prefix="", subset="test"):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    logger.info("***** Running evaluation on {} dataset *****".format(subset))

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_dataloader = build_dataload_and_cache_examples(args, [encoder_tokenizer, decoder_tokenizer], evaluate=True)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model_vae.eval()

    model_vae = model_vae.module if hasattr(model_vae,
                                            'module') else model_vae  # Take care of distributed/parallel training
    mi = calc_mi(model_vae, eval_dataloader, args)
    au = calc_au(model_vae, eval_dataloader, delta=0.01, args=args)[0]
    ppl, elbo, nll, kl = calc_iwnll(model_vae, eval_dataloader, args, ns=100)

    result = {
        "perplexity": ppl, "elbo": elbo, "kl": kl, "nll": nll, "au": au, "mi": mi
    }

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    row = {
        'PartitionKey': 'MILU_Rule_Rule_Template',
        'RowKey': str(datetime.now()),
        'ExpName': args.ExpName,
        'test_perplexity': str(ppl),
        'test_elbo': str(elbo),
        'test_nll': str(nll),
        'test_au': str(au),
        'test_mi': str(mi)
    }
    # pdb.set_trace()
    # ts.insert_entity(table_name, row)

    return result


def calc_rec_lgy(model_vae, encoder_tokenizer, decoder_tokenizer, args, ns=1):
    from modules import sample_sequence_conditional
    eval_dataloader = build_dataload_and_cache_examples(args, [encoder_tokenizer, decoder_tokenizer], evaluate=True)
    count = 0
    result = defaultdict(str)
    ref = []
    cand = []
    for batch in tqdm(eval_dataloader, desc="Evaluating recontruction", disable=args.disable_bar):
        x0, x1, x_lengths = batch
        max_len_values, _ = x_lengths.max(0)
        x0 = x0[:, :max_len_values[0]]
        x1 = x1[:, :max_len_values[1]]
        x0 = x0.to(args.device)
        x1 = x1.to(args.device)
        x_lengths = x_lengths.to(args.device)
        context_tokens = decoder_tokenizer.encode('<BOS>')
        with torch.no_grad():
            # text_x0 = encoder_tokenizer.decode(x0[0,:x_lengths[0,0]].tolist(), clean_up_tokenization_spaces=True)[0]
            # result["INPUT TEXT " + str(count)].append(text_x0)
            attention_mask = (x0 != encoder_tokenizer.pad_token_id).float()

            pooled_hidden_fea = model_vae.encoder(x0, attention_mask)[1]

            # Connect hidden feature to the latent space
            # latent_z, loss_kl = model_vae.connect(pooled_hidden_fea)
            mean, logvar = model_vae.encoder.linear(pooled_hidden_fea).chunk(2, -1)
            # latent_z = model_vae.reparameterize(mean, logvar, nsamples=1).squeeze(1)
            latent_z = mean.squeeze(1)

            past = latent_z
            out = sample_sequence_conditional(
                model=model_vae.decoder,
                context=context_tokens,
                past=past,
                length=x_lengths[0, 1],  # Chunyuan: Fix length; or use <EOS> to complete a sentence
                num_samples=latent_z.size(0),
                device=args.device,
                decoder_tokenizer=decoder_tokenizer,
                eos_id=model_vae.eos_token_id
            )

        for i in range(latent_z.size(0)):
            text_x0_ = decoder_tokenizer.decode(x1[i, :].tolist(), clean_up_tokenization_spaces=False).split('<EOS>')[
                0].replace('<BOS>', '').strip()
            text_x0_ = text_x0_.split()
            text_x1 = decoder_tokenizer.decode(out[i, :].tolist(), clean_up_tokenization_spaces=False).split('<EOS>')[
                0].replace('<BOS>', '').strip()
            text_x1 = text_x1.split()

            count += 1
            ref.append([text_x0_])
            cand.append(text_x1)

        if count > 1000:
            break
    bleu = corpus_bleu(ref, cand) * 100
    logger.info("  BLEU = %s", str(round(bleu, 2)))
    output_eval_file = os.path.join(args.output_dir, "eval_results_bleu.txt")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(output_eval_file, "w") as writer:
        writer.write("%s = %s\n" % ('bleu', str(bleu)))
    return {'bleu': bleu}


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--checkpoint_dir", default=None, type=str,
                        help="The directory where checkpoints are saved.")
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
    parser.add_argument("--ratio_zero", default=0.5, type=float,
                        help="Learning schedule, the percentage for the pure auto-encoding stage.")
    parser.add_argument("--fb_mode", default=1, type=int,
                        help="free bit training mode.")
    parser.add_argument("--dim_target_kl", default=3.0, type=float,
                        help="dim_target_kl free bit training mode.")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-4, type=float,
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
    ############## Mine
    parser.add_argument('--fix_model', type=int, default=0,
                        help="0: no fix; 1: fix both bert & gpt; 2: fix gpt; 3: fix both bert & gpt, extra layers")
    parser.add_argument('--disable_bar', action='store_true')
    parser.add_argument('--no_save', action='store_true')
    args = parser.parse_args()
    if args.fix_model == 3 or args.fix_model == 4:
        MODEL_CLASSES['gpt2'] = GPT2ForLatentConnectorNew
        MODEL_CLASSES['bert'] = BertForLatentConnectorNew
        MODEL_CLASSES['roberta'] = RobertaForLatentConnectorNew
        MODEL_CLASSES['deberta'] = DebertaForLatentConnectorNew
    elif args.fix_model == 5:
        # gpt2 unchanged
        MODEL_CLASSES['bert'] = BertForLatentConnectorNew
        MODEL_CLASSES['roberta'] = RobertaForLatentConnectorNew
        MODEL_CLASSES['deberta'] = DebertaForLatentConnectorNew
    elif args.fix_model == 6 or args.fix_model == 8 or args.fix_model == 8 or args.fix_model == 83 or args.fix_model == 881  or args.fix_model == 882 or args.fix_model == 883:
        MODEL_CLASSES['gpt2'] = GPT2ForLatentConnectorNew#
    elif args.fix_model == 84:
        MODEL_CLASSES['bertu'] = BertForLatentConnectorAVG
        if 'large' in args.decoder_model_name_or_path:
            MODEL_CLASSES['gpt2'] = GPT2ForLatentConnectorNew
        else:
            MODEL_CLASSES['gpt2'] = GPT2ForLatentConnectorNew2

    elif args.fix_model == 85:
        MODEL_CLASSES['bertu'] = BertForLatentConnectorAVG
        MODEL_CLASSES['gpt2'] = GPT2ForLatentConnectorNew
    elif args.fix_model == 13 or args.fix_model == 14 or args.fix_model == 82:
        MODEL_CLASSES['gpt2'] = GPT2ForLatentConnectorNew2

    # if args.decoder_model_type in ["bert", "roberta"] and not args.mlm:
    #     raise ValueError("BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
    #                      "flag (masked language modeling).")
    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument.")

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    args.ExpName = 'Vae_' + args.dataset + '_Nz_' + str(args.latent_size) + '_Beta_' + str(args.beta) + '_Dkl_' + str(
        args.dim_target_kl) + '_Ra_' + str(args.ratio_increase) + '_R0_' + str(args.ratio_zero)
    table_name = 'Vae' + args.dataset + 'Nz' + str(args.latent_size)
    try:
        ts.create_table(table_name)
    except:
        pass

    # Set seed
    set_seed(args)
    # if 'roberta' in args.encoder_model_type:
    #     print("This is ROBERTA, block size modified")
    #     args.block_size = args.block_size + 1
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    # Load Optimius pre-trained model and tokenizer
    checkpoint = None
    if args.use_pretrained_model:
        global_step = args.gloabl_step_eval
        output_full_dir = os.path.join(args.checkpoint_dir, 'checkpoint-full-{}'.format(global_step))
        checkpoint = torch.load(os.path.join(output_full_dir, 'training.bin'))

    encoder_model_class = MODEL_CLASSES[args.encoder_model_type]
    # encoder_config = encoder_config_class.from_pretrained(
    #     args.encoder_config_name if args.encoder_config_name else args.encoder_model_name_or_path)

    tokenizer_encoder = AutoTokenizer.from_pretrained(
        args.encoder_tokenizer_name if args.encoder_tokenizer_name else args.encoder_model_name_or_path,
        do_lower_case=args.do_lower_case)
    if args.block_size <= 0:
        args.block_size = tokenizer_encoder.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer_encoder.max_len_single_sentence)

    model_encoder = encoder_model_class.from_pretrained(args.encoder_model_name_or_path, latent_size=args.latent_size,
                                                        pad_id=tokenizer_encoder.pad_token_id)

    # model_encoder.to(args.device)

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

    # setattr(decoder_config, "latent_size", args.latent_size)
    model_decoder = decoder_model_class.from_pretrained(args.decoder_model_name_or_path, latent_size=args.latent_size,
                                                        latent_as_gpt_emb=latent_as_gpt_emb,
                                                        latent_as_gpt_memory=latent_as_gpt_memory)

    # model_decoder = decoder_model_class.from_pretrained(args.decoder_model_name_or_path,
    #                                                     from_tf=bool('.ckpt' in args.decoder_model_name_or_path),
    #                                                     config=decoder_config, latent_size=args.latent_size,
    #                                                     latent_as_gpt_emb=latent_as_gpt_emb,
    #                                                     latent_as_gpt_memory=latent_as_gpt_memory)
    decoder_n_layer = model_decoder.transformer.config.n_layer
    if args.fix_model == 3 or args.fix_model == 4:
        print("Initialize the Extra Layer.")
        model_encoder.linear_forbert.load_state_dict(model_encoder.encoder.layer[-1].state_dict())
        model_decoder.transformer.h[decoder_n_layer].load_state_dict(model_decoder.transformer.h[0].state_dict())
        model_decoder.transformer.change_order()
        print('Change the Order of Decoder Layers')
    elif args.fix_model == 5:
        print("Initialize the Extra Layer.")
        model_encoder.linear_forbert.load_state_dict(model_encoder.encoder.layer[0].state_dict())
    elif args.fix_model == 6 or args.fix_model == 8 or args.fix_model == 85 or args.fix_model == 881  or args.fix_model == 882 or args.fix_model == 883:
        print("Initialize the Extra Layer.")
        model_decoder.transformer.h[decoder_n_layer].load_state_dict(model_decoder.transformer.h[0].state_dict())
        model_decoder.transformer.change_order()
    elif args.fix_model == 84:
        print("Initialize the Extra Layer.")
        model_decoder.transformer.change_order()
    elif args.fix_model == 10 or args.fix_model == 11:
        from transformers.adapters import CompacterConfig
        config = CompacterConfig(reduction_factor=4)
        model_decoder.transformer.add_adapter("dummy", config=config)
        model_decoder.transformer.train_adapter("dummy")
        # model_decoder.transformer.train_adapter("poem")
    elif args.fix_model == 12:
        aa = model_decoder.transformer.load_adapter("/home/guangyiliu/yiwen_Optimus/output/adapters", model_name='gpt2')
        model_decoder.transformer.train_adapter(aa)
    elif args.fix_model == 13 or args.fix_model == 14 or args.fix_model == 82:
        model_decoder.transformer.h[decoder_n_layer+1].load_state_dict(model_decoder.transformer.h[0].state_dict())
        model_decoder.transformer.h[decoder_n_layer].load_state_dict(model_decoder.transformer.h[11].state_dict())
        model_decoder.transformer.change_order(extra_num=2)
    elif args.fix_model == 83:
        model_decoder.transformer.h[decoder_n_layer].load_state_dict(model_decoder.transformer.h[11].state_dict())
        model_decoder.transformer.config.n_layer += 1
    # Save the init weights of BERT and GPT-2, so that we can load from local (Some infra requires so)
    if args.save_bert_gpt_init:
        encoder_path = os.path.join(args.output_dir, f"initial-models-tokenization-enoder-{args.latent_size}")
        if not os.path.exists(encoder_path): os.makedirs(encoder_path)
        model_encoder.save_pretrained(encoder_path)
        tokenizer_encoder.save_pretrained(encoder_path)

        decoder_path = os.path.join(args.output_dir, f"initial-models-tokenization-decoder-{args.latent_size}")
        if not os.path.exists(decoder_path): os.makedirs(decoder_path)
        model_decoder.save_pretrained(decoder_path)
        tokenizer_decoder.save_pretrained(decoder_path)

        return

        # Chunyuan: Add Padding token to GPT2
    special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>', }
    num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)
    print('We have added', num_added_toks, 'tokens to GPT2')
    model_decoder.resize_token_embeddings(
        len(tokenizer_decoder))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
    assert tokenizer_decoder.pad_token == '<PAD>'

    # model_decoder.to(args.device)

    model_vae = VAE(model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, args)

    # pdb.set_trace()
    if args.use_random_weight:
        model_vae.apply(weights_init_rondom)

    if args.use_pretrained_model:
        if args.fix_model == 4 or args.fix_model == 6:
            key_list = [n for n in checkpoint['model_state_dict']]
            for key in key_list:
                if 'linear' in key:
                    checkpoint['model_state_dict'].pop(key)
                    print('drop', key)

        if args.fix_model == 7:
            key_list = [n for n in checkpoint['model_state_dict']]
            for key in key_list:
                if 'linear' in key:
                    checkpoint['model_state_dict'].pop(key)
                    print('drop', key)

        model_vae.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info("Pre-trained Optimus is successfully loaded")
    model_vae.to(args.device)  #

    # on_gpu = next(model_vae.parameters()).is_cuda

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    ##############################
    # Training
    global_step = 0
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataloader = build_dataload_and_cache_examples(args, [tokenizer_encoder, tokenizer_decoder],
                                                             evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss, optimizer = train(args, train_dataloader, model_vae, tokenizer_encoder, tokenizer_decoder,
                                                table_name, checkpoint)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    # if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    #     if not args.no_save:
    #         save_checkpoint(model_vae, optimizer, global_step, args)

    ##############################
    # Evaluation the metrics of VAE models, including PPL, MI, AU
    # results = {}
    # if args.do_eval and args.local_rank in [-1, 0]:
    #     if global_step == 0:
    #         global_step = args.gloabl_step_eval
    #
    #     output_encoder_dir = os.path.join(args.output_dir, 'checkpoint-encoder-{}'.format(global_step))
    #     output_decoder_dir = os.path.join(args.output_dir, 'checkpoint-decoder-{}'.format(global_step))
    #     output_full_dir = os.path.join(args.output_dir, 'checkpoint-full-{}'.format(global_step))
    #     checkpoint_dir = [output_encoder_dir, output_decoder_dir, output_full_dir]
    #
    #     logger.info("Evaluate the following checkpoint: %s", checkpoint_dir[-1])
    #     global_step = checkpoint_dir[-1].split('-')[-1] if len(checkpoint_dir) > 1 else ""
    #
    #     checkpoint = torch.load(os.path.join(output_full_dir, 'training.bin'))
    #     model_vae.load_state_dict(checkpoint['model_state_dict'])
    #     logger.info(f"Pre-trained Optimus is successfully loaded: {output_full_dir}")
    #     model_vae.to(args.device)
    #
    #     result = evaluate(args, model_vae, tokenizer_encoder, tokenizer_decoder, table_name, prefix=global_step,
    #                       subset='test')
    #     result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
    #     results.update(result)
    #
    #     output_eval_file = os.path.join(args.output_dir, "eval_vae_results.txt")
    #     with open(output_eval_file, "w") as writer:
    #         logger.info("***** Eval results *****")
    #         for key in sorted(results.keys()):
    #             logger.info("%s = %s", key, str(results[key]))
    #             writer.write("%s = %s\n" % (key, str(results[key])))
    #     logger.info(f"The testing results are successfully saved: {output_eval_file}")

    return None


if __name__ == "__main__":
    main()
