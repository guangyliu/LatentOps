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

import pdb
import argparse
import glob
import logging

import os
import pickle
import random

import numpy as np
import torch
import torch.nn.init as init
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from collections import defaultdict

from datetime import datetime
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer as GPT2Tokenizer_LM
from run_latent_generation import sample_sequence_conditional
from nltk.translate.bleu_score import corpus_bleu

from pytorch_transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,
                                  BertConfig, BertForLatentConnector, BertTokenizer,
                                  GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer,
                                  OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                                  RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)

from utils_simcse import (weight_init, calc_iwnll, calc_rec, calc_mi, calc_au, BucketingDataLoader, TextDataset_Split,
                   TextDataset_2Tokenizers, frange_cycle_linear, frange_cycle_zero_linear)

from modules import VAE, DenseEmbedder, GAN, Similarity
from run_lm_vae_training import  set_seed, mask_tokens, weights_init_rondom
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForLatentConnector, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)
}


def build_dataload_and_cache_examples(args, tokenizer, evaluate=False):
    if isinstance(tokenizer, list):
        if not evaluate:
            args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
            file_path=args.train_data_file
        else:
            args.batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
            file_path=args.eval_data_file
        dataloader = BucketingDataLoader(file_path, args.batch_size, args.max_seq_length, tokenizer, args, bucket=100, shuffle=False)
    else:
        pass
    return dataloader


def save_cls_checkpoint(classifier, optimizer, global_step, args, gan=None):
    # Create output directory if needed
    # Save model checkpoint
    save_last = args.save_step

    output_cls_dir = os.path.join(args.output_dir, 'checkpoint-cls-{}'.format(save_last))
    if not os.path.exists(output_cls_dir) and args.local_rank in [-1, 0]:
        os.makedirs(output_cls_dir)

    if classifier is not None and 'cls' in args.train_cls_gan:
        logger.info("Saving classifier model checkpoint to %s", output_cls_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`

        model_cls_to_save = classifier.module if hasattr(classifier,
                                                         'module') else classifier  # Take care of distributed/parallel training
        checkpoint = {
            'iter': global_step,
            'model_state_dict': model_cls_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': args
        }
        torch.save(checkpoint, os.path.join(output_cls_dir, 'training_cls.bin'))
        logger.info("Saving cls checkpoint to %s", output_cls_dir)
    if gan is not None and 'gan' in args.train_cls_gan:
        output_gan_dir = os.path.join(args.output_dir, 'checkpoint-gan-{}'.format('1'))
        if not os.path.exists(output_gan_dir) and args.local_rank in [-1, 0]:
            os.makedirs(output_gan_dir)
        logger.info("Saving GAN model checkpoint to %s", output_gan_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`

        model_gan_to_save = gan.module if hasattr(gan,
                                                  'module') else gan  # Take care of distributed/parallel training
        checkpoint_gan = {
            'iter': global_step,
            'model_state_dict': model_gan_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': args
        }
        torch.save(checkpoint_gan, os.path.join(output_gan_dir, 'training_gan.bin'))
        logger.info("Saving GAN checkpoint to %s", output_gan_dir)



def train(args, train_dataloader, model_vae, encoder_tokenizer, decoder_tokenizer, table_name, classifier, gan):
    """ Train the model """

    #     LM_tokenizer, LM_model = LM_ppl()
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(
            './runs/' + args.output_dir.split('/')[-2] + '/' + args.output_dir.split('/')[-1] + '_cls_gan')

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)

    # model_encoder, model_decoder, model_connector = model_vae.encoder,  model_vae.decoder, model_vae.linear
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in classifier.named_parameters()],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in gan.latent_discriminator.named_parameters()],
         'weight_decay': args.weight_decay},
        # {'params': [p for n, p in z_mapping.named_parameters()],
        #  'weight_decay': args.weight_decay}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # optimizer = torch.optim.RMSprop(optimizer_grouped_parameters,  lr=args.learning_rate)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    optimizer_grouped_parameters_G = [
        {'params': [p for n, p in gan.latent_generator.named_parameters()],
         'weight_decay':args.weight_decay},
    ]
    optimizer_G = torch.optim.RMSprop(optimizer_grouped_parameters_G, lr=args.learning_rate)
    # optimizer_G = AdamW(optimizer_grouped_parameters_G, lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler_G = WarmupLinearSchedule(optimizer_G, warmup_steps=args.warmup_steps, t_total=t_total)

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

    # model_vae = model_vae.module if hasattr(model_vae, 'module') else model_vae  # Take care of distributed/parallel training   

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])

    n_iter = int(args.num_train_epochs) * len(train_dataloader)
    beta_t_list = frange_cycle_zero_linear(n_iter, start=0.0, stop=args.beta, n_cycle=int(args.num_train_epochs),
                                           ratio_increase=args.ratio_increase, ratio_zero=args.ratio_zero)
    tmp_list = []
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    #     result_new = calc_rec_lgy(model_vae, encoder_tokenizer, decoder_tokenizer,args,ns=100)
    args.logging_steps = len(train_dataloader) * args.logging_steps
    #     logger.info('logging_steps is ',args.logging_steps)
    args.save_steps = args.logging_steps
    best_cls_acc = -10
    best_gan_diff = 10
    best_acc_cnt = 0
    best_diff_cnt = 0
    loss_gan_g = torch.tensor(0)
    # loss = torch.tensor(0)
    model_vae.eval()
    model_vae.encoder.train()
    sim = Similarity(temp=0.05)
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator):
            tokenized_text0, tokenized_text1, tokenized_text_lengths = batch
            latent_labels = tokenized_text_lengths[:, -1]

            inputs, _ = mask_tokens(tokenized_text0, encoder_tokenizer, args) if args.mlm else (
                tokenized_text0, tokenized_text1)

            # tokenized_text1 = tokenized_text1.to(args.device)
            inputs = inputs.to(args.device)
            latent_labels = latent_labels.to(args.device)


            if 'cls' in args.train_cls_gan:
                # z_mapping.train()
                classifier.train()
            else:
                # z_mapping.eval()
                classifier.eval()
            if 'gan' in args.train_cls_gan:
                gan.train()
            else:
                gan.eval()
            beta_t = beta_t_list[step + epoch * len(epoch_iterator)]
            if args.n_gpu == 1:
                model_vae.args.beta = beta_t

                if beta_t == 0.0:
                    model_vae.args.fb_mode = 0
                else:
                    model_vae.args.fb_mode = 1

                if args.use_deterministic_connect:
                    model_vae.args.fb_mode = 2
            else:
                model_vae.module.args.beta = beta_t

                if beta_t == 0.0:
                    model_vae.module.args.fb_mode = 0
                else:
                    model_vae.module.args.fb_mode = 1

                if args.use_deterministic_connect:
                    model_vae.module.args.fb_mode = 2
            #####  CLS start
            with torch.no_grad():
                latent_z = model_vae.encode_x(inputs)
            if 'cls' in args.train_cls_gan:
                # latent_z = z_mapping(latent_z)
                logits = classifier(latent_z).view(latent_z.size(0)//2, 2, -1)
                z1, z2 = logits[:, 0], logits[:, 1]
                cos_sim = sim(z1.unsqueeze(1), z2.unsqueeze(0))
                labels = torch.arange(cos_sim.size(0)).long().to(args.device)
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(cos_sim, labels)
                tb_writer.add_scalar('loss_cls', loss.mean().item(), train_step)
                train_step += 1

                loss = loss.mean()
            else:
                loss = 0

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.max_grad_norm)
                    # torch.nn.utils.clip_grad_norm_(gan.parameters(), args.max_grad_norm)

                optimizer.step()
                # scheduler.step()  # Update learning rate schedule

                classifier.zero_grad()
                gan.latent_discriminator.zero_grad()

                if step % args.n_cyc == 0:
                # for _ in range(args.n_cyc):
                    loss_gan_g = gan.g_loss(latent_z)
                    loss_gan_g.backward()
                    torch.nn.utils.clip_grad_norm_(gan.parameters(), args.max_grad_norm)
                    optimizer_G.step()
                    # scheduler_G.step()
                    gan.latent_generator.zero_grad()
                epoch_iterator.set_description(
                    (
                        f'iter: {step + epoch * len(epoch_iterator)}; loss: {loss.item():.3f}; '
                    )
                )
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate_acc(model_vae, encoder_tokenizer, decoder_tokenizer, args, ns=1,
                                               classifier=classifier, gan=gan)
                        logger.info("ACC = %f", results['acc'])
                        logger.info("GAN ACC Diff = %f", results['gan_acc_diff'])
                        if -results['acc'] > best_cls_acc:
                            best_cls_acc = -results['acc']
                            best_acc_cnt = 0
                            save_cls_checkpoint(classifier, optimizer, global_step, args, gan=None)
                        else:
                            best_acc_cnt += 1
                        if results['gan_acc_diff'] < best_gan_diff:
                            best_gan_diff = results['gan_acc_diff']
                            best_diff_cnt = 0
                            save_cls_checkpoint(None, optimizer, global_step, args, gan=gan)
                        else:
                            best_diff_cnt += 1
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                # if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                #     save_cls_checkpoint(classifier, optimizer, global_step, args, gan=gan)
            if best_acc_cnt >= 3 and  best_diff_cnt >5:
                break
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    results = evaluate_acc(model_vae, encoder_tokenizer, decoder_tokenizer, args, ns=1, classifier=classifier, gan=gan)
    #     results['ppl_sample'] = sampling_lgy(model_vae, decoder_tokenizer, args, LM_model, LM_tokenizer, cnt=1000)['ppl']
    for key, value in results.items():
        tb_writer.add_scalar('final_{}'.format(key), value, global_step)
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, optimizer


def evaluate(args, model_vae, encoder_tokenizer, decoder_tokenizer, table_name, prefix="", subset="test"):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    # if subset == 'test':
    #     eval_dataset = load_and_cache_examples(args, [encoder_tokenizer, decoder_tokenizer], evaluate=True)
    # elif subset == 'train':
    #     eval_dataset = load_and_cache_examples(args, [encoder_tokenizer, decoder_tokenizer], evaluate=False)
    logger.info("***** Running evaluation on {} dataset *****".format(subset))

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    # eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    # eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

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

    return result


def calc_rec_lgy(model_vae, encoder_tokenizer, decoder_tokenizer, args, ns=1):
    from nltk.translate.bleu_score import corpus_bleu
    eval_dataloader = build_dataload_and_cache_examples(args, [encoder_tokenizer, decoder_tokenizer], evaluate=True)
    count = 0
    result = defaultdict(str)
    ref = []
    cand = []
    for batch in tqdm(eval_dataloader, desc="Evaluating recontruction"):
        x0, x1, x_lengths = batch
        max_len_values, _ = x_lengths.max(0)
        x0 = x0[:, :max_len_values[0]]
        x1 = x1[:, :max_len_values[1]]
        x0 = x0.to(args.device)
        x1 = x1.to(args.device)
        x_lengths = x_lengths.to(args.device)
        context_tokens = decoder_tokenizer.encode('<BOS>')
        with torch.no_grad():
            text_x0 = encoder_tokenizer.decode(x0[0, :x_lengths[0, 0]].tolist(), clean_up_tokenization_spaces=True)[0]
            # result["INPUT TEXT " + str(count)].append(text_x0)

            pooled_hidden_fea = model_vae.encoder(x0, attention_mask=(x0 > 0).float())[1]

            # Connect hidden feature to the latent space
            # latent_z, loss_kl = model_vae.connect(pooled_hidden_fea)
            mean, logvar = model_vae.encoder.linear(pooled_hidden_fea).chunk(2, -1)
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
            text_x0_ = decoder_tokenizer.decode(x1[i, :].tolist(), clean_up_tokenization_spaces=False).split(' <EOS>')[
                0]
            text_x0_ = text_x0_.split()[1:]
            text_x1 = decoder_tokenizer.decode(out[i, :].tolist(), clean_up_tokenization_spaces=False).split(' <EOS>')[
                0]
            text_x1 = text_x1.split()[1:]

            count += 1
            ref.append([text_x0_])
            cand.append(text_x1)
        #             writer.write("%s\n%s\n" % (text_x0.strip(), str(text_x1).strip()))
        if count > 1000:
            break
    bleu = corpus_bleu(ref, cand) * 100
    return {'bleu': bleu}


def evaluate_acc(model_vae, encoder_tokenizer, decoder_tokenizer, args, ns=1, classifier=None, gan=None):
    eval_dataloader = build_dataload_and_cache_examples(args, [encoder_tokenizer, decoder_tokenizer], evaluate=True)
    corrects = []
    acc_diff_list = []
    sim = Similarity(temp=0.05)
    for batch in tqdm(eval_dataloader, desc="Evaluating acc"):
        x0, x1, x_lengths = batch
        max_len_values, _ = x_lengths.max(0)
        latent_labels = x_lengths[:, -1]
        x0 = x0[:, :max_len_values[0]]
        x0 = x0.to(args.device)
        latent_labels = latent_labels.to(args.device)
        with torch.no_grad():
            pooled_hidden_fea = model_vae.encoder(x0, attention_mask=(x0 > 0).float())[1]
            mean, _ = model_vae.encoder.linear(pooled_hidden_fea).chunk(2, -1)
            latent_z = mean.squeeze(1)
            logits = classifier(latent_z).view(latent_z.size(0) // 2, 2, -1)
            z1, z2 = logits[:, 0], logits[:, 1]
            cos_sim = sim(z1.unsqueeze(1), z2.unsqueeze(0))
            labels = torch.arange(cos_sim.size(0)).long().to(args.device)
            loss_fct = torch.nn.CrossEntropyLoss()
            loss =  loss_fct(cos_sim, labels)
            corrects.append(loss.mean().float().cpu().numpy())
            ### GAN
            acc_gen, acc_enc, acc_diff = gan.discriminate(latent_z)
            acc_diff_list.append(acc_gen.float().cpu().numpy())
    correct = np.mean(corrects)
    gan_acc_diff = np.mean(acc_diff_list)
    return {'acc': correct, 'gan_acc_diff': gan_acc_diff}


def evaluate_rec(args, model_vae, encoder_tokenizer, decoder_tokenizer, table_name, prefix="", subset="test"):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    if subset == 'test':
        eval_dataloader = build_dataload_and_cache_examples(args, [encoder_tokenizer, decoder_tokenizer], evaluate=True)
    elif subset == 'train':
        eval_dataloader = build_dataload_and_cache_examples(args, [encoder_tokenizer, decoder_tokenizer],
                                                            evaluate=False)
    logger.info("***** Running evaluation on {} dataset *****".format(subset))

    # Note that DistributedSampler samples randomly
    # eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    # eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # eval_dataloader = build_dataload_and_cache_examples(args, [encoder_tokenizer, decoder_tokenizer], evaluate=True)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model_vae.eval()
    model_vae = model_vae.module if hasattr(model_vae,
                                            'module') else model_vae  # Take care of distributed/parallel training
    nll_s, nll_w = calc_rec(model_vae, eval_dataloader, args, ns=1)

    result = {
        "rec_w": nll_w, "rec_s": nll_s
    }

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("%s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def sampling_lgy(model_vae, tokenizer_decoder, args, LM_model, LM_tokenizer=None, cnt=200):
    ccc = cnt // args.per_gpu_eval_batch_size
    valid_lines = []
    for i in tqdm(range(ccc)):
        latent_z = torch.normal(0, 1, size=(args.per_gpu_eval_batch_size, args.latent_size)).cuda()
        past = latent_z
        context_tokens = tokenizer_decoder.encode('<BOS>')
        length = 50  # maximum length, but not used
        out = sample_sequence_conditional(
            model=model_vae.decoder,
            context=context_tokens,
            past=past,
            length=length,  # Chunyuan: Fix length; or use <EOS> to complete a sentence
            num_samples=latent_z.size(0),
            device=args.device,
            decoder_tokenizer=tokenizer_decoder,
            eos_id=model_vae.eos_token_id
        )
        for i in range(latent_z.size(0)):
            text_x1 = tokenizer_decoder.decode(out[i, :].tolist(), clean_up_tokenization_spaces=False).split(' <EOS>')[
                0]
            text_x1 = text_x1.split()[1:]
            text_x1 = ' '.join(text_x1)
            valid_lines.append('[BOS] ' + text_x1 + ' [EOS]')
    batch_size = args.per_gpu_eval_batch_size
    valid_step = int(np.ceil(len(valid_lines) / batch_size))
    with torch.no_grad():
        loss_all = 0
        for i in range(valid_step):
            if i == valid_step - 1:
                text_batch = valid_lines[i * batch_size:]
            else:
                text_batch = valid_lines[i * batch_size:(i + 1) * batch_size]
            encoding = LM_tokenizer(text_batch, return_tensors='pt', padding=True)
            input_ids = encoding['input_ids'].cuda()
            attention_mask = encoding['attention_mask'].cuda()
            labels = input_ids[:, 1:]
            outputs = LM_model(input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(outputs.logits[:, :-1].reshape(-1, 50260).contiguous(), labels.reshape(-1),
                                   ignore_index=50257)
            loss_all += loss.item() * len(text_batch)
        ppl = round(np.exp(loss_all / len(valid_lines)), 2)
    return {'ppl': ppl}


def evaluate_rec_sample(args, model_vae, encoder_tokenizer, decoder_tokenizer, table_name, prefix="", subset="test"):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    if subset == 'test':
        eval_dataloader = build_dataload_and_cache_examples(args, [encoder_tokenizer, decoder_tokenizer], evaluate=True)
    elif subset == 'train':
        eval_dataloader = build_dataload_and_cache_examples(args, [encoder_tokenizer, decoder_tokenizer],
                                                            evaluate=False)
    logger.info("***** Running evaluation on {} dataset *****".format(subset))

    # Note that DistributedSampler samples randomly
    # eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    # eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # eval_dataloader = build_dataload_and_cache_examples(args, [encoder_tokenizer, decoder_tokenizer], evaluate=True)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model_vae.eval()
    model_vae = model_vae.module if hasattr(model_vae,
                                            'module') else model_vae  # Take care of distributed/parallel training
    result = calc_rec(model_vae, eval_dataloader, args, ns=1, rec_sample=True)
    return result


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
    parser.add_argument('--train_cls_gan', type=str, default='cls_gan')
    parser.add_argument('--n_cyc', type=int, default=5)
    parser.add_argument('--save_step', type=int, default=1)
    args = parser.parse_args()

    if args.decoder_model_type in ["bert", "roberta"] and not args.mlm:
        raise ValueError("BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
                         "flag (masked language modeling).")
    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument.")

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    global_step = args.gloabl_step_eval

    output_full_dir = os.path.join(args.checkpoint_dir, 'checkpoint-full-{}'.format(global_step))

    checkpoint = torch.load(os.path.join(output_full_dir, 'training.bin'))

    ## Encoder
    encoder_config_class, encoder_model_class, encoder_tokenizer_class = MODEL_CLASSES[args.encoder_model_type]
    encoder_config = encoder_config_class.from_pretrained(
        args.encoder_config_name if args.encoder_config_name else args.encoder_model_name_or_path)
    tokenizer_encoder = encoder_tokenizer_class.from_pretrained(
        args.encoder_tokenizer_name if args.encoder_tokenizer_name else args.encoder_model_name_or_path,
        do_lower_case=args.do_lower_case)
    if args.block_size <= 0:
        args.block_size = tokenizer_encoder.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer_encoder.max_len_single_sentence)
    model_encoder = encoder_model_class.from_pretrained(args.encoder_model_name_or_path,
                                                        from_tf=bool('.ckpt' in args.encoder_model_name_or_path),
                                                        config=encoder_config, latent_size=args.latent_size)
    # model_encoder.to(args.device)

    ## Decoder
    decoder_config_class, decoder_model_class, decoder_tokenizer_class = MODEL_CLASSES[args.decoder_model_type]
    decoder_config = decoder_config_class.from_pretrained(
        args.decoder_config_name if args.decoder_config_name else args.decoder_model_name_or_path)
    tokenizer_decoder = decoder_tokenizer_class.from_pretrained(
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

    setattr(decoder_config, "latent_size", args.latent_size)
    model_decoder = decoder_model_class.from_pretrained(args.decoder_model_name_or_path,
                                                        from_tf=bool('.ckpt' in args.decoder_model_name_or_path),
                                                        config=decoder_config, latent_size=args.latent_size,
                                                        latent_as_gpt_emb=latent_as_gpt_emb,
                                                        latent_as_gpt_memory=latent_as_gpt_memory)

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
    model_decoder.resize_token_embeddings(len(
        tokenizer_decoder))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
    assert tokenizer_decoder.pad_token == '<PAD>'

    # model_decoder.to(args.device)

    model_vae = VAE(model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, args)
    # z_mapping = torch.nn.Sequential(
    #         torch.nn.Linear(args.latent_size, 2 * args.latent_size),
    #         torch.nn.LeakyReLU(0.2)
    # )
    classifier = DenseEmbedder(args.latent_size , up_dim=args.latent_size*2,depth=2, num_classes=args.latent_size*4)
    gan = GAN(args)
    if args.use_random_weight:
        model_vae.apply(weights_init_rondom)
        classifier.apply(weights_init_rondom)
        gan.apply(weights_init_rondom)
        # z_mapping.apply(weights_init_rondom)
    if args.use_pretrained_model:
        model_vae.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info("Pre-trained Optimus is successfully loaded")
    model_vae.to(args.device)  #
    classifier.to(args.device)
    gan.to(args.device)
    # z_mapping.to(args.device)
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
                                                None, classifier=classifier, gan=gan)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)



if __name__ == "__main__":
    main()
