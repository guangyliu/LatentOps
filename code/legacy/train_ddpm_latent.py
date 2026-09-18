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
import logging
import os
import random

import numpy as np
import torch
import torch.nn.init as init
from my_transformers import *
# from pytorch_transformers import AdamW
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AdamW
# from run_latent_generation import sample_sequence_conditional
from transformers import get_polynomial_decay_schedule_with_warmup

from modules import GAN  # GANVAE as GAN
from modules import VAE, DenseEmbedder, sample_sequence_conditional, DDPM, LinearModel
from utils import (BucketingDataLoader, TextDataset_Split,
                   TextDataset_2Tokenizers, frange_cycle_zero_linear)
import time
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': GPT2ForLatentConnector,
    # 'openai-gpt': (None, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': BertForLatentConnector,
    'bertu': BertForLatentConnector,
    'roberta': RobertaForLatentConnector,
    'deberta': DebertaForLatentConnector,
    't5': T5EncoderForLatentConnector,
}

#### GPT2 for ppl
from transformers import GPT2LMHeadModel as GPT2_
from transformers import GPT2TokenizerFast, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader

# model_id = '../output/gpt2_styleptb'  # sentiment'  # _sentiment' #amazon'
# model_ppl = GPT2_.from_pretrained(model_id).cuda()
# tokenizer_ppl = GPT2TokenizerFast.from_pretrained(model_id)


#### GPT2 for ppl
start_time = time.time()
class LatentDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, latent_z, labels):
        self.latent_z = latent_z
        self.labels = labels

    def __len__(self):
        return len(self.latent_z)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'latent_z': self.latent_z[idx], 'labels': self.labels[idx]}
        return sample


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


def distinct(lines):
    for i, line in enumerate(lines):
        lines[i] = line.strip().split()
    grams = lines
    grams_list1 = []
    for sen in grams:
        for g in sen:
            grams_list1.append(g)

    grams_list2 = []
    for sen in grams:
        for i in range(len(sen) - 1):
            grams_list2.append(str(sen[i]) + ' ' + str(sen[i + 1]))
    dist1 = round(len(set(grams_list1)) / len(grams_list1), 4)
    dist2 = round(len(set(grams_list2)) / len(grams_list2), 4)
    return (dist1, dist2)


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


def save_cls_checkpoint(classifier, optimizer, global_step, args, gan=None, eval_loss=False):
    # Create output directory if needed
    # Save model checkpoint
    save_last = args.save_step

    output_cls_dir = os.path.join(args.output_dir, 'checkpoint-cls-{}'.format(save_last))
    if not os.path.exists(output_cls_dir) and args.local_rank in [-1, 0]:
        os.makedirs(output_cls_dir)

    output_gan_dir = os.path.join(args.output_dir, 'checkpoint-ddpm-{}'.format('1'))
    if not os.path.exists(output_gan_dir) and args.local_rank in [-1, 0]:
        os.makedirs(output_gan_dir)
    logger.info("Saving DDPM model checkpoint to %s", output_gan_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`

    model_gan_to_save = gan.module if hasattr(gan,
                                              'module') else gan  # Take care of distributed/parallel training

    checkpoint_gan = {
        'iter': global_step,
        'model_state_dict': model_gan_to_save.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        'args': args
    }
    torch.save(checkpoint_gan, os.path.join(output_gan_dir, 'training_ddpm.bin'))
    logger.info("Saving DDPM checkpoint to %s", output_gan_dir)


def access_latent_label(args, train_dataloader, model_vae, train=True):
    """ Train the model """
    npy_file_path = args.train_data_file if train else args.eval_data_file
    if os.path.exists(npy_file_path+'.npy'):
        with open(npy_file_path+'.npy', 'rb') as f:
            all_data = np.load(f)
            all_z = all_data[:,:-1]
            all_label = all_data[:,-1]
    else:
        all_z = np.zeros((0, args.latent_size))
        all_label = np.zeros((0), )
        epoch_iterator = tqdm(train_dataloader, desc="Creating Latent data")
        for step, batch in enumerate(epoch_iterator):
            tokenized_text0, tokenized_text1, tokenized_text_lengths = batch
            latent_labels = tokenized_text_lengths[:, -1]

            inputs = tokenized_text0
            inputs = inputs.to(args.device)
            model_vae.eval()
            with torch.no_grad():
                latent_z = model_vae.encode_x(inputs)
                all_z = np.append(all_z, latent_z.cpu().numpy(), 0)
                all_label = np.append(all_label, latent_labels.numpy(), 0)
        all_data = np.append(all_z,all_label[:,None],1)
        with open(npy_file_path+'.npy', 'wb') as f:
            np.save(f,all_data)
    return [all_z, all_label]



def train_ddpm(args, train_dataloader, model_vae, encoder_tokenizer, decoder_tokenizer, ddpm, eval_latent_dataset):
    """ Train the ddpm model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    optimizer_grouped_parameters = [
        {'params': [p for n, p in ddpm.named_parameters()],
         'weight_decay': 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        ddpm, optimizer = amp.initialize(ddpm, optimizer,
                                         opt_level=args.fp16_opt_level)

        # if 'cls' in args.train_cls_gan:
        #     classifier, optimizer = amp.initialize(classifier, optimizer, opt_level=args.fp16_opt_level)

    # Train!
    logger.info("***** Running training *****")
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

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])

    n_iter = int(args.num_train_epochs) * len(train_dataloader)
    beta_t_list = frange_cycle_zero_linear(n_iter, start=0.0, stop=args.beta, n_cycle=int(args.num_train_epochs),
                                           ratio_increase=args.ratio_increase, ratio_zero=args.ratio_zero)
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    args.logging_steps = int(np.floor(len(train_dataloader)))
    args.save_steps = args.logging_steps

    stop_flag = False
    best_gan_diff = 200
    best_diff_cnt = 0
    start_time = time.time()
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        if best_gan_diff < 200:
            use_time = time.time() - start_time
            start_time = time.time()
            logger.info("Time for this epoch = %f", use_time)
        for step, batch in enumerate(epoch_iterator):

            # tokenized_text0, _, tokenized_text_lengths = batch
            # latent_labels = tokenized_text_lengths[:, -1]
            latent_z = batch['latent_z'].float().to(args.device)
            # latent_labels = batch['labels'].to(args.device)

            model_vae.eval()
            ddpm.train()

            loss = ddpm(latent_z)

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
                    torch.nn.utils.clip_grad_norm_(ddpm.parameters(), args.max_grad_norm)

                optimizer.step()
                ddpm.zero_grad()

                epoch_iterator.set_description(
                    (
                        f'iter: {step + epoch * len(epoch_iterator)}; loss: {loss.item():.3f}; '
                        f'loss_d: {loss.item():.3f};'
                    )
                )
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics1
                    # results = evaluate_acc(model_vae, encoder_tokenizer, decoder_tokenizer, args, ns=1,
                    #                            classifier=None, gan=gan, eval_latent_dataset=eval_latent_dataset)
                    results = calc_ppl_lgy_ddpm(
                        model_vae, encoder_tokenizer, decoder_tokenizer, args, 1, 
                        ddpm, model_ppl,tokenizer_ppl, z=latent_z
                    )

                    logger.info("PPL = %f", results['ppl'])
                    logger.info("sBLEU = %f", results['sbleu'])
                    logger.info("PPL+sBLEU = %f", results['ppl+sbleu'])
                    logger.info("Length = %f", results['length'])
                    logger.info("z norm = %f", results['norm_z'])
                    if results['ppl+sbleu'] < best_gan_diff and results['ppl'] > 11 and results['norm_z'] < 10:
                        best_gan_diff = results['ppl+sbleu']
                        best_diff_cnt = 0
                        save_cls_checkpoint(None, optimizer, global_step, args, gan=ddpm)

                    else:
                        best_diff_cnt += 1

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return 0



def calc_ppl_lgy_ddpm(model_vae, encoder_tokenizer, decoder_tokenizer, args, ns=1, ddpm=None, model=None, tokenizer=None,
                 z=None):
    generate_text = []
    bz = 50
    num_epoch = 100 // bz

    def out_(zz):
        generate_text1 = []
        context_tokens = decoder_tokenizer.encode('<BOS>')
        with torch.no_grad():
            out = sample_sequence_conditional(
                model=model_vae.decoder,
                context=context_tokens,
                past=zz,
                length=20,  # Chunyuan: Fix length; or use <EOS> to complete a sentence
                num_samples=zz.size(0),
                device=args.device,
                decoder_tokenizer=decoder_tokenizer,
                eos_id=model_vae.eos_token_id
            )
        for i in range(zz.size(0)):
            text_x1 = decoder_tokenizer.decode(out[i, :].tolist(), clean_up_tokenization_spaces=False).split('<EOS>')[
                0].replace('<BOS>', '').strip()
            text_x1 = ' '.join(text_x1.split())
            generate_text1.append(text_x1 + '\n')
        return generate_text1

    for _ in trange(num_epoch, desc="Evaluating PPL", disable=True):
        latent_z = ddpm.sample(bz,(64,), args.device )
        # latent_z = gan.generate_z(bz, eval=True)
        context_tokens = decoder_tokenizer.encode('<BOS>')
        with torch.no_grad():
            out = sample_sequence_conditional(
                model=model_vae.decoder,
                context=context_tokens,
                past=latent_z,
                length=20,
                num_samples=latent_z.size(0),
                device=args.device,
                decoder_tokenizer=decoder_tokenizer,
                eos_id=model_vae.eos_token_id
            )
        for i in range(latent_z.size(0)):
            text_x1 = decoder_tokenizer.decode(out[i, :].tolist(), clean_up_tokenization_spaces=False).split('<EOS>')[
                0].replace('<BOS>', '').strip()
            text_x1 = ' '.join(text_x1.split())
            generate_text.append(text_x1 + '\n')
        print(text_x1)
    encodings = tokenizer('\n\n'.join(generate_text), return_tensors='pt')
    max_length = model.config.n_positions
    stride = 512

    nlls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].cuda()
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    list_of_references = []
    len_list = []
    for jj, line in enumerate(generate_text):
        if jj < 10:
            print(line)
        split = line.strip().split(' ')
        list_of_references.append(split)
        len_list.append(len(split))
    # dist1,dist2 = distinct(generate_text)
    # score  = 10*(-dist2-dist1)
    sbleu = []
    num_all = len(list_of_references)
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    for i in range(num_all):
        refs = [list_of_references[j] for j in range(num_all) if i != j]
        bleu_ = sentence_bleu(refs, list_of_references[i], smoothing_function=SmoothingFunction().method1)
        sbleu.append(bleu_ * 100)
    score = np.mean(sbleu)
    # weights = {'4gram': (1 / 4., 1 / 4., 1 / 4., 1 / 4.)}
    # from fast_bleu import SelfBLEU
    # self_bleu = SelfBLEU(list_of_references, weights)
    # score = np.mean(self_bleu.get_score()['4gram']) * 100
    len_mean = np.mean(len_list)
    norm_z = latent_z.norm(dim=-1).mean().item()
    return {'ppl': ppl, 'sbleu': round(score, 2), 'length': round(len_mean, 2), 'norm_z': norm_z,
            'ppl+sbleu': ppl + round(score, 2)}




def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default='../data/datasets/yelp_data/train.shuf.txt', type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--checkpoint_dir", default='../ckpts/base_yelp', type=str,
                        help="The directory where checkpoints are saved.")
    parser.add_argument("--output_dir", default='../ckpts/base_yelp', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--dataset", default='Yelp_cls', type=str, help="The dataset.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default='../data/datasets/yelp_data/test.txt', type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--ExpName", default="", type=str,
                        help="The experiment name used in Azure Table.")
    parser.add_argument("--save_bert_gpt_init", action='store_true',
                        help="Use Philly for computing.")
    parser.add_argument("--length_weighted_loss", default=1, type=int,
                        help="Use sentence length re-weight the reconstruction loss.")

    ## Encoder options
    parser.add_argument("--encoder_model_type", default="bertu", type=str,
                        help="The encoder model architecture to be fine-tuned.")
    parser.add_argument("--encoder_model_name_or_path", default="prajjwal1/bert-small", type=str,
                        help="The encoder model checkpoint for weights initialization.")
    parser.add_argument("--encoder_config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--encoder_tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    ## Decoder options
    parser.add_argument("--decoder_model_type", default="gpt2", type=str,
                        help="The decoder model architecture to be fine-tuned.")
    parser.add_argument("--decoder_model_name_or_path", default="gpt2", type=str,
                        help="The decoder model checkpoint for weights initialization.")
    parser.add_argument("--decoder_config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--decoder_tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    ## Variational auto-encoder
    parser.add_argument("--latent_size", default=64, type=int, help="Latent space dimension.")
    parser.add_argument("--use_deterministic_connect", action='store_true',
                        help="Use deterministic inference to generate latent codes, i.e., standard auto-encoders.")
    parser.add_argument("--use_pretrained_model", type=int, default= 1.0,
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
    parser.add_argument("--block_size", default=30, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", default=1, type=int,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=1, type=int,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval_rec", action='store_true',
                        help="Whether to run eval reconstruction on a set of models.")
    parser.add_argument("--evaluate_during_training", default=1, type=int,
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    # Training Schedules
    parser.add_argument("--ratio_increase", default=0.25, type=float,
                        help="Learning schedule, the percentage for the annealing stage.")
    parser.add_argument("--ratio_zero", default=0.25, type=float,
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
    parser.add_argument("--num_train_epochs", default=5, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--use_philly", action='store_true',
                        help="Use Philly for computing.")
    parser.add_argument("--use_pretrained_vae", type=int, default= 1.0,
                        help="Use use_pretrained_vae as initialization, where beta value is specified in the folder")
    parser.add_argument("--use_random_weight", action='store_true',
                        help="Use random weights as initialization")

    ## IO: Logging and Saving
    parser.add_argument('--logging_steps', type=float, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=898,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', default=1, type=int,
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gloabl_step_eval', type=int, default=1,
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
    parser.add_argument('--train_cls_gan', type=str, default='cls')
    parser.add_argument('--n_cyc', type=int, default=5)
    parser.add_argument('--save_step', type=str, default=1)
    parser.add_argument('--fix_model', type=int, default=84,
                        help="0: no fix; 1: fix both bert & gpt; 2: fix gpt; 3: fix both bert & gpt, extra layers")
    parser.add_argument('--nt', type=int, default=2000, help="T for diffusion process")
    args = parser.parse_args()
    model_id = '../../Optimus-ODE/output/gpt2_sentiment' # + args.output_dir.split('/')[-1]  # sentiment'  # _sentiment' #amazon'
    print(model_id)
    global model_ppl
    model_ppl = GPT2_.from_pretrained(model_id).cuda()
    global tokenizer_ppl
    tokenizer_ppl = GPT2TokenizerFast.from_pretrained(model_id)
    MODEL_CLASSES['bertu'] = BertForLatentConnectorAVG
    if 'large' in args.decoder_model_name_or_path:
        MODEL_CLASSES['gpt2'] = GPT2ForLatentConnectorNew
    else:
        MODEL_CLASSES['gpt2'] = GPT2ForLatentConnectorNew2

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    # Set seed
    set_seed(args)

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

    latent_as_gpt_emb = True if args.latent_as_gpt_emb == 1 else False
    latent_as_gpt_memory = True if args.latent_as_gpt_memory == 1 else False

    # setattr(decoder_config, "latent_size", args.latent_size)
    model_decoder = decoder_model_class.from_pretrained(args.decoder_model_name_or_path, latent_size=args.latent_size,
                                                        latent_as_gpt_emb=latent_as_gpt_emb,
                                                        latent_as_gpt_memory=latent_as_gpt_memory)
    model_decoder.transformer.change_order()

    special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>', }
    num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)
    print('We have added', num_added_toks, 'tokens to GPT2')
    model_decoder.resize_token_embeddings(len(tokenizer_decoder)) 
    model_vae = VAE(model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, args)
    model_vae.load_state_dict(checkpoint['model_state_dict'], strict=False)
    logger.info("Pre-trained Optimus is successfully loaded")
    model_vae.to(args.device)  #

    ddpm = DDPM(eps_model=LinearModel(64), betas=(1e-4, 0.02), n_T=args.nt,)
    ddpm.to(args.device)
    ddpm.apply(weights_init_rondom)
    import copy
    args_ = copy.deepcopy(args)
    args_.per_gpu_train_batch_size = args.per_gpu_train_batch_size * 8
    train_dataloader = build_dataload_and_cache_examples(args_, [tokenizer_encoder, tokenizer_decoder],
                                                         evaluate=False)
    all_z, all_label = access_latent_label(args, train_dataloader, model_vae, train=True)
    latent_dataset = LatentDataset(all_z, all_label)

    dataloader = DataLoader(latent_dataset, batch_size=args.per_gpu_train_batch_size,
                            shuffle=True, num_workers=0)
    eval_dataloader = build_dataload_and_cache_examples(args_, [tokenizer_encoder, tokenizer_decoder],
                                                        evaluate=True)
    eval_z, eval_label = access_latent_label(args, eval_dataloader, model_vae, train=False)
    eval_latent_dataset = LatentDataset(eval_z, eval_label)
    train_ddpm(args, dataloader, model_vae, tokenizer_encoder, tokenizer_decoder, ddpm=ddpm,
          eval_latent_dataset=eval_latent_dataset)

if __name__ == "__main__":
    main()
