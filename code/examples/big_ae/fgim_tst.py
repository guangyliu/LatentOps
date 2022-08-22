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
from torch import nn
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

from nltk.translate.bleu_score import corpus_bleu
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint as odeint_normal

from pytorch_transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,
                                  BertConfig, BertForLatentConnector, BertTokenizer,
                                  GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer,
                                  OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                                  RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)

from utils_eval import (weight_init, calc_iwnll, calc_rec, calc_mi, calc_au, BucketingDataLoader, TextDataset_Split,
                        TextDataset_2Tokenizers, frange_cycle_linear, frange_cycle_zero_linear)

from modules import VAE, DenseEmbedder, CCF
from modules import ConditionalSampling, ConditionalTransfer
from run_latent_generation import sample_sequence_conditional

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForLatentConnector, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)
}
STEP_CLASSES = {
    '1': 2,
    '2': 2,
    '3': 2,
    '4': 2,
    '5': 1,
    '6': 2
}


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


class VPODE(nn.Module):
    def __init__(self, ccf, y, beta_min=0.1, beta_max=20, T=1.0, save_path=None, plot=None, every_n_plot=5,
                 args=None, kwargs=None):
        super().__init__()
        self.ccf = ccf
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.T = T
        self.y = y
        self.n_evals = 0
        self.every_n_plot = every_n_plot
        self.plot = plot
        self.model_vae = kwargs['model']
        self.decoder_tokenizer = kwargs['dec_tokenizer']
        self.device = kwargs['device']
        ####
        self.reg_z = args.reg_z
        self.reg_logits = args.reg_logits
        self.z_anchor = None
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.generate_z = []

    def set_anchor_z(self, z):
        self.z_anchor = None if z is None else z.detach().clone()

    def forward(self, t_k, states):
        z = states[0]
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)

            beta_t = self.beta_0 + t_k * (self.beta_1 - self.beta_0)
            cond_energy_neg = self.ccf.get_cond_energy(z, self.y)  # get_cond_energy
            # ---------------- Add regularizations for sequential editing -------------
            z_diff_norm = None
            if self.reg_z > 0:
                assert self.z_anchor is not None
                z_diff_norm = torch.norm(z - self.z_anchor, dim=1) ** 2

                # z_diff_norm = -self.cos(z, self.z_anchor)
                assert z_diff_norm.ndim == cond_energy_neg.ndim == 1, (z_diff_norm.ndim, cond_energy_neg.ndim)

                cond_energy_neg -= self.reg_z * z_diff_norm
            if self.reg_logits >= 0:
                ori_logits = self.ccf.f[0](self.z_anchor)
                trans_logits = self.ccf.f[0](z)
                tmp_logits = torch.ones_like(ori_logits).cuda()
                tmp_logits[:, 1] = -1
                # tmp_logits = -tmp_logits
                # logits_dist = torch.norm(tmp_logits - trans_logits, dim=1) ** 2
                logits_dist = torch.norm(ori_logits - trans_logits, dim=1) ** 2
                cond_energy_neg += self.reg_logits * logits_dist
            # if self.reg_logits > 0:
            #     ori_logits = self.ccf.f[0](self.z_anchor)
            #     trans_logits = self.ccf.f[0](z)
            #     logits_dist =self.cos(ori_logits,trans_logits)
            #     import pdb
            #     pdb.set_trace()
            #     cond_energy_neg -= self.reg_logits * logits_dist
            # ---------------- [End] Add regularizations for sequential editing -------------

            cond_f_prime = torch.autograd.grad(cond_energy_neg.sum(), [z])[0]
            dz_dt = -0.5 * beta_t * cond_f_prime
            acc = (self.ccf.f[0](z).max(1)[1]).sum().item()
            norm = torch.norm(z - self.z_anchor, dim=1).mean().item()
            logits_diff = self.cos(tmp_logits, trans_logits).mean().item()
            z_diff = self.cos(z, self.z_anchor).mean().item()
            print('n_eval:', self.n_evals, '\tacc:', acc, '\tnorm:', round(norm, 2), 'cos_logits:',
                  round(logits_diff, 2), 'cos_z:', round(z_diff, 2))
            # self.generate_z.append(z.detach())
        self.n_evals += 1

        return dz_dt,


def sample_q_ode(ccf, y, device=torch.device('cuda'), save_path=None, plot=None, every_n_plot=5, **kwargs):
    """sampling in the z space"""
    ccf.eval()
    kwargs['model'].eval()
    latent_dim = kwargs['latent_dim']
    atol = 2e-3  # kwargs['atol']
    rtol = 2e-3  # kwargs['rtol']
    method = kwargs['method']
    use_adjoint = kwargs['use_adjoint']
    kwargs['device'] = device
    # generate initial samples
    # z_k = torch.FloatTensor(y.size(0), latent_dim).normal_(0, 1).to(device)
    z_k = kwargs['z_k']
    # z_k: batch x latent_dim,
    # y: batch
    # ODE function
    vpode = VPODE(ccf, y, plot=plot, every_n_plot=every_n_plot, args=kwargs['args'], kwargs=kwargs)
    vpode.set_anchor_z(z_k)
    states = (z_k,)
    integration_times = torch.linspace(vpode.T, 0., 2).type(torch.float32).to(device)

    # ODE solver
    odeint = odeint_adjoint if use_adjoint else odeint_normal
    state_t = odeint(
        vpode,  # model
        states,  # (z,)
        integration_times,
        atol=atol,
        rtol=rtol,
        method=method)
    ccf.train()
    kwargs['model'].eval()
    z_t0 = state_t[0][-1]
    print(f'#ODE steps for {y[0].item()}: {vpode.n_evals}')
    # import pdb
    # pdb.set_trace()
    return z_t0.detach()  # , vpode.generate_z


def sample_q_sgld(ccf, y, device=torch.device('cuda'), save_path=None, plot=None, every_n_plot=5, **kwargs):
    """sampling in the z space"""
    ccf.eval()

    latent_dim = kwargs['latent_dim']
    sgld_lr = kwargs['sgld_lr']
    sgld_std = kwargs['sgld_std']
    n_steps = kwargs['n_steps']

    # generate initial samples
    init_sample = torch.randn(y.size(0), latent_dim).to(device)
    x_k = torch.autograd.Variable(init_sample, requires_grad=True)

    # sgld
    for k in range(n_steps):
        #
        # if save_path is not None and k % every_n_plot == 0:
        #     g_z_sampled = ccf.g(x_k.detach())
        #     x_sampled = ccf.generate_images(g_z_sampled)
        #     plot('{}/samples_class{}_nsteps{}.png'.format(save_path, y[0].item(), k), x_sampled)

        energy_neg = ccf(x_k, y=y)
        f_prime = torch.autograd.grad(energy_neg.sum(), [x_k])[0]
        x_k.data += sgld_lr * f_prime + sgld_std * torch.randn_like(x_k)

    ccf.train()
    final_samples = x_k.detach()

    return final_samples


def _first_nonzero(x, axis=0):
    nonz = (x > 0)
    return ((nonz.cumsum(axis) == 1) & nonz).max(axis)


def sample_q_fgim(ccf, y, device=torch.device('cuda'), save_path=None, plot=None, every_n_plot=5, **kwargs):
    """sampling in the z space"""
    ccf.eval()
    latent_dim = kwargs['latent_dim']
    weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    max_steps = 30
    decay_factor = 1.0
    t = float(kwargs['args'].reg_logits)
    print('t is ', t)
    with torch.enable_grad():
        x = kwargs['z_k'].detach().clone()
        x = x.unsqueeze(0)
        x = x.repeat(len(weights), 1, 1)
        y_ = y.repeat(len(weights), 1, 1)
        weights = torch.tensor(np.array(weights), device=x.device).float()
        weights = weights.unsqueeze(1).unsqueeze(1)
        cnt = 0
        while True:
            cnt = cnt + 1
            x = x.detach().clone()
            x.requires_grad = True
            preds = ccf.f[0](x.view(-1, latent_dim)).squeeze().view(len(weights), -1, 2)
            correct_classification = torch.abs(1 - torch.softmax(preds, -1)[:, :, [y[0]]]) <= t
            if (correct_classification.sum(dim=0) >= 1).all() or cnt == max_steps:
                _, first_success = _first_nonzero(correct_classification,
                                                  axis=0)
                if cnt == max_steps:
                    print('LAST one')
                break
            loss = torch.nn.CrossEntropyLoss(reduction='sum')(preds.view(-1, 2), y_.view(-1))
            # compute the loss from which to obtain gradients
            # loss = loss_function(x)
            # import pdb
            # pdb.set_trace()
            weights = weights * decay_factor * \
                      (1 - correct_classification.float())
            ccf.zero_grad()
            loss.sum().backward()
            # import pdb
            # pdb.set_trace()
            x = x - weights * x.grad
    x = x.gather(0, first_success.view(1, -1, 1).expand_as(x))[0, :, :]
    ccf.train()
    final_samples = x.detach()
    return final_samples


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--checkpoint_dir", default=None, type=str,
                        help="The directory where checkpoints are saved.")
    parser.add_argument("--cls_dir", default=None, type=str,
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
    parser.add_argument('--reg_z', type=float, default=0.04)
    parser.add_argument('--reg_logits', type=str, default='0.01')
    parser.add_argument('--cls_step', type=str, default='1')
    parser.add_argument('--data_type', type=str, default='sentiment')
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
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
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

    # Load Optimius pre-trained model and tokenizer
    # if args.use_pretrained_model:
    #     # args.encoder_model_type = args.encoder_model_type.lower()
    #     # args.decoder_model_type = args.decoder_model_type.lower()
    #
    #     global_step = args.gloabl_step_eval
    #
    #     # output_encoder_dir = os.path.join(args.checkpoint_dir, 'checkpoint-encoder-{}'.format(global_step))
    #     # output_decoder_dir = os.path.join(args.checkpoint_dir, 'checkpoint-decoder-{}'.format(global_step))
    #     output_full_dir = os.path.join(args.checkpoint_dir, 'checkpoint-full-{}'.format(global_step))
    #
    #     # checkpoints = [[output_encoder_dir, output_decoder_dir]]
    #     # logger.info("Evaluate the following checkpoints: %s", checkpoints)
    #
    #     # Load a trained Encoder model and vocabulary
    #     encoder_config_class, encoder_model_class, encoder_tokenizer_class = MODEL_CLASSES[args.encoder_model_type]
    #     encoder_config = encoder_config_class.from_pretrained(
    #         args.encoder_config_name if args.encoder_config_name else args.encoder_model_name_or_path)
    #
    #     model_encoder = encoder_model_class(config=encoder_config, latent_size=args.latent_size)
    #     tokenizer_encoder = encoder_tokenizer_class.from_pretrained(
    #         args.encoder_tokenizer_name if args.encoder_tokenizer_name else args.encoder_model_name_or_path,
    #         do_lower_case=args.do_lower_case)
    #
    #     model_encoder.to(args.device)
    #     args.block_size = min(args.block_size, tokenizer_encoder.max_len_single_sentence)
    #
    #     # Load a trained Decoder model and vocabulary
    #     decoder_config_class, decoder_model_class, decoder_tokenizer_class = MODEL_CLASSES[args.decoder_model_type]
    #     decoder_config = decoder_config_class.from_pretrained(
    #         args.decoder_config_name if args.decoder_config_name else args.decoder_model_name_or_path)
    #     model_decoder = decoder_model_class(config=decoder_config, latent_size=args.latent_size)
    #     tokenizer_decoder = decoder_tokenizer_class.from_pretrained(
    #         args.decoder_tokenizer_name if args.decoder_tokenizer_name else args.decoder_model_name_or_path,
    #         do_lower_case=args.do_lower_case)
    #     model_decoder.to(args.device)
    #     args.block_size = min(args.block_size, tokenizer_decoder.max_len_single_sentence)
    #
    #     # Load full model
    #     checkpoint = torch.load(os.path.join(output_full_dir, 'training.bin'))
    #     # cls_dir = os.path.join(args.cls_dir, 'training_cls.bin')
    #     # cls_checkpoint = torch.load(os.path.join(cls_dir))
    # import pdb
    # pdb.set_trace()
    # # Chunyuan: Add Padding token to GPT2
    # special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>', }
    # num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)
    # print('We have added', num_added_toks, 'tokens to GPT2')
    # model_decoder.resize_token_embeddings(len(
    #     tokenizer_decoder))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
    # assert tokenizer_decoder.pad_token == '<PAD>'
    #
    # model_vae = VAE(model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, args)
    # model_vae.load_state_dict(checkpoint['model_state_dict'], strict=False)
    # logger.info("Pre-trained Optimus is successfully loaded")
    # model_vae.to(args.device)  #
    #####################
    global_step = args.gloabl_step_eval

    output_full_dir = os.path.join(args.checkpoint_dir, 'checkpoint-full-{}'.format(global_step))
    checkpoint = torch.load(os.path.join(output_full_dir, 'training.bin'))

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

    # Chunyuan: Add Padding token to GPT2
    special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>', }
    num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)
    print('We have added', num_added_toks, 'tokens to GPT2')
    model_decoder.resize_token_embeddings(len(
        tokenizer_decoder))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
    assert tokenizer_decoder.pad_token == '<PAD>'

    # model_decoder.to(args.device)

    model_vae = VAE(model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, args)
    model_vae.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model_vae.to(args.device)
    cls_step_list = args.cls_step.split(',')
    classifier_list = []
    for cls_ in cls_step_list:
        classifier_ = DenseEmbedder(args.latent_size, up_dim=2, depth=4, num_classes=STEP_CLASSES[cls_])
        cls_dir = os.path.join(args.cls_dir + cls_, 'training_cls.bin')
        cls_checkpoint = torch.load(cls_dir)
        classifier_.load_state_dict(cls_checkpoint['model_state_dict'], strict=False)
        classifier_list.append(classifier_)
    classifier = CCF(classifier_list)
    classifier.to(args.device)

    # on_gpu = next(model_vae.parameters()).is_cuda

    model_kwargs = {
        'model': model_vae,
        'enc_tokenizer': tokenizer_encoder,
        'dec_tokenizer': tokenizer_decoder,
        'reg_z': args.reg_z,
        'args': args
    }
    sde_kwargs = {}
    ode_kwargs = {'atol': 1e-3, 'rtol': 1e-3, 'method': 'dopri5', 'use_adjoint': True, 'latent_dim': args.latent_size}
    ld_kwargs = {'batch_size': args.per_gpu_eval_batch_size, 'sgld_lr': 1,
                 'sgld_std': 1e-2, 'n_steps': 20}
    sample_q = sample_q_fgim  # sample_q_ode
    latent_dim = args.latent_size
    n_classes = args.n_classes
    save_path = os.path.join(args.output_dir, 'sample')
    eval_dataloader = build_dataload_and_cache_examples(args, [tokenizer_encoder, tokenizer_decoder], evaluate=True)
    logits_list = [float(i) for i in args.reg_logits.split(',')]
    for logit in logits_list:
        args.reg_logits = logit
        model_kwargs['args'] = args
        print('logit is', logit)
        condSampling = ConditionalTransfer(sample_q, args.per_gpu_eval_batch_size, latent_dim, n_classes, classifier,
                                           device, save_path, ode_kwargs, ld_kwargs, sde_kwargs,
                                           every_n_plot=5, model_kwargs=model_kwargs, test_data_batch=eval_dataloader)
        # condSampling = ConditionalSampling(sample_q, args.per_gpu_eval_batch_size, latent_dim, n_classes, classifier,
        #                                    device, save_path, ode_kwargs, ld_kwargs, sde_kwargs,
        #                                    every_n_plot=5, model_kwargs=model_kwargs)
        condSampling.get_samples()


if __name__ == "__main__":
    main()
