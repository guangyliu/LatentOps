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

from __future__ import absolute_import, division, print_function

from my_transformers import *
import argparse
import logging

import os
import random

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint as odeint_normal

from tqdm import tqdm

from modules import VAE, DenseEmbedder, CCF, sample_sequence_conditional
from modules import GAN
from modules import ConditionalSampling

logger = logging.getLogger(__name__)

from run_lm_vae_training import MODEL_CLASSES

STEP_CLASSES = {
    '1': 2,
    '2': 2,
    '4': 3,
    '33':2,
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def latent2text(model_vae, decoder_tokenizer,z,print_text=False, logits=None, word='dog'):
    device = z.device
    context_tokens = decoder_tokenizer.encode('<BOS>')
    out = sample_sequence_conditional(
        model=model_vae.decoder,
        context=context_tokens,
        past=z.detach(),
        length=50,  # Chunyuan: Fix length; or use <EOS> to complete a sentence
        num_samples=z.size(0),
        device=device,
        decoder_tokenizer=decoder_tokenizer,
        eos_id=model_vae.eos_token_id
    )
    text_list = []
    for i in range(z.size(0)):
        text_x1 = decoder_tokenizer.decode(out[i,:].tolist(), clean_up_tokenization_spaces=False).split(
            '<EOS>')[0].replace('<BOS>', '').strip()
        text_x1 = text_x1.split()
        text_x1 = ' '.join(text_x1)
        text_list.append(text_x1)
        if print_text:
            if logits is not None:
                print(i,round(logits[i].item(),4),(word in text_x1),'\n',text_x1)
            else:
                print(i,text_x1)
    return text_list

class VPODE(nn.Module):
    def __init__(self, ccf, y, beta_min=0.1, beta_max=20, T=1.0, save_path=None, plot=None, every_n_plot=5,
                 kwargs=None):
        super().__init__()
        self.ccf = ccf
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.T = T
        self.y = y
        self.save_path = save_path
        self.n_evals = 0
        self.every_n_plot = every_n_plot
        self.plot = plot
        self.model_vae = kwargs['model']
        self.decoder_tokenizer = kwargs['dec_tokenizer']
        self.device = kwargs['device']
        self.args = kwargs['args']
        self.reg_logits = self.args.reg_logits

    def forward(self, t_k, states):
        z = states[0]
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            beta_t = self.beta_0 + t_k * (self.beta_1 - self.beta_0)
            cond_energy_neg = self.ccf.get_cond_energy(z, self.y)
            cond_f_prime = torch.autograd.grad(cond_energy_neg.sum(), [z])[0]
            dz_dt = -0.5 * beta_t * cond_f_prime
        self.n_evals += 1
        return dz_dt,


def sample_q_ode(ccf, y, device=torch.device('cuda'), save_path=None, plot=None, every_n_plot=5, **kwargs):
    """sampling in the z space"""
    ccf.eval()
    kwargs['model'].eval()
    atol = kwargs['atol']
    rtol = kwargs['rtol']
    method = kwargs['method']
    use_adjoint = kwargs['use_adjoint']
    kwargs['device'] = device
    # generate initial samples
    z_k = kwargs['z_k']
    # z_k: batch x latent_dim,
    # y: batch
    # ODE function
    vpode = VPODE(ccf, y, save_path=save_path, plot=plot, every_n_plot=every_n_plot, kwargs=kwargs)
    states = (z_k,)
    integration_times = torch.linspace(vpode.T, 0., 2).type(torch.float32).to(device)

    # ODE solver
    odeint = odeint_adjoint if use_adjoint else odeint_normal
    state_t = odeint(
        vpode,  # model
        states,  # (z,)
        integration_times,
        atol=atol,  # tolerance
        rtol=rtol,
        method=method)

    ccf.train()
    kwargs['model'].eval()
    z_t0 = state_t[0][-1]
    # print(f'#ODE steps : {vpode.n_evals}')
    return z_t0.detach(), vpode.n_evals


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
        energy_neg = ccf(x_k, y=y)
        f_prime = torch.autograd.grad(energy_neg.sum(), [x_k])[0]
        x_k.data += sgld_lr * f_prime + sgld_std * torch.randn_like(x_k)

    ccf.train()
    final_samples = x_k.detach()

    return final_samples,k

def sample_q_vpsde(ccf, y, device=torch.device('cuda'), save_path=None, plot=None, every_n_plot=5,
                   beta_min=0.1, beta_max=20, T=1, eps=1e-3, **kwargs):
    """sampling in the z space"""
    ccf.eval()

    latent_dim = kwargs['latent_dim']
    N = kwargs['N']
    correct_nsteps = kwargs['correct_nsteps']
    target_snr = kwargs['target_snr']

    # generate initial samples
    z_init = torch.FloatTensor(y.size(0), latent_dim).normal_(0, 1).to(device)
    z_k = torch.autograd.Variable(z_init, requires_grad=True)

    discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    alphas = 1. - discrete_betas
    timesteps = torch.linspace(T, eps, N, device=device)

    # vpsde
    for k in range(N):
        energy_neg = ccf(z_k, y=y)

        # predictor
        t_k = timesteps[k]
        timestep = (t_k * (N - 1) / T).long()
        beta_t = discrete_betas[timestep]
        alpha_t = alphas[timestep]

        score_t = torch.autograd.grad(energy_neg.sum(), [z_k])[0]

        z_k = (2 - torch.sqrt(alpha_t)) * z_k + beta_t * score_t
        noise = torch.FloatTensor(y.size(0), latent_dim).normal_(0, 1).to(device)
        z_k = z_k + torch.sqrt(beta_t) * noise

        # corrector
        for j in range(correct_nsteps):
            noise = torch.FloatTensor(y.size(0), latent_dim).normal_(0, 1).to(device)

            grad_norm = torch.norm(score_t.reshape(score_t.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha_t

            assert step_size.ndim == 0, step_size.ndim

            z_k_mean = z_k + step_size * score_t
            z_k = z_k_mean + torch.sqrt(step_size * 2) * noise

    ccf.train()
    final_samples = z_k.detach()

    return final_samples, k

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
    parser.add_argument("--dataset", default='Yelp_cls', type=str, help="The dataset.")

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
    parser.add_argument('--reg_logits', type=float, default='0.0')
    parser.add_argument('--load_weight_from_json', type=str, default='')
    args = parser.parse_args()
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
    print("Initialize the Extra Layer.")
    model_decoder.transformer.change_order()
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
    sde_kwargs = {'N': 1000, 'correct_nsteps': 2, 'target_snr': 0.16}
    ode_kwargs = {'atol': 1e-3, 'rtol': 1e-3, 'method': 'dopri5', 'use_adjoint': True, 'latent_dim': args.latent_size}
    ld_kwargs = {'batch_size': args.per_gpu_eval_batch_size, 'sgld_lr': 1,
                 'sgld_std': 1e-2, 'n_steps': 200}
    sample_q = sample_q_ode
    latent_dim = args.latent_size
    n_classes = args.n_classes
    save_path = os.path.join(args.output_dir, 'sample')
    cls_list_all = args.cls_step.split(';')
    if cls_list_all[0] == '':
        cls_list_all = cls_list_all[1:]
    weight_dict = None
    if args.load_weight_from_json != '':
        import json
        with open(args.load_weight_from_json) as f:
            weight_dict = json.load(f)
    pbar = tqdm(cls_list_all)
    for cls_list in pbar:
        args.cls_ = cls_list
        cls_step_list = cls_list.split(',')
        classifier_list = []
        for ii, cls_ in enumerate(cls_step_list):
            if cls_.isdigit(): # attributes
                num_classes = STEP_CLASSES[cls_]
            else: # keyword
                num_classes = 2
            classifier_ = DenseEmbedder(args.latent_size, up_dim=2, depth=4, num_classes=num_classes)
            cls_dir = os.path.join(args.cls_dir + cls_, 'training_cls.bin')
            cls_checkpoint = torch.load(cls_dir)
            classifier_.load_state_dict(cls_checkpoint['model_state_dict'], strict=False)
            if weight_dict != None:
                classifier_.set_energy_weight(weight_dict[cls_][0])
            else:
                classifier_.set_energy_weight(args.weight_energy)
            classifier_list.append(classifier_)
        classifier = CCF(classifier_list)
        classifier.to(args.device)
        mode = [1, 0, 0]
        condSampling = ConditionalSampling(sample_q, args.per_gpu_eval_batch_size, latent_dim, n_classes, classifier,
                                           device, save_path, ode_kwargs, ld_kwargs, sde_kwargs,
                                           every_n_plot=5, model_kwargs=model_kwargs, gan=gan, disable_bar=True,mode=mode)
        condSampling.get_samples_multiple(args.att_val_list.split(';'), mode=mode)

if __name__ == "__main__":
    main()
