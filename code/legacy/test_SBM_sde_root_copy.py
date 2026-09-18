import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
import os
from my_transformers import *
from transformers import AutoTokenizer, AdamW
import copy
from examples.big_ae.modules import VAE,sample_sequence_conditional, DDPM, LinearModel
import logging
from tqdm import tqdm, trange
import random
import time
import torch.nn.init as init

class Args:
    latent_size = 64
    fb_mode = 1
    beta = 1.0
    nt = 2000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_gpu_train_batch_size = 128
    per_gpu_eval_batch_size = 128
    n_gpu = 1
    train_data_file = '../data/datasets/yelp_data/train.shuf.txt'
    test_data_file = '../data/datasets/yelp_data/test.txt'
    eval_data_file = '../data/datasets/yelp_data/test.txt'
    max_seq_length=512
    block_size=30
    decoder_model_type='gpt2'
    encoder_model_type='bertu'
    dataset = 'Yelp_cls'
    num_train_epochs = 10
    gradient_accumulation_steps = 1
    learning_rate = 5e-4
    adam_epsilon = 1e-8
    seed = 42
    fp16 = True
    fp16_opt_level = 'O1'
    max_grad_norm = 1.0
    max_steps = -1

args = Args()
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device


output_full_dir = '../ckpts/base_yelp/checkpoint-full-1'
checkpoint = torch.load(os.path.join(output_full_dir, 'training.bin'))

encoder_model='prajjwal1/bert-small'
tokenizer_encoder = AutoTokenizer.from_pretrained(encoder_model)
model_encoder = BertForLatentConnectorAVG.from_pretrained(encoder_model, latent_size=64,
                                                    pad_id=tokenizer_encoder.pad_token_id)
decoder_model = 'gpt2'
tokenizer_decoder = AutoTokenizer.from_pretrained(decoder_model)
model_decoder = GPT2ForLatentConnectorNew2.from_pretrained(decoder_model, latent_size=64,
                                                        latent_as_gpt_emb=True,
                                                        latent_as_gpt_memory=True)
model_decoder.transformer.change_order()

special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>', }
num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)

model_decoder.resize_token_embeddings(len(tokenizer_decoder))
model_vae = VAE(model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, args)
model_vae.load_state_dict(checkpoint['model_state_dict'], strict=False)
model_vae.to(args.device)  #


ddpm = DDPM(eps_model=LinearModel(args.latent_size), betas=(1e-4, 0.02), n_T=args.nt,)
ddpm.to(args.device)
def weights_init_rondom(model):
    model = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_state_dict = model.state_dict()
    for key in model_state_dict:
        #         pdb.set_trace()
        if 'encoder' in key:
            init.normal_(model_state_dict[key].data)
            # weight_init(item)
ddpm.apply(weights_init_rondom)



from examples.big_ae.utils import (BucketingDataLoader, TextDataset_Split,
                   TextDataset_2Tokenizers, frange_cycle_zero_linear)
from torch.utils.data import Dataset, DataLoader

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


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
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


from transformers import GPT2LMHeadModel as GPT2_
from transformers import GPT2TokenizerFast
model_id = '../../Optimus-ODE/output/gpt2_sentiment'
model_ppl = GPT2_.from_pretrained(model_id).cuda()
tokenizer_ppl = GPT2TokenizerFast.from_pretrained(model_id)


args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
t_total = len(train_dataloader)//args.gradient_accumulation_steps * args.num_train_epochs

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



global_step = 0
train_step = 0
tr_loss, logging_loss = 0.0, 0.0
model_vae.zero_grad()

train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=False)

n_iter = int(args.num_train_epochs) * len(train_dataloader)
set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
args.logging_steps = int(np.floor(len(train_dataloader)))
args.save_steps = args.logging_steps

stop_flag = False
best_gan_diff = 200
best_diff_cnt = 0
start_time = time.time()
for epoch in train_iterator:
    epoch_iterator = tqdm(dataloader, desc="Iteration", disable=False)
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
                    model_vae, tokenizer_encoder, tokenizer_decoder, args, 1,
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

