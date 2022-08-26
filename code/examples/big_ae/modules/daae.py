import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import log_sum_exp
from .vae import DenseEmbedder
import pdb
import numpy as np
import logging
logger = logging.getLogger(__name__)


class DAAE(nn.Module):
    """DAAE with normal prior"""
    def __init__(self, encoder, decoder,  tokenizer_encoder, tokenizer_decoder, args): # 
        super(DAAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        # self.D = nn.Sequential(nn.Linear(args.latent_size, args.latent_size*4), nn.ReLU(),
        #     nn.Linear(args.latent_size*4, 1), nn.Sigmoid())
        self.D= DenseEmbedder(args.latent_size, 4, 4, 1)

        self.args = args
        self.nz = args.latent_size

        self.eos_token_id = tokenizer_decoder.convert_tokens_to_ids([tokenizer_decoder.eos_token])[0]
        self.pad_token_id = tokenizer_decoder.convert_tokens_to_ids([tokenizer_decoder.pad_token])[0]
        self.bos_token_id = tokenizer_decoder.convert_tokens_to_ids([tokenizer_decoder.bos_token])[0]
        self.tokenizer_decoder = tokenizer_decoder
        self.tokenizer_encoder = tokenizer_encoder
        self.pad_enc = tokenizer_encoder.convert_tokens_to_ids([tokenizer_encoder.pad_token])[0]
        self.cls_enc = tokenizer_encoder.convert_tokens_to_ids([tokenizer_encoder.cls_token])[0]
        self.sep_enc = tokenizer_encoder.convert_tokens_to_ids([tokenizer_encoder.sep_token])[0]
        # self.mask_enc = tokenizer_encoder.convert_tokens_to_ids([tokenizer_encoder.mask_token])[0]
#         special_tokens_dict = {'bos_token': '<BOS>','eos_token':'<EOS>','sep_token':'<SEP>','mask_token':'<MASK>'}
#         self.tokenizer_decoder.add_special_tokens(special_tokens_dict)
        # connector: from Bert hidden units to the latent space
        # self.linear = nn.Linear(args.nz, 2 * args.nz, bias=False)

        # Standard Normal prior
        loc = torch.zeros(self.nz, device=args.device)
        scale = torch.ones(self.nz, device=args.device)
        self.prior = torch.distributions.normal.Normal(loc, scale)

    def reparameterize(self, mu, logvar, nsamples=1):
        """sample from posterior Gaussian family
        Args:
            mu: Tensor
                Mean of gaussian distribution with shape (batch, nz)
            logvar: Tensor
                logvar of gaussian distibution with shape (batch, nz)
        Returns: Tensor
            Sampled z with shape (batch, nsamples, nz)
        """
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()

        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)

        eps = torch.zeros_like(std_expd).normal_()

        return mu_expd + torch.mul(eps, std_expd)

    def word_drop(self, inputs, p):
        x_ = []
        for i in range(inputs.size(0)):
            words = inputs[i].tolist()
            keep = np.random.rand(len(words)) > p
            keep[0] = True
            sent = [w for j, w in enumerate(words) if (keep[j] or words[j] == self.sep_enc)]
            sent += [self.pad_enc] * (len(words) - len(sent))
            x_.append(sent)
#         pdb.set_trace() 
        return torch.LongTensor(x_).contiguous().to(inputs.device)
    
    def forward(self, inputs, labels):
        if self.args.noise:
            inputs_ = self.word_drop(inputs,0)
        else:
            inputs_ = inputs
#         pdb.set_trace() 
        attention_mask=(inputs_ > 0).float()
        reconstrution_mask=(labels != 50257).float() # 50257 is the padding token for GPT2
        sent_length = torch.sum(reconstrution_mask, dim=1)
        outputs = self.encoder(inputs_, attention_mask)

        pooled_hidden_fea = outputs[1]  # model outputs are always tuple in pytorch-transformers (see doc)

        # Connect hidden feature to the latent space
        mean, logvar = self.encoder.linear(pooled_hidden_fea).chunk(2, -1)
        latent_z = self.reparameterize(mean, logvar, 1).squeeze(1)
        # Decoding
#         import ipdb
#         ipdb.set_trace()
        outputs = self.decoder(input_ids=labels, past=latent_z, labels=labels, label_ignore=self.pad_token_id)
        loss_rec = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
        if self.args.adv:
            loss_d, adv = self.loss_adv(latent_z) #
        else:
            loss_d = adv = torch.zeros_like(loss_rec).detach()
            
        if self.args.length_weighted_loss: # 需要 adv loss 
            loss = loss_rec / sent_length  + self.args.dim_target_kl* adv
        else:
            loss = loss_rec + self.args.dim_target_kl* adv

        return loss_rec, adv, loss, loss_d

    def loss_adv(self, z):
        zn = torch.randn_like(z) # zn 是latent space 的vector，z是sentence->latent 的vector
        zeros = torch.zeros(len(zn),1,device=z.device)
        ones = torch.ones(len(zn),1,device=z.device)
        loss_d = F.binary_cross_entropy(self.D(z.detach()), zeros,reduction='none') + \
            F.binary_cross_entropy(self.D(zn), ones,reduction='none') # zn 真的 -> 1, z 假的 -> 0, fix encoder, train D
        loss_g = F.binary_cross_entropy(self.D(z), ones,reduction='none') # 错误的entropy
        return loss_d.squeeze(-1), loss_g.squeeze(-1)
    

    def encoder_sample(self, bert_fea, nsamples):
        """sampling from the encoder
        Returns: Tensor1
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
        """

        # (batch_size, nz)

        mu, logvar = self.encoder.linear(bert_fea).chunk(2, -1)
        mu, logvar = mu.squeeze(0), logvar.squeeze(0)

        # (batch, nsamples, nz)
        z = self.reparameterize(mu, logvar, nsamples)

        return z, (mu, logvar)


    def encode_stats(self, x):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the mean of latent z with shape [batch, nz]
            Tensor2: the logvar of latent z with shape [batch, nz]
        """

        return self.encoder.encode_stats(x)

    def decode(self, z, strategy, K=10):
        """generate samples from z given strategy
        Args:
            z: [batch, nsamples, nz]
            strategy: "beam" or "greedy" or "sample"
            K: the beam width parameter
        Returns: List1
            List1: a list of decoded word sequence
        """

        if strategy == "beam":
            return self.decoder.beam_search_decode(z, K)
        elif strategy == "greedy":
            return self.decoder.greedy_decode(z)
        elif strategy == "sample":
            return self.decoder.sample_decode(z)
        else:
            raise ValueError("the decoding strategy is not supported")
            
    def decode_eval_greedy(self, x, z):
#         n_sample, length = x.size()
        x_shape = list(x.size())
        z_shape = list(z.size())
        if len(z_shape) == 3:
            x = x.unsqueeze(1).repeat(1, z_shape[1], 1).contiguous().view(x_shape[0]*z_shape[1], x_shape[-1]) 
            z = z.contiguous().view(x_shape[0]*z_shape[1], z_shape[-1]) 
        batch_size = z.size()[0]
        decoded_batch = [[] for _ in range(batch_size)]
        x_ = torch.zeros_like(z[:,:1],dtype=torch.long) + self.bos_token_id
#         for i in range(length):
        mask = torch.zeros_like(z[:,0],dtype=torch.long) + 1
        length_c = 1
        end_symbol = torch.zeros_like(mask,dtype=torch.long) + self.eos_token_id
        while mask.sum().item() != 0 and length_c < 100:
            output = self.decoder(input_ids=x_, past=z)
            out_token = output[0][:,-1:].max(-1)[1]
            x_ = torch.cat((x_, out_token),-1)
            length_c += 1
            mask =  torch.mul((out_token.squeeze(-1) != end_symbol), mask)
            for i in range(batch_size):
#                 word = self.tokenizer_decoder.decode(out_token[i].tolist())
                if mask[i].item():
                    decoded_batch[i].append(self.tokenizer_decoder.
                                decode(out_token[i].item()))
#         out_tokens = x_[:,1:]
        for i in range(batch_size):
            decoded_batch[i] = ''.join(decoded_batch[i])
        if False:
            out = self.log_probability_out(x,z)
            can_dict = {}
            can_dict['GT'] = self.tokenizer_decoder.decode(x[0][1:].tolist()).split('<EOS>')[0]
            can_dict['TF'] = self.tokenizer_decoder.decode(out[1].max(-1)[1][0].tolist()).split('<EOS>')[0]
            can_dict['GD'] = decoded_batch[0]
            for key in can_dict.keys():
                print(key,'\t',can_dict[key])
#             import ipdb
#             ipdb.set_trace()
        return decoded_batch
    
    def decode_eval_greedy_tf(self, x, z):
#         n_sample, length = x.size()
        x_shape = list(x.size())
        z_shape = list(z.size())
        if len(z_shape) == 3:
            x = x.unsqueeze(1).repeat(1, z_shape[1], 1).contiguous().view(x_shape[0]*z_shape[1], x_shape[-1]) 
            z = z.contiguous().view(x_shape[0]*z_shape[1], z_shape[-1]) 
        batch_size = z.size()[0]
        decoded_batch = [[] for _ in range(batch_size)]
        x_ = torch.zeros_like(z[:,:1],dtype=torch.long) + self.bos_token_id
#         for i in range(length):
        mask = torch.zeros_like(z[:,0],dtype=torch.long) + 1
        length_c = 1
        end_symbol = torch.zeros_like(mask,dtype=torch.long) + self.eos_token_id
        while mask.sum().item() != 0 and length_c < 100 and length_c <= x_shape[-1]:
            output = self.decoder(input_ids=x_, past=z)
            out_token = output[0][:,-1:].max(-1)[1]
            x_ = torch.cat((x_, x[:,length_c:length_c+1]),-1)
            length_c += 1
            mask =  torch.mul((out_token.squeeze(-1) != end_symbol), mask)
            for i in range(batch_size):
#                 word = self.tokenizer_decoder.decode(out_token[i].tolist())
                if mask[i].item():
                    decoded_batch[i].append(self.tokenizer_decoder.
                                decode(out_token[i].item()))
#         out_tokens = x_[:,1:]
        for i in range(batch_size):
            decoded_batch[i] = ''.join(decoded_batch[i])
        return decoded_batch
    
    def reconstruct(self, x, decoding_strategy="greedy", K=5):
        """reconstruct from input x
        Args:
            x: (batch, *)
            decoding_strategy: "beam" or "greedy" or "sample"
            K: the beam width parameter
        Returns: List1
            List1: a list of decoded word sequence
        """
        z = self.sample_from_inference(x).squeeze(1)

        return self.decode(z, decoding_strategy, K)

    def log_probability(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            log_p: (batch_size, n_sample).
                log_p(x|z) across different x and z
        """
#         outputs_ = self.decode_eval_gy(x,z)
        outputs = self.decoder(input_ids=x, past=z, labels=x, label_ignore=self.pad_token_id)
        loss_rec = outputs[0]
        return -loss_rec
    
    def log_probability_out(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            log_p: (batch_size, n_sample).
                log_p(x|z) across different x and z
        """
#         outputs_ = self.decode_eval_gy(x,z)
        outputs = self.decoder(input_ids=x, past=z, labels=x, label_ignore=self.pad_token_id)
        return outputs


    def loss_iw(self, x0, x1, nsamples=50, ns=1):
        """
        Args:
            x: if the data is constant-length, x is the data tensor with
                shape (batch, *). Otherwise x is a tuple that contains
                the data tensor and length list
        Returns: Tensor1, Tensor2, Tensor3
            Tensor1: total loss [batch]
            Tensor2: reconstruction loss shape [batch]
            Tensor3: KL loss shape [batch]
        """

        # encoding into bert features
        bert_fea = self.encoder(x0)[1]
        # (batch_size, nz)

        mu, logvar = self.encoder.linear(bert_fea).chunk(2, -1)
        

        ##################
        # compute KL
        ##################
        # pdb.set_trace()
        KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)

        # mu, logvar = mu.squeeze(0), logvar.squeeze(0)
        ll_tmp, rc_tmp = [], []
        for _ in range(int(nsamples / ns)):

            # (batch, nsamples, nz)
            z = self.reparameterize(mu, logvar, ns)
            # past = self.decoder.linear(z)
            past = z
         
            # [batch, nsamples]
            log_prior = self.eval_prior_dist(z)
            log_gen = self.eval_cond_ll(x1, past)
            log_infer = self.eval_inference_dist(z, (mu, logvar))

            # pdb.set_trace()
            log_gen = log_gen.unsqueeze(0).contiguous().view(z.shape[0],-1)


            # pdb.set_trace()
            rc_tmp.append(log_gen)
            ll_tmp.append(log_gen + log_prior - log_infer)

            
        
        log_prob_iw = log_sum_exp(torch.cat(ll_tmp, dim=-1), dim=-1) - math.log(nsamples)
        log_gen_iw = torch.mean(torch.cat(rc_tmp, dim=-1), dim=-1)

        return log_prob_iw, log_gen_iw , KL
    
    
    def rec_sample(self, x0, x1, sample=False):
        bert_fea = self.encoder(x0)[1]
        mu, logvar = self.encoder.linear(bert_fea).chunk(2, -1)
        z = self.reparameterize(mu, logvar, 1)
        rec='rec'
        if sample:
            z = torch.tensor(np.random.normal(size=(z.size()[0], z.size()[-1])),dtype=torch.double).cuda().unsqueeze(1)
            x1 = torch.zeros_like(x1)
            rec='sample'
        decoded_batch = self.decode_eval_greedy(x1, z)
        with open('/home/lptang/Optimus/samples/'+self.args.output_dir.split('/')[-1]+'.'+str(self.args.gloabl_step_eval)+'.'+rec,'a+') as f:
            for sent in decoded_batch:
                f.write(sent+'\n')
        
    def nll_iw(self, x0, x1, nsamples, ns=1):
        """compute the importance weighting estimate of the log-likelihood
        Args:
            x0, x1:  two different tokenization results of x, where x is the data tensor with shape (batch, *). 
            nsamples: Int
                the number of samples required to estimate marginal data likelihood
        Returns: Tensor1
            Tensor1: the estimate of log p(x), shape [batch]
        """

        # compute iw every ns samples to address the memory issue
        # nsamples = 500, ns = 100
        # nsamples = 500, ns = 10

        # TODO: note that x is forwarded twice in self.encoder.sample(x, ns) and self.eval_inference_dist(x, z, param)
        #.      this problem is to be solved in order to speed up

        tmp = []
        for _ in range(int(nsamples / ns)):
            # [batch, ns, nz]

            # Chunyuan:
            # encoding into bert features
            pooled_hidden_fea = self.encoder(x0)[1]

            # param is the parameters required to evaluate q(z|x)
            z, param = self.encoder_sample(pooled_hidden_fea, ns)

            # [batch, ns]
            log_comp_ll = self.eval_complete_ll(x1, z)
            log_infer_ll = self.eval_inference_dist(z, param)

            tmp.append(log_comp_ll - log_infer_ll)

        ll_iw = log_sum_exp(torch.cat(tmp, dim=-1), dim=-1) - math.log(nsamples)

        return ll_iw

    def KL(self, x):
        _, KL = self.encode(x, 1)

        return KL

    def eval_prior_dist(self, zrange):
        """perform grid search to calculate the true posterior
        Args:
            zrange: tensor
                different z points that will be evaluated, with
                shape (k^2, nz), where k=(zmax - zmin)/space
        """

        # (k^2)
        return self.prior.log_prob(zrange).sum(dim=-1)

    def eval_complete_ll(self, x, z):
        """compute log p(z,x)
        Args:
            x: Tensor
                input with shape [batch, seq_len]
            z: Tensor
                evaluation points with shape [batch, nsamples, nz]
        Returns: Tensor1
            Tensor1: log p(z,x) Tensor with shape [batch, nsamples]
        """

        # [batch, nsamples]
        log_prior = self.eval_prior_dist(z)
        log_gen = self.eval_cond_ll(x, z)

        return log_prior + log_gen



    def eval_cond_ll(self, x, z):
        """compute log p(x|z)
        """
        x_shape = list(x.size())
        z_shape = list(z.size())
        if len(z_shape) == 3:
            x = x.unsqueeze(1).repeat(1, z_shape[1], 1).contiguous().view(x_shape[0]*z_shape[1], x_shape[-1]) 
            z = z.contiguous().view(x_shape[0]*z_shape[1], z_shape[-1]) 

        return self.log_probability(x, z)



    def eval_log_model_posterior(self, x, grid_z):
        """perform grid search to calculate the true posterior
         this function computes p(z|x)
        Args:
            grid_z: tensor
                different z points that will be evaluated, with
                shape (k^2, nz), where k=(zmax - zmin)/pace
        Returns: Tensor
            Tensor: the log posterior distribution log p(z|x) with
                    shape [batch_size, K^2]
        """
        try:
            batch_size = x.size(0)
        except:
            batch_size = x[0].size(0)

        # (batch_size, k^2, nz)
        grid_z = grid_z.unsqueeze(0).expand(batch_size, *grid_z.size()).contiguous()

        # (batch_size, k^2)
        log_comp = self.eval_complete_ll(x, grid_z)

        # normalize to posterior
        log_posterior = log_comp - log_sum_exp(log_comp, dim=1, keepdim=True)

        return log_posterior

    def sample_from_inference(self, x, nsamples=1):
        """perform sampling from inference net
        Returns: Tensor
            Tensor: samples from infernece nets with
                shape (batch_size, nsamples, nz)
        """
        bert_fea = self.encoder(x)[1]
        mu, logvar = self.encoder.linear(bert_fea).chunk(2, -1)
#         z, _ = self.encoder.sample(x, nsamples)
        z = self.reparameterize(mu, logvar, nsamples)
        return z


    def sample_from_posterior(self, x, nsamples):
        """perform MH sampling from model posterior
        Returns: Tensor
            Tensor: samples from model posterior with
                shape (batch_size, nsamples, nz)
        """

        # use the samples from inference net as initial points
        # for MCMC sampling. [batch_size, nsamples, nz]
        cur = self.encoder.sample_from_inference(x, 1)
        cur_ll = self.eval_complete_ll(x, cur)
        total_iter = self.args.mh_burn_in + nsamples * self.args.mh_thin
        samples = []
        for iter_ in range(total_iter):
            next = torch.normal(mean=cur,
                std=cur.new_full(size=cur.size(), fill_value=self.args.mh_std))
            # [batch_size, 1]
            next_ll = self.eval_complete_ll(x, next)
            ratio = next_ll - cur_ll

            accept_prob = torch.min(ratio.exp(), ratio.new_ones(ratio.size()))

            uniform_t = accept_prob.new_empty(accept_prob.size()).uniform_()

            # [batch_size, 1]
            mask = (uniform_t < accept_prob).float()
            mask_ = mask.unsqueeze(2)

            cur = mask_ * next + (1 - mask_) * cur
            cur_ll = mask * next_ll + (1 - mask) * cur_ll

            if iter_ >= self.args.mh_burn_in and (iter_ - self.args.mh_burn_in) % self.args.mh_thin == 0:
                samples.append(cur.unsqueeze(1))

        return torch.cat(samples, dim=1)


    def calc_model_posterior_mean(self, x, grid_z):
        """compute the mean value of model posterior, i.e. E_{z ~ p(z|x)}[z]
        Args:
            grid_z: different z points that will be evaluated, with
                    shape (k^2, nz), where k=(zmax - zmin)/pace
            x: [batch, *]
        Returns: Tensor1
            Tensor1: the mean value tensor with shape [batch, nz]
        """

        # [batch, K^2]
        log_posterior = self.eval_log_model_posterior(x, grid_z)
        posterior = log_posterior.exp()

        # [batch, nz]
        return torch.mul(posterior.unsqueeze(2), grid_z.unsqueeze(0)).sum(1)

    def calc_infer_mean(self, x):
        """
        Returns: Tensor1
            Tensor1: the mean of inference distribution, with shape [batch, nz]
        """

        mean, logvar = self.encoder.forward(x)

        return mean


 

    def eval_inference_dist(self, z, param):
        """this function computes log q(z | x)
        Args:
            z: tensor
                different z points that will be evaluated, with
                shape [batch, nsamples, nz]
        Returns: Tensor1
            Tensor1: log q(z|x) with shape [batch, nsamples]
        """

        nz = z.size(2)
        mu, logvar = param

        # (batch_size, 1, nz)
        mu, logvar = mu.unsqueeze(1), logvar.unsqueeze(1)
        var = logvar.exp()

        # (batch_size, nsamples, nz)
        dev = z - mu

        # (batch_size, nsamples)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        return log_density



    def calc_mi(self, test_data_batch, args):
        # calc_mi_v3
        import math 
        from modules.utils import log_sum_exp

        mi = 0
        num_examples = 0

        mu_batch_list, logvar_batch_list = [], []
        neg_entropy = 0.
        for batch_data in test_data_batch:

            x0, _, _ = batch_data
            x0 = x0.to(args.device)

            # encoding into bert features
            bert_fea = self.encoder(x0)[1]

            (batch_size, nz)
            mu, logvar = self.encoder.linear(bert_fea).chunk(2, -1)

            x_batch, nz = mu.size()

            #print(x_batch, end=' ')

            num_examples += x_batch

            # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)

            neg_entropy += (-0.5 * nz * math.log(2 * math.pi)- 0.5 * (1 + logvar).sum(-1)).sum().item()
            mu_batch_list += [mu.cpu()]
            logvar_batch_list += [logvar.cpu()]

#             pdb.set_trace()

        neg_entropy = neg_entropy / num_examples
        ##print()

        num_examples = 0
        log_qz = 0.
        for i in range(len(mu_batch_list)):
            ###############
            # get z_samples
            ###############
            mu, logvar = mu_batch_list[i].cuda(), logvar_batch_list[i].cuda()
            
            # [z_batch, 1, nz]

            z_samples = self.reparameterize(mu, logvar, 1)

            z_samples = z_samples.view(-1, 1, nz)
            num_examples += z_samples.size(0)

            ###############
            # compute density
            ###############
            # [1, x_batch, nz]
            #mu, logvar = mu_batch_list[i].cuda(), logvar_batch_list[i].cuda()
            #indices = list(np.random.choice(np.arange(len(mu_batch_list)), 10)) + [i]
            indices = np.arange(len(mu_batch_list))
            mu = torch.cat([mu_batch_list[_] for _ in indices], dim=0).cuda()
            logvar = torch.cat([logvar_batch_list[_] for _ in indices], dim=0).cuda()
            x_batch, nz = mu.size()

            mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
            var = logvar.exp()

            # (z_batch, x_batch, nz)
            dev = z_samples - mu

            # (z_batch, x_batch)
            log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
                0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

            # log q(z): aggregate posterior
            # [z_batch]
            log_qz += (log_sum_exp(log_density, dim=1) - math.log(x_batch)).sum(-1)

        log_qz /= num_examples
        mi = neg_entropy - log_qz

        return mi



    def calc_au(self, eval_dataloader, args, delta=0.01):
        """compute the number of active units
        """
        cnt = 0
        for batch_data in eval_dataloader:

            x0, _, _ = batch_data
            x0 = x0.to(args.device)

            # encoding into bert features
            bert_fea = self.encoder(x0)[1]

            # (batch_size, nz)
            mean, logvar = self.encoder.linear(bert_fea).chunk(2, -1)

            if cnt == 0:
                means_sum = mean.sum(dim=0, keepdim=True)
            else:
                means_sum = means_sum + mean.sum(dim=0, keepdim=True)
            cnt += mean.size(0)

        # (1, nz)
        mean_mean = means_sum / cnt

        cnt = 0
        for batch_data in eval_dataloader:

            x0, _, _ = batch_data
            x0 = x0.to(args.device)

            # encoding into bert features
            bert_fea = self.encoder(x0)[1]

            # (batch_size, nz)
            mean, _ = self.encoder.linear(bert_fea).chunk(2, -1)

            if cnt == 0:
                var_sum = ((mean - mean_mean) ** 2).sum(dim=0)
            else:
                var_sum = var_sum + ((mean - mean_mean) ** 2).sum(dim=0)
            cnt += mean.size(0)

        # (nz)
        au_var = var_sum / (cnt - 1)

        return (au_var >= delta).sum().item(), au_var

