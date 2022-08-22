import copy
from abc import ABC

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
# from sentence_transformers import SentenceTransformer
from transformers.models.bert.modeling_bert import BertModel, BertPooler, BertLayer
from transformers.models.deberta.modeling_deberta import DebertaModel, ContextPooler, DebertaLayer
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Model, CausalLMOutputWithCrossAttentions, \
    GPT2PreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaLayer
from transformers.models.t5.modeling_t5 import T5Model, T5Stack
from transformers.models.albert.modeling_albert import AlbertModel


class BertForLatentConnector(BertModel, ABC):
    def __init__(self, config, latent_size=64, pad_id=None):
        super().__init__(config)
        self.linear = nn.Linear(config.hidden_size, 2 * latent_size, bias=False)

        self.pad_id = pad_id
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            if_pool=True
    ):
        # no_grad = True
        avg_pool = False
        # if no_grad:
        #     with torch.no_grad():
        #         outputs = super().forward(input_ids=input_ids,
        #                                   attention_mask=attention_mask,
        #                                   token_type_ids=token_type_ids,
        #                                   position_ids=position_ids,
        #                                   head_mask=head_mask,
        #                                   inputs_embeds=inputs_embeds,
        #                                   encoder_hidden_states=encoder_hidden_states,
        #                                   encoder_attention_mask=encoder_attention_mask,
        #                                   past_key_values=past_key_values,
        #                                   use_cache=use_cache,
        #                                   output_attentions=output_attentions,
        #                                   output_hidden_states=output_hidden_states,
        #                                   return_dict=return_dict, )
        # else:
        outputs = super().forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  position_ids=position_ids,
                                  head_mask=head_mask,
                                  inputs_embeds=inputs_embeds,
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=encoder_attention_mask,
                                  past_key_values=past_key_values,
                                  use_cache=use_cache,
                                  output_attentions=output_attentions,
                                  output_hidden_states=output_hidden_states,
                                  return_dict=return_dict, )
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).float()
        if if_pool:
            if avg_pool:
                ave_pool = attention_mask / attention_mask.sum(-1, keepdim=True)
                pooled_out = torch.bmm(outputs[0].transpose(1, 2), ave_pool.unsqueeze(-1).half()).transpose(1, 2)
                pooled_out_final = self.pooler(pooled_out)
            else:
                pooled_out = outputs[0]
                pooled_out_final = outputs[1]

            return outputs[0], pooled_out_final, pooled_out
        return outputs[0], attention_mask,

class BertForLatentConnectorAVG(BertModel, ABC):
    def __init__(self, config, latent_size=64, pad_id=None):
        super().__init__(config)
        self.linear = nn.Linear(config.hidden_size, 2 * latent_size, bias=False)

        self.pad_id = pad_id
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            if_pool=True
    ):
        # no_grad = True1
        avg_pool = True
        outputs = super().forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  position_ids=position_ids,
                                  head_mask=head_mask,
                                  inputs_embeds=inputs_embeds,
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=encoder_attention_mask,
                                  past_key_values=past_key_values,
                                  use_cache=use_cache,
                                  output_attentions=output_attentions,
                                  output_hidden_states=output_hidden_states,
                                  return_dict=return_dict, )
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id)
        if if_pool:
            if avg_pool:
                ave_pool = (attention_mask / attention_mask.sum(-1, keepdim=True)).to(outputs[0].dtype)
                pooled_out = torch.bmm(outputs[0].transpose(1, 2), ave_pool.unsqueeze(-1)).transpose(1, 2)
                pooled_out_final = self.pooler(pooled_out)
            else:
                pooled_out = outputs[0]
                pooled_out_final = outputs[1]

            return outputs[0], pooled_out_final, pooled_out
        return outputs[0], attention_mask,


class BertForLatentConnectorNew(BertForLatentConnector, ABC):
    def __init__(self, config, latent_size=64, pad_id=None):
        super().__init__(config=config, latent_size=latent_size, pad_id=pad_id)
        self.linear_forbert = BertLayer(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            if_pool=False
    ):
        avg_pool = True
        with torch.no_grad():
            outputs = super().forward(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids,
                                      position_ids=position_ids,
                                      head_mask=head_mask,
                                      inputs_embeds=inputs_embeds,
                                      encoder_hidden_states=encoder_hidden_states,
                                      encoder_attention_mask=encoder_attention_mask,
                                      past_key_values=past_key_values,
                                      use_cache=use_cache,
                                      output_attentions=output_attentions,
                                      output_hidden_states=output_hidden_states,
                                      return_dict=return_dict,
                                      if_pool=if_pool)
        # outputs[0]: hidden, outputs[1]: attention_mask
        if attention_mask is None:
            attention_mask = outputs[1]
        input_shape = input_ids.size()
        device = input_ids.device
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
        layer_outputs = self.linear_forbert(hidden_states=outputs[0], attention_mask=extended_attention_mask)
        hidden_states = layer_outputs[0]
        if avg_pool:
            ave_pool = attention_mask / attention_mask.sum(-1, keepdim=True)
            pooled_out = torch.bmm(hidden_states.transpose(1, 2), ave_pool.unsqueeze(-1)).transpose(1, 2)
            pooled_out_final = self.pooler(pooled_out)
        else:
            pooled_out = layer_outputs[0]
            pooled_out_final = self.pooler(pooled_out)
        return hidden_states, pooled_out_final, pooled_out


class RobertaForLatentConnector(RobertaModel, ABC):
    def __init__(self, config, latent_size=64, pad_id=None):
        super().__init__(config)
        self.linear = nn.Linear(config.hidden_size, 2 * latent_size, bias=False)
        self.pad_id = pad_id
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            if_pool=True
    ):
        outputs = super().forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  position_ids=position_ids,
                                  head_mask=head_mask,
                                  inputs_embeds=inputs_embeds,
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=encoder_attention_mask,
                                  past_key_values=past_key_values,
                                  use_cache=use_cache,
                                  output_attentions=output_attentions,
                                  output_hidden_states=output_hidden_states,
                                  return_dict=return_dict, )
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).float()
        if if_pool:
            ave_pool = attention_mask / attention_mask.sum(-1, keepdim=True)
            pooled_out = torch.bmm(outputs[0].transpose(1, 2), ave_pool.unsqueeze(-1).half()).transpose(1, 2)
            pooled_out_final = self.pooler(pooled_out)
            return outputs[0], pooled_out_final, pooled_out
            # pooled_out_final = outputs[1]
            # pooled_out = outputs[0]
            # return outputs[0], pooled_out_final, pooled_out
        return outputs[0], attention_mask


class RobertaForLatentConnectorNew(RobertaForLatentConnector, ABC):
    def __init__(self, config, latent_size=64, pad_id=None, ):
        super().__init__(config=config, latent_size=latent_size, pad_id=pad_id)
        self.linear_forbert = RobertaLayer(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            if_pool=False
    ):
        with torch.no_grad():
            outputs = super().forward(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids,
                                      position_ids=position_ids,
                                      head_mask=head_mask,
                                      inputs_embeds=inputs_embeds,
                                      encoder_hidden_states=encoder_hidden_states,
                                      encoder_attention_mask=encoder_attention_mask,
                                      past_key_values=past_key_values,
                                      use_cache=use_cache,
                                      output_attentions=output_attentions,
                                      output_hidden_states=output_hidden_states,
                                      return_dict=return_dict,
                                      if_pool=if_pool)
        # outputs[0]: hidden, outputs[1]: attention_mask
        if attention_mask is None:
            attention_mask = outputs[1]
        input_shape = input_ids.size()
        device = input_ids.device
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
        layer_outputs = self.linear_forbert(hidden_states=outputs[0], attention_mask=extended_attention_mask)
        hidden_states = layer_outputs[0]
        ave_pool = attention_mask / attention_mask.sum(-1, keepdim=True)
        pooled_out = torch.bmm(hidden_states.transpose(1, 2), ave_pool.unsqueeze(-1)).transpose(1, 2)
        pooled_out_final = self.pooler(pooled_out)
        return hidden_states, pooled_out_final, pooled_out


class DebertaForLatentConnector(DebertaModel, ABC):
    def __init__(self, config, latent_size=64, pad_id=None):
        super().__init__(config)
        self.linear = nn.Linear(config.hidden_size, 2 * latent_size, bias=False)
        self.pooler = ContextPooler(config)
        self.pad_id = pad_id
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            if_pool=True
    ):
        outputs = super().forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  position_ids=position_ids,
                                  inputs_embeds=inputs_embeds,
                                  output_attentions=output_attentions,
                                  output_hidden_states=output_hidden_states,
                                  return_dict=return_dict, )
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).float()
        if if_pool:
            # ave_pool = attention_mask / attention_mask.sum(-1, keepdim=True)
            # pooled_out = torch.bmm(outputs[0].transpose(1, 2), ave_pool.unsqueeze(-1).half()).transpose(1, 2)
            # pooled_out_final = self.pooler(pooled_out)
            encoder_layer = outputs[0]
            pooled_out_final = self.pooler(encoder_layer)
            return outputs[0], pooled_out_final, None
        return outputs[0], attention_mask


class DebertaForLatentConnectorNew(DebertaForLatentConnector, ABC):
    def __init__(self, config, latent_size=64, pad_id=None):
        super().__init__(config=config, latent_size=latent_size, pad_id=pad_id)
        self.linear_forbert = DebertaLayer(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            if_pool=False
    ):
        with torch.no_grad():
            outputs = super().forward(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids,
                                      position_ids=position_ids,
                                      inputs_embeds=inputs_embeds,
                                      output_attentions=output_attentions,
                                      output_hidden_states=output_hidden_states,
                                      return_dict=return_dict,
                                      if_pool=if_pool
                                      )
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).float()

        extended_attention_mask = self.encoder.get_attention_mask(attention_mask)
        relative_pos = self.encoder.get_rel_pos(outputs[0], None, None)
        rel_embeddings = self.encoder.get_rel_embedding()

        layer_outputs = self.linear_forbert(outputs[0], extended_attention_mask, relative_pos=relative_pos,
                                            rel_embeddings=rel_embeddings)
        hidden_states = layer_outputs
        ave_pool = attention_mask / attention_mask.sum(-1, keepdim=True)
        pooled_out = torch.bmm(hidden_states.transpose(1, 2), ave_pool.unsqueeze(-1)).transpose(1, 2)
        pooled_out_final = self.pooler(pooled_out)
        return hidden_states, pooled_out_final, pooled_out


class T5EncoderForLatentConnector(T5Model, ABC):
    def __init__(self, config, latent_size=64, pad_id=None):
        super(T5Model, self).__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self.linear = nn.Linear(config.hidden_size, 2 * latent_size, bias=False)
        self.pad_id = pad_id
        self.pooler = BertPooler(config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = None  # T5Stack(decoder_config, self.shared)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
        if self.decoder:
            # Decode
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            return decoder_outputs + encoder_outputs
        else:
            if attention_mask is None:
                attention_mask = (input_ids != self.pad_id).float()
            ave_pool = attention_mask / attention_mask.sum(-1, keepdim=True)
            pooled_out = torch.bmm(encoder_outputs[0].transpose(1, 2), ave_pool.unsqueeze(-1)).transpose(1, 2)
            pooled_out_final = self.pooler(pooled_out)
            return encoder_outputs, pooled_out_final, pooled_out


# class SentenceForLatentConnector(nn.Sequential, ABC):
#     def __init__(self, model_name_or_path: Optional[str] = None,
#                  modules: Optional[Iterable[nn.Module]] = None,
#                  device: Optional[str] = None,
#                  cache_folder: Optional[str] = None,
#                  use_auth_token: Union[bool, str, None] = None
#                  ):
#         self.encoder = SentenceTransformer(model_name_or_path=model_name_or_path, modules=modules, device=device,
#                                            cache_folder=cache_folder, use_auth_token=use_auth_token)
#

class GPT2ModelForVAE(GPT2Model, ABC):
    def __init__(self, config, latent_size=64):
        super().__init__(config)
        self.latent_size = latent_size
        self.linear = nn.Linear(self.latent_size, config.hidden_size * config.n_layer,
                                bias=False)  # different latent vector for each layer
        self.linear_emb = nn.Linear(self.latent_size, config.hidden_size,
                                    bias=False)  # share the same latent vector as the embeddings
        self.config = config
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            latent_as_gpt_emb=True,
            latent_as_gpt_memory=True,
    ):

        # def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
        #             latent_as_gpt_emb=False, latent_as_gpt_memory=True):
        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.h)
        else:
            if latent_as_gpt_emb:
                past_emb = self.linear_emb(past_key_values)  # used as embeddings to add on other three embeddings
                inputs_embeds = self.wte(input_ids) + past_emb.unsqueeze(1)
            if latent_as_gpt_memory:
                past_key_values = self.linear(past_key_values)
                past_split = torch.split(past_key_values.unsqueeze(1), self.config.hidden_size, dim=2)
                past_split = [self.h[0].attn._split_heads(past, self.h[0].attn.num_heads, self.h[0].attn.head_dim)
                              for past in past_split]
                past_key_values = list(zip(past_split, past_split))
                past_length = 1  # past[0][0].size(-2)
            else:
                past_length = 0
                past_key_values = [None] * len(self.h)

        return super().forward(inputs_embeds=inputs_embeds, past_key_values=past_key_values,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)

# #
# class GPT2ForLatentConnector(GPT2LMHeadModel, ABC):
#     def __init__(self, config, latent_size=64, latent_as_gpt_emb=True, latent_as_gpt_memory=True):
#         super().__init__(config)
#         self.transformer = GPT2ModelForVAE(config, latent_size=latent_size)
#         self.latent_as_gpt_emb = latent_as_gpt_emb
#         self.latent_as_gpt_memory = latent_as_gpt_memory
#         self.init_weights()
#         self.tie_weights()
#
#     def tie_weights(self):
#         """ Make sure we are sharing the input and output embeddings.
#             Export to TorchScript can't handle parameter sharing so we are cloning them instead.
#         """
#         self._tie_or_clone_weights(self.lm_head,
#                                    self.transformer.wte)
#     def forward(
#             self,
#             input_ids=None,
#             past=None,
#             attention_mask=None,
#             token_type_ids=None,
#             position_ids=None,
#             head_mask=None,
#             inputs_embeds=None,
#             encoder_hidden_states=None,
#             encoder_attention_mask=None,
#             labels=None,
#             use_cache=None,
#             output_attentions=None,
#             output_hidden_states=None,
#             return_dict=None,
#             label_ignore=None,
#     ):
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         transformer_outputs = self.transformer(
#             input_ids,
#             past_key_values=past,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_attention_mask,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             latent_as_gpt_emb=self.latent_as_gpt_emb,
#             latent_as_gpt_memory=self.latent_as_gpt_memory
#         )
#         hidden_states = transformer_outputs[0]
#
#         # Set device for model parallelism
#         if self.model_parallel:
#             torch.cuda.set_device(self.transformer.first_device)
#             hidden_states = hidden_states.to(self.lm_head.weight.device)
#         # print(hidden_states.sum(),'\n')
#         # import pdb
#         # pdb.set_trace()
#         lm_logits = self.lm_head(hidden_states)
#
#         loss = None
#         if labels is not None:
#             # Shift so that tokens < n predict n
#             shift_logits = lm_logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             # Flatten the tokens
#             # loss_fct = CrossEntropyLoss()
#             loss_fct = CrossEntropyLoss(ignore_index=label_ignore, reduction='none')
#             loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#             loss = torch.sum(loss.view(-1, shift_labels.shape[-1]), -1)
#
#         if not return_dict:
#             output = (lm_logits,) + transformer_outputs[1:]
#             return ((loss,) + output) if loss is not None else output
#         return CausalLMOutputWithCrossAttentions(
#             loss=loss,
#             logits=lm_logits,
#             past_key_values=transformer_outputs.past_key_values,
#             hidden_states=transformer_outputs.hidden_states,
#             attentions=transformer_outputs.attentions,
#             cross_attentions=transformer_outputs.cross_attentions,
#         )

class GPT2ForLatentConnector(GPT2PreTrainedModel, ABC):
    def __init__(self, config, latent_size=64, latent_as_gpt_emb=True, latent_as_gpt_memory=True):
        super().__init__(config)
        self.transformer = GPT2ModelForVAE(config, latent_size=latent_size)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.init_weights()
        self.tie_weights()
        self.latent_as_gpt_emb = latent_as_gpt_emb
        self.latent_as_gpt_memory = latent_as_gpt_memory

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.transformer.wte)
    def forward(
            self,
            input_ids=None,
            past=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            label_ignore=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            latent_as_gpt_emb=self.latent_as_gpt_emb,
            latent_as_gpt_memory=self.latent_as_gpt_memory
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        # if self.model_parallel:
        #     torch.cuda.set_device(self.transformer.first_device)
        #     hidden_states = hidden_states.to(self.lm_head.weight.device)
        # print(hidden_states.sum(),'\n')
        # import pdb
        # pdb.set_trace()
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            # loss_fct = CrossEntropyLoss()
            loss_fct = CrossEntropyLoss(ignore_index=label_ignore, reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = torch.sum(loss.view(-1, shift_labels.shape[-1]), -1)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


class GPT2ModelForVAENew(GPT2ModelForVAE, ABC):
    def __init__(self, config, latent_size=64):
        super().__init__(config=config, latent_size=latent_size)
        self.linear = nn.Linear(self.latent_size, config.hidden_size * (config.n_layer + 1), bias=False)
        # config1 = copy.deepcopy(config)
        # config1.n_inner = config.hidden_size*12
        # net1=GPT2Block(config1)
        # def count_parameters(model):
        #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
        # import pdb
        # pdb.set_trace()
        self.h.append(GPT2Block(config))
        self.init_weights()
        self.tie_weights()

    def change_order(self, extra_num=1):
        self.h = nn.ModuleList([self.h[-1], *self.h[:-1]])
        self.config.n_layer += extra_num


class GPT2ForLatentConnectorNew(GPT2ForLatentConnector, ABC):
    def __init__(self, config, latent_size=64, latent_as_gpt_emb=True, latent_as_gpt_memory=True):
        super().__init__(config=config, latent_size=latent_size, latent_as_gpt_emb=latent_as_gpt_emb,
                         latent_as_gpt_memory=latent_as_gpt_memory)
        self.transformer = GPT2ModelForVAENew(config, latent_size=latent_size)
        self.latent_as_gpt_emb = latent_as_gpt_emb
        self.latent_as_gpt_memory = latent_as_gpt_memory

class GPT2ModelForVAENew2(GPT2ModelForVAE, ABC):
    def __init__(self, config, latent_size=64):
        super().__init__(config=config, latent_size=latent_size)
        self.linear = nn.Linear(self.latent_size, config.hidden_size * (config.n_layer + 1), bias=False)
        config1 = copy.deepcopy(config)
        config1.n_inner = config.hidden_size*12
        self.h.append(GPT2Block(config1))
        self.init_weights()
        self.tie_weights()

    def change_order(self, extra_num=1):
        self.h = nn.ModuleList([self.h[-1], *self.h[:-1]])
        self.config.n_layer += extra_num


class GPT2ForLatentConnectorNew2(GPT2ForLatentConnector, ABC):
    def __init__(self, config, latent_size=64, latent_as_gpt_emb=True, latent_as_gpt_memory=True):
        super().__init__(config=config, latent_size=latent_size, latent_as_gpt_emb=latent_as_gpt_emb,
                         latent_as_gpt_memory=latent_as_gpt_memory)
        self.transformer = GPT2ModelForVAENew2(config, latent_size=latent_size)
        self.latent_as_gpt_emb = latent_as_gpt_emb
        self.latent_as_gpt_memory = latent_as_gpt_memory

class AlbertForLatentConnector(AlbertModel, ABC):
    def __init__(self, config, latent_size=64, pad_id=None):
        super().__init__(config)
        self.linear = nn.Linear(config.hidden_size, 2 * latent_size, bias=False)
        self.pad_id = pad_id
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            if_pool=True
    ):
        avg_pool = True
        outputs = super().forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  position_ids=position_ids,
                                  head_mask=head_mask,
                                  inputs_embeds=inputs_embeds,
                                  output_attentions=output_attentions,
                                  output_hidden_states=output_hidden_states,
                                  return_dict=return_dict, )
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).float()
        if if_pool:
            if avg_pool:
                ave_pool = attention_mask / attention_mask.sum(-1, keepdim=True)
                pooled_out = torch.bmm(outputs[0].transpose(1, 2), ave_pool.unsqueeze(-1).half()).transpose(1, 2)

                pooled_out_final = self.pooler_activation(self.pooler(pooled_out[:,0]))
            else:
                pooled_out = outputs[0]
                pooled_out_final = outputs[1]

            return outputs[0], pooled_out_final, pooled_out
        return outputs[0], attention_mask,


# class Debertav2ForLatentConnector(DebertaV2Model, ABC):
#     def __init__(self, config, latent_size=64, pad_id=None):
#         super().__init__(config)
#         self.linear = nn.Linear(config.hidden_size, 2 * latent_size, bias=False)
#         self.pooler = ContextPooler(config)
#         self.pad_id = pad_id
#         self.init_weights()
#
#     def forward(
#             self,
#             input_ids=None,
#             attention_mask=None,
#             token_type_ids=None,
#             position_ids=None,
#             inputs_embeds=None,
#             output_attentions=None,
#             output_hidden_states=None,
#             return_dict=None,
#             if_pool=True
#     ):
#         outputs = super().forward(input_ids=input_ids,
#                                   attention_mask=attention_mask,
#                                   token_type_ids=token_type_ids,
#                                   position_ids=position_ids,
#                                   inputs_embeds=inputs_embeds,
#                                   output_attentions=output_attentions,
#                                   output_hidden_states=output_hidden_states,
#                                   return_dict=return_dict, )
#         if attention_mask is None:
#             attention_mask = (input_ids != self.pad_id).float()
#         if if_pool:
#             ave_pool = attention_mask / attention_mask.sum(-1, keepdim=True)
#             pooled_out = torch.bmm(outputs[0].transpose(1, 2), ave_pool.unsqueeze(-1).half()).transpose(1, 2)
#             pooled_out_final = self.pooler(pooled_out)
#             # encoder_layer = outputs[0]
#             # pooled_out_final = self.pooler(encoder_layer)
#             return outputs[0], pooled_out_final, None
#         return outputs[0], attention_mask
