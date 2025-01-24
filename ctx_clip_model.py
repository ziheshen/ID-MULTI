"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.configuration_clip import CLIPTextConfig
from transformers.models.clip.modeling_clip import (
    CLIPEncoder,
    CLIPPreTrainedModel,
    CLIPTextModel,
    _expand_mask,
)

class Mapper(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            )
    def forward(self, x):
        x1 = self.net(x)
        return x1

# class FeatureFusionMLP(nn.Module):
#     def __init__(self, image_feature_dim=16 * 768, text_feature_dim=768, hidden_dim=1024, output_dim=768):
#         super(FeatureFusionMLP, self).__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(image_feature_dim + text_feature_dim, hidden_dim),  # 輸入拼接後的特徵
#             nn.GELU(),  # 平滑激活
#             nn.Dropout(0.1),  # 正則化
#             nn.Linear(hidden_dim, hidden_dim),  # 隱藏層繼續處理
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, output_dim)  # 最終映射至 1 * 768
#         )
    
#     def forward(self, image_features, text_features):
#         """
#         image_features: torch.Tensor, 形狀 [batch_size, 16, 768]
#         text_features: torch.Tensor, 形狀 [batch_size, 1, 768]
#         """
#         # 展平 Q-Former 特徵
#         image_features_flat = image_features.view(image_features.size(0), -1)  # [batch_size, 16 * 768]
        
#         # 展平文字特徵
#         text_features_flat = text_features.view(text_features.size(0), -1)  # [batch_size, 768]
        
#         # 拼接特徵
#         combined_features = torch.cat([image_features_flat, text_features_flat], dim=1)  # [batch_size, 16 * 768 + 768]
        
#         # 通過 MLP 進行特徵融合
#         output = self.mlp(combined_features)  # [batch_size, 768]
#         return output

class CtxCLIPTextModel(CLIPPreTrainedModel):
    config_class = CLIPTextConfig

    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, modifier_token, config: CLIPTextConfig):
        super().__init__(config)
        self.modifier_token = [i[1:-1] for i in modifier_token]
        self.text_model = CtxCLIPTextTransformer(self.modifier_token, config)
        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, modifier_token, *model_args, **kwargs):
        # 獲取 config
        config = kwargs.pop("config", None)
        if config is None:
            config = CLIPTextConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # 初始化模型
        model = cls(modifier_token=modifier_token, config=config)

        # 加載預訓練的模型權重
        state_dict = CLIPTextModel.from_pretrained(pretrained_model_name_or_path).state_dict()
        model.load_state_dict(state_dict, strict=False)

        return model
    
    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    def forward(
        self,
        modifier_token_id = None,
        images_embeds = None,
        subjects_position = None,
        ctx_embeddings: dict = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import CLIPTokenizer, CLIPTextModel

        >>> model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        return self.text_model(
            modifier_token_id,
            images_embeds,
            subjects_position = subjects_position,
            ctx_embeddings=ctx_embeddings,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class CtxCLIPTextTransformer(nn.Module):
    def __init__(self, modifier_token, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = CtxCLIPTextEmbeddings(modifier_token, config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        
        

    def forward(
        self,
        modifier_token_id,
        images_embeds,
        subjects_position,
        ctx_embeddings: dict,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is None:
            raise ValueError("You have to specify either input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            ctx_embeddings=ctx_embeddings,
            modifier_token_id=modifier_token_id,
            images_embeds=images_embeds,
            subjects_position = subjects_position,
        )

        # indices = torch.zeros(input_ids.shape, dtype=torch.bool, device="cuda")
        
        # for token_id, token in zip(modifier_token_id, self.modifier_token):
        #     print(indices, token_id, token, hidden_states.shape)
        #     indices |= input_ids == token_id
        #     idx = torch.where(token_id == input_ids)
        #     if min(idx[0].shape) != 0:
        #         t = []
        #         mapper = getattr(self, f'mapper{token}')
        #         for i in range(input_ids.shape[0]):
        #             t.append(mapper(images_embeds[i].view(1,768)).view(768))
        #         t = torch.stack(t).view(input_ids.shape[0], 768)
        #         hidden_states[idx] = t
        
        # indices = (indices*1).unsqueeze(-1)
        # hidden_states = (1-indices)*hidden_states.detach() + indices*hidden_states  # detach

        bsz, _ = input_shape
        # print(bsz)
        seq_len = hidden_states.shape[-2]
        # print("之前", seq_len)
        # if ctx_embeddings is not None:
        #     # print(seq_len)
        #     seq_len += list(ctx_embeddings.values())[0].size(1)
        # print("後來p1:", seq_len)

        # print("後來p2:", seq_len)
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(
            bsz, seq_len, hidden_states.dtype
        ).to(hidden_states.device)
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=input_ids.device),
            input_ids.to(torch.int).argmax(dim=-1),
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        ), hidden_states

    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask


class CtxCLIPTextEmbeddings(nn.Module):
    def __init__(self, modifier_token, config: CLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, embed_dim
        )

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        self.modifier_token = modifier_token
        for i in modifier_token:
            setattr(self, f"mapper_{i}", Mapper(dim=768))
        
        # self.mlp1 = MLP(embed_dim * 2, embed_dim, embed_dim, use_residual=False)
        # self.mlp2 = MLP(embed_dim, embed_dim, embed_dim, use_residual=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        ctx_embeddings: dict,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        modifier_token_id = None,
        images_embeds = None,
        subjects_position = None,
    ) -> torch.Tensor:
        
        # print(input_ids.shape)
        if inputs_embeds is None:
            
            inputs_embeds = self.token_embedding(input_ids)

            # print(inputs_embeds.shape)
            if modifier_token_id is not None and images_embeds is not None:
                indices = torch.zeros(input_ids.shape, dtype=torch.bool, device="cuda")
            
                for token_id, token, (subject,_) in zip(modifier_token_id, self.modifier_token, images_embeds.items()):
                    indices |= input_ids == token_id
                    idx = torch.where(token_id == input_ids)
                    if min(idx[0].shape) != 0:
                        mapper = getattr(self, f'mapper_{token}')
                        t = mapper(images_embeds[subject])
                        inputs_embeds[idx] = t
                
                indices = (indices*1).unsqueeze(-1)
                inputs_embeds = (1-indices)*inputs_embeds.detach() + indices*inputs_embeds  # detach
                # print('inputs_embeds:', inputs_embeds.shape)

            # for each input embeddings, add the ctx embeddings at the correct position
            input_embeds_ctx = []
            bsz = inputs_embeds.shape[0]
            if ctx_embeddings:
                for i in range(bsz):
                    sample_input_ids = input_ids[i]
                    sample_inputs_embeds = inputs_embeds[i]
                    sample_subjects_position = subjects_position[i]
                    for j, (subject, pos) in enumerate(sample_subjects_position):
                        # 获取对应的ctx_embedding
                        
                        ctx_embed = ctx_embeddings[subject][i]
                        prefix = sample_inputs_embeds[:(pos+ j*16)]
                        suffix = sample_inputs_embeds[(pos+ j*16):]
                        # print(subject, pos+ j*16, sample_inputs_embeds.shape, prefix.shape, ctx_embed.shape, suffix.shape)
                        # 插入ctx_embed
                        sample_inputs_embeds = torch.cat([prefix, ctx_embed, suffix], dim=0)
                    input_embeds_ctx.append(sample_inputs_embeds)
                    # print(input_embeds_ctx[-1].shape)
                inputs_embeds = torch.stack(input_embeds_ctx, dim=0)
                # print(inputs_embeds.shape)
            
            seq_length = inputs_embeds.shape[-2]
        else:
            seq_length = input_ids.shape[-1]
            print("else:",seq_length)
        # if not ctx_embeddings:
        #     ctx_len = 0
        # else:
        #     ctx_len = list(ctx_embeddings.values())[0].shape[1]

        # seq_length = (
        #     input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        # ) + ctx_len
        # seq_length = seq_length-1 if ctx_begin_pos[0]>77 and ctx_embeddings else seq_length
        # print(seq_length)

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        position_embeddings = self.position_embedding(position_ids)
        # print(seq_length, inputs_embeds.shape, position_embeddings.shape)
        embeddings = inputs_embeds + position_embeddings

        return embeddings
    
    @torch.no_grad()
    def embed( self, input_ids: Optional[torch.LongTensor] = None )-> torch.Tensor:
        input_ids = input_ids.to("cuda")
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        position_ids = self.position_ids[:, :seq_length]
        inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings
