import itertools
from packaging import version
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import transformers
from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPProcessor
from transformers.models.clip.modeling_clip import _make_causal_mask
from ctx_clip_model import CtxCLIPTextModel

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class CLIPEmbedderWrapper(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, modifier_token: list, num_imgs: int, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.transformer = CtxCLIPTextModel.from_pretrained(version,modifier_token)
        self.image_encoder = CLIPModel.from_pretrained(version)
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        

        self.num_imgs = num_imgs

        self.device = device
        self.max_length = max_length
        self.modifier_token = modifier_token

        # self.add_token()
        # self.freeze()

    def add_token(self):
        self.modifier_token_id = []
        token_embeds1 = self.transformer.get_input_embeddings().weight.data
        for each_modifier_token in self.modifier_token:
            num_added_tokens = self.tokenizer.add_tokens(each_modifier_token)
            modifier_token_id = self.tokenizer.convert_tokens_to_ids(each_modifier_token)
            self.modifier_token_id.append(modifier_token_id)

        self.transformer.resize_token_embeddings(len(self.tokenizer))

    # def custom_forward(self, hidden_states, input_ids):
    #     r"""
    #     Returns:
    #     """
    #     input_shape = hidden_states.size()
    #     bsz, seq_len = input_shape[:2]
    #     if version.parse(transformers.__version__) >= version.parse('4.21'):
    #         causal_attention_mask = self.transformer.text_model._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
    #             hidden_states.device
    #         )
    #     else:
    #         causal_attention_mask = self.transformer.text_model._build_causal_attention_mask(bsz, seq_len).to(
    #             hidden_states.device
    #         )

    #     encoder_outputs = self.transformer.text_model.encoder(
    #         inputs_embeds=hidden_states,
    #         causal_attention_mask=causal_attention_mask,
    #     )

    #     last_hidden_state = encoder_outputs[0]
    #     last_hidden_state = self.transformer.text_model.final_layer_norm(last_hidden_state)

    #     return last_hidden_state

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.transformer.text_model.encoder.parameters():
            param.requires_grad = False
        for param in self.transformer.text_model.final_layer_norm.parameters():
            param.requires_grad = False
        for param in self.transformer.text_model.embeddings.position_embedding.parameters():
            param.requires_grad = False

    def forward(
        self,
        text,
        subjects_position,
        input_img=None,
        ctx_embeddings: dict = None,
        ):
        # print(len(self.tokenizer))
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="do_not_pad", return_tensors="pt")
        tokens_ids = batch_encoding["input_ids"].to(self.device)
        # print(self.device)
        bsz = tokens_ids.shape[0]
 
        # input_shape = tokens.size()
        # hidden_states = self.transformer.text_model.embeddings(input_ids=tokens.view(-1, input_shape[-1]))
        
        # get CLIP img embedding
        
        images_embeds = {}
        if input_img is not None:
            for subject, image_embeds in input_img[0].items():
                images_embeds[subject] = self.image_encoder.get_image_features(pixel_values=image_embeds.unsqueeze(0).expand(bsz, -1, -1, -1).to(self.device))
                images_embeds[subject] = images_embeds[subject] / images_embeds[subject].norm(p=2, dim=-1, keepdim=True)
            # image_embeds = []
            # for index in range(bs):
            #     img = input_img[index]
            #     print(tokens[0].shape, img.shape)
            #     image_embeds.append(self.image_encoder(input_ids = tokens[0], pixel_values=img.to(self.device)).image_embeds)
            # image_embeds = torch.stack(image_embeds).to(self.device)

        # indices = torch.tensor([[False]*tokens.shape[1]]*bs).to("cuda")
        # for token_id, token in zip(self.modifier_token_id, self.modifier_token):
        #     indices |= tokens == token_id
        #     if input_img is not None:
        #         idx = torch.where(token_id == tokens)
        #         if min(idx[0].shape) != 0:
        #             t = []
        #             mapper = getattr(self, f'mapper{token}')
        #             for i in range(bs):
        #                 t.append(mapper(image_embeds[i].view(1,768)).view(768))
        #             t = torch.stack(t).view(bs, 768)
        #             hidden_states[idx] = t

        # indices = (indices*1).unsqueeze(-1)

        # hidden_states = (1-indices)*hidden_states.detach() + indices*hidden_states  # detach
        

        encoder_hidden_states, hidden_states = self.transformer(
            modifier_token_id = self.modifier_token_id,
            images_embeds = images_embeds,
            ctx_embeddings = ctx_embeddings,
            input_ids = tokens_ids,
            subjects_position = subjects_position,
        )

        return encoder_hidden_states, hidden_states

    def encode(self, text, input_img):
        return self(text, input_img)
    
    def get_index(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        index = []
        for token_id in self.modifier_token_id:
            if token_id in tokens:
                idx = torch.where(token_id == tokens)
                index.append(int(idx[1]))
        return index
        
    
    def return_parameters(self):
        token_embeds = self.transformer.get_input_embeddings().weight.data
        param = list(itertools.chain(token_embeds[self.modifier_token_id[0]]))
        for i in range(self.num_imgs):
            param += itertools.chain(getattr(self, f'mapperp{i+1}').parameters())
        for i in range(self.num_imgs):
            param += itertools.chain(getattr(self, f'mapperb{i+1}').parameters())

        return param

