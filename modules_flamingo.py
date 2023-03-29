### This file contains all the modules / components characterizing the Flamingo model

## Features:
# 1. Perceiver resampler: This module is used to compress the media features obtained from a frozen pre-trained vision model to a small set of latents. The architecture is just stacked blocks of perceiver xattn (no masks). The set of learnt latents form the query, while media features concatenated with the learnt latents form the keys and values.
# 2. Gated xattn: This is a typical xattn block but with two unique components: (a) tanh gating after the attn and ff blocks with learnt gating parameters and (b) media specific attn masks that unmask just the previous media features (as keys and values) for xattn with the following text features (as queries)
# 3. Flamingo Base Class: this class holds the core functionality of flamingo - takes in a pretrained LLM and modifies its layers. The modification involves freezing the self-attn layers of the pretrained LLM and pre-pending gated_xattn_dense layers. So this is a general class, agnostic to the LLM that is being 'flamingo-ed'. This class also holds the perceiver_resampler as a component.
# 4. Flamingo LLM specific derived class: This class holds all the LLM specific functionality required for flamingo 

## Todos / Questions:
# 1. Efficient way to form the media specific mask 
# 2. Modular way to hook the gated xattn blocks in between the self-attn blocks of the frozen pretrained LLM 
# 3. Do the perceiver xattn layers in the perceiver resampler share weights [Ans: no - the code snippet in figure 4 specifies each attention layer as a unique 'attention_i' layer]
# 4. In xattn, should all three inputs (q,k,v) be normalized, or just the query [Ans: all three. Even in the original transformer implementation, the encoder input to the decoder xattn is normalized]
# 5. Positional embeddings to be added to the inputs in perceiver resampler ? (Guess: yes, add positional embeddings to the latents, and add temporal encodings to the media features)
# 6. Handling temporal dimension for video inputs
# 7. Suggestive idea: the model can be more powerful if the gating parameters are learnt as a function of input (so the flamingo model can adjust its gating of cross-attn layers based on nature of input)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from time import time 
import cv2 
from typing import Optional, Tuple, Union

from utils_transformer import * 

# class implementing perceiver cross-attention layer (component of the perceiver resampler)
class Perceiver_xattn(nn.Module):
    def __init__(self, xattn, ff, dim, dropout):
        super().__init__()
        self.xattn = xattn
        self.ff = ff
        self.ln1_q = nn.LayerNorm(dim)
        self.ln1_kv = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x_f, x, mask_padding):
        x_fx = torch.cat([x_f,x], dim=-2)
        x_fx = self.ln1_kv(x_fx)
        x1 = self.ln1_q(x)
        x1 = self.xattn(x_fx, x1, x_fx, mask_padding=mask_padding) # xattn: (key=x_fx, query=x, value=x_fx)
        x = self.dropout(x1) + x 
        x1 = self.ln2(x)
        x1 = self.ff(x1)
        x = self.dropout(x1) + x 
        return x
    
# class implementing perceiver resampler (stacks perceiver_xattn layers)
class Perceiver_resampler(nn.Module):
    def __init__(self, layer, n_layers, d_model, n_latents, media_seqlen, media_dim):
        super().__init__()
        self.layers = clones(layer, n_layers)
        self.latents = nn.Parameter(torch.rand(n_latents, d_model))
        self.latents_pos_emb = nn.Parameter(torch.rand(n_latents, d_model)) # learnt positional embeddings for latents
        self.media_emb = nn.Linear(media_dim, d_model, bias=False)
        self.media_tem_emb = nn.Parameter(torch.rand(media_seqlen, d_model)) # learnt temporal embeddings for media
    def forward(self, x_f): # x_f.shape: [batch, media_seqlen, media_dim]
        batch_size = x_f.shape[0]
        x_f = self.media_emb(x_f) + self.media_tem_emb # x_f.shape: [batch, media_seqlen, d_model]
        x = self.latents + self.latents_pos_emb
        x = x.unsqueeze(0).expand(batch_size, -1, -1) # x.shape: [batch, n_latents, d_model]
        for layer in self.layers:
            x = layer(x_f, x, mask_padding=None)
        return x
    
# class implementing the gated xattn dense layer 
class Gated_xattn_dense(nn.Module):
    def __init__(self, xattn, ff, dim, dropout):
        super().__init__()
        self.xattn = xattn 
        self.ff = ff 
        self.alpha_xattn = nn.Parameter(torch.tensor(0.)) # init at 0
        self.alpha_ff = nn.Parameter(torch.tensor(0.)) # init at 0
        self.ln1_q = nn.LayerNorm(dim)
        self.ln1_kv = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.kv = None # placeholder for key/value (used for pre-populating / conditioning)
    def forward(self, y, mask_padding, mask_causal):
        x = self.kv 
        x = self.ln1_kv(x)
        y1 = self.ln1_q(y)
        y1 = self.xattn(x, y1, x, mask_padding=mask_padding, mask_causal=mask_causal) # xattn: (key=x_fx, query=x, value=x_fx)
        y = self.dropout(y1) * torch.tanh(self.alpha_xattn) + y 
        y1 = self.ln2(y)
        y1 = self.ff(y1)
        y = self.dropout(y1) * torch.tanh(self.alpha_ff) + y  
        # print('gate_attn:{} \t gate_ff:{}'.format(torch.tanh(self.alpha_xattn).item(), torch.tanh(self.alpha_ff).item()))
        return y 
    
# class implementing modified LM Block 
class Modified_LMBlock(nn.Module):
    def __init__(self, gxd_block, original_block):
        super().__init__()
        self.gxd_block = gxd_block
        self.original_block = original_block
    # function to pre-populate / condition the keys and values for xattn 
    def condition(self, x):
        self.gxd_block.kv = x 
    # note that the input arguments of forward call should match the input arguments of the forward call of the original block (gpt2 block for now)
    def forward(
        self,
        hidden_states,
        layer_past,
        attention_mask,
        head_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        use_cache,
        output_attentions,
    ):
        # pass through gxd layer 
        # TODO: no xattn mask needed for one_image prompt, but this will change when a prompt has multiple images
        hidden_states = self.gxd_block(hidden_states, mask_padding=None, mask_causal=None)
        # standard pass through original layer (gpt2 block)
        y = self.original_block(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        ) 
        return y 
        
    
# function to instantiate perceiver resampler
def init_perceiver_resampler(n_latents, media_seqlen, media_dim, d_model, n_layers, d_ff, dropout, device):
    attn_singleHead = MultiHeadAttention(1, d_model, d_model, d_model, dropout) # multi head attention block - setup as single head attn block
    ff = FeedForward(d_model, d_ff, dropout) # feed forward block for each encoder / decoder block
    perceiver_xattn = Perceiver_xattn(attn_singleHead, ff, d_model, dropout)
    perceiver_resampler = Perceiver_resampler(perceiver_xattn, n_layers, d_model, n_latents, media_seqlen, media_dim)
    # initialize params - Xavier initialization
    for p in perceiver_resampler.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return perceiver_resampler 

# function to instantiate a gated xattn dense layer 
def init_gated_xattn_dense_layer(n_heads, d_k, d_v, d_model, d_ff, dropout, device):
    attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout) # multi head attention block
    ff = FeedForward(d_model, d_ff, dropout) # feed forward block 
    gxd_layer = Gated_xattn_dense(attn, ff, d_model, dropout)
    return gxd_layer