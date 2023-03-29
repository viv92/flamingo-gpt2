### This program impelements the entire flamingo model pipeline, but supports only gpt2 backbone and one image per prompt.

## Features:
# 1. CLIP based pre-trained resnet is used as the frozen vision encoder.
# 2. Output from the frozen resnet is fed to perceiver resampler to obtain a compressed latent vector for media inputs
# 3. The media latent vector and the corresponding prompt text are fed to the pretrained frozen LLM to autoregressively predict the prompt at output
# 4. The frozen LLM is interspersed with trainable gated cross-attn layers that perform cross-attn between the media latents and the text tokens, with media specific masks and tanh gating controlled by learnable parameters

## Todos / Questions:
# 1. Are the media latents to be averaged, as done in original perceiver paper
# 2. Modular way to hook the gated xattn blocks in between the self-attn blocks of the frozen pretrained LLM 
# 3. Efficient way to form the media specific mask - not a problem for this implementation as it assumes one image per prompt 
# 4. Handling temporal dimension for video inputs
# 5. For the frozen vision encoder, is it enough to keep the model in eval mode, or we need to set requires_grad = False on all its parameters
# 6. Is it correct to normalize dataset images: torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # zero mean, unit std

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
import json 
from copy import deepcopy
from time import time 
import cv2 

# import GPT2 
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from utils_transformer import * 
from CLIP_resnet import ImageEmbeddings, CaptionEmbeddings, ClipTextEncoder, CLIP, init_text_encoder, init_CLIP, forward_hook
from modules_flamingo import Perceiver_xattn, Perceiver_resampler, Gated_xattn_dense, Modified_LMBlock, init_perceiver_resampler, init_gated_xattn_dense_layer


# utility function to load img and captions data
def load_data(img_tag):
    captions_file_path = 'dataset_coco_val2017/annotations/captions_val2017.json'
    captions_file = open(captions_file_path)
    captions = json.load(captions_file)
    img_dict, img_cap_pairs = {}, []
    max_caption_len = 0

    print('Loading Data...')
    num_iters = len(captions['images']) + len(captions['annotations'])
    pbar = tqdm(total=num_iters)

    for img in captions['images']:
        id, file_name = img['id'], img['file_name']
        img_dict[id] = file_name
        pbar.update(1)
    for cap in captions['annotations']:
        id, caption = cap['image_id'], cap['caption']
        # update max_caption_len
        caption_len = len(caption.split(' '))
        if caption_len > max_caption_len:
            max_caption_len = caption_len
        # # process caption - no need to do this as its handled by gpt2 tokenizer
        # caption = unidecode.unidecode(caption) # strip accents
        # caption = caption.lower()
        # use img_name as key for img_cap_dict
        img_filename = img_dict[id]
        # # pre-pend img_tag to caption
        # caption = img_tag + caption 
        img_cap_pairs.append([img_filename, caption])
        pbar.update(1)
    pbar.close()
    max_caption_len += 3 # for <s>, </s> and a precautionary <pad>
    return img_cap_pairs, max_caption_len


# utility function to process minibatch - convert img_filenames to augmented_imgs and convert caption_text to tokenized_captions 
def process_batch(minibatch, tokenizer, img_size, device):
    augmented_imgs = []
    img_files, captions = list(map(list, zip(*minibatch)))
    # tokenize captions 
    caption_tokens_dict = tokenizer(captions, return_tensors='pt', padding=True, truncation=True)
    # get augmented imgs
    imgs_folder = 'dataset_coco_val2017/images/'
    for img_filename in img_files:
        img_path = imgs_folder + img_filename
        img = cv2.imread(img_path, 1)
        resize_shape = (img_size, img_size)
        img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_LINEAR)
        img = np.float32(img) / 255
        img = torch.tensor(img)
        img = img.permute(2, 1, 0) # [w,h,c] -> [c,h,w]
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize( int(1.25*img_size) ),  # image_size + 1/4 * image_size
            torchvision.transforms.RandomResizedCrop(resize_shape, scale=(0.8, 1.0)),
            # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # imagenet stats for normalization
            # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # zero mean, unit std
        ])
        img = transforms(img)
        augmented_imgs.append(img)
    augmented_imgs = torch.stack(augmented_imgs, dim=0).to(device)
    return augmented_imgs, caption_tokens_dict.to(device)


# utility function to load model weights from checkpoint - loads to the device passed as 'device' argument
def load_ckpt(checkpoint_path, model, optimizer=None, scheduler=None, device=torch.device('cpu'), mode='eval'):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if mode == 'eval':
        model.eval() 
        return model
    else:
        model.train()
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            return model, optimizer, scheduler
        else:
            return model, optimizer


# utility function to save a checkpoint (model_state, optimizer_state, scheduler_state) - saves on cpu (to save gpu memory)
def save_ckpt(device, checkpoint_path, model, optimizer, scheduler=None):
    # transfer model to cpu
    model = model.to('cpu')
    # prepare dicts for saving
    save_dict = {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}
    if scheduler is not None:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(save_dict, checkpoint_path)
    # load model back on original device 
    model = model.to(device)


# utility function to freeze model 
def freeze(model):
    for p in model.parameters():
        p.requires_grad = False 
    return model 


# utility function to append trainable gated xattn dense blockss in the frozen gpt2 model
def append_gxd_blocks(model, gxd_block):
    for i, original_block in enumerate(model.transformer.h):
        new_gxd_block = deepcopy(gxd_block)
        model.transformer.h[i] = Modified_LMBlock(new_gxd_block, original_block)
    return model 


### main 
if __name__ == '__main__':

    # hyperparams for GPT2
    d_model = 768 # d_model for GPT2
    max_seq_len_gpt2 = 1024 # required to init GPT2 Tokenizer

    # hyperparams for gated xattn dense (gxd) layer 
    n_heads_gxd = 2 
    d_k_gxd = d_model // n_heads_gxd 
    d_v_gxd = d_k_gxd
    d_ff_gxd = 2048

    # hyperparams for CLIP (as required by the clip checkpoint)
    img_size = 224 # resize for resnet
    d_model_clip = 512
    d_k_clip = 64
    d_v_clip = 64
    n_heads_clip = 8
    n_layers_clip = 6
    d_ff_clip = 2048

    # hyperparams for perceiver resampler 
    n_latents = 16 # 64
    n_layers_perceiver = 6 # 12
    d_ff_perceiver = 2048
    media_seqlen = d_model_clip # since our clip image encoder flattens the output internally
    media_dim = 1 # since our clip image encoder flattens the output internally

    dropout = 0.1
    lr = 3e-6
    batch_size = 16 # 8
    num_epochs = 2500
    epochs_done = 0
    random_seed = 10

    gpt2_model_name = 'gpt2'

    ckpt_path_clip = 'ckpts/clip_resnet.pt'

    resume_training_from_ckpt = False 

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # result containers for plotting 
    result_losses = []
    result_accuracy = []

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # clip_device = torch.device('cpu') # clip model will be loaded for inference on the cpu (to save memory on the gpu for diffusion prior model)
    clip_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # dummy variables for clip text encoder (only for initialization, we don't use clip text encoder)
    vocab_size_clip = 10000 # as required by the clip checkpoint
    eos_token = -1

    # load voc data and create img_cap_dict
    img_tag = '<image>'
    dataset, max_seq_len_clip = load_data(img_tag) # loads data and pre-pends img_tag to each caption
    dataset_len = len(dataset)

    # init clip model
    clip_image_encoder = ImageEmbeddings(forward_hook, d_model_clip).to(clip_device) # resnet
    clip_text_encoder = init_text_encoder(vocab_size_clip, max_seq_len_clip, d_model_clip, d_k_clip, d_v_clip, n_heads_clip, n_layers_clip, d_ff_clip, dropout, clip_device).to(clip_device) # not used - just for initialization and loading the pretrained clip model
    clip_model = init_CLIP(clip_text_encoder, clip_image_encoder, d_model_clip, eos_token).to(clip_device)
    # load clip_model weights and put clip_model in eval mode
    clip_model = load_ckpt(ckpt_path_clip, clip_model, device=clip_device, mode='eval')
    # freeze clip model 
    clip_model = freeze(clip_model)

    # init perceiver resampler 
    perceiver_resampler = init_perceiver_resampler(n_latents, media_seqlen, media_dim, d_model, n_layers_perceiver, d_ff_perceiver, dropout, device).to(device)

    # init GPT2 tokenizer 
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name, model_max_length=max_seq_len_gpt2)
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token # necessary to init the pad_token for padding
    # gpt2_tokenizer.add_special_tokens({'img_token': img_tag}) # add token for <image> tag to the tokenizer vocab 

    # init GPT2 model
    gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name).to(device)
    # freeze gpt2 model - note that we don't put gpt2 in eval_model because we still need to train the added xattn layers (gxd layers)
    gpt2_model = freeze(gpt2_model)

    # instantiate gated xattn dense layer
    gxd_layer = init_gated_xattn_dense_layer(n_heads_gxd, d_k_gxd, d_v_gxd, d_model, d_ff_gxd, dropout, device).to(device)
    # append gxd layers in frozen gpt2 
    gpt2_model = append_gxd_blocks(gpt2_model, gxd_layer)

    # optimizer and params to be optimized
    perceiver_resampler_params = perceiver_resampler.parameters()
    gpt2_model_params = filter(lambda x: x.requires_grad, gpt2_model.parameters())
    params = list(perceiver_resampler_params) + list(gpt2_model_params)
    optimizer = torch.optim.AdamW(params=params, lr=lr)

    # loss function 
    loss_fn = nn.CrossEntropyLoss(reduction='mean')


    # train loop
    for ep in tqdm(range(num_epochs)):
        ep += epochs_done

        # fetch minibatch
        idx = np.arange(dataset_len)
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        minibatch = [dataset[i] for i in idx]

        # process minibatch - convert img_filenames to augmented_imgs and convert caption_text to tokenized_captions
        imgs, cap_tokens_dict = process_batch(minibatch, gpt2_tokenizer, img_size, device) # imgs.shape:[b, 3, 224, 224], captions.shape:[b, batch_seq_len]
        cap_tokens, cap_attn_mask = cap_tokens_dict.input_ids, cap_tokens_dict.attention_mask

        # append begginning of sentence <bos> token to cap_tokens
        bos_id = gpt2_tokenizer(gpt2_tokenizer.bos_token, return_tensors='pt', padding=False, truncation=True).input_ids
        bos_id = bos_id.expand(batch_size, -1).to(device) # start_token_id.shape: [b, 1]
        cap_tokens_rshifted = torch.cat((bos_id, cap_tokens), dim=-1) # cap_tokens_rshifted.shape: [b, seq_len+1]
        cap_tokens_rshifted = cap_tokens_rshifted[:, :-1] # cap_tokens_rshifted.shape: [b, seq_len]

        # pass image through frozen clip image encoder 
        clip_img_emb = clip_model.image_encoder(imgs) # clip_img_emb.shape: [b, d_model]

        # pass img_emb through perceiver resampler 
        # for perceiver, the img_emb is treated as a byte array with seqlen = d_model and num_channels = 1
        clip_img_emb = clip_img_emb.unsqueeze(-1) # clip_img_emb.shape: [b, d_model, 1]
        img_latent = perceiver_resampler(clip_img_emb) # img_latent.shape: [b, n_latents, d_model]
        # average out latents to have one latent per image 
        img_latent = img_latent.mean(dim=-2) # img_latent.shape: [b, d_model]
        img_latent = img_latent.unsqueeze(-2) # img_latent.shape[b, 1, d_model]

        # condition xattn layers with img_latents (populate keys and values of the gxd_blocks with img_latents)
        for i in range(len(gpt2_model.transformer.h)):
            gpt2_model.transformer.h[i].condition(img_latent)
        
        # forward prop the caption (query) through the 'flamingo-ed' LLM
        # TODO: gpt2_model creates a causal mask internally if none passed, but check how does it create a padding_mask
        out = gpt2_model(input_ids=cap_tokens_rshifted)

        # loss
        targets = cap_tokens
        scores = out.logits
        scores = scores.permute(0, 2, 1) # scores.shape: [b, vocab_size, batch_seq_len] - required for crossEntropyLoss
        loss = loss_fn(scores, targets)

        # optimization step 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calculate batch accuracy 
        pred_cap_tokens = torch.argmax(scores, dim=1) # shape: [batch_size, seq_len]
        batch_accuracy = (pred_cap_tokens == cap_tokens).float().mean() 

        # sample prediction for display
        sample_gt = gpt2_tokenizer.decode(cap_tokens[0], skip_special_tokens=True)
        sample_pred = gpt2_tokenizer.decode(pred_cap_tokens[0], skip_special_tokens=True)

        # store loss for plotting 
        result_losses.append(loss.item())
        result_accuracy.append(batch_accuracy.item())

        if (ep+1) % (num_epochs // 20) == 0:
                # print metrics
                print('ep:{} \t loss:{:.3f} \t batch_accuracy:{:.3f}'.format(ep, loss.item(), batch_accuracy.item()))
                print('gt:\t ', sample_gt)
                print('pred:\t ', sample_pred)

                # # save checkpoint
                # save_ckpt(device, checkpoint_path, net, optimizer)

    # hyperparam dict 
    hyperparam_dict = {}
    hyperparam_dict['n_heads_gxd'] = n_heads_gxd
    hyperparam_dict['n_latents'] = n_latents
    hyperparam_dict['n_layers_perceiver'] = n_layers_perceiver
    hyperparam_dict['lr'] = lr  
    hyperparam_dict['batchsz'] = batch_size 
    hyperparam_dict['n_epochs'] = num_epochs
    hyperparam_dict['seed'] = random_seed 
    
    hyperstr = ''
    for k,v in hyperparam_dict.items():
        hyperstr += '|' + k + ':' + str(v) 

    # plot 
    plt.plot(result_losses, label='loss')
    plt.plot(result_accuracy, label='accuracy')
    plt.legend()
    plt.xlabel('train iters')
    plt.ylabel('value')
    plt.title('flamingo one image - training curves')
    plt.savefig('plots/flamingo_oneimage' + hyperstr +  '.png')


