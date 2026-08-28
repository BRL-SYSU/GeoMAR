import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pytorch_lightning as pl
import torch.optim as optim
from torch.distributions import Gumbel
from GeoMAR.modules.vqvae.vqvae_arch import MultiHeadAttnBlock, MultiHeadEncoder, MultiHeadDecoder, VectorQuantizer

class maskedVQVAEGAN(nn.Module):
    def __init__(self, n_embed=1024, embed_dim=256, ch=128, out_ch=3, ch_mult=(1,2,4,8), 
                 num_res_blocks=2, attn_resolutions=16, dropout=0.0, in_channels=3, 
                 resolution=512, z_channels=256, double_z=False, enable_mid=True, 
                 fix_decoder=False, fix_codebook=False, fix_encoder=False, head_size=1, **ignore_kwargs):
        super(maskedVQVAEGAN, self).__init__()

        #LQ
        self.encoder = MultiHeadEncoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, double_z=double_z, 
                               enable_mid=enable_mid, head_size=head_size)
        #HQ
        self.HQ_encoder = MultiHeadEncoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, double_z=double_z, 
                               enable_mid=enable_mid, head_size=head_size) 
               
        #HQ
        self.decoder = MultiHeadDecoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, enable_mid=enable_mid, head_size=head_size)

        # HQ_codebook
        self.HQ_quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)

        #LQ
        self.quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1) 

        #HQ
        self.HQ_quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)

        #HQ
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1) # 1x1 conv for conv feature map to z_channels

        if fix_decoder:
            for _, param in self.decoder.named_parameters():
                param.requires_grad = False
            for _, param in self.post_quant_conv.named_parameters():
                param.requires_grad = False
            for _, param in self.HQ_quantize.named_parameters():
                param.requires_grad = False
        elif fix_codebook:
            for _, param in self.HQ_quantize.named_parameters():
                param.requires_grad = False

        if not fix_encoder:
            for _, param in self.encoder.named_parameters():
                param.requires_grad = True
            for _, param in self.HQ_encoder.named_parameters():
                param.requires_grad = False    
            for _, param in self.quant_conv.named_parameters():
                param.requires_grad = True
            for _, param in self.HQ_quant_conv.named_parameters():
                param.requires_grad = False 

    # HQ encoder 
    def HQ_encode(self, x):

        hs = self.HQ_encoder(x)
        h = self.HQ_quant_conv(hs['out'])
        quant, emb_loss, info, dictionary = self.HQ_quantize(h)
        # return quant, emb_loss, info, hs
        return quant, emb_loss, info, hs, h, dictionary

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)

        return dec

    def forward(self, input):
        # quant, diff, info, hs = self.encode(input)
        quant, diff, info, hs, h, dictionary = self.HQ_encode(input)
        dec = self.decode(quant)

        # return dec, diff, info, hs
        # info = (perplexity, min_encodings, min_encoding_indices, d)
        return dec, diff, info, hs, h, quant, dictionary            
