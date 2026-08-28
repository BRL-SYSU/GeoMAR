import os, math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, List
import pytorch_lightning as pl
from main_GeoMAR import instantiate_from_config
from alignment_hq import alignmodel
from ..modules.vqvae.utils import get_roi_regions
import pyiqa
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.optim.lr_scheduler import LambdaLR

class MaskedEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, mask_token_id=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.mask_token_id = mask_token_id or vocab_size
        
        # Extend the vocabulary to include the mask token.
        extended_vocab_size = vocab_size + 1
        self.word_embeddings = nn.Embedding(extended_vocab_size, embedding_dim)
        
        # Initialize the mask-token embedding with a truncated normal distribution.
        with torch.no_grad():
            self.word_embeddings.weight[self.mask_token_id].normal_(mean=0.0, std=0.02)
    
    def forward(self, input_ids):
        # Ensure that indices are within the valid range.
        valid_ids = torch.clamp(input_ids, 0, self.mask_token_id)
        return self.word_embeddings(valid_ids)
    

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
    
class TransformerSALayer(nn.Module): 
    # Transformer encoder layer with self-attention and a feed-forward MLP.
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout) # Multi-head attention.
        # Implementation of Feedforward model - MLP
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)
        # LayerNormapplied to the sums of the self attention and the input embedding
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # Activation function.
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        
        # self attention
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos) # Add positional encodings to the query and key.
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, 
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2) 

        # ffn 
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2) # Residual connection.
        return tgt


class GeoMARModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 ckpt_path_HQ=None, # HQ checkpoint path
                 ckpt_path_LQ=None, # LQ checkpoint path
                 ignore_keys=[],
                 image_key="lq",
                 colorize_nlabels=None,
                 monitor=None,
                 special_params_lr_scale=1.0,
                 comp_params_lr_scale=1.0,
                 schedule_step=[80000, 200000],

                 mask_token_id=1024,  # Mask-token ID.
                 timesteps=8,        # Number of iterations.
                 mask_scheduling_method='cosine',  # Mask scheduling method.
                 use_alignment=True,               # Whether to use alignment.
                 is_training=True,    
                 train_text_features_dir=None,
                 eval_text_features_dir=None,              
                 ):
        super().__init__()

        # import pdb
        # pdb.set_trace()
        self.mask_token_id = mask_token_id
        self.timesteps = timesteps
        self.mask_scheduling_method = mask_scheduling_method
        self.vocab_size = 1024  # Vocabulary size; identical to codebook_size.
        print("timesteps = ", self.timesteps)    

        self.image_key = image_key
        self.vqvae = instantiate_from_config(ddconfig)

        lossconfig['params']['distill_param'] = ddconfig['params']
        # get the weights from HQ and LQ checkpoints
        if (ckpt_path_HQ is not None) and (ckpt_path_LQ is not None):
            print('loading HQ and LQ checkpoints')
            self.init_from_ckpt_two(
                ckpt_path_HQ, ckpt_path_LQ, ignore_keys=ignore_keys)

        if ('comp_weight' in lossconfig['params'] and lossconfig['params']['comp_weight']) or ('comp_style_weight' in lossconfig['params'] and lossconfig['params']['comp_style_weight']):
            self.use_facial_disc = True
        else:
            self.use_facial_disc = False

        self.fix_decoder = ddconfig['params']['fix_decoder']

        self.disc_start = lossconfig['params']['disc_start']
        self.special_params_lr_scale = special_params_lr_scale
        self.comp_params_lr_scale = comp_params_lr_scale
        self.schedule_step = schedule_step

        # codeformer code-----------------------------------
        dim_embd=512
        n_head=8
        n_layers=9 # Number of Transformer layers.
        codebook_size=1024
        latent_size=256
        concat_size=512  # Sequence length after concatenation.

        connect_list=['32', '64', '128', '256']
        fix_modules=['quantize','generator']
        self.connect_list = connect_list
        self.n_layers = n_layers
        self.dim_embd = dim_embd
        self.dim_mlp = dim_embd*2
        self.feat_emb = nn.Linear(256, self.dim_embd)
        self.position_emb = nn.Parameter(torch.zeros(512, self.dim_embd))
       
        # Use a dedicated masked embedding layer instead of a standard embedding layer.
        self.token_emb = MaskedEmbedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.dim_embd,
            mask_token_id=self.mask_token_id
        )
        # Build ft_layers from n_layers TransformerSALayer instances.
        self.ft_layers = nn.Sequential(*[TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0) 
                                    for _ in range(self.n_layers)])

        # Linear logits-prediction head for index prediction.
        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim_embd),
            nn.Linear(dim_embd, codebook_size, bias=False))
        self.use_alignment = use_alignment
        
        # Initialize the alignment model.
        self.alignmodel = alignmodel(
                dim=256,
                train_text_features_dir=train_text_features_dir,
                eval_text_features_dir=eval_text_features_dir,
                is_training=is_training,
            )

         # ==================== Initialize all validation metrics ====================
        self.fid_metric = FrechetInceptionDistance(feature=2048, normalize=True)
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.niqe_metric = pyiqa.create_metric('niqe', device=self.device)

       
    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def init_from_ckpt_two(self, path_HQ, path_LQ, ignore_keys=list()):
        """Load component weights from HQ and LQ checkpoints by direct copying."""
        print('='*50)
        print('Loading HQ and LQ checkpoints...')
        
        try:
            # Load the HQ checkpoint.
            print(f"Loading HQ checkpoint: {path_HQ}")
            sd_HQ = torch.load(path_HQ, map_location="cpu")
            if "state_dict" in sd_HQ:
                sd_HQ = sd_HQ["state_dict"]
            
            # Load the LQ checkpoint.
            print(f"Loading LQ checkpoint: {path_LQ}")
            sd_LQ = torch.load(path_LQ, map_location="cpu")
            if "state_dict" in sd_LQ:
                sd_LQ = sd_LQ["state_dict"]
            
            # Print the original key counts.
            print(f"Number of keys in HQ checkpoint: {len(sd_HQ)}")
            print(f"Number of keys in LQ checkpoint: {len(sd_LQ)}")
            
            # ----- Part 1: Define the component mapping -----
            # Tuple format: (checkpoint prefix, model component, checkpoint object).
            component_mapping = [
                # HQ components loaded from the HQ checkpoint.
                ('vqvae.quantize', self.vqvae.HQ_quantize, sd_HQ),
                ('vqvae.encoder', self.vqvae.HQ_encoder, sd_HQ),
                ('vqvae.quant_conv', self.vqvae.HQ_quant_conv, sd_HQ),
                ('vqvae.decoder', self.vqvae.decoder, sd_HQ),
                ('vqvae.post_quant_conv', self.vqvae.post_quant_conv, sd_HQ),
                
                # LQ components loaded from the LQ checkpoint.
                ('vqvae.encoder', self.vqvae.encoder, sd_LQ),
                ('vqvae.quant_conv', self.vqvae.quant_conv, sd_LQ),
            ]
            
            # ----- Part 2: Load key component weights -----
            # Load the embedding directly.
            hq_embed_key = 'vqvae.quantize.embedding.weight'
            if hq_embed_key in sd_HQ:
                # print(f"\nLoading HQ codebook embedding weights directly:")
                # print(f"  Checkpoint embedding: min={sd_HQ[hq_embed_key].min().item():.4f}, max={sd_HQ[hq_embed_key].max().item():.4f}")
                self.vqvae.HQ_quantize.embedding.weight.data.copy_(sd_HQ[hq_embed_key])
                # print(f"  Loaded embedding: min={self.vqvae.HQ_quantize.embedding.weight.min().item():.4f}, max={self.vqvae.HQ_quantize.embedding.weight.max().item():.4f}")
            else:
                print(f"Warning: HQ embedding weight key '{hq_embed_key}' was not found")
            
            # ----- Part 3: Load the remaining weights component by component -----
            loaded_params = 0
            
            for prefix, component, checkpoint in component_mapping:
                component_loaded = 0
                
                # Iterate over all named parameters in each component.
                for name, param in component.named_parameters():
                    # Build the corresponding checkpoint key.
                    ckpt_key = f"{prefix}.{name}"
                    
                    # Check whether the key exists in the checkpoint.
                    if ckpt_key in checkpoint:
                        # Copy the weights directly.
                        param.data.copy_(checkpoint[ckpt_key])
                        component_loaded += 1
                        loaded_params += 1
                
                # Print the number of parameters loaded for each component.
                total_params = sum(1 for _ in component.parameters())
                # print(f"Component {prefix}: loaded parameters {component_loaded}/{total_params}")
                print(f"Component {prefix} ({'HQ' if checkpoint is sd_HQ else 'LQ'}): loaded parameters {component_loaded}/{total_params}")

            # ----- Part 4: Verify the loaded weights -----

            
            # ----- Part 5: Print final HQ quantizer statistics -----
            print('Finished loading checkpoints')
            print('='*50)
        except Exception as e:
            print(f"Error loading checkpoints: {e}")
            import traceback
            traceback.print_exc()
            raise

    # 3. Add the mask scheduling function.
    def _get_mask_ratio(self, t, method='cosine'):
        """Return the mask ratio for a given timestep."""
        if method == 'cosine':
            return math.cos(math.pi/2 * t)
        elif method == 'linear':
            return 1.0 - t
        else:
            return math.cos(math.pi/2 * t)  # Use cosine scheduling by default.

        # ============ Masked autoregressive section ============
    def maskgit_train_step(self, lq_feat, gt_indices):
        """
        lq_feat: [B, L, C]
        gt_indices: [B, L], where L=256
        """
        device = lq_feat.device
        B, L = gt_indices.shape
         # quant_feat, codebook_loss, quant_stats = self.quantize(lq_feat)
        pos_emb = self.position_emb.unsqueeze(1).repeat(1,B,1)
        
        feat_emb = self.feat_emb(lq_feat.permute(1,0,2)) # [seq_len, batch, embed_dim]
        
        # --------------------------------------------------- 
        # 1. Sample a random mask ratio r ∈ [0, 1].
        # --------------------------------------------------- 
        r = torch.rand(1, device=device).item()
        mask_ratio = self._get_mask_ratio(r, self.mask_scheduling_method)
            # Select the exact number of masks instead of masking probabilistically.
        num_mask = max(1, int(mask_ratio * L))

        # ---------------------------------------------------
        # 2. Randomly sample mask positions.
        # ---------------------------------------------------
        rand_perm = torch.rand(B, L, device=device).argsort(dim=-1)
        mask_pos = rand_perm[:, :num_mask]             # [B, num_mask]
        mask_bool = torch.zeros(B, L, dtype=torch.bool, device=device)
        mask_bool.scatter_(1, mask_pos, True)

        # ---------------------------------------------------
        # 3. Generate the masked-token sequence.
        # ---------------------------------------------------
        input_tokens = gt_indices.clone()
        input_tokens[mask_bool] = self.mask_token_id   # Replace masked positions with the mask token.
        # ---------------------------------------------------
        # 4. token embedding + lq_feat embedding
        # ---------------------------------------------------
        # ---- LQ feature embedding used as a prefix.
        # ---- token embedding
        tok_emb = self.token_emb(input_tokens).permute(1, 0, 2) # [B,seq_len,dim] -> [seq_len,B,dim]
        # ---- concat prefix + token
        input_emb = torch.cat([feat_emb, tok_emb], dim=0) # [2*seq_len, batch, embed_dim]

        # ---------------------------------------------------
        # 5. Fully parallel Transformer forward pass.
        # ---------------------------------------------------
        # Transformer forward pass.
        query_emb = input_emb
        for layer in self.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)  # Use the full positional encoding directly.
        
        # Use only the HQ-token output for prediction.
        hq_output = query_emb[L:]  # Take the second half: [seq_len, batch, embed_dim].

        # ---------------------------------------------------
        # 6. Predict logits.
        # ---------------------------------------------------
        logits = self.idx_pred_layer(hq_output)       # [seq_len, B, vocab_size]
        logits = logits.permute(1, 0, 2)  # [B, seq_len, vocab_size]
        # ---------------------------------------------------
        # 7. CE loss with mask positions only
        # ---------------------------------------------------
        loss = F.cross_entropy(
           logits[mask_bool], gt_indices[mask_bool])

        return loss, logits

    @torch.no_grad()
    def maskgit_generate(self, x, filenames, gt=None):
        """
        Inference via iterative parallel decoding.
        Input: LQ features.
        Output: Final generated token sequence [B, L].
        """
        
        gt_indices = None
        quant_gt = None
        L2_loss = torch.tensor(0.0, device=x.device)
        if gt is not None:   # Check whether GT is provided.
            quant_gt, gt_indices, gt_info, gt_hs, gt_h, gt_dictionary = self.encode_to_gt(gt) # Encode GT.
        

        # LQ feature from LQ encoder and quantizer 
        z_hs = self.vqvae.encoder(x)
        
        z_h = self.vqvae.quant_conv(z_hs['out'])
       
        # Use the original HQ codebook for indexing.
        quant_z, emb_loss, z_info, z_dictionary = self.vqvae.HQ_quantize(z_h)
        indices = z_info[2].view(quant_z.shape[0], -1)
        z_indices = indices

        # Apply feature alignment.
        batch_size, c, h, w = z_h.shape
        z_seq = z_h.permute(0, 2, 3, 1).reshape(batch_size, h*w, c)
        
        if self.alignmodel is not None and filenames is not None:
            # Type check to avoid treating a Tensor as a file path.
            if isinstance(filenames, torch.Tensor):
                print("Tensor-valued file paths detected; skipping feature alignment")
                lq_feat = z_seq  # Use z_seq directly without recomputing it.
            else:
                # Process string paths normally.
                batch_with_gt_path = {"gt_path": filenames}
                text_features = self.alignmodel.get_text_features(z_h, batch=batch_with_gt_path)
               
                # Detect and match data types.
                model_dtype = next(self.alignmodel.parameters()).dtype
                z_seq = z_seq.to(dtype=model_dtype)
                text_features = text_features.to(dtype=model_dtype)
        
                
                lq_feat,_ = self.alignmodel(z_seq, text_features)

                # Convert back to the original data type for consistency.
                lq_feat = lq_feat.to(dtype=z_h.dtype)
            
        else:
            # Use the original features directly when no alignment model is available.
            lq_feat = z_seq  # Use z_seq directly without recomputing it.
            print("Alignment model not used; using the original features directly")

        B, L, C = lq_feat.shape
        device = lq_feat.device
        # Initialize all positions as masked.
        current_hq_tokens = torch.full((B, L), fill_value=self.mask_token_id, dtype=torch.long, device=device)
        # Prefix with the LQ feature embedding, as during training.
        feat_emb = self.feat_emb(lq_feat.permute(1,0,2)) # [seq_len, batch, embed_dim]
        pos_emb = self.position_emb.unsqueeze(1).repeat(1, B, 1)  # [seq_len, B, dim]

        # # ------------------------------
        # Multi-step iterative reveal.
        # ------------------------------
        total_ce = 0.0
        for t in range(self.timesteps):
            # print("Inference step {}/{}".format(t+1, self.timesteps))
            
            # token embedding
            tok_emb = self.token_emb(current_hq_tokens).permute(1, 0, 2)  # [B,seq_len,dim] -> [seq_len,B,dim]    
            query_emb = torch.cat([feat_emb, tok_emb], dim=0) # [2*seq_len, batch, embed_dim]
            
            
            # Transformer forward pass.
            for layer in self.ft_layers:
                query_emb = layer(query_emb, query_pos=pos_emb)

            hq_output = query_emb[L:]  # Take the second half: [seq_len, batch, embed_dim].

            logits = self.idx_pred_layer(hq_output)
            logits = logits.permute(1, 0, 2)


            k_max = 1024  # Maximum k in early steps to increase diversity.
            k_min = 1 # Minimum k in later steps (k=1 is equivalent to argmax) for consistency.
            if t<2:
                current_k=k_max
            # --- 2. Compute current_k from step t using linear decay. ---
            # progress_ratio changes from 0.0 (t=0) to 1.0 (t=timesteps-1).
            else:

                progress_ratio = (t + 1) / self.timesteps  # From 0.0 to 1.0.

                current_k = k_max - (k_max - k_min) * progress_ratio
                current_k = int(current_k)
                current_k = max(k_min, current_k) # Ensure that k is at least 1.

 
            topk_logits, topk_indices = torch.topk(logits, k=current_k, dim=-1)

            # Step 2: Apply softmax to the top-k logits to create a new probability distribution.
            topk_probs = F.softmax(topk_logits, dim=-1)

            # Step 3: Randomly sample from the new k-dimensional distribution.
            B, L, _ = topk_probs.shape
            # topk_sample_idx: [B * L, 1], with each value in [0, k-1].
            topk_sample_idx = torch.multinomial(topk_probs.view(B * L, -1), num_samples=1)

            # Reshape the sampled indices back to [B, L].
            topk_sample_idx = topk_sample_idx.view(B, L)

            # Step 4: Use the sampled indices to select the final tokens from topk_indices.
            pred_tokens = torch.gather(topk_indices, dim=-1, index=topk_sample_idx.unsqueeze(-1)).squeeze(-1)

            # Step 5: Compute confidence scores.
            probs = F.softmax(logits, dim=-1)
            confidence = torch.gather(probs, dim=-1, index=pred_tokens.unsqueeze(-1)).squeeze(-1)


            # ---- freeze revealed tokens ----
            mask = (current_hq_tokens == self.mask_token_id)
            confidence = confidence.masked_fill(~mask, -float("inf"))
            

            # Compute the reveal ratio for this iteration.
            progress_ratio = (t + 1) / self.timesteps
            if t == self.timesteps -1:
                mask_ratio_t_1 = 0.0  # Reveal all tokens in the final step.
            else:   
                mask_ratio_t_1 =self._get_mask_ratio(progress_ratio, self.mask_scheduling_method)
            
            batch_step_ce = 0.0
            # Process confidence scores globally.
            batch_tokens = []

            for b in range(B):
                cur_tokens = current_hq_tokens[b]
                current_mask = mask[b]
                unmasked_count = (~current_mask).sum().item()
                num_reveal = int((1 - mask_ratio_t_1) * L)

                # Compute how many more tokens must be revealed.
                tokens_to_reveal = max(0, num_reveal - unmasked_count)
                # Create a new token state, starting with all revealed tokens.
                new_tokens =  cur_tokens.clone()
                
                 # Reveal more tokens if masked positions remain.
                if tokens_to_reveal > 0 and current_mask.any():
                    # Consider confidence scores only at masked positions.
                    mask_positions = torch.where(current_mask)[0]
                    masked_confidence = confidence[b][mask_positions]


                    # Select the k highest-confidence positions to reveal.
                    _, top_indices = torch.topk(masked_confidence, k=min(tokens_to_reveal, len(mask_positions)))
                    positions_to_reveal = mask_positions[top_indices]
                                
                    # Reveal the selected positions.
                    new_tokens[positions_to_reveal] = pred_tokens[b][positions_to_reveal]
                    
                    if gt_indices is not None:
                        # Compute CE for this step.
                        step_ce = F.cross_entropy(logits[b, positions_to_reveal], gt_indices[b, positions_to_reveal])
                        batch_step_ce += step_ce            
                batch_tokens.append(new_tokens)
                
            # Update the current tokens.
            current_hq_tokens = torch.stack(batch_tokens)
            total_ce += batch_step_ce / B
            # ===== 4. Final safety check =====
        # Ensure that no mask tokens remain.
        mask_positions = (current_hq_tokens == self.mask_token_id)
        if mask_positions.any():
            print(f"Warning: {mask_positions.sum().item()} masked positions remain; filling them randomly")
            current_hq_tokens[mask_positions] = torch.randint(
                0, self.vocab_size, (mask_positions.sum().item(),), device=device
            )
        
        # Ensure that indices are within the valid range.
        current_hq_tokens = torch.clamp(current_hq_tokens, 0, self.vocab_size-1)


         # 4. Decode into an image.
        quant_feat = self.vqvae.HQ_quantize.get_codebook_entry(current_hq_tokens.view(-1), 
                                                           shape=[B, 16,16,256])
       
        dec_img = self.vqvae.decode(quant_feat)


        lq_feat = lq_feat.reshape(batch_size, h, w, c).permute(0, 3, 1, 2)
        if quant_gt is not None:
            L2_loss = F.mse_loss(lq_feat, quant_gt)

        return current_hq_tokens, dec_img, total_ce, L2_loss 


        
    def forward(self, input, gt=None,filenames=None, save_features=False, features_dir="features"):
        
        if gt is not None:   # Check whether GT is provided.
            quant_gt, gt_indices, gt_info, gt_hs, gt_h, gt_dictionary = self.encode_to_gt(gt) # Encode GT.
        
        # LQ feature from LQ encoder and quantizer 
        z_hs = self.vqvae.encoder(input)
        
        z_h = self.vqvae.quant_conv(z_hs['out'])
       
        # Use the original HQ codebook for indexing.
        quant_z, emb_loss, z_info, z_dictionary = self.vqvae.HQ_quantize(z_h)
        indices = z_info[2].view(quant_z.shape[0], -1)
        z_indices = indices

        if gt is None:
            quant_gt = quant_z
            gt_indices = z_indices
            self.alignmodel.eval()
            self.alignmodel.set_eval_mode(True)

        # Add feature alignment. =============================================
        batch_size, c, h, w = z_h.shape
        z_seq = z_h.permute(0, 2, 3, 1).reshape(batch_size, h*w, c)
        
        if self.alignmodel is not None and filenames is not None:
            # Type check to avoid treating a Tensor as a file path.
            if isinstance(filenames, torch.Tensor):
                print("Tensor-valued file paths detected; skipping feature alignment")
                lq_feat = z_seq  # Use z_seq directly without recomputing it.
            else:
                # Process string paths normally.
                batch_with_gt_path = {"gt_path": filenames}
                text_features = self.alignmodel.get_text_features(z_h, batch=batch_with_gt_path)
               
                # Detect and match data types.
                model_dtype = next(self.alignmodel.parameters()).dtype
                z_seq = z_seq.to(dtype=model_dtype)
                text_features = text_features.to(dtype=model_dtype)
        
                # Call forward and request contrastive features.
                lq_feat, contrastive_loss  = self.alignmodel(z_seq, text_features)

                # Convert back to the original data type for consistency.
                lq_feat= lq_feat.to(dtype=z_h.dtype)
            
        else:
            # Use the original features directly when no alignment model is available.
            lq_feat = z_seq  # Use z_seq directly without recomputing it.
            print("Alignment model not used; using the original features directly")



         # ============ Replace the original method with iterative masked autoregression ============
        BCE_loss, logits = self.maskgit_train_step(
            lq_feat=lq_feat, 
            gt_indices=gt_indices
        )

        soft_one_hot = F.softmax(logits, dim=2)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)

        quant_feat = self.vqvae.HQ_quantize.get_codebook_entry(top_idx.reshape(-1), shape=[z_h.shape[0],16,16,256])
        

        lq_feat = lq_feat.reshape(batch_size, h, w, c).permute(0, 3, 1, 2)
        L2_loss = F.mse_loss(lq_feat, quant_gt)
        
        # Preserve gradients using shape-compatible tensors.
        quant_feat = lq_feat + (quant_feat - lq_feat).detach()
        dec = self.vqvae.decode(quant_feat)
        # Obtain features and decode them.

        return dec, BCE_loss, L2_loss, z_info, z_hs, z_h, quant_gt, z_dictionary, contrastive_loss
    



    @torch.no_grad()
    def encode_to_gt(self, gt):
        quant_gt, _, info, hs, h, dictionary = self.vqvae.HQ_encode(gt)
        indices = info[2].view(quant_gt.shape[0], -1)
        return quant_gt, indices, info, hs, h, dictionary

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        self.alignmodel.train()
        self.alignmodel.set_eval_mode(False)
        if optimizer_idx == None:
            optimizer_idx = 0

        x = batch[self.image_key]
        gt = batch['gt']
        filenames = batch.get('gt_path', None)
        xrec, BCE_loss, L2_loss, info, hs,_,_,_, contrastive_loss = self(x, gt, filenames)

        # qloss = BCE_loss + 10*L2_loss + contrastive_loss

        if self.image_key != 'gt':
            x = batch['gt']

        if self.use_facial_disc:
            loc_left_eyes = batch['loc_left_eye']
            loc_right_eyes = batch['loc_right_eye']
            loc_mouths = batch['loc_mouth']
            face_ratio = xrec.shape[-1] / 512
            components = get_roi_regions(
                x, xrec, loc_left_eyes, loc_right_eyes, loc_mouths, face_ratio)
        else:
            components = None

        if optimizer_idx == 0:

            aeloss = BCE_loss + 10*L2_loss + contrastive_loss

            rec_loss = (torch.abs(gt.contiguous() - xrec.contiguous()))

            log_dict_ae = {
                    "train/total_aeloss": aeloss.detach().mean(),
                   "train/BCE_loss": BCE_loss.detach().mean(),
                   "train/L2_loss": L2_loss.detach().mean(),
                   "train/Rec_loss": rec_loss.detach().mean(),
                   "train/Contrastive_loss": contrastive_loss.detach().mean()
                }

            self.log_dict(
            log_dict_ae,
            prog_bar=True,   
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True   
            )
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, components, optimizer_idx, self.global_step,
                                                last_layer=None, split="train")
            self.log("train/discloss", discloss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False,
                          logger=True, on_step=True, on_epoch=True)
            return discloss

        if self.disc_start <= self.global_step:

            # left eye
            if optimizer_idx == 2:
                # discriminator
                disc_left_loss, log_dict_disc = self.loss(qloss, x, xrec, components, optimizer_idx, self.global_step,
                                                          last_layer=None, split="train")
                self.log("train/disc_left_loss", disc_left_loss,
                         prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False,
                              logger=True, on_step=True, on_epoch=True)
                return disc_left_loss

            # right eye
            if optimizer_idx == 3:
                # discriminator
                disc_right_loss, log_dict_disc = self.loss(qloss, x, xrec, components, optimizer_idx, self.global_step,
                                                           last_layer=None, split="train")
                self.log("train/disc_right_loss", disc_right_loss,
                         prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False,
                              logger=True, on_step=True, on_epoch=True)
                return disc_right_loss

            # mouth
            if optimizer_idx == 4:
                # discriminator
                disc_mouth_loss, log_dict_disc = self.loss(qloss, x, xrec, components, optimizer_idx, self.global_step,
                                                           last_layer=None, split="train")
                self.log("train/disc_mouth_loss", disc_mouth_loss,
                         prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False,
                              logger=True, on_step=True, on_epoch=True)
                return disc_mouth_loss

    def validation_step(self, batch, batch_idx):
     
        x = batch[self.image_key]

        gt = batch['gt']
        filenames = batch.get('gt_path', batch.get('filename', None))

        _, xrec, BCE_loss, L2_loss = self.maskgit_generate(x=x,  filenames=filenames, gt = gt)

        # qloss = BCE_loss + L2_loss

        if self.image_key != 'gt':
            x = batch['gt']


        xrec = torch.clamp(xrec, -1.0, 1.0) 
        xrec_norm = torch.clamp((xrec + 1.0) / 2.0, 0.0, 1.0)
        gt_norm = torch.clamp((gt + 1.0) / 2.0, 0.0, 1.0)

        self.fid_metric.update(gt_norm, real=True)
        self.fid_metric.update(xrec_norm, real=False)
        self.psnr_metric.update(xrec_norm, gt_norm)
        self.ssim_metric.update(xrec_norm, gt_norm)
       
         # ================= Compute NIQE (no-reference metric) =================
        # NIQE can be averaged across the batch.
        try:
            if self.niqe_metric.device != xrec_norm.device:
                self.niqe_metric = self.niqe_metric.to(xrec_norm.device)
            
            # The pyiqa NIQE metric generally expects input in [0, 1].
            val_niqe = self.niqe_metric(xrec_norm).mean()
        except Exception as e:
            if self.global_rank == 0:
                print(f"NIQE Error: {e}")
            val_niqe = torch.tensor(0.0, device=xrec_norm.device)

        
        rec_loss = (torch.abs(gt.contiguous() - xrec.contiguous())).mean()
        
        self.log("val_niqe", val_niqe.detach(), prog_bar=True,
                 logger=True, on_step=False, on_epoch=True, sync_dist=False)
              
        log_dict_ae = {
                "val_BCE_loss": BCE_loss.detach().mean(),
                "val_L2_loss": L2_loss.detach().mean(),
                "val_Rec_loss": rec_loss.detach(),
            }

        # self.log_dict(log_dict_ae)
        self.log_dict(log_dict_ae, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)

        if self.global_rank == 0:
            # print(f'niqe: {val_niqe.item():.4f}')
            print(f'Validation Step {batch_idx}: niqe: {val_niqe.item():.4f}')

        return self.log_dict
  

    def on_validation_epoch_end(self):
        # ===========================================================================
        # 1. Compute and log torchmetrics metrics (the most likely missing section).
        # ===========================================================================
        try:
            val_fid = self.fid_metric.compute()
            val_psnr = self.psnr_metric.compute()
            val_ssim = self.ssim_metric.compute()

            # Reset metric states for the next validation epoch.
            self.fid_metric.reset()
            self.psnr_metric.reset()
            self.ssim_metric.reset()
        except Exception as e:
            # Supply defaults on failure to prevent the program from crashing.
            if self.global_rank == 0:
                print(f"Error computing torchmetrics: {e}")
            val_fid = torch.tensor(float('inf'), device=self.device) # Use a very large value.
            val_psnr = torch.tensor(0.0, device=self.device)
            val_ssim = torch.tensor(0.0, device=self.device)

            
        self.trainer.callback_metrics["val_fid"] = val_fid
        self.trainer.callback_metrics["val_psnr"] = val_psnr
        self.trainer.callback_metrics["val_ssim"] = val_ssim


        if self.global_rank == 0:
            print(f"\nEpoch End Validation: FID={val_fid:.4f} PSNR={val_psnr:.4f} SSIM={val_ssim:.4f}\n")

    def on_validation_epoch_start(self):
        self.alignmodel.set_eval_mode(True)
        

    def configure_optimizers(self):
        lr = self.learning_rate
        print(f"Configuring optimizer with base learning rate: {lr}")
        normal_params = []
        special_params = []
        fixed_params = []
        fixed_parameter = 0
        test_count = 0
        # schedules = []
        # autoencoder part -------------------------------
        for name, param in self.vqvae.named_parameters():
            if not param.requires_grad:
                continue

            if 'HQ' in name:
                special_params.append(param)
                fixed_parameter = fixed_parameter + 1
                continue
            if 'decoder' in name or 'post_quant_conv' in name or 'quantize' in name:
                test_count = test_count + 1
                # continue
                special_params.append(param)
                # print(name)
            else:
                normal_params.append(param)

        # Add alignmodel parameters.
        if self.alignmodel is not None:
            print("Adding alignmodel parameters to the optimizer...")
            for name, param in self.alignmodel.named_parameters():
                if not param.requires_grad:
                    continue
                else:
                    normal_params.append(param)
                    print(f"Adding alignmodel parameter: {name}")
                
        # transformer part--------------------------------
        
        normal_params.append(self.position_emb)   
        
        for name, param in self.feat_emb.named_parameters():
            if not param.requires_grad:
                continue
            else:
                normal_params.append(param) 

        for name, param in self.ft_layers.named_parameters():
            if not param.requires_grad:
                continue
            else:
                normal_params.append(param) 
        for name, param in self.token_emb.named_parameters():
            if not param.requires_grad:
                continue
            else:
                normal_params.append(param)

        for name, param in self.idx_pred_layer.named_parameters():
            if not param.requires_grad:
                continue
            else:
                normal_params.append(param)                 
        
        opt_ae_params = [{'params': normal_params, 'lr': lr}]

        opt_ae = torch.optim.Adam(opt_ae_params, betas=(0.5, 0.9))

        optimizations = opt_ae

        if self.use_facial_disc:
            opt_l = torch.optim.Adam(self.loss.net_d_left_eye.parameters(),
                                     lr=lr*self.comp_params_lr_scale, betas=(0.9, 0.99))
            opt_r = torch.optim.Adam(self.loss.net_d_right_eye.parameters(),
                                     lr=lr*self.comp_params_lr_scale, betas=(0.9, 0.99))
            opt_m = torch.optim.Adam(self.loss.net_d_mouth.parameters(),
                                     lr=lr*self.comp_params_lr_scale, betas=(0.9, 0.99))
            optimizations += [opt_l, opt_r, opt_m]

            s2 = torch.optim.lr_scheduler.MultiStepLR(
                opt_l, milestones=self.schedule_step, gamma=0.1, verbose=True)
            s3 = torch.optim.lr_scheduler.MultiStepLR(
                opt_r, milestones=self.schedule_step, gamma=0.1, verbose=True)
            s4 = torch.optim.lr_scheduler.MultiStepLR(
                opt_m, milestones=self.schedule_step, gamma=0.1, verbose=True)
            schedules += [s2, s3, s4]

        # return optimizations, schedules
        return optimizations

    def get_last_layer(self):
        if self.fix_decoder:
            return self.vqvae.quant_conv.weight
        return self.vqvae.decoder.conv_out.weight

    def log_images(self, batch, split, **kwargs):
        log = dict()
        x = batch[self.image_key]
        x = x.to(self.device)
   
        gt = batch['gt'].to(self.device)
        filenames = batch.get('gt_path', batch.get('filename', None))
    
        if split == 'train':
            xrec, _, _, _, _, _, _, _,_ = self(x, gt, filenames)

        elif split == 'val':
            self.alignmodel.set_eval_mode(True)
            _, xrec, BCE_loss,_ = self.maskgit_generate(x= x, filenames=filenames, gt=gt)

        log["inputs"] = x
        log["reconstructions"] = xrec
        
        if self.image_key != 'gt':
            x = batch['gt']
            log["gt"] = x
        
        return log
