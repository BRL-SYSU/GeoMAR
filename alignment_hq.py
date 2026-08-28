import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import traceback
from pathlib import Path
import gc
import numpy as np
import torch.distributed as dist
# --- 1. Visual Attention Pool  ---
class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        # Positional embedding: (H*W + 1, Dim)
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        # Input x expected: [B, C, H, W]
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # BCHW -> (HW)BC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        
        # Add positional embeddings.
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        
        # Multi-head Attention
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0) # [B, Output_Dim]

# --- 2. Text Attention Pool  ---
class AttentionPool1d(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        # Positional embeddings: [Sequence_Length, Input_Dim].
        self.positional_embedding = nn.Parameter(torch.randn(256, input_dim) / input_dim ** 0.5)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.c_proj = nn.Linear(input_dim, embed_dim) # Output projection directly to embed_dim.
        self.num_heads = num_heads

    def forward(self, x):
        # x: [B, Seq_Len, Dim]
        x = x.permute(1, 0, 2)  # [Seq, B, Dim] -> [Seq, B, Dim]
        
        # Add positional embeddings.
        seq_len = x.shape[0]
        x = x + self.positional_embedding[:seq_len, None, :].to(x.dtype)
        
        # Generate the query (mean token).
        mean_token = x.mean(dim=0, keepdim=True) # [1, B, Dim]
        x = torch.cat([mean_token, x], dim=0)    # [Seq+1, B, Dim]
        
        # Attention
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight, # This layer projects to embed_dim.
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0) # [B, Embed_Dim]


# --- 3. Encapsulated Contrastive Learning Model ---
class SiAContrastiveModel(nn.Module):
    def __init__(self, 
                 visual_input_dim=256,   
                 visual_seq_len=256,     
                 text_input_dim=256,     # Text encoder output dimension.
                 embed_dim=256,          # Dimension of the final shared alignment space.
                 num_heads=4,            # Number of attention-pooling heads.
                 temperature=0.07):
        super().__init__()
        
        # A. Visual branch.
        self.spacial_dim = 16
        self.visual_pool = AttentionPool2d(
            spacial_dim=self.spacial_dim,
            embed_dim=visual_input_dim,
            num_heads=num_heads,
            output_dim=embed_dim
        )
        
        # B. Text branch.
        self.text_pool = AttentionPool1d(
            input_dim=text_input_dim, 
            embed_dim=embed_dim, 
            num_heads=8 # The number of heads may differ from the visual branch.
        )
        
        
        # C. Temperature coefficient.
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def forward(self, visual_feat, text_feat):
        """
        Args:
            visual_feat: [B, L, C] 
            text_feat:   [B, 128, 256]
        """
        device = visual_feat.device
        h = w = self.spacial_dim  # 16x16
        # --- 1. Process visual features. ---
        B, L, C = visual_feat.shape

        visual_feat = visual_feat.reshape(B, h, w, C).permute(0, 3, 1, 2)
        visual_embed = self.visual_pool(visual_feat) # [B, 256]

        # --- 2. Process text features. ---
        # Pass [B, 128, 256] directly; attention pooling handles it automatically.
        text_embed = self.text_pool(text_feat) # [B, 256]

        # --- 3. Normalize. ---
        visual_embed = F.normalize(visual_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)

        def concat_all_gather_no_grad(tensor):
            """Gather features from all GPUs as negative samples without backpropagating gradients."""
            if not dist.is_initialized():
                return tensor
            tensors_gather = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(tensors_gather, tensor)
            return torch.cat(tensors_gather, dim=0)

        all_visual = concat_all_gather_no_grad(visual_embed)
        all_text   = concat_all_gather_no_grad(text_embed)

        # ----------------------------
        # 5. Compute the contrastive loss.
        # ----------------------------
        logit_scale = torch.clamp(self.logit_scale.exp(), max=50)

        logits_i2t = logit_scale * visual_embed @ all_text.t()   # [B, B*world_size]
        logits_t2i = logit_scale * text_embed @ all_visual.t()   # [B, B*world_size]
        # print("logits_i2t:", logits_i2t.max(), logits_i2t.min())
        
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
            
        # Labels must include the rank offset.
        # Rank 0: [0, 1, 2, 3]
        # Rank 1: [4, 5, 6, 7]
        labels = torch.arange(B, device=device, dtype=torch.long) + rank * B
        
        loss_i2t = F.cross_entropy(logits_i2t, labels)
        loss_t2i = F.cross_entropy(logits_t2i, labels)
        

        loss = (loss_i2t + loss_t2i) / 2
        # print("dist initialized:", dist.is_initialized())

        return loss


class FeatureMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        output_dim = output_dim or input_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim*2),
            nn.GELU(),  # Use GELU; alternatives include ReLU and LeakyReLU.
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim*2),  # Additional hidden layer.
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, output_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """Preprocess features."""
        param = next(self.mlp.parameters())
        x = x.to(device=param.device, dtype=param.dtype)
        return self.mlp(x)
    
class SFTModule(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.scale_conv = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
        )
        
        self.shift_conv = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
        )
    
    def forward(self, features, conditions):
        """Transform features by conditionally scaling and shifting them."""
        # Pool the conditioning features into a single vector.

        pooled_conditions = conditions.mean(dim=1, keepdim=True)
  
        # Rearrange dimensions for convolution operations.
        features_t = features.transpose(1, 2)
        conditions_t = pooled_conditions.transpose(1, 2)

        # Compute scale and shift factors.
        scale = self.scale_conv(conditions_t)
        shift = self.shift_conv(conditions_t)

        # Apply SFT: features * scale + shift.
        transformed_features = features_t * (scale + 1.0) + shift

        # Restore the original dimension order.
        transformed_features = transformed_features.transpose(1, 2)

        return transformed_features

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Multi-head attention projections.
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention.
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention.
        output = torch.matmul(attn_weights, v)
        output = output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        output = self.out_proj(output)
        
        return output

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, q, k, v):
        batch_size, q_len, _ = q.shape
        _, kv_len, _ = k.shape
        
        # Multi-head attention projections.
        q = self.q_proj(q).reshape(batch_size, q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(k).reshape(batch_size, kv_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(v).reshape(batch_size, kv_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention.
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention.
        output = torch.matmul(attn_weights, v)
        output = output.permute(0, 2, 1, 3).reshape(batch_size, q_len, -1)
        output = self.out_proj(output)
        
        return output

class FeatureProjector(nn.Module):
    """Feature projection module with learnable parameters."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # Special handling for reducing 512-dimensional CLIP features to 256 dimensions.
        
        self.projection = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.GELU(),
                nn.LayerNorm(output_dim)
            )
        print(f"Creating learnable T5-optimized projection: {input_dim} -> {output_dim} (with LayerNorm)")

        # Initialize.
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Project features."""
        return self.projection(x)


class alignmodel(nn.Module):
    """Text-image feature alignment model using cross-attention and learnable dimensionality reduction."""
    def __init__(self, dim=256,
                 aligned_features_dir=None, 
                 train_text_features_dir="./datasets/text_feature/ffhq",
                 eval_text_features_dir="./datasets/text_feature/celeba144",
                 is_training=True,
                 contrastive_dim=256):
        super().__init__()
        self.dim = dim
        self.contrastive_dim = contrastive_dim
        # self.text_folder = text_folder
        self.aligned_features_dir = aligned_features_dir
       
        # Feature preprocessing MLP.
        self.text_mlp = FeatureMLP(input_dim=dim)


        self.train_text_features_dir = train_text_features_dir
        self.eval_text_features_dir = eval_text_features_dir

        self.text_features_dir = (
            self.train_text_features_dir
            if is_training
            else self.eval_text_features_dir
        )

        # print(f"Using text feature directory: {self.text_features_dir} ({'training' if is_training else 'validation/test'} mode)")
        
        self.seq_len = 128 # T5 sequence length.
        
        # Learnable feature projection layers.
        self.projectors = nn.ModuleDict()
    
        
        # Initialize commonly used projection layers.
        self._create_projector(1024, dim, prefix="text")  # Dedicated text-feature projection.


        # Cross-attention and SFT modules.
        self.cross_attention = CrossAttention(dim=dim)  
        self.sft_module = SFTModule(dim)

        self.contrastive_loss = SiAContrastiveModel(visual_input_dim=256, 
                                                    visual_seq_len=256,
                                                    text_input_dim=256,  
                                                    embed_dim=256,       
                                                    num_heads=8
                                                )
        
        
        # Feature cache; only raw features are cached.
        self._feature_cache = {}

    def _create_projector(self, input_dim, output_dim, prefix=""):
        """Create a feature-type-specific projector."""
        key = f"{prefix}_{input_dim}_{output_dim}" if prefix else f"{input_dim}_{output_dim}"
        if key not in self.projectors:
            self.projectors[key] = FeatureProjector(input_dim, output_dim)
            prefix_str = f"{prefix} " if prefix else ""
            print(f"Creating {prefix_str}feature projector: {input_dim} -> {output_dim}")
        return self.projectors[key]

    def forward(self, img_features, text_features):
        """Run the model forward pass with learnable feature projection."""
   
        device = text_features.device
        # 1. Project feature dimensions using learnable dimensionality reduction.
        text_dim = text_features.shape[-1]

        if text_dim != self.dim:
            proj_key = f"text_{text_dim}_{self.dim}"
            if proj_key not in self.projectors:
                self._create_projector(text_dim, self.dim, prefix="text")
            
            # Ensure that the projection layer is on the correct device.
            projector = self.projectors[proj_key].to(device)
            text_features = projector(text_features)
    
        param = next(self.text_mlp.parameters())
        text_features = text_features.to(device=param.device, dtype=param.dtype)
        # 2. Preprocess features with the MLP.
        text_features_processed = self.text_mlp(text_features)


        contrastive_loss = self.contrastive_loss(img_features, text_features_processed)
        # 3. Process image features with the SFT module.
        sft_aligned_features = self.sft_module(img_features, text_features_processed)

        # 5. Self-attention alignment.
        final_aligned_features = self.cross_attention(sft_aligned_features,img_features, img_features)


        return final_aligned_features, contrastive_loss


    def _load_feature(self, path):
        """Load features without any processing."""
        # Check the cache.
        path_str = str(path)
        if path_str in self._feature_cache:
            return self._feature_cache[path_str]
        
        # Load the feature.
        feature = torch.load(path, map_location="cpu")
        # print(f"Loading feature file: {path_str}, shape: {feature.shape}")
        # Handle the batch dimension.
        if len(feature.shape) == 2:
            feature = feature.unsqueeze(0)
        
        # Handle the sequence length.
        seq_len = feature.shape[1]
        if seq_len != self.seq_len:
            if seq_len > self.seq_len:
                # Truncate.
                feature = feature[:, :self.seq_len, :]
            else:
                # Pad.
                padding = torch.zeros(
                    feature.shape[0], 
                    self.seq_len-seq_len, 
                    feature.shape[2],
                    device="cpu"
                )
                feature = torch.cat([feature, padding], dim=1)
        
        # Cache and return.
        self._feature_cache[path_str] = feature
        return feature

    def get_text_features(self, images=None, filenames=None, batch=None):
        """Load text features without reducing dimensions; reduction occurs in forward."""
        # Determine the device.
        caller_device = images.device if images is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Prefer gt_path from the batch.
        if batch is not None and 'gt_path' in batch:
            filenames = batch['gt_path']
        
        # Return random features when no filenames are available.
        if filenames is None:
            batch_size = images.shape[0] if images is not None else 1
            return self._get_random_features(caller_device, batch_size)
        
        # Normalize the filenames format.
        if isinstance(filenames, str):
            filenames = [filenames]
            
        # Handle the batch size.
        batch_size = len(filenames)
        if images is not None and images.shape[0] != batch_size:
            print(f"Warning: Image batch size ({images.shape[0]}) does not match the number of filenames ({batch_size})")
            batch_size = images.shape[0]
            # Handle the batch-size mismatch.
            filenames = filenames[:batch_size] if len(filenames) > batch_size else filenames + [filenames[-1]] * (batch_size - len(filenames))
        
        # Process text features in batches.
        text_features_list = []
        found_count = 0
        
        for i in range(batch_size):
            # Manage memory.
            if i > 0 and i % 4 == 0:
                torch.cuda.empty_cache()
                
            filename = filenames[i]
            img_id = self._extract_image_id(filename)
            
            # print(self.text_features_dir)
            # Construct possible feature paths.
            potential_paths = [
                Path(self.text_features_dir) / f"{img_id}_text.pt",
                Path(self.text_features_dir) / f"{img_id.zfill(5)}_text.pt",
                Path(self.text_features_dir) / f"{os.path.basename(filename)}_text.pt"
            ]
        
            feature_loaded = False
            for feature_path in potential_paths:
                if feature_path.exists():
                    try:
                        # Load the feature without reducing its dimensions.
                        text_feature = self._load_feature(feature_path)
                        # Move to the target device.
                        text_feature = text_feature.to(caller_device, dtype=torch.float16)
                        text_features_list.append(text_feature)
                        found_count += 1
                        feature_loaded = True
                        break
                    except Exception as e:
                        print(f"Error processing feature {feature_path}: {str(e)}")
            
            if not feature_loaded:
                # Use random features when no feature file is found.
                print(f"Text feature file not found; using random features: {potential_paths}")
                text_features_list.append(self._get_random_features(caller_device, 1))
        
        # Combine features.
        if found_count == 0:
            print("No text feature files found; using random features for all items")
        
        try:
            return torch.cat(text_features_list, dim=0)
        except Exception as e:
            print(f"Error combining features: {e}")
            return self._get_random_features(caller_device, batch_size)

    def get_tag_features(self, filenames=None, batch=None):
        """Load tag features without reducing dimensions; reduction occurs in forward."""
        # Determine the device.
        caller_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Prefer gt_path from the batch.
        if batch is not None and 'gt_path' in batch:
            filenames = batch['gt_path']
        
        # Return random features when no filenames are available.
        if filenames is None:
            batch_size = len(batch['gt_path']) if batch is not None else 1
            return self._get_random_features(caller_device, batch_size)
        
        # Normalize the filenames format.
        if isinstance(filenames, str):
            filenames = [filenames]
        
        # Handle the batch size.
        batch_size = len(filenames)
        tag_features_list = []
        found_count = 0
        
        for i in range(batch_size):
            filename = filenames[i]
            tag_id = self._extract_image_id(filename)
            
            # Construct the tag-feature path.
            tag_feature_path = Path(self.tag_features_dir) / f"{tag_id}_tag.pt"
            
            if tag_feature_path.exists():
                try:
                    # Load the feature without reducing its dimensions.
                    tag_feature = self._load_feature(tag_feature_path)
                    # Move to the target device.
                    tag_feature = tag_feature.to(caller_device, dtype=torch.float16)
                    tag_features_list.append(tag_feature)
                    found_count += 1
                except Exception as e:
                    print(f"Error loading tag feature {tag_feature_path}: {str(e)}")
                    tag_features_list.append(self._get_random_features(caller_device, 1))
            else:
                # Use random features when no feature file is found.
                print(f"Tag feature file not found; using random features: {tag_feature_path}")
                tag_features_list.append(self._get_random_features(caller_device, 1))
        
        # Combine features.
        if found_count == 0:
            print("No tag feature files found; using random features for all items")

        try:
            return torch.cat(tag_features_list, dim=0)
        except Exception as e:
            print(f"Error combining tag features: {e}")
            return self._get_random_features(caller_device, batch_size)

    def _extract_image_id(self, path, keep_zeros=True):
        """Extract an image ID from a path."""
        basename = os.path.basename(path)
        basename = os.path.splitext(basename)[0]
        return basename

    def _get_random_features(self, device, batch_size=1):
        """Generate random features."""
        if device.type == 'cuda' and torch.cuda.is_available():
            features = []
            chunk_size = min(batch_size, 4)
            
            for i in range(0, batch_size, chunk_size):
                end_idx = min(i + chunk_size, batch_size)
                chunk_size_actual = end_idx - i
                chunk = torch.randn(chunk_size_actual, self.seq_len, self.dim, 
                                  device=device, dtype=torch.float16)
                features.append(chunk)
                
                if i > 0 and i % 8 == 0:
                    torch.cuda.empty_cache()
            
            return torch.cat(features, dim=0)
        else:
            return torch.randn(batch_size, self.seq_len, self.dim, 
                              device=device, dtype=torch.float16)

    def set_eval_mode(self, eval_mode=True):
        """Set evaluation mode and switch the text-feature directory."""
        self.text_features_dir = (
            self.eval_text_features_dir
            if eval_mode
            else self.train_text_features_dir
        )
        
        self._feature_cache = {}
    
