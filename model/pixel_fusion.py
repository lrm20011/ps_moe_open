"""Fused coordinate-token embedding: learned PE + Fourier PE + image patch feature.

Enabled by `use_fused_pixel_embedding: true` in the model config; when it is
false or absent, coordinate tokens keep the learned-PE embedding only.

The embedding is a weighted sum of three branches:

- learned PE: pixel_embed_mlp2 of the interpolated vision position table
  (the existing embedding);
- Fourier features of (x, y), through a two-layer MLP;
- the vision tower's pre-merger patch feature (last_hidden_state, one per
  16-px patch) at (x, y), bilinearly sampled from the last image before the
  token in the same sample.

Each branch has one learned weight, shared by all tokens: it records how
useful that source is, not anything about the token. The branch layers are
randomly initialized; the weights start at (learned PE 1, Fourier 0, image 0),
so a checkpoint without these weights produces exactly the learned-PE
embedding until the weights move. The fused vector is rescaled to the
learned-PE norm, so the branches change its direction but not the scale the
LM expects; the weights therefore act as relative contributions. The weights
are stored as offsets from that start, and the LayerNorm has no affine scale,
because pure-bf16 training cannot move a parameter near 1 by less than ~0.004
per step.
"""
import math

import torch
import torch.nn as nn
from transformers import initialization as hf_init

# sin/cos(2^k * pi * x) for k = 0..11: periods from 2 (whole image) down to
# 2^-10 ~ 0.001, about one of 1000 bins.
FOURIER_FREQS = 12
# Learned PE, Fourier, image feature: only the learned PE contributes at first.
BRANCH_WEIGHTS_START = (1.0, 0.0, 0.0)


def use_fused_pixel_embedding(config):
    return bool(getattr(config, "use_fused_pixel_embedding", False))


class PixelFusionEmbedding(nn.Module):
    def __init__(self, vision_hidden_size, text_hidden_size, num_freqs=FOURIER_FREQS):
        super().__init__()
        self.num_freqs = num_freqs
        self.fourier_mlp = nn.Sequential(
            nn.Linear(4 * num_freqs, text_hidden_size), nn.SiLU(), nn.Linear(text_hidden_size, text_hidden_size))
        self.feat_norm = nn.LayerNorm(vision_hidden_size, elementwise_affine=False)
        self.feat_proj = nn.Linear(vision_hidden_size, text_hidden_size)
        # Weights of learned PE, Fourier, image feature, as offsets from BRANCH_WEIGHTS_START.
        self.branch_weight_offsets = nn.Parameter(torch.zeros(3))

    @property
    def branch_weights(self):
        start = torch.tensor(BRANCH_WEIGHTS_START, device=self.branch_weight_offsets.device)
        return start + self.branch_weight_offsets.float()

    def forward(self, learned, xy, patch_feat, has_image):
        """learned [N, D] learned-PE embeddings; xy [N, 2] in [0, 1];
        patch_feat [N, C] sampled patch features; has_image [N] bool."""
        dtype = self.feat_proj.weight.dtype
        freqs = math.pi * 2.0 ** torch.arange(self.num_freqs, device=xy.device, dtype=torch.float32)
        angles = xy.float().unsqueeze(-1) * freqs
        fourier = self.fourier_mlp(torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(1).to(dtype))
        image = self.feat_proj(self.feat_norm(patch_feat.to(dtype))) * has_image.unsqueeze(-1).to(dtype)
        weights, learned = self.branch_weights, learned.float()
        fused = weights[0] * learned + weights[1] * fourier.float() + weights[2] * image.float()
        # Keep each token at the learned-PE norm the LM was trained with: the
        # branches change the direction, not the scale (exactly 1 at the start).
        scale = learned.norm(dim=-1, keepdim=True) / fused.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return (fused * scale).to(dtype)


def init_pixel_fusion_weights(module):
    """For PreTrainedModel._init_weights: a checkpoint without these weights
    starts from BRANCH_WEIGHTS_START (the branch layers get the usual init)."""
    if isinstance(module, PixelFusionEmbedding):
        hf_init.zeros_(module.branch_weight_offsets)


def coordinate_image_index(input_ids, image_token_id, seq_idx=None):
    """[B, S] index of the last image at or before each token in the same sample, -1 if none.

    Images are runs of image tokens numbered in row-major order, the order of
    image_grid_thw and of the vision features. seq_idx marks the samples of a
    padding-free packed row, so a token never refers to another sample's image.
    """
    is_image = input_ids == image_token_id
    starts = is_image.clone()
    starts[:, 1:] &= ~is_image[:, :-1]
    starts = starts.flatten()
    index = starts.long().cumsum(0) - 1
    rows = torch.arange(input_ids.shape[0], device=input_ids.device).unsqueeze(1).expand_as(input_ids)
    segment = rows * (input_ids.shape[1] + 1)
    if seq_idx is not None:
        segment = segment + seq_idx.long()
    segment = segment.flatten()
    image_segment = segment[starts]
    if image_segment.numel() == 0:
        return torch.full_like(input_ids, -1)
    same_sample = image_segment[index.clamp(min=0)] == segment
    return torch.where((index >= 0) & same_sample, index, -1).view_as(input_ids)


def sample_patch_features(patch_feats, image_grid_thw, merge_size, image_index, xy):
    """[N, C] bilinear patch feature of image image_index[n] at xy[n]; zeros where image_index is -1.

    patch_feats [sum(t*h*w), C] is the vision tower's last_hidden_state. Per image,
    patches are block-major over merge_size x merge_size windows (as in
    get_vision_position_ids); only the first temporal slice is used. xy follows
    align_corners=False: 0 and 1 are the image edges, patch centers sit at (i + 0.5) / w.
    """
    has_image = image_index >= 0
    out = patch_feats.new_zeros(xy.shape[0], patch_feats.shape[-1], dtype=torch.float32)
    if not has_image.any():
        return out
    thw = image_grid_thw.to(xy.device).long()
    sizes = thw.prod(-1)
    offsets = sizes.cumsum(0) - sizes
    image = image_index.clamp(min=0)
    h, w, offset = thw[image, 1], thw[image, 2], offsets[image]
    fx = torch.minimum((xy[:, 0].float() * w - 0.5).clamp(min=0), (w - 1).float())
    fy = torch.minimum((xy[:, 1].float() * h - 0.5).clamp(min=0), (h - 1).float())
    x0, y0 = fx.floor().long(), fy.floor().long()
    x1, y1 = torch.minimum(x0 + 1, w - 1), torch.minimum(y0 + 1, h - 1)
    dx, dy = (fx - x0).unsqueeze(-1), (fy - y0).unsqueeze(-1)
    m = merge_size

    def patch(r, c):
        index = offset + ((r // m) * (w // m) + c // m) * m * m + (r % m) * m + c % m
        return patch_feats[index].float()

    out = (patch(y0, x0) * (1 - dx) * (1 - dy) + patch(y0, x1) * dx * (1 - dy)
           + patch(y1, x0) * (1 - dx) * dy + patch(y1, x1) * dx * dy)
    return out * has_image.unsqueeze(-1)


def fused_coordinate_embedding(model, learned, input_ids, pix_mask, xy, patch_feats, image_grid_thw,
                               past_key_values=None, seq_idx=None):
    """Fused embeddings [N, D] of the coordinate tokens input_ids[pix_mask].

    model is the Qwen3_5p backbone (pixel_fusion, visual, config). Outside
    training, each prefill keeps its patch features on the model so that decode
    steps, which receive no pixel_values, still sample the row's last image.
    """
    image_index = coordinate_image_index(input_ids, model.config.image_token_id, seq_idx)
    prefill = past_key_values is None or past_key_values.get_seq_length() == 0
    if prefill:
        if not model.training:
            model._pixel_feature_cache = (
                None if patch_feats is None else (patch_feats.detach(), image_grid_thw, image_index[:, -1]))
    elif patch_feats is None and model._pixel_feature_cache is not None:
        patch_feats, image_grid_thw, last_image = model._pixel_feature_cache
        image_index = last_image.unsqueeze(1).expand_as(input_ids)
    token_image = image_index[pix_mask]
    if patch_feats is None:
        feats = learned.new_zeros(token_image.shape[0], model.config.vision_config.hidden_size)
    else:
        feats = sample_patch_features(patch_feats, image_grid_thw, model.visual.spatial_merge_size, token_image, xy)
    return model.pixel_fusion(learned, xy, feats, token_image >= 0)
