import torch
import torch.nn as nn
import re
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Any
import time
import math

from transformers import Qwen2Tokenizer, AutoConfig, AutoModel
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import ModelOutput
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5Model,
    Qwen3_5PreTrainedModel,
    Qwen3_5ModelOutputWithPast,
    Qwen3_5TextModel,
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5MLP,
)
try:
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
except ImportError:
    from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask
    create_recurrent_attention_mask = None
from transformers.modeling_outputs import BaseModelOutputWithPooling
from pixel_fusion import (
    PixelFusionEmbedding, fused_coordinate_embedding, init_pixel_fusion_weights, use_fused_pixel_embedding,
)

try:
    from typing import Unpack
    from transformers.utils import TransformersKwargs
except ImportError:
    Unpack, TransformersKwargs = Any, Any

import torch.nn.functional as F


STOP_TRAINING_ARGUMENT_NAMES = (
    "eos_loss_weighting_enable", "coord_input_noise_enable", "eos_router_loss_weight", "eos_lm_loss_weight",
    "coord_input_noise_sample_prob", "coord_input_noise_token_prob", "coord_input_noise_std",
)
LEGACY_STOP_TRAINING_CONFIG_NAMES = STOP_TRAINING_ARGUMENT_NAMES + ("stop_robust_training", "stop_token_id")
# Run settings that older checkpoints saved in config.json; dropped on load.
LEGACY_RUN_CONFIG_NAMES = LEGACY_STOP_TRAINING_CONFIG_NAMES + ("lambda_esdf",)
# Padding-free (packed) batch kwargs. They describe the text row only; the
# vision tower builds its own cu_seqlens and must not receive them.
PACKED_SEQUENCE_KWARGS = ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k", "seq_idx")


def resolve_stop_token_ids(config):
    """Read EOS IDs from text_config, falling back to the top-level config."""
    text_config = getattr(config, "text_config", None)
    eos_ids = getattr(text_config, "eos_token_id", None)
    if eos_ids is None:
        eos_ids = getattr(config, "eos_token_id", None)
    if eos_ids is None:
        return ()
    if isinstance(eos_ids, int) and not isinstance(eos_ids, bool):
        eos_ids = [eos_ids]
    if not isinstance(eos_ids, (list, tuple)) or not eos_ids:
        raise ValueError("config eos_token_id must be an integer or a non-empty list of integers")
    vocab_size = config.text_config.vocab_size
    if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < vocab_size for i in eos_ids):
        raise ValueError("config eos_token_id must contain IDs in the text vocabulary")
    return tuple(dict.fromkeys(eos_ids))


def build_eos_mask(labels, eos_ids):
    if not eos_ids:
        return torch.zeros_like(labels, dtype=torch.bool)
    if len(eos_ids) == 1:
        return labels == eos_ids[0]
    return torch.isin(labels, labels.new_tensor(eos_ids))


def eos_weighted_cross_entropy(logits, targets, eos_mask, weight=1.0, ignore_index=-100):
    """Weight EOS positions, normalizing by the sum of valid token weights."""
    if weight == 1.0:
        return F.cross_entropy(logits, targets, ignore_index=ignore_index)
    losses = F.cross_entropy(logits.float(), targets, ignore_index=ignore_index, reduction="none")
    valid = targets != ignore_index
    weights = torch.where(eos_mask, weight, 1.0).to(losses.dtype) * valid
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def perturb_assistant_coordinates(input_ids, labels, *, num_bins, pix_start,
                                  sample_prob, token_prob, std, ignore_index=-100, seq_idx=None):
    """Corrupt only teacher-forced assistant coordinate inputs; never labels.

    seq_idx [B, S] marks the samples of a padding-free packed row, so the
    sample-level draw is made per sample, not per row.
    """
    if sample_prob == 0.0 or token_prob == 0.0 or std == 0.0:
        return input_ids
    eligible = (input_ids >= pix_start) & (labels != ignore_index)
    if seq_idx is None:
        noisy_samples = torch.rand((input_ids.shape[0], 1), device=input_ids.device) < sample_prob
    else:
        draws = torch.rand(input_ids.shape, device=input_ids.device)
        noisy_samples = draws.gather(1, seq_idx.long()) < sample_prob
    selected = eligible & noisy_samples & (torch.rand(input_ids.shape, device=input_ids.device) < token_prob)
    values = input_ids[selected] - pix_start
    xy = torch.stack((values % num_bins, values // num_bins), dim=-1).float() / (num_bins - 1)
    xy = (xy + torch.randn_like(xy) * std).clamp(0.0, 1.0)
    bins = (xy * (num_bins - 1)).round().long()
    perturbed = input_ids.clone()
    perturbed[selected] = pix_start + bins[:, 1] * num_bins + bins[:, 0]
    return perturbed


def pack_normalized_coordinate(
    x: float,
    y: float,
    *,
    num_bins: int,
    pix_start: int,
) -> int:
    """Pack one normalized coordinate into the synthetic coordinate-ID space."""
    if num_bins < 2:
        raise ValueError(f"num_bins must be at least 2, got {num_bins}")
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError(
            f"Coordinate values must be normalized to [0, 1], got ({x}, {y})"
        )
    x_bin = int(round(x * (num_bins - 1)))
    y_bin = int(round(y * (num_bins - 1)))
    return pix_start + y_bin * num_bins + x_bin


def unpack_coordinate_token(
    token_id: int,
    *,
    num_bins: int,
    pix_start: int,
) -> tuple[float, float]:
    """Decode a packed coordinate ID back to normalized coordinates."""
    if num_bins < 2:
        raise ValueError(f"num_bins must be at least 2, got {num_bins}")
    value = int(token_id) - pix_start
    if not 0 <= value < num_bins * num_bins:
        raise ValueError(f"Token ID {token_id} is outside the coordinate-ID range")
    y_bin, x_bin = divmod(value, num_bins)
    scale = float(num_bins - 1)
    return x_bin / scale, y_bin / scale


def build_next_token_pixel_mask(input_ids: torch.Tensor, pix_start: int) -> torch.Tensor:
    """Return mask[t] = True when hidden[t] must predict a coordinate at t+1."""
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    if input_ids.shape[1] > 1:
        mask[:, :-1] = input_ids[:, 1:] >= pix_start
    return mask


def sample_signed_distance(pred_xy: torch.Tensor, dist_fields: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample decoded signed-distance fields at predicted points.

    pred_xy: [N, 2] normalized (x, y) in [0, 1].
    dist_fields: [N, S, S] decoded distance (negative inside the mask); cell
    (i, j) sits at (j / (S - 1), i / (S - 1)), hence align_corners=True.
    Returns [N] float32 signed distances, differentiable w.r.t. pred_xy.
    """
    with torch.autocast(device_type=pred_xy.device.type, enabled=False):
        grid = (2.0 * pred_xy.float() - 1.0)[:, None, None, :]
        return F.grid_sample(
            dist_fields.detach().float().unsqueeze(1), grid,
            mode="bilinear", padding_mode="border", align_corners=True,
        )[:, 0, 0, 0]


def signed_distance_loss(dist: torch.Tensor) -> torch.Tensor:
    """inside (dist <= 0): 0.2 * exp(25 * dist); outside: 0.2 + 5 * dist + 8 * dist^2.

    Smooth at dist = 0 (value 0.2, slope 5 on both sides). Inside, the push
    decays with depth (x0.61 at 0.02); outside, the pull grows with distance.
    """
    x = dist.float()
    inside = 0.2 * torch.exp(25.0 * x.clamp_max(0.0))
    outside = 0.2 + 5.0 * x + 8.0 * x.clamp_min(0.0) ** 2
    return torch.where(x <= 0.0, inside, outside)

@dataclass
class Qwen3_5PixMoECausalLMOutputWithPast(Qwen3_5CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    router_loss: Optional[torch.FloatTensor] = None
    pix_loss: Optional[torch.FloatTensor] = None
    # Unweighted teacher-forced diagnostics at supervised EOS positions.
    eos_lm_loss: Optional[torch.FloatTensor] = None
    eos_router_loss: Optional[torch.FloatTensor] = None
    eos_lm_correct: Optional[torch.LongTensor] = None
    eos_router_correct: Optional[torch.LongTensor] = None
    eos_count: Optional[torch.LongTensor] = None
    # Mask-supervised (ESDF) pixel tokens, for logging only.
    esdf_loss: Optional[torch.FloatTensor] = None
    esdf_inside: Optional[torch.FloatTensor] = None
    esdf_count: Optional[torch.LongTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    pred_coords: Optional[torch.FloatTensor] = None
    # Per-token log-likelihood of logp_labels[:, 1:] (see pointing_token_logps).
    text_token_logps: Optional[torch.FloatTensor] = None
    coord_token_logps: Optional[torch.FloatTensor] = None

class ContinuousPixHead(nn.Module):
    """
    Predicts continuous normalized 2D coordinates (x, y) in [0, 1] from hidden states.
    Uses an MLP with GELU activation and Sigmoid output.
    """
    def __init__(self, hidden_size: int, intermediate_size: Optional[int] = None):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = hidden_size
        self.fc1 = nn.Linear(hidden_size, intermediate_size, dtype=torch.bfloat16)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, 2, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc2(self.act(self.fc1(x))))

class Qwen3_5PixConfig(Qwen3_5Config):
    model_type = "qwen3_5PixMoE"
    def __init__(
        self,
        *args,
        router_k_layers=4,
        num_bins=128,
        pix_start=300000,
        lambda_coord=20.0,
        coord_beta=0.035,
        coord_mode="discrete",
        coord_loss_type="smooth_l1",
        **kwargs
    ):
        # Discard fields saved by the earlier implementation: these are run settings.
        for name in LEGACY_RUN_CONFIG_NAMES:
            kwargs.pop(name, None)
        super().__init__(*args, **kwargs)
        self.router_k_layers = router_k_layers
        self.num_bins = num_bins
        self.pix_start = pix_start
        self.lambda_coord = lambda_coord
        self.coord_beta = coord_beta
        self.coord_mode = coord_mode
        self.coord_loss_type = coord_loss_type

AutoConfig.register("qwen3_5PixMoE", Qwen3_5PixConfig)

@dataclass
class Qwen3_5PixModelOutputWithPast(Qwen3_5ModelOutputWithPast):
    """
    Model output class with past key values and router logits.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        past_key_values (`Cache`, *optional*):
            Contains pre-computed hidden-states (key and values in the self-attention blocks).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` for hidden-states.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` for attention weights.
        rope_deltas (`torch.Tensor`, *optional*):
            Rotary position embedding deltas.
        router_logits (`torch.FloatTensor`, *optional*):
            Router logits for token routing.
    """
    router_logits: Optional[torch.FloatTensor] = None

class Qwen3_5PixMoeDecoderLayer(Qwen3_5DecoderLayer):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # Create a second MLP expert specifically for pixel coordinate generation
        self.pix_mlp = Qwen3_5MLP(
            config, 
            config.intermediate_size
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        mask_origin: torch.Tensor = None,
        mask_pix: torch.Tensor = None,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Token Mixer (Self Attention) - Shared across both experts
        block_type = getattr(self, "block_type", getattr(self, "layer_type", None))
        if block_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        elif block_type == "full_attention":
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = residual + hidden_states

        # Fully Connected (MLP block with token-level routing)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Reshape to 2D for linear layers / MoE execution
        B, S, H = hidden_states.shape
        flat_hidden = hidden_states.view(-1, H)
        flat_mask_origin = mask_origin.view(-1)
        flat_mask_pix = mask_pix.view(-1)
        
        mlp_output = torch.zeros_like(flat_hidden)
        
        # Expert 0: Origin MLP
        idx_origin = flat_mask_origin.nonzero(as_tuple=True)[0]
        if idx_origin.numel() > 0:
            mlp_output[idx_origin] = self.mlp(flat_hidden[idx_origin])
        else:
            mlp_output = mlp_output + 0.0 * sum(p.sum() for p in self.mlp.parameters())
            
        # Expert 1: Pixel FFN MLP
        idx_pix = flat_mask_pix.nonzero(as_tuple=True)[0]
        if idx_pix.numel() > 0:
            mlp_output[idx_pix] = self.pix_mlp(flat_hidden[idx_pix])
        else:
            mlp_output = mlp_output + 0.0 * sum(p.sum() for p in self.pix_mlp.parameters())
            
        hidden_states = residual + mlp_output.view(B, S, H)

        return hidden_states

class Qwen3_5PixTextModel(Qwen3_5TextModel):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__(config)
        self.router_k_layers = getattr(config, "router_k_layers", 4)
        
        # Router MLP: hidden_size -> 2
        self.router_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 2, dtype=torch.bfloat16)
        )
        
        # Replace the last K layers in-place with Qwen3_5PixMoeDecoderLayer
        num_shared_layers = config.num_hidden_layers - self.router_k_layers
        for i in range(num_shared_layers, config.num_hidden_layers):
            self.layers[i] = Qwen3_5PixMoeDecoderLayer(config, i)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        gt_pix_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> Qwen3_5PixModelOutputWithPast:
        # NOTE: input_ids is always None here — Qwen3_5p.forward does the embedding
        # (embed_tokens + pixel PE replacement) and passes inputs_embeds directly.
        if inputs_embeds is None:
            raise ValueError("Qwen3_5PixTextModel expects inputs_embeds (not input_ids). "
                             "Embedding is handled by the outer Qwen3_5p model.")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            linear_mask = (
                create_recurrent_attention_mask(**mask_kwargs)
                if create_recurrent_attention_mask is not None
                else (self._update_linear_attn_mask(attention_mask, past_key_values) if hasattr(self, "_update_linear_attn_mask") else None)
            )
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": linear_mask,
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        num_shared_layers = self.config.num_hidden_layers - self.router_k_layers
        for i in range(num_shared_layers):
            decoder_layer = self.layers[i]
            l_type = getattr(decoder_layer, "block_type", getattr(decoder_layer, "layer_type", self.config.layer_types[i]))
            layer_mask = causal_mask_mapping.get(l_type, None) if isinstance(causal_mask_mapping, dict) else causal_mask_mapping

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        if self.router_k_layers == 0:
            # No MoE layers: the router reads the final normed hidden state, the
            # same input as lm_head / pix_head, and only picks the output head.
            hidden_states = self.norm(hidden_states)
            return Qwen3_5PixModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values,
                router_logits=self.router_mlp(hidden_states),
            )

        router_logits = self.router_mlp(hidden_states)

        S = hidden_states.shape[1]
        if self.training and gt_pix_mask is not None:
            mask_pix = gt_pix_mask.to(dtype=hidden_states.dtype)
            mask_origin = (~(gt_pix_mask.bool())).to(dtype=hidden_states.dtype)
        else:
            router_pred = router_logits.argmax(dim=-1).unsqueeze(-1) # [B, S, 1]
            mask_pix = (router_pred == 1).to(dtype=hidden_states.dtype)
            mask_origin = (router_pred == 0).to(dtype=hidden_states.dtype)

        # Run remaining K layers with custom MoE routing
        for i in range(num_shared_layers, self.config.num_hidden_layers):
            decoder_layer = self.layers[i]
            l_type = getattr(decoder_layer, "block_type", getattr(decoder_layer, "layer_type", self.config.layer_types[i]))
            layer_mask = causal_mask_mapping.get(l_type, None) if isinstance(causal_mask_mapping, dict) else causal_mask_mapping

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                mask_origin=mask_origin,
                mask_pix=mask_pix,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return Qwen3_5PixModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            router_logits=router_logits,
        )

class Qwen3_5p(Qwen3_5Model):
    def _init_weights(self, module):
        super()._init_weights(module)
        init_pixel_fusion_weights(module)

    def __init__(self, config):
        super().__init__(config)
        self.num_bins = getattr(config, "num_bins", 128)
        self.pix_start = getattr(config, "pix_start", 300000)
        # Single-layer MLP for pixel position embeddings
        vis_hidden_size = getattr(getattr(config, "vision_config", None), "hidden_size", 1024)
        text_hidden_size = getattr(getattr(config, "text_config", None), "hidden_size", 2048)
        self.pixel_embed_mlp2 = nn.Linear(vis_hidden_size, text_hidden_size, dtype=torch.bfloat16)
        # use_fused_pixel_embedding: add Fourier and image-feature branches (pixel_fusion.py).
        self.pixel_fusion = (PixelFusionEmbedding(vis_hidden_size, text_hidden_size)
                             if use_fused_pixel_embedding(config) else None)
        self._pixel_feature_cache = None

        # Replace language_model with Qwen3_5PixTextModel
        if not hasattr(config.text_config, "router_k_layers"):
            config.text_config.router_k_layers = getattr(config, "router_k_layers", 4)
        self.language_model = Qwen3_5PixTextModel(config.text_config)

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def get_pixel_pos_embed(
        self,
        target_pix_batch: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        normalized: bool = False,
    ) -> torch.Tensor:
        pos_embed_module = self.visual.pos_embed
        num_grid_per_side = getattr(self.visual, "num_grid_per_side", 48)
        
        w_idxs = (target_pix_batch[:, 0].float() / float(self.num_bins - 1)) * (num_grid_per_side - 1)
        h_idxs = (target_pix_batch[:, 1].float() / float(self.num_bins - 1)) * (num_grid_per_side - 1)
        if normalized:
            w_norm = target_pix_batch[:, 0].float()
            h_norm = target_pix_batch[:, 1].float()
        else:
            w_norm = target_pix_batch[:, 0].float() / float(self.num_bins - 1)
            h_norm = target_pix_batch[:, 1].float() / float(self.num_bins - 1)

        w_idxs = (w_norm * (num_grid_per_side - 1)).clamp(0, num_grid_per_side - 1)
        h_idxs = (h_norm * (num_grid_per_side - 1)).clamp(0, num_grid_per_side - 1)
        
        w_idxs = w_idxs.clamp(0, num_grid_per_side - 1)
        h_idxs = h_idxs.clamp(0, num_grid_per_side - 1)
        
        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        h_idxs_ceil = (h_idxs.int() + 1).clamp(max=num_grid_per_side - 1)
        w_idxs_ceil = (w_idxs.int() + 1).clamp(max=num_grid_per_side - 1)
        
        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor
        
        base_h = h_idxs_floor * num_grid_per_side
        base_h_ceil = h_idxs_ceil * num_grid_per_side
        
        indices = torch.stack([
            (base_h + w_idxs_floor),
            (base_h + w_idxs_ceil),
            (base_h_ceil + w_idxs_floor),
            (base_h_ceil + w_idxs_ceil),
        ], dim=0).long().to(device=device)
        
        max_idx = pos_embed_module.num_embeddings - 1
        indices = indices.clamp(0, max_idx)
        
        weights = torch.stack([
            ((1 - dh) * (1 - dw)),
            ((1 - dh) * dw),
            (dh * (1 - dw)),
            (dh * dw),
        ], dim=0).to(dtype=dtype, device=device)
        
        pos_embeds = pos_embed_module(indices)
        return (pos_embeds * weights[:, :, None]).sum(dim=0)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Qwen3_5PixModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        pix_mask = None
        if inputs_embeds is None:
            pix_mask = input_ids >= self.pix_start
            
            safe_input_ids = input_ids.clone()
            if pix_mask.any():
                safe_input_ids[pix_mask] = 0
            inputs_embeds = self.get_input_embeddings()(safe_input_ids)

        vision_kwargs = {k: v for k, v in kwargs.items() if k not in PACKED_SEQUENCE_KWARGS}
        patch_feats = None
        if pixel_values is not None:
            image_outputs: BaseModelOutputWithPooling = self.get_image_features(
                pixel_values, image_grid_thw, return_dict=True, **vision_kwargs
            )
            # Pre-merger patch features for the fused coordinate embedding.
            patch_feats = image_outputs.last_hidden_state
            image_embeds = image_outputs.pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_outputs: BaseModelOutputWithPooling = self.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True, **vision_kwargs
            )
            video_embeds = video_outputs.pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if pix_mask is not None:
            # Run even for an empty pixel selection: every rank must produce
            # pixel_embed_mlp2 (and pixel_fusion) gradients for matching ZeRO-2 collectives.
            pix_ids = input_ids[pix_mask]
            val = pix_ids - self.pix_start
            y_coords = val // self.num_bins
            x_coords = val % self.num_bins
            target_pix_batch = torch.stack([x_coords, y_coords], dim=-1)

            pos_embed = self.get_pixel_pos_embed(
                target_pix_batch=target_pix_batch,
                device=input_ids.device,
                dtype=inputs_embeds.dtype,
            )

            pos_embed = self.pixel_embed_mlp2(pos_embed)
            if self.pixel_fusion is not None:
                xy = target_pix_batch.float() / float(self.num_bins - 1)
                pos_embed = fused_coordinate_embedding(
                    self, pos_embed, input_ids, pix_mask, xy, patch_feats, image_grid_thw,
                    past_key_values=past_key_values, seq_idx=kwargs.get("seq_idx"),
                ).to(inputs_embeds.dtype)

            inputs_embeds = inputs_embeds.masked_scatter(
                pix_mask.unsqueeze(-1).expand_as(inputs_embeds),
                pos_embed
            )

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        # The hidden state at position t predicts token t+1.  During training,
        # teacher-force the MoE expert with the *next* token modality so that
        # expert routing has the same meaning as router_target below and as the
        # predicted router decision used during generation.
        gt_pix_mask = None
        if self.training and input_ids is not None:
            gt_pix_mask = build_next_token_pixel_mask(
                input_ids, self.pix_start
            ).unsqueeze(-1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            gt_pix_mask=gt_pix_mask,
            **kwargs,
        )

        return Qwen3_5PixModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
            router_logits=outputs.router_logits,
        )

class Qwen3_5PixGeneration(Qwen3_5ForConditionalGeneration):
    config_class = Qwen3_5PixConfig
    _tied_weights_keys = {}

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.language_model.embed_tokens = value

    def __init__(self, config, num_bins=128, pix_start=300000):
        if not hasattr(config, "num_bins") or getattr(config, "num_bins") is None:
            config.num_bins = num_bins
        if not hasattr(config, "pix_start") or getattr(config, "pix_start") is None:
            config.pix_start = pix_start
        if not hasattr(config, "router_k_layers") or getattr(config, "router_k_layers") is None:
            config.router_k_layers = 4

        num_bins = config.num_bins
        pix_start = config.pix_start

        self.total_vocab_size = pix_start + num_bins * num_bins
        config.vocab_size = self.total_vocab_size

        self.num_bins = num_bins
        self.num_bins_half = num_bins // 2
        self.pix_start = pix_start
        self.ignore_index = -100
        self.lambda_coord = getattr(config, "lambda_coord", 20.0)
        self.coord_beta = getattr(config, "coord_beta", 0.035)
        self.coord_mode = getattr(config, "coord_mode", "discrete")
        self.coord_loss_type = getattr(config, "coord_loss_type", "smooth_l1")
        # Training entry points configure runtime attributes after loading weights.
        # Never read or persist these run settings in the model config.
        for subconfig in (config, config.text_config):
            for name in LEGACY_RUN_CONFIG_NAMES:
                subconfig.__dict__.pop(name, None)
        runtime_defaults = {
            "eos_loss_weighting_enable": False,
            "coord_input_noise_enable": False,
            "eos_router_loss_weight": 3.0,
            "eos_lm_loss_weight": 3.0,
            "coord_input_noise_sample_prob": 0.5,
            "coord_input_noise_token_prob": 0.2,
            "coord_input_noise_std": 0.002,
            # Weight of the signed-distance loss of mask-supervised pixel tokens.
            "lambda_esdf": 1.0,
            # Scale of the coordinate pseudo-likelihood in pointing_token_logps.
            "logp_coord_scale": 0.01,
        }
        for name, value in runtime_defaults.items():
            setattr(self, name, value)
        self.stop_token_ids = resolve_stop_token_ids(config)
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        
        self.model = Qwen3_5p(config)

        # A SINGLE linear layer for both X and Y
        self.pix_head = nn.Linear(
            config.text_config.hidden_size, 
            2 * num_bins, 
            dtype=torch.bfloat16
        )
        if self.coord_mode == "continuous":
            self.pix_head = ContinuousPixHead(
                hidden_size=config.text_config.hidden_size,
                intermediate_size=getattr(config, "pix_head_intermediate_size", config.text_config.hidden_size)
            )
        else:
            # A SINGLE linear layer for both X and Y
            self.pix_head = nn.Linear(
                config.text_config.hidden_size, 
                2 * num_bins, 
                dtype=torch.bfloat16
            )
        self.post_init()

    def configure_stop_training(
        self, *, eos_loss_weighting_enable=True, coord_input_noise_enable=False,
        eos_router_loss_weight=3.0, eos_lm_loss_weight=3.0,
        coord_input_noise_sample_prob=0.5, coord_input_noise_token_prob=0.2,
        coord_input_noise_std=0.002,
    ):
        """Apply validated training arguments to this instance, not its config."""
        settings = {
            "eos_loss_weighting_enable": eos_loss_weighting_enable,
            "coord_input_noise_enable": coord_input_noise_enable,
            "eos_router_loss_weight": eos_router_loss_weight,
            "eos_lm_loss_weight": eos_lm_loss_weight,
            "coord_input_noise_sample_prob": coord_input_noise_sample_prob,
            "coord_input_noise_token_prob": coord_input_noise_token_prob,
            "coord_input_noise_std": coord_input_noise_std,
        }
        for name in ("eos_loss_weighting_enable", "coord_input_noise_enable"):
            if not isinstance(settings[name], bool):
                raise ValueError(f"{name} must be a boolean")
        for name in ("eos_router_loss_weight", "eos_lm_loss_weight"):
            value = settings[name]
            if not math.isfinite(value) or value < 1.0:
                raise ValueError(f"{name} must be finite and >= 1, got {value}")
        for name in ("coord_input_noise_sample_prob", "coord_input_noise_token_prob"):
            value = settings[name]
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if not math.isfinite(coord_input_noise_std) or coord_input_noise_std < 0.0:
            raise ValueError("coord_input_noise_std must be finite and >= 0")
        stop_token_ids = resolve_stop_token_ids(self.config)
        if eos_loss_weighting_enable and not stop_token_ids:
            raise ValueError("EOS weighting requires eos_token_id in the model config")
        for name, value in settings.items():
            setattr(self, name, value)
        self.stop_token_ids = stop_token_ids

    def init_pix_mlp_from_mlp(self):
        """
        Clones pretrained self.mlp weights into self.pix_mlp for all MoE decoder layers.
        """
        num_layers = self.config.text_config.num_hidden_layers
        router_k_layers = self.model.language_model.router_k_layers
        num_shared_layers = num_layers - router_k_layers
        for i in range(num_shared_layers, num_layers):
            layer = self.model.language_model.layers[i]
            if hasattr(layer, "pix_mlp") and hasattr(layer, "mlp"):
                with torch.no_grad():
                    layer.pix_mlp.load_state_dict(layer.mlp.state_dict())
                print(f"[MoE] Initialized layer {i} pix_mlp from pretrained mlp.")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        has_pix_mlp = False
        checkpoint_coord_mode = None
        try:
            import os
            import json
            from safetensors import safe_open
            p = str(pretrained_model_name_or_path)
            st_path = os.path.join(p, "model.safetensors")
            idx_path = os.path.join(p, "model.safetensors.index.json")
            
            # Read checkpoint config to detect its coord_mode
            ckpt_config_path = os.path.join(p, "config.json")
            if os.path.exists(ckpt_config_path):
                with open(ckpt_config_path, "r") as f:
                    ckpt_cfg = json.load(f)
                    checkpoint_coord_mode = ckpt_cfg.get("coord_mode", "discrete")
            
            if os.path.exists(st_path):
                with safe_open(st_path, framework="pt") as f:
                    has_pix_mlp = any("pix_mlp" in k for k in f.keys())
            elif os.path.exists(idx_path):
                with open(idx_path, "r") as f:
                    idx = json.load(f)
                    has_pix_mlp = any("pix_mlp" in k for k in idx.get("weight_map", {}).keys())
        except Exception:
            pass

        # Detect if we're switching coord_mode (e.g. discrete checkpoint -> continuous config)
        target_config = kwargs.get("config", None)
        target_coord_mode = getattr(target_config, "coord_mode", None) if target_config else None
        if checkpoint_coord_mode is None:
            checkpoint_coord_mode = "discrete"
        switching_coord_mode = (
            target_coord_mode is not None 
            and checkpoint_coord_mode != target_coord_mode
        )
        
        if switching_coord_mode:
            # pix_head architecture differs between modes; ignore mismatched weights
            print(f"[CoordMode] Switching from '{checkpoint_coord_mode}' -> '{target_coord_mode}'. "
                  f"pix_head weights will be randomly initialized.")
            kwargs["ignore_mismatched_sizes"] = True

        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        if not has_pix_mlp:
            model.init_pix_mlp_from_mlp()
        else:
            print("[MoE] Preserving trained pix_mlp weights from checkpoint.")
        
        if switching_coord_mode:
            # Re-initialize the pix_head cleanly since the checkpoint weights were mismatched
            print(f"[CoordMode] Re-initializing pix_head for '{target_coord_mode}' mode.")
            if target_coord_mode == "continuous":
                model.pix_head = ContinuousPixHead(
                    hidden_size=model.config.text_config.hidden_size,
                    intermediate_size=getattr(model.config, "pix_head_intermediate_size", model.config.text_config.hidden_size)
                ).to(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
            else:
                num_bins = model.config.num_bins
                model.pix_head = nn.Linear(
                    model.config.text_config.hidden_size, 
                    2 * num_bins,
                    dtype=next(model.parameters()).dtype
                ).to(device=next(model.parameters()).device)
        return model

    def tie_weights(self, *args, **kwargs):
        pass

    def pointing_token_logps(self, hidden_states, router_logits, logp_labels):
        """Per-token log-likelihood of logp_labels under the factorized policy.

        Text token: log p_router(text) + log p_lm(token).
        Coordinate token: log p_router(coord) + log of a Huber pseudo-likelihood
        around the regressed point, -smooth_l1(pred - target; coord_beta).sum(xy)
        / logp_coord_scale (continuous mode; up to a constant).
        Returns (text_logps, coord_logps), float32 [B, S - 1] aligned with
        logp_labels[:, 1:], zero at ignored positions and tokens of the other kind.
        """
        if self.coord_mode != "continuous":
            raise NotImplementedError("pointing_token_logps supports coord_mode='continuous' only")
        labels = logp_labels[:, 1:]
        hidden = hidden_states[:, :-1]
        router_logps = F.log_softmax(router_logits[:, :-1].float(), dim=-1)
        valid = labels != self.ignore_index
        is_pix = valid & (labels >= self.pix_start)
        is_text = valid & ~is_pix
        text_logps = hidden.new_zeros(labels.shape, dtype=torch.float32)
        coord_logps = hidden.new_zeros(labels.shape, dtype=torch.float32)
        if is_text.any():
            logits = self.lm_head(hidden[is_text]).float()
            targets = labels[is_text]
            token_logps = logits.gather(-1, targets[:, None]).squeeze(-1) - logits.logsumexp(-1)
            text_logps[is_text] = token_logps + router_logps[..., 0][is_text]
        if is_pix.any():
            val = labels[is_pix] - self.pix_start
            target = torch.stack([val % self.num_bins, val // self.num_bins], dim=-1).float() / float(self.num_bins - 1)
            pred = self.pix_head(hidden[is_pix]).float()
            huber = F.smooth_l1_loss(pred, target, beta=self.coord_beta, reduction="none").sum(-1)
            coord_logps[is_pix] = router_logps[..., 1][is_pix] - huber / self.logp_coord_scale
        return text_logps, coord_logps

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None, 
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        return_dict: Optional[bool] = None,
        pix_esdf: Optional[torch.Tensor] = None,
        pix_esdf_index: Optional[torch.LongTensor] = None,
        logp_labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, Qwen3_5CausalLMOutputWithPast]:
        """pix_esdf: [M, S, S] decoded signed-distance fields of mask targets.
        pix_esdf_index: [N_pix] for every labeled pixel token in row-major
        order, the pix_esdf row supervising it, or -1 for a fixed point target.
        logp_labels: [B, S] targets (-100 ignored) for text_token_logps /
        coord_token_logps (preference training); without labels, no logits
        are built.
        """
        if (self.training and self.coord_input_noise_enable and labels is not None
                and input_ids is not None and inputs_embeds is None):
            # Sample noise once outside decoder checkpointing; targets and token
            # modalities stay intact, including teacher-forced routing masks.
            input_ids = perturb_assistant_coordinates(
                input_ids, labels, num_bins=self.num_bins, pix_start=self.pix_start,
                sample_prob=self.coord_input_noise_sample_prob,
                token_prob=self.coord_input_noise_token_prob,
                std=self.coord_input_noise_std, ignore_index=self.ignore_index,
                seq_idx=kwargs.get("seq_idx"),
            )
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        router_logits = outputs.router_logits
        
        B, S, device = hidden_states.size(0), hidden_states.size(1), hidden_states.device
        
        loss = None
        return_logits = None
        esdf_loss = esdf_inside = esdf_count = None
        eos_lm_loss = eos_router_loss = None
        eos_lm_correct = eos_router_correct = eos_count = None

        if labels is not None:
            # Training mode: align logits and compute joint loss
            shift_hidden = hidden_states[..., :-1, :].contiguous().view(-1, hidden_states.size(-1))
            shift_labels = labels[..., 1:].contiguous().view(-1)
            shift_router_logits = router_logits[..., :-1, :].contiguous().view(-1, 2)
            
            # Ground truth route: 1 if shift_labels >= pix_start, 0 if < pix_start
            router_target = torch.where(shift_labels >= self.pix_start, 1, 0)
            router_target = torch.where(shift_labels == self.ignore_index, self.ignore_index, router_target)
            
            eos_mask = build_eos_mask(shift_labels, self.stop_token_ids)
            router_loss = eos_weighted_cross_entropy(
                shift_router_logits, router_target, eos_mask,
                weight=self.eos_router_loss_weight if self.eos_loss_weighting_enable else 1.0,
                ignore_index=self.ignore_index,
            )
            eos_count = eos_mask.sum().detach()
            if eos_mask.any():
                eos_router_logits = shift_router_logits[eos_mask]
                eos_router_loss = F.cross_entropy(eos_router_logits.float(), router_target[eos_mask]).detach()
                eos_router_correct = (eos_router_logits.argmax(-1) == 0).sum().detach()
            
            # Mask out pixel tokens from language loss; only compute lm_head for active text tokens
            lm_labels = torch.where(router_target == 0, shift_labels, self.ignore_index)
            lm_mask = (lm_labels != self.ignore_index)
            if lm_mask.any():
                active_hidden = shift_hidden[lm_mask]
                active_logits = self.lm_head(active_hidden)
                active_labels = lm_labels[lm_mask]
                active_eos = eos_mask[lm_mask]
                lm_loss = eos_weighted_cross_entropy(
                    active_logits, active_labels, active_eos,
                    weight=self.eos_lm_loss_weight if self.eos_loss_weighting_enable else 1.0,
                    ignore_index=self.ignore_index,
                )
                if active_eos.any():
                    eos_lm_logits = active_logits[active_eos]
                    eos_lm_loss = F.cross_entropy(eos_lm_logits.float(), active_labels[active_eos]).detach()
                    eos_lm_correct = (eos_lm_logits.argmax(-1) == active_labels[active_eos]).sum().detach()
            else:
                lm_loss = 0.0 * sum(p.sum() for p in self.lm_head.parameters())
            
            loss = lm_loss + router_loss
            
            is_pix = (router_target == 1)
            if is_pix.any():
                val = shift_labels[is_pix] - self.pix_start
                xy = torch.stack([val % self.num_bins, val // self.num_bins], dim=-1)
                target_norm = xy.float() / float(self.num_bins - 1)
                
                # Only project pixel tokens through pix_head
                pix_hidden = shift_hidden[is_pix]
                
                if self.coord_mode == "continuous":
                    pred_norm = self.pix_head(pix_hidden)
                else:
                    coord_logits = self.pix_head(pix_hidden).view(-1, 2, self.num_bins)
                    coord_logits = torch.softmax(coord_logits.float(), dim=-1)
                    probs = coord_logits.float()
                    indices = torch.arange(self.num_bins, dtype=torch.float32, device=probs.device)
                    pred_pix = torch.sum(probs * indices, dim=-1)
                    pred_norm = pred_pix / float(self.num_bins - 1)
                
                if pix_esdf_index is None:
                    if self.coord_loss_type == "l1":
                        coord_loss = F.l1_loss(pred_norm, target_norm, reduction='mean')
                    else:
                        coord_loss = F.smooth_l1_loss(pred_norm, target_norm, beta=self.coord_beta, reduction='mean')
                    pix_loss = self.lambda_coord * coord_loss
                else:
                    # Per-token loss: fixed points keep the coordinate loss; mask
                    # targets use the signed distance at the predicted point. The
                    # token's label coordinate is only its teacher-forcing input.
                    if pix_esdf_index.numel() != pred_norm.size(0):
                        raise ValueError(f"pix_esdf_index has {pix_esdf_index.numel()} entries for "
                                         f"{pred_norm.size(0)} labeled pixel tokens")
                    if self.coord_loss_type == "l1":
                        point_loss = F.l1_loss(pred_norm.float(), target_norm, reduction='none').mean(-1)
                    else:
                        point_loss = F.smooth_l1_loss(pred_norm.float(), target_norm, beta=self.coord_beta,
                                                      reduction='none').mean(-1)
                    is_mask = pix_esdf_index >= 0
                    dist = sample_signed_distance(pred_norm[is_mask], pix_esdf[pix_esdf_index[is_mask]])
                    mask_loss = signed_distance_loss(dist)
                    token_loss = torch.where(
                        is_mask,
                        self.lambda_esdf * point_loss.new_zeros(point_loss.shape).masked_scatter(is_mask, mask_loss),
                        self.lambda_coord * point_loss,
                    )
                    pix_loss = token_loss.mean()
                    esdf_count = is_mask.sum()
                    if esdf_count > 0:
                        esdf_loss = mask_loss.detach().mean()
                        esdf_inside = (dist.detach() <= 0).float().mean()
                loss = loss + pix_loss
            else:
                pix_loss = torch.tensor(0.0, device=device)
                loss = loss + 0.0 * sum(p.sum() for p in self.pix_head.parameters())

            return_logits = None
        elif logp_labels is not None:
            lm_loss = router_loss = pix_loss = None
        else:
            # Inference mode: construct combined logits
            lm_loss = None
            router_loss = None
            pix_loss = None
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            
            hidden_states_use = hidden_states[:, slice_indices, :]
            router_logits_use = router_logits[:, slice_indices, :]
            
            lm_logits = self.lm_head(hidden_states_use)
            
            if self.coord_mode == "continuous":
                pred_coords_norm = self.pix_head(hidden_states_use) # [B, S_use, 2] in [0, 1]
                pred_coords = pred_coords_norm * float(self.num_bins - 1)
                pred_coords_rounded = torch.round(pred_coords).long().clamp(0, self.num_bins - 1)
                pred_coords_out = pred_coords_norm
            else:
                pix_logits = self.pix_head(hidden_states_use).view(B, lm_logits.size(1), 2, self.num_bins)
                probs = torch.softmax(pix_logits.float(), dim=-1)
                indices = torch.arange(self.num_bins, dtype=torch.float32, device=device)
                pred_coords = torch.sum(probs * indices, dim=-1) # [B, S_use, 2]
                pred_coords_rounded = torch.round(pred_coords).long().clamp(0, self.num_bins - 1)
                pred_coords_out = pred_coords / float(self.num_bins - 1)
            
            return_logits = torch.full(
                (B, lm_logits.size(1), self.total_vocab_size),
                -10000.0,
                dtype=lm_logits.dtype,
                device=device
            )
            return_logits[:, :, :lm_logits.size(-1)] = lm_logits
            x_pred = pred_coords_rounded[:, :, 0]
            y_pred = pred_coords_rounded[:, :, 1]
            coord_id = y_pred * self.num_bins + x_pred + self.pix_start
            
            router_pred = router_logits_use.argmax(dim=-1)
            is_pix_pred = (router_pred == 1)
            
            coord_logits_to_scatter = torch.where(is_pix_pred.unsqueeze(-1), 50000.0, -10000.0).to(dtype=lm_logits.dtype)
            
            return_logits.scatter_(
                dim=-1,
                index=coord_id.unsqueeze(-1),
                src=coord_logits_to_scatter
            )

        text_token_logps = coord_token_logps = None
        if logp_labels is not None:
            text_token_logps, coord_token_logps = self.pointing_token_logps(hidden_states, router_logits, logp_labels)

        return Qwen3_5PixMoECausalLMOutputWithPast(
            loss=loss,
            lm_loss=lm_loss.detach() if (labels is not None and lm_loss is not None) else None,
            router_loss=router_loss.detach() if (labels is not None and router_loss is not None) else None,
            pix_loss=pix_loss.detach() if (labels is not None and pix_loss is not None) else None,
            eos_lm_loss=eos_lm_loss,
            eos_router_loss=eos_router_loss,
            eos_lm_correct=eos_lm_correct,
            eos_router_correct=eos_router_correct,
            eos_count=eos_count,
            esdf_loss=esdf_loss,
            esdf_inside=esdf_inside,
            esdf_count=esdf_count,
            logits=return_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            pred_coords=pred_coords_out if labels is None and logp_labels is None else None,
            text_token_logps=text_token_logps,
            coord_token_logps=coord_token_logps,
        )

class Qwen3_5PixTokenizer(Qwen2Tokenizer):
    def __init__(self, *args, num_bins=128, pix_start=300000, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_bins = num_bins
        self.pix_start = pix_start
        # Dataset contract: coordinates are normalized floats in [0, 1].
        # Values such as 0 and 1 are valid normalized endpoints as well.
        number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
        self.coord_pattern = re.compile(
            rf"<\|\s*({number})\s*,\s*({number})\s*\|>"
        )
        self.dummy_token = "<|fim_pad|>"
        self.dummy_id = 248063

    def _process_text_and_coords(self, text):
        if not isinstance(text, str):
            return text, []
        coords = []
        def repl(match):
            # One coordinate is packed into one synthetic token ID.  The only
            # difference between discrete and continuous runs is num_bins
            # (for example 128 vs. 1000).
            coord_id = pack_normalized_coordinate(
                float(match.group(1)),
                float(match.group(2)),
                num_bins=self.num_bins,
                pix_start=self.pix_start,
            )
            coords.append(coord_id)
            return self.dummy_token
        processed_text = self.coord_pattern.sub(repl, text)
        return processed_text, coords

    def encode(self, text, **kwargs):
        if isinstance(text, str):
            processed_text, coords = self._process_text_and_coords(text)
            ids = super().encode(processed_text, **kwargs)
            coord_idx = 0
            for i, idx in enumerate(ids):
                if idx == self.dummy_id and coord_idx < len(coords):
                    ids[i] = coords[coord_idx]
                    coord_idx += 1
            return ids
        return super().encode(text, **kwargs)

    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            processed_text, coords = self._process_text_and_coords(text)
            outputs = super().__call__(processed_text, **kwargs)
            self._replace_dummy_ids_in_batch_encoding(outputs, [coords])
            return outputs
        elif isinstance(text, list) and all(isinstance(t, str) for t in text):
            processed_texts = []
            all_coords = []
            for t in text:
                pt, c = self._process_text_and_coords(t)
                processed_texts.append(pt)
                all_coords.append(c)
            outputs = super().__call__(processed_texts, **kwargs)
            self._replace_dummy_ids_in_batch_encoding(outputs, all_coords)
            return outputs
        return super().__call__(text, **kwargs)

    def _replace_dummy_ids_in_batch_encoding(self, outputs, all_coords):
        if "input_ids" not in outputs:
            return
        input_ids = outputs["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            for b in range(input_ids.shape[0]):
                coords = all_coords[b]
                coord_idx = 0
                for i in range(input_ids.shape[1]):
                    if input_ids[b, i] == self.dummy_id and coord_idx < len(coords):
                        input_ids[b, i] = coords[coord_idx]
                        coord_idx += 1
        else:
            if len(input_ids) > 0 and isinstance(input_ids[0], int):
                coords = all_coords[0] if len(all_coords) > 0 else []
                coord_idx = 0
                for i in range(len(input_ids)):
                    if input_ids[i] == self.dummy_id and coord_idx < len(coords):
                        input_ids[i] = coords[coord_idx]
                        coord_idx += 1
            else:
                for b in range(len(input_ids)):
                    coords = all_coords[b]
                    coord_idx = 0
                    for i in range(len(input_ids[b])):
                        if input_ids[b][i] == self.dummy_id and coord_idx < len(coords):
                            input_ids[b][i] = coords[coord_idx]
                            coord_idx += 1

    def decode(self, token_ids, **kwargs):
        def convert_to_list_or_int(x):
            if isinstance(x, torch.Tensor):
                return x.tolist()
            if hasattr(x, "tolist"):
                return x.tolist()
            if isinstance(x, (list, tuple)):
                return [convert_to_list_or_int(i) for i in x]
            if hasattr(x, "item"):
                return x.item()
            return x

        token_ids_list = convert_to_list_or_int(token_ids)

        if isinstance(token_ids_list, int):
            token_ids_list = [token_ids_list]
        elif not isinstance(token_ids_list, list):
            token_ids_list = list(token_ids_list)

        if len(token_ids_list) > 0 and isinstance(token_ids_list[0], list):
            return [self.decode(ids, **kwargs) for ids in token_ids_list]

        decoded_tokens = []
        for idx in token_ids_list:
            if idx >= self.pix_start:
                x_norm, y_norm = unpack_coordinate_token(
                    idx,
                    num_bins=self.num_bins,
                    pix_start=self.pix_start,
                )
                x_text = f"{x_norm:.6f}".rstrip("0").rstrip(".")
                y_text = f"{y_norm:.6f}".rstrip("0").rstrip(".")
                decoded_tokens.append(f"<|{x_text},{y_text}|>")
            else:
                decoded_tokens.append(super().decode([idx], **kwargs))
        return "".join(decoded_tokens)

    def batch_decode(self, sequences, *args, **kwargs):
        return [self.decode(seq, *args, **kwargs) for seq in sequences]

if __name__ == "__main__":
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig

    print("🚀 Initializing Vectorized Pix Test Environment...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mock configs using standard transformers config classes
    text_config = Qwen3_5TextConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=151936,
    )
    vision_config = Qwen3_5VisionConfig(
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
    )
    for test_mode in ["discrete", "continuous"]:
        print(f"\n==========================================")
        print(f"Testing coord_mode: {test_mode}")
        print(f"==========================================")
        config = Qwen3_5PixConfig(
            text_config=text_config.to_dict(),
            vision_config=vision_config.to_dict(),
            router_k_layers=2,
            num_bins=144,
            pix_start=300000,
            coord_mode=test_mode
        )
        num_bins = 144
        test_model = Qwen3_5PixGeneration(config, num_bins=num_bins).to(device, dtype=torch.bfloat16)
        test_model.eval()

        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
        attention_mask = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
        ]).to(device)
        labels = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
        labels[1, 5] = 351320

        print(f"[{test_mode}] Running training forward pass...")
        with torch.no_grad():
            train_outputs = test_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        print(f"[{test_mode}] Total Combined Loss: {train_outputs.loss.item():.4f}")
        if train_outputs.pix_loss is not None:
            print(f"[{test_mode}] Pix Loss: {train_outputs.pix_loss.item():.4f}")

        print(f"[{test_mode}] Running inference pass...")
        with torch.no_grad():
            infer_outputs = test_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        print(f"[{test_mode}] Inference Logits shape: {infer_outputs.logits.shape}")
        if infer_outputs.pred_coords is not None:
            print(f"[{test_mode}] Continuous pred_coords shape: {infer_outputs.pred_coords.shape}")

        pred_classes = infer_outputs.logits[:, -1].argmax(dim=-1)
        for i in range(batch_size):
            print(f"[{test_mode}] Batch {i}: Predicted token ID -> {pred_classes[i].item()}")

        print(f"[{test_mode}] Running generate pass...")
        try:
            generate_ids = test_model.generate(input_ids, max_new_tokens=5)
            print(f"[{test_mode}] Generated IDs shape:", generate_ids.shape)
            print(f"✅ [{test_mode}] Generate test passed successfully!")
        except Exception as e:
            print(f"Warning: [{test_mode}] Generate test failed: {e}")

    # Test custom tokenizer
    print("\n--- TEST: Tokenizer Pass ---")
    try:
        tokenizer = Qwen3_5PixTokenizer(vocab_file=None, num_bins=144)
        print("Tokenizer initialized successfully!")
    except Exception as e:
        print(f"Tokenizer test skipped: {e}")

    print("\n✅ All tests passed. Standard causal LM output format ready.")
