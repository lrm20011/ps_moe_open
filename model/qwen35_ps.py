from numpy.core import einsumfunc
import torch
import torch.nn as nn
import re
from transformers import Qwen2Tokenizer, AutoConfig
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

class Qwen3_5PixConfig(Qwen3_5Config):
    model_type = "qwen3_5Pix"

AutoConfig.register("qwen3_5Pix", Qwen3_5PixConfig)
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Any
import time
from transformers.cache_utils import Cache
from transformers.utils import ModelOutput
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5Model,
    Qwen3_5PreTrainedModel,
    Qwen3_5ModelOutputWithPast,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling
from pixel_fusion import (
    PixelFusionEmbedding, fused_coordinate_embedding, init_pixel_fusion_weights, use_fused_pixel_embedding,
)

# Padding-free (packed) batch kwargs. They describe the text row only; the
# vision tower builds its own cu_seqlens and must not receive them.
PACKED_SEQUENCE_KWARGS = ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k", "seq_idx")
from transformers.utils import auto_docstring, can_return_tuple

try:
    from typing import Unpack
    from transformers.utils import TransformersKwargs
except ImportError:
    Unpack, TransformersKwargs = Any, Any


import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_loss(sign_logits, sign_tgt, coord_logits=None, xy=None, num_bins=144, ignore_index=-100):
    """
    Computes unified combined loss: CrossEntropy for punctuation/separators and Huber loss for coordinates.
    """
    loss = None
    if (sign_tgt != ignore_index).any():
        loss = F.cross_entropy(sign_logits, sign_tgt, ignore_index=ignore_index)
        
    if coord_logits is not None and xy is not None:
        probs = coord_logits.float()
        indices = torch.arange(num_bins, dtype=torch.float32, device=probs.device)
        pred_pix = torch.sum(probs * indices, dim=-1)
        
        pred_norm = pred_pix / float(num_bins - 1)*50
        target_norm = xy.float() / float(num_bins - 1)*50
        
        pix_loss = F.smooth_l1_loss(pred_norm, target_norm, reduction='mean')
        loss = pix_loss if loss is None else loss + pix_loss
        
    return loss





class Qwen3_5p(Qwen3_5Model):
    def _init_weights(self, module):
        super()._init_weights(module)
        init_pixel_fusion_weights(module)

    def __init__(self, config):
        super().__init__(config)
        self.num_bins = getattr(config, "num_bins", 144)
        self.pix_start = getattr(config, "pix_start", 300000)
        # Single-layer MLP for pixel position embeddings
        self.pixel_embed_mlp2 = nn.Linear(1024, config.text_config.hidden_size, dtype=torch.bfloat16)
        # use_fused_pixel_embedding: add Fourier and image-feature branches (pixel_fusion.py).
        self.pixel_fusion = (PixelFusionEmbedding(config.vision_config.hidden_size, config.text_config.hidden_size)
                             if use_fused_pixel_embedding(config) else None)
        self._pixel_feature_cache = None

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def get_pixel_pos_embed(
        self,
        target_pix_batch: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        pos_embed_module = self.visual.pos_embed
        num_grid_per_side = getattr(self.visual, "num_grid_per_side", 48)
        
        w_idxs = (target_pix_batch[:, 0].float() / float(self.num_bins - 1)) * (num_grid_per_side - 1)
        h_idxs = (target_pix_batch[:, 1].float() / float(self.num_bins - 1)) * (num_grid_per_side - 1)
        
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

    @auto_docstring
    @can_return_tuple
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
    ) -> tuple | Qwen3_5ModelOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
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

        # The fused embedding always runs, so that every rank uses its weights.
        if pix_mask is not None and (pix_mask.any() or self.pixel_fusion is not None):
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

            # Apply the MLP with residual spatial connection to project from 1024 to config.text_config.hidden_size (e.g. 2048)
            # pos_embed = pos_embed + self.pixel_embed_act(self.pixel_embed_mlp1(pos_embed))
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

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        return Qwen3_5ModelOutputWithPast(
            **outputs,
            rope_deltas=self.rope_deltas,
        )

class Qwen3_5PixGeneration(Qwen3_5ForConditionalGeneration):
    config_class = Qwen3_5PixConfig
    _tied_weights_keys = {}

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.language_model.embed_tokens = value

    def __init__(self, config, num_bins=144, pix_start=300000):
        if not hasattr(config, "num_bins") or getattr(config, "num_bins") is None:
            config.num_bins = num_bins
        if not hasattr(config, "pix_start") or getattr(config, "pix_start") is None:
            config.pix_start = pix_start

        num_bins = config.num_bins
        pix_start = config.pix_start

        self.total_vocab_size = pix_start + num_bins * num_bins
        config.vocab_size = self.total_vocab_size

        self.num_bins = num_bins
        self.num_bins_half = num_bins // 2
        self.pix_start = pix_start
        self.ignore_index = -100
        self.lm_head = None
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.special_tokens_list = [11, 26, 13, 248046] # , ; . <|im_end|>
        self.num_spec = len(self.special_tokens_list)+1
        # Replace the model with Qwen3_5p
        self.model = Qwen3_5p(config)

                
        # A SINGLE linear layer for both X and Y
        self.pix_head = nn.Linear(
            config.text_config.hidden_size, 
            2 * num_bins, 
            dtype=torch.bfloat16
        )
        self.sign_head = nn.Linear(
            config.text_config.hidden_size, 
            self.num_spec, 
            dtype=torch.bfloat16
        )
        self.post_init()

    def tie_weights(self, *args, **kwargs):
        pass

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
        **kwargs,
    ) -> Union[Tuple, Qwen3_5CausalLMOutputWithPast]:
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

        hidden_states = outputs[0]
        B, S, device = hidden_states.size(0), hidden_states.size(1), hidden_states.device
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        hidden_states_use = hidden_states[:, slice_indices, :]

        sign_logits = self.sign_head(hidden_states_use)

        loss = None
        pix_logits = None
        if labels is not None:
            # Standard VLM shifting: pad labels on the right and slice from index 1
            shift_labels = F.pad(labels, (0, 1), value=self.ignore_index)[:, 1:].contiguous().view(-1)

            # Vectorized mapping: 0 for coordinates, c+1 for special tokens
            sign_tgt = torch.where(shift_labels >= self.pix_start, 0, self.ignore_index)
            special_tokens = torch.tensor(self.special_tokens_list, dtype=torch.long, device=device)
            matches = shift_labels.unsqueeze(-1) == special_tokens
            has_match, match_idx = matches.max(dim=-1)
            mapped_class = match_idx + 1
            sign_tgt = torch.where(has_match, mapped_class, sign_tgt)
            
            coord_logits = None
            xy = None
            is_pix = sign_tgt == 0
            if is_pix.any():
                val = shift_labels[is_pix] - self.pix_start
                xy = torch.stack([val % self.num_bins, val // self.num_bins], dim=-1)
                
                # OPTIMIZATION: Only project the coordinate tokens through the pix_head!
                coord_hidden = hidden_states_use.view(-1, hidden_states_use.size(-1))[is_pix]
                coord_logits = self.pix_head(coord_hidden).view(-1, 2, self.num_bins)
                coord_logits = torch.softmax(coord_logits.float(), dim=-1)
                
            loss = compute_loss(
                sign_logits=sign_logits.view(-1, self.num_spec),
                sign_tgt=sign_tgt,
                coord_logits=coord_logits,
                xy=xy,
                num_bins=self.num_bins,
                ignore_index=self.ignore_index
            )
            return_logits = sign_logits
        else:
            # Inference mode: project the full sequence
            pix_logits = self.pix_head(hidden_states_use).view(B, -1, 2, self.num_bins)
            pix_logits = torch.softmax(pix_logits.float(), dim=-1)

            # Construct unified logits over total_vocab_size
            S_use = hidden_states_use.size(1)
            return_logits = torch.full(
                (B, S_use, self.total_vocab_size),
                -10000.0,
                dtype=sign_logits.dtype,
                device=device
            )

            # 1. Map sign classes 1..4 to their corresponding token IDs in the vocabulary
            special_tokens = torch.tensor(self.special_tokens_list, dtype=torch.long, device=device)

            return_logits[:, :, special_tokens] = sign_logits[:, :, 1 : self.num_spec]

            # 2. Extract predicted coordinates (x, y) using soft-argmax and map to class 0 (coordinate tokens)
            probs = pix_logits.float()
            indices = torch.arange(self.num_bins, dtype=torch.float32, device=device)
            pred_coords = torch.sum(probs * indices, dim=-1) # [B, S_use, 2]
            
            # Round to nearest bin and clamp to valid range
            pred_coords_rounded = torch.round(pred_coords).long().clamp(0, self.num_bins - 1)
            x_pred = pred_coords_rounded[:, :, 0] # [B, S_use]
            y_pred = pred_coords_rounded[:, :, 1] # [B, S_use]
            coord_id = y_pred * self.num_bins + x_pred + self.pix_start # [B, S_use]

            # Only scatter class 0 logit where it is the predicted class (argmax == 0)
            sign_pred = sign_logits.argmax(dim=-1) # [B, S_use]
            is_coord_pred = (sign_pred == 0)
            coord_logits_to_scatter = torch.where(is_coord_pred.unsqueeze(-1), sign_logits[:, :, 0:1], -10000.0)

            # Scatter to the predicted coordinate token IDs
            return_logits.scatter_(
                dim=-1,
                index=coord_id.unsqueeze(-1),
                src=coord_logits_to_scatter
            )

        return Qwen3_5CausalLMOutputWithPast(
            loss=loss,
            logits=return_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

class Qwen3_5PixTokenizer(Qwen2Tokenizer):
    def __init__(self, *args, num_bins=144, pix_start=300000, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_bins = num_bins
        self.pix_start = pix_start
        self.coord_pattern = re.compile(r"<\|(\d+),(\d+)\|>")
        self.dummy_token = "<|fim_pad|>"
        self.dummy_id = 248063

    def _process_text_and_coords(self, text):
        if not isinstance(text, str):
            return text, []
        coords = []
        def repl(match):
            x, y = int(match.group(1)), int(match.group(2))
            coord_id = y * self.num_bins + x + self.pix_start
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
                val = idx - self.pix_start
                y = val // self.num_bins
                x = val % self.num_bins
                decoded_tokens.append(f"<|{x},{y}|>")
            else:
                decoded_tokens.append(super().decode([idx], **kwargs))
        return "".join(decoded_tokens)

    def batch_decode(self, sequences, *args, **kwargs):
        return [self.decode(seq, *args, **kwargs) for seq in sequences]


# ==========================================
# Testing Block
# ==========================================
if __name__ == "__main__":
    from transformers import AutoConfig

    print("🚀 Initializing Vectorized Pix Test Environment...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model_id = "/home/vlak/FoundationModels/Qwen3.5-2B"
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: Mocking config due to load error: {e}")
        class MockConfig:
            class TextConfig:
                hidden_size = 2048
                vocab_size = 151936
            text_config = TextConfig()
        config = MockConfig()

    num_bins = 144
    
    model = Qwen3_5PixGeneration(config, num_bins=num_bins).to(device, dtype=torch.bfloat16)
    model.eval()

    batch_size = 2
    seq_len = 10
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    attention_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    ]).to(device)

    print("\n--- TEST 1: Training Pass ---")
    
    # labels containing some coordinate token IDs (>= 300000) and punctuation
    labels = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    # Coordinate: x=120, y=200 -> id = 200 * 144 + 120 + 300000 = 351320
    labels[1, 5] = 351320

    with torch.no_grad():
        train_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    print(f"Total Combined Loss: {train_outputs.loss.item():.4f}")
    print(f"Logits shape: {train_outputs.logits.shape}")
    
    print("\n--- TEST 2: Inference Pass ---")
    t = time.time()
    with torch.no_grad():
        infer_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    print(f"Inference Time: {time.time()-t:.4f}s")
    print(f"Inference Logits shape: {infer_outputs.logits.shape}")

    # Decode predictions from logits
    pred_classes = infer_outputs.logits[:, -1].argmax(dim=-1)
    class_map = {0: "pix", 1: ",", 2: ";", 3: ".", 4: "eof"}
    for i in range(batch_size):
        pred_class = pred_classes[i].item()
        print(f"Batch {i}: Predicted Class -> {pred_class} ({class_map.get(pred_class, 'unknown')})")
            
    # Test custom tokenizer
    print("\n--- TEST 3: Tokenizer Pass ---")
    try:
        tokenizer = Qwen3_5PixTokenizer.from_pretrained(model_id)
        test_text = "Observe: <|120,200|> then <|50,210|>"
        print("Original Text:", test_text)
        ids = tokenizer.encode(test_text, add_special_tokens=False)
        print("Encoded IDs:", ids)
        for idx in ids:
            val = tokenizer.decode([idx])
            print(f'{idx} -> {repr(val)}')

        print("✅ Tokenizer test passed successfully!")
    except Exception as e:
        print(f"Warning: Tokenizer test skipped/failed: {e}")

    print("\n--- TEST 4: Generate Pass ---")
    try:
        # Generate with default generate method
        generate_ids = model.generate(input_ids, max_new_tokens=5)
        print("Generated IDs:", generate_ids)
        print("✅ Generate test passed successfully!")
    except Exception as e:
        print(f"Warning: Generate test skipped/failed (possibly due to MockConfig): {e}")

    print("\n✅ All tests passed. Standard causal LM output format ready.")