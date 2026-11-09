class Qwen3_5p(Qwen3_5Model):
    def __init__(self, config):
        super().__init__(config)
        self.num_bins = getattr(config, "num_bins", 128)
        self.pix_start = getattr(config, "pix_start", 300000)
        # Single-layer MLP for pixel position embeddings
        vis_hidden_size = getattr(getattr(config, "vision_config", None), "hidden_size", 1024)
        text_hidden_size = getattr(getattr(config, "text_config", None), "hidden_size", 2048)
        self.pixel_embed_mlp2 = nn.Linear(vis_hidden_size, text_hidden_size, dtype=torch.bfloat16)

        self.pix_build_weights = nn.Parameter(torch.zeros(3))
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

        if inputs_embeds is None:
            pix_mask = input_ids >= self.pix_start
            
            safe_input_ids = input_ids.clone()
            if pix_mask.any():
                safe_input_ids[pix_mask] = 0
            inputs_embeds = self.get_input_embeddings()(safe_input_ids)

            # Run even for an empty pixel selection: every rank must produce
            # pixel_embed_mlp2 gradients for matching ZeRO-2 collectives.
            pix_ids = input_ids[pix_mask]
            val = pix_ids - self.pix_start
            y_coords = val // self.num_bins
            x_coords = val % self.num_bins
            target_pix_batch = torch.stack([x_coords, y_coords], dim=-1)
            
            learn_pos_embed = self.get_pixel_pos_embed(
                target_pix_batch=target_pix_batch,
                device=input_ids.device,
                dtype=inputs_embeds.dtype,
            )
            


        vision_kwargs = {k: v for k, v in kwargs.items() if k not in PACKED_SEQUENCE_KWARGS}
        if pixel_values is not None:
            image_outputs: BaseModelOutputWithPooling = self.get_image_features(
                pixel_values, image_grid_thw, return_dict=True, **vision_kwargs
            )
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

        ax = self.pixel_embed_mlp2(learn_pos_embed)
        w = F.softmax(self.pix_build_weights, dim=0)
        s_pe= self.spe_mlp(cat( self.fourier_embed(x_coords), self.fourier_embed(y_coords), dim=1))
        img_pix_feat = self.img_pix_mlp(self.visual(pixel_values))
        pos_embed = w[0] * ax + w[1] * img_pix_feat + w[2] * s_pe
            
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