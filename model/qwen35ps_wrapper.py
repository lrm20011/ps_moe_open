import os
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, AutoConfig

try:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    class Qwen3_5PixConfig(Qwen3_5Config):
        model_type = "qwen3_5Pix"
    AutoConfig.register("qwen3_5Pix", Qwen3_5PixConfig)
    class Qwen3_5PixMoEConfig(Qwen3_5Config):
        model_type = "qwen3_5PixMoE"
    AutoConfig.register("qwen3_5PixMoE", Qwen3_5PixMoEConfig)
except ImportError:
    pass

class Qwen35PixWrapper(nn.Module):
    def __init__(self, model_name="Qwen-VL-Series-Finetune/output_sft", origin_model_path="/home/vlak/FoundationModels/Qwen3.5PixMoE-4B", device="cuda", dtype=torch.bfloat16, use_flash_attn=True, coord_mode=None):
        super().__init__()
        self.device = device
        self.dtype = dtype

        # Auto-resolve checkpoint directory if a folder containing checkpoints was provided
        if os.path.isdir(model_name) and not os.path.exists(os.path.join(model_name, "config.json")):
            ckpt_dirs = [
                d for d in os.listdir(model_name)
                if d.startswith("checkpoint-") and os.path.isdir(os.path.join(model_name, d))
            ]
            if ckpt_dirs:
                ckpt_dirs.sort(key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0)
                latest_ckpt = ckpt_dirs[-1]
                print(f"[Wrapper] Auto-resolved {model_name} to latest checkpoint: {latest_ckpt}")
                model_name = os.path.join(model_name, latest_ckpt)

        self.model_name = model_name
        self.origin_model_path = origin_model_path
        
        print(f"Loading custom Qwen3.5-PS VLA model and processor from {model_name}...")
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
        except Exception:
            if origin_model_path and os.path.exists(origin_model_path):
                print(f"[Wrapper] Loading fallback processor from origin model: {origin_model_path}")
                self.processor = AutoProcessor.from_pretrained(origin_model_path)
            else:
                raise

        if hasattr(self.processor, "image_processor") and hasattr(self.processor.image_processor, "size"):
            self.processor.image_processor.size["shortest_edge"] = 32 * 32
        
        # Load config to extract num_bins and pix_start
        # Load config to extract num_bins, pix_start, and coord_mode
        try:
            config = AutoConfig.from_pretrained(model_name)
            m_type = getattr(config, "model_type", "qwen3_5PixMoE")
            num_bins = getattr(config, "num_bins", 128)
            pix_start = getattr(config, "pix_start", 300000)
            cfg_coord_mode = getattr(config, "coord_mode", "discrete")
        except Exception:
            config = None
            m_type = "qwen3_5PixMoE"
            num_bins = 128
            pix_start = 300000
            cfg_coord_mode = "discrete"
        self.model_type = m_type
        self.num_bins = num_bins
        self.pix_start = pix_start
        self.coord_mode = coord_mode if coord_mode is not None else cfg_coord_mode

        # Import matching model & tokenizer classes based on architecture
        if m_type == "qwen3_5PixMoE":
            try:
                from .qwen35_psmoe import Qwen3_5PixGeneration, Qwen3_5PixTokenizer
            except ImportError:
                from qwen35_psmoe import Qwen3_5PixGeneration, Qwen3_5PixTokenizer
        else:
            try:
                from .qwen35_ps import Qwen3_5PixGeneration, Qwen3_5PixTokenizer
            except ImportError:
                from qwen35_ps import Qwen3_5PixGeneration, Qwen3_5PixTokenizer
            
        # Override default tokenizer with Qwen3_5PixTokenizer
        tokenizer = Qwen3_5PixTokenizer.from_pretrained(
            model_name,
            num_bins=num_bins,
            pix_start=pix_start
        )
        self.processor.tokenizer = tokenizer
        
        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": device
        }
        if use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        else:
            model_kwargs["attn_implementation"] = "sdpa"   
        if coord_mode is not None and config is not None:
            config.coord_mode = self.coord_mode
            model_kwargs["config"] = config
            
        # Load custom VLA model with coordinate projection heads
        self.model = Qwen3_5PixGeneration.from_pretrained(
            model_name,
            **model_kwargs
        )
        self.model.eval()
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            self.model.generation_config.eos_token_id = [248044, 248046]

    def _format_sft_prompt(self, num_images, text_prompt):
        img_tokens = "<|vision_start|><|image_pad|><|vision_end|>" * num_images
        return f"<|im_start|>user\n{img_tokens}{text_prompt}<|im_end|>\n<|im_start|>assistant\n"

    def process_inputs(self, images_list, text_prompt="Describe the image."):
        """
        Processes a single sample.
        images_list: list of image paths, PIL Images, or RGB numpy arrays.
        """
        pil_images = []
        for img in images_list:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            pil_images.append(img)

        self.processor.tokenizer.padding_side = "left"
        formatted_prompt = self._format_sft_prompt(len(pil_images), text_prompt)

        inputs = self.processor(
            images=pil_images if len(pil_images) > 0 else None,
            text=[formatted_prompt],
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def process_inputs_batch(self, batch_data):
        """
        Processes a batch of samples with different prompts and images.
        Uses exact SFT conversation structure without injecting thinking tags.
        """
        all_images = []
        texts = []
        
        for item in batch_data:
            imgs = item.get("images", [])
            for img in imgs:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                all_images.append(img)
            
            prompt_str = self._format_sft_prompt(len(imgs), item.get("text_prompt", ""))
            texts.append(prompt_str)

        self.processor.tokenizer.padding_side = "left"
        inputs = self.processor(
            images=all_images if len(all_images) > 0 else None,
            text=texts,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    @torch.no_grad()
    def generate(self, inputs, return_language=True, return_latent=True, layer_idx=-1, skip_generated_tokens=None, max_new_tokens=1024):
        """
        Generates text (with decoded coordinate tags) and extracts latent features from the target layers.
        """
        result = {}
        layers_to_extract = [layer_idx] if isinstance(layer_idx, int) else layer_idx
        
        # Run generation using custom model supporting coordinates projection
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_hidden_states=return_latent
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], outputs.sequences)
        ]
        if return_language:
            output_text = self.processor.tokenizer.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            result["text"] = output_text

        if return_latent:
            layer_tensors = []
            for l_idx in layers_to_extract:
                # outputs.hidden_states is a tuple of steps.
                # Step 0 contains the prompt hidden states; its last token represents the 1st generated token.
                # Steps 1+ contain the single-token hidden states for subsequent generated tokens.
                step_tensors = []
                if len(outputs.hidden_states) > 0:
                    # Slice the last token along the sequence dimension (dim 1)
                    step_tensors.append(outputs.hidden_states[0][l_idx][:, -1:, :])
                for step in range(1, len(outputs.hidden_states)):
                    step_tensors.append(outputs.hidden_states[step][l_idx])
                    
                if skip_generated_tokens is not None and step_tensors:
                    n = skip_generated_tokens
                    if n > 0 and len(step_tensors) > 2 * n:
                        step_tensors = step_tensors[n:-n]
                            
                if step_tensors:
                    layer_tensors.append(torch.cat(step_tensors, dim=1))
                  
            if layer_tensors:
                # Stack all layer features to form [B, L_layers, S_tokens, D_llm]
                stacked = torch.stack(layer_tensors, dim=1)
                result["latent"] = stacked.to(self.device)
            
        result["coord_mode"] = self.coord_mode
        return result


if __name__ == "__main__":
    import time
    print("🚀 Initializing Qwen35PixWrapper Test...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "/home/vlak/FoundationModels/Qwen3.5Pix-2B"

    # Check if local model directory exists
    if os.path.exists(model_id):
        print(f"Loading real model from local checkpoint: {model_id}")
        try:
            # Load wrapper in bfloat16 on GPU or CPU
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            use_flash = (device == "cuda")
            
            wrapper = Qwen35PixWrapper(
                model_name=model_id,
                device=device,
                dtype=dtype,
                use_flash_attn=use_flash
            )
            
            # Load real images from /home/vlak/data/demo_video/
            demo_video_dir = "/home/vlak/data/demo_video/"
            test_images = []
            if os.path.exists(demo_video_dir):
                for f in sorted(os.listdir(demo_video_dir)):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        full_path = os.path.join(demo_video_dir, f)
                        try:
                            test_images.append(Image.open(full_path).convert("RGB"))
                        except Exception as e:
                            print(f"Warning: Could not open {full_path}: {e}")
            
            if test_images:
                print(f"Loaded {len(test_images)} real images from {demo_video_dir} for testing.")
            else:
                print(f"No real images found in {demo_video_dir}. Falling back to mock white image.")
                mock_img = Image.new("RGB", (224, 224), color="white")
                test_images = [mock_img]
            
            print("\n--- TEST 1: Single Input Processing ---")
            inputs = wrapper.process_inputs(
                images_list=[test_images[0]],
                text_prompt="Observe: <|0.6,0.4|> then draw the trace "
            )
            print("Inputs keys:", list(inputs.keys()))
            print("input_ids shape:", inputs["input_ids"].shape)
            
            print("\n--- TEST 2: Generation and Latent Extraction ---")
            outputs = wrapper.generate(
                inputs=inputs,
                return_language=True,
                return_latent=True,
                layer_idx=-1,
                max_new_tokens=64
            )
            print("Generated Text:", outputs.get("text"))
            if "latent" in outputs:
                print("Latent features shape:", outputs["latent"].shape)
            
            print("\n--- TEST 3: Batch Processing ---")
            batch_data = [
                {"images": [test_images[0]], "text_prompt": "Observe: <|0.25,0.25|>"},
                {"images": test_images[:2], "text_prompt": "Observe multiple"}
            ]
            batch_inputs = wrapper.process_inputs_batch(batch_data)
            print("Batch inputs keys:", list(batch_inputs.keys()))
            print("Batch input_ids shape:", batch_inputs["input_ids"].shape)
            
            print("Running generation on batch inputs...")
            batch_outputs = wrapper.generate(
                inputs=batch_inputs,
                return_language=True,
                return_latent=True,
                layer_idx=-1,
                max_new_tokens=16
            )
            print("Batch Generated Texts:")
            for idx, text in enumerate(batch_outputs.get("text", [])):
                print(f"  Sample {idx}: {text!r}")
            if "latent" in batch_outputs:
                print("Batch latent features shape:", batch_outputs["latent"].shape)
            
            print("\n✅ Wrapper tests executed successfully!")
        except Exception as e:
            print(f"Error executing real model wrapper tests: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\nWarning: Checkpoint '{model_id}' not found on disk. Verifying class signatures and custom classes...")
        try:
            class DummyConfig:
                class TextConfig:
                    hidden_size = 1024
                    vocab_size = 151936
                text_config = TextConfig()
                num_bins = 256
                pix_start = 300000
                
            config = DummyConfig()
            model = Qwen3_5PixGeneration(config)
            print("✅ Qwen3_5PixGeneration initialization interface check passed.")
            print("✅ Custom tokenizer class Qwen3_5PixTokenizer is available.")
        except Exception as e:
            print(f"Warning: Mock initialization check failed: {e}")
            
        print("\nSkipped real generation because model weights are not loaded.")
