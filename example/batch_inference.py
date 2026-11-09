"""Batched inference with different images and prompts in one forward pass.

    python example/batch_inference.py --model-path /path/to/PhysMoE-9B

Text-only questions and coordinate questions can be mixed in one batch: the
router decides per token whether to emit text or a coordinate.
"""
import argparse
from pathlib import Path

from PIL import Image

from ps_utils import load_model, parse_coordinates

HERE = Path(__file__).resolve().parent

BATCH = [
    ("figs/whiteboard_marker.jpg", "I need to pick up the whiteboard marker. Locate the target object. Type: bbox"),
    ("figs/glass_cups.jpg", "Where can I get more glass cups? Predict the related area. Type: point"),
    ("figs/mushroom_pot.jpg", "Move the mushroom and put it in the pot. "
                              "Predict the visual trajectory of the target object. Type: trajectory. Lim: 8"),
    ("figs/bowl_fill.jpg", "Describe the scene in one sentence."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    wrapper = load_model(args.model_path, device=args.device)

    batch = [{"images": [Image.open(HERE / p).convert("RGB")], "text_prompt": q} for p, q in BATCH]
    inputs = wrapper.process_inputs_batch(batch)          # left-padded
    out = wrapper.generate(
        inputs,
        return_language=True,
        return_latent=False,
        max_new_tokens=args.max_new_tokens,
    )

    for (path, prompt), text in zip(BATCH, out["text"]):
        print(f"\n{path}\n  prompt : {prompt}\n  answer : {text}")
        groups = parse_coordinates(text)
        if groups:
            print(f"  points : {groups}")


if __name__ == "__main__":
    main()
