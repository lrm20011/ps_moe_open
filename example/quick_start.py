"""Quick start: one image + one instruction -> answer + visualization.

    python example/quick_start.py --model-path /path/to/PhysMoE-9B \
        --image example/figs/mug_grasp.jpg \
        --prompt "You need to grasp the mug. Predict the affordance area for grasp. Type: point"

The task type (point / bbox / trajectory) is read from the "Type:" suffix of the
prompt. Prompts without it are answered as plain text.
"""
import argparse
from pathlib import Path

from PIL import Image

from ps_utils import draw_prediction, generate, infer_task_type, load_model, parse_coordinates

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True, help="local checkpoint directory or Hugging Face repo id")
    ap.add_argument("--image", default=str(HERE / "figs" / "mug_grasp.jpg"))
    ap.add_argument("--prompt", default="You need to grasp the mug. Predict the affordance area for grasp. Type: point")
    ap.add_argument("--device", default=None, help="cuda / cuda:0 / cpu (default: cuda if available)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", default=str(HERE / "outputs" / "quick_start.jpg"))
    args = ap.parse_args()

    wrapper = load_model(args.model_path, device=args.device)
    image = Image.open(args.image).convert("RGB")

    answer = generate(wrapper, image, args.prompt, max_new_tokens=args.max_new_tokens)
    task = infer_task_type(args.prompt)
    print(f"Prompt : {args.prompt}")
    print(f"Answer : {answer}")

    if task != "text":
        w, h = image.size
        for i, group in enumerate(parse_coordinates(answer)):
            pix = [(round(x * w, 1), round(y * h, 1)) for x, y in group]
            print(f"  group {i}: normalized={group}  pixels={pix}")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        draw_prediction(image, answer, task, prompt=args.prompt).save(out, quality=92)
        print(f"Saved visualization to {out}")


if __name__ == "__main__":
    main()
