"""Run every demo in prompt/examples.json and save visualizations.

    # with the model (GPU recommended)
    python example/run_examples.py --model-path /path/to/PhysMoE-9B

    # no model / no GPU: draw the reference outputs recorded from PhysMoE-9B
    python example/run_examples.py --reference

Outputs go to example/outputs/<name>.jpg plus a results.jsonl with the prompt,
the model answer and the reference answer for each demo.
"""
import argparse
import json
import time
from pathlib import Path

from PIL import Image

from ps_utils import draw_prediction, generate

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", help="local checkpoint directory or Hugging Face repo id")
    ap.add_argument("--reference", action="store_true",
                    help="skip the model and visualize the recorded reference outputs")
    ap.add_argument("--examples", default=str(HERE / "prompt" / "examples.json"))
    ap.add_argument("--only", nargs="*", help="run only these example names")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=str(HERE / "outputs"))
    args = ap.parse_args()
    if not args.reference and not args.model_path:
        ap.error("pass --model-path, or --reference to draw the recorded outputs")

    examples = json.loads(Path(args.examples).read_text())
    if args.only:
        examples = [e for e in examples if e["name"] in set(args.only)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wrapper = None
    if not args.reference:
        from ps_utils import load_model
        wrapper = load_model(args.model_path, device=args.device)

    records = []
    for ex in examples:
        image = Image.open(HERE / ex["image"]).convert("RGB")
        start = time.time()
        if wrapper is None:
            answer = ex["reference_output"]
        else:
            answer = generate(wrapper, image, ex["prompt"])
        seconds = time.time() - start
        vis = draw_prediction(image, answer, ex["task"], prompt=ex["prompt"])
        vis.save(out_dir / f"{ex['name']}.jpg", quality=92)
        records.append({**ex, "output": answer, "seconds": round(seconds, 2)})
        print(f"[{ex['task']:>10}] {ex['name']:<18} {answer[:90]}")

    with open(out_dir / "results.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(records)} visualizations to {out_dir}")


if __name__ == "__main__":
    main()
