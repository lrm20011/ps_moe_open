"""Shared helpers for the PhysMoE examples.

- load_model(): builds Qwen35PixWrapper with sensible device / dtype defaults.
- parse_coordinates(): turns the model's `<|x,y|>` tags into groups of points.
- draw_prediction(): renders points, boxes or trajectories on the input image.

Everything except load_model() runs without torch weights, so the parsing and
drawing code can be used (and tested) on a machine without a GPU.
"""
from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "model"
# qwen35_psmoe.py imports `pixel_fusion` as a top-level module.
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

Point = Tuple[float, float]

# One coordinate tag, e.g. <|0.48048,0.508509|>. Values are normalized to [0, 1].
_NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
COORD_TAG = re.compile(rf"<\|\s*({_NUM})\s*,\s*({_NUM})\s*\|>")


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model(model_path: str, device: str | None = None, flash_attn: bool | None = None):
    """Load the PhysMoE checkpoint through Qwen35PixWrapper.

    device: "cuda", "cuda:1", "cpu"... defaults to cuda when available.
    flash_attn: defaults to True on CUDA when flash-attn is installed, else SDPA.
    The model always runs in bfloat16: several heads are created in bf16.
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        use_torch_linear_attention()
    from qwen35ps_wrapper import Qwen35PixWrapper

    if flash_attn is None:
        flash_attn = device.startswith("cuda") and _has_flash_attn()
    return Qwen35PixWrapper(
        model_name=model_path,
        origin_model_path=None,
        device=device,
        dtype=torch.bfloat16,
        use_flash_attn=flash_attn,
    )


def use_torch_linear_attention():
    """Force the pure-PyTorch Gated DeltaNet path (needed for CPU inference).

    transformers picks `causal_conv1d` / `fla` kernels when they are installed,
    and those kernels only accept CUDA tensors. Hiding the packages before the
    Qwen3.5 modeling file is imported makes transformers use its torch fallback.
    """
    if "transformers.models.qwen3_5.modeling_qwen3_5" in sys.modules:
        print("[ps_utils] Qwen3.5 modeling already imported; CPU kernels cannot be switched now.")
        return
    for name in ("causal_conv1d", "fla"):
        sys.modules[name] = None  # makes `import name` raise ImportError


def _has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


def generate(wrapper, image: Image.Image | str, prompt: str, max_new_tokens: int = 256,
             repetition_penalty: float = 1.05) -> str:
    """Single image + prompt -> decoded answer.

    Greedy decoding with repetition_penalty=1.05, the setting used to record the
    reference outputs in prompt/examples.json.
    """
    import torch

    if isinstance(image, (str, os.PathLike)):
        image = Image.open(image).convert("RGB")
    inputs = wrapper.process_inputs([image], text_prompt=prompt)
    with torch.inference_mode():
        out = wrapper.model.generate(**inputs, do_sample=False, repetition_penalty=repetition_penalty,
                                     max_new_tokens=max_new_tokens)
    new_tokens = out[:, inputs["input_ids"].shape[1]:]
    return wrapper.processor.tokenizer.batch_decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


# --------------------------------------------------------------------------- #
# Output parsing
# --------------------------------------------------------------------------- #
def parse_coordinates(text: str) -> List[List[Point]]:
    """Split the answer into groups of normalized (x, y) points.

    Groups are separated by commas between tags:
        point       "<|x,y|>"                        -> [[(x, y)]]
        bbox        "<|x1,y1|><|x2,y2|>"             -> [[(x1, y1), (x2, y2)]]
        many boxes  "<|..|><|..|>,<|..|><|..|>"      -> [[p1, p2], [p3, p4]]
        trajectory  "<|..|>,<|..|>,<|..|>"           -> [[p1], [p2], [p3]]
    Use flatten_points() when the grouping does not matter.
    """
    groups: List[List[Point]] = []
    current: List[Point] = []
    last_end = 0
    for m in COORD_TAG.finditer(text):
        between = text[last_end:m.start()]
        if current and "," in between:
            groups.append(current)
            current = []
        current.append((float(m.group(1)), float(m.group(2))))
        last_end = m.end()
    if current:
        groups.append(current)
    return groups


def flatten_points(groups: Sequence[Sequence[Point]]) -> List[Point]:
    return [p for g in groups for p in g]


def to_pixels(points: Sequence[Point], width: int, height: int) -> List[Tuple[float, float]]:
    return [(x * width, y * height) for x, y in points]


def boxes_from_groups(groups: Sequence[Sequence[Point]]) -> List[Tuple[Point, Point]]:
    """Every consecutive pair of points is one box (top-left, bottom-right)."""
    pts = flatten_points(groups)
    return [(pts[i], pts[i + 1]) for i in range(0, len(pts) - 1, 2)]


def infer_task_type(prompt: str) -> str:
    m = re.search(r"Type:\s*(point|bbox|trajectory)", prompt, re.I)
    return m.group(1).lower() if m else "text"


# --------------------------------------------------------------------------- #
# Visualization: prompt banner on top, only the model prediction on the image
# --------------------------------------------------------------------------- #
RED = (235, 50, 50)
ORANGE = (245, 110, 20)
WHITE = (255, 255, 255)
BANNER_BG = (22, 27, 38)
VIS_WIDTH = 1000


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_prediction(image: Image.Image, answer: str, task_type: str, prompt: str | None = None,
                    width: int = VIS_WIDTH) -> Image.Image:
    """Resize `image` to `width`, draw the parsed answer and put `prompt` in a banner on top."""
    im = image.convert("RGB")
    im = im.resize((width, round(im.height * width / im.width)), Image.Resampling.LANCZOS)
    s = width / VIS_WIDTH                  # stroke scale
    draw = ImageDraw.Draw(im)
    pts = to_pixels(flatten_points(parse_coordinates(answer)), im.width, im.height)

    if task_type == "bbox":
        for (x1, y1), (x2, y2) in zip(pts[0::2], pts[1::2]):
            box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            draw.rectangle([box[0] - 2 * s, box[1] - 2 * s, box[2] + 2 * s, box[3] + 2 * s],
                           outline=WHITE, width=max(1, round(2 * s)))
            draw.rectangle(box, outline=RED, width=max(1, round(4 * s)))
    elif task_type == "trajectory" and pts:
        if len(pts) > 1:
            draw.line(pts, fill=WHITE, width=round(11 * s), joint="curve")
            draw.line(pts, fill=ORANGE, width=round(6 * s), joint="curve")
        for i, (x, y) in enumerate(pts):
            r = (11 if i == 0 else 9) * s
            draw.ellipse([x - r, y - r, x + r, y + r], fill=ORANGE, outline=WHITE, width=round(3 * s))
        if len(pts) > 1:                   # arrow head at the end
            (x0, y0), (x1, y1) = pts[-2], pts[-1]
            a = math.atan2(y1 - y0, x1 - x0)
            head = [(x1 + 22 * s * math.cos(a), y1 + 22 * s * math.sin(a)),
                    (x1 + 8 * s * math.cos(a + 2.2), y1 + 8 * s * math.sin(a + 2.2)),
                    (x1 + 8 * s * math.cos(a - 2.2), y1 + 8 * s * math.sin(a - 2.2))]
            draw.polygon(head, fill=ORANGE, outline=WHITE)
    elif task_type == "point":
        for x, y in pts:
            r = 16 * s
            draw.ellipse([x - r - 3 * s, y - r - 3 * s, x + r + 3 * s, y + r + 3 * s],
                         outline=WHITE, width=round(4 * s))
            draw.ellipse([x - r, y - r, x + r, y + r], outline=RED, width=round(5 * s))
            c = 5 * s
            draw.ellipse([x - c, y - c, x + c, y + c], fill=RED, outline=WHITE, width=max(1, round(2 * s)))

    if not prompt:
        return im
    banner = _banner(re.sub(r"\.\.+", ".", prompt), im.width)
    out = Image.new("RGB", (im.width, banner.height + im.height), BANNER_BG)
    out.paste(banner, (0, 0))
    out.paste(im, (0, banner.height))
    return out


def _banner(text: str, width: int) -> Image.Image:
    font = _font(max(12, round(26 * width / VIS_WIDTH)))
    pad = round(24 * width / VIS_WIDTH)
    lines, cur = [], ""
    for word in text.split():
        test = (cur + " " + word).strip()
        if cur and font.getlength(test) > width - 2 * pad:
            lines.append(cur)
            cur = word
        else:
            cur = test
    lines.append(cur)
    lh = round(font.size * 1.35)
    im = Image.new("RGB", (width, pad + lh * len(lines) + pad // 2), BANNER_BG)
    d = ImageDraw.Draw(im)
    for i, line in enumerate(lines):
        d.text((pad, round(pad * 0.75) + i * lh), line, font=font, fill=WHITE)
    return im
