"""
Build layer-wise prefill/decode hidden_states distribution shift statistics
for LLaVA v1.6 using LLaVA-NeXT codebase.

Supports multiple datasets by swapping the dataset loader.

Datasets supported:
  - synthdog (json, like your current)
  - docvqa   (local parquet from /data/model/datasets--lmms-lab--DocVQA)

Output:
  layer_statistics_table.json
  statistics_report.txt

Example (SynthDog):
python build_stats_llava_next_multi_dataset.py \
  --pretrained liuhaotian/llava-v1.6-vicuna-7b \
  --dataset synthdog \
  --data_json /data/dataset/synthdog-en/synthdog-en.json \
  --num_samples 200 \
  --max_decode_tokens 64 \
  --output_dir ./statistics_table/llava_next_v16_synthdog

Example (DocVQA parquet, already downloaded):
python build_stats_llava_next_multi_dataset.py \
  --pretrained liuhaotian/llava-v1.6-vicuna-7b \
  --dataset docvqa \
  --docvqa_root /data/model/datasets--lmms-lab--DocVQA \
  --docvqa_subset DocVQA \
  --docvqa_split validation \
  --num_samples 200 \
  --max_decode_tokens 64 \
  --output_dir ./statistics_table/llava_next_v16_docvqa
"""

import os
import json
import glob
import argparse
from typing import Dict, List, Tuple, Any, Optional, Union

import torch
from tqdm import tqdm
from PIL import Image

# For DocVQA parquet
from datasets import load_dataset

import sys
sys.path.append("./visual_head/LLaVA-NeXT")

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates


# ---------------------------
# Helpers
# ---------------------------
def compute_global_statistics(hidden_states: torch.Tensor) -> Tuple[float, float]:
    flat = hidden_states.reshape(-1)
    return flat.mean().item(), flat.std().item()


def build_prompt(question: str, conv_mode: str = "vicuna_v1") -> str:
    q = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], q)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


# ---------------------------
# Dataset loaders
# Return: List[(image, text)] where image is PIL.Image
# ---------------------------
def load_synthdog_pairs(json_path: str, num_samples: Optional[int]) -> List[Tuple[Image.Image, str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data[:num_samples] if num_samples else data
    pairs: List[Tuple[Image.Image, str]] = []

    for item in items:
        img = item.get("image_name")
        if not img:
            continue
        img_path = img if os.path.isabs(img) else os.path.join(os.path.dirname(json_path), img)
        if not os.path.exists(img_path):
            continue

        text = ""
        coord = item.get("2_coord", [])
        if isinstance(coord, list) and len(coord) > 0:
            chunks = [c.get("chunk", "") for c in coord if isinstance(c, dict) and c.get("chunk")]
            if chunks:
                text = " ".join(chunks[:5])

        if not text:
            text = item.get("text") or item.get("question") or ""

        if not text:
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        pairs.append((image, str(text)))

    return pairs


def load_docvqa_parquet_pairs(
    docvqa_root: str,
    subset: str = "DocVQA",
    split: str = "validation",
    num_samples: Optional[int] = None,
) -> List[Tuple[Image.Image, str]]:
    """
    docvqa_root: /data/model/datasets--lmms-lab--DocVQA
    subset: DocVQA or InfographicVQA
    split: train / validation / test (depends on what you downloaded)
    """
    data_files = sorted(glob.glob(os.path.join(docvqa_root, subset, f"{split}-*.parquet")))
    if not data_files:
        raise FileNotFoundError(f"No parquet found: {docvqa_root}/{subset}/{split}-*.parquet")

    ds = load_dataset("parquet", data_files={split: data_files})[split]

    pairs: List[Tuple[Image.Image, str]] = []
    n = len(ds) if num_samples is None else min(len(ds), num_samples)

    for i in range(n):
        row = ds[i]

        # question field name may vary slightly
        q = row.get("question") or row.get("query") or row.get("text") or row.get("prompt")
        if not q:
            continue
        q = str(q)

        img = row.get("image")
        if img is None:
            continue

        # datasets Image feature often returns PIL.Image already
        try:
            if isinstance(img, Image.Image):
                pil = img.convert("RGB")
            elif isinstance(img, dict):
                # sometimes {"bytes":..., "path":...}
                if img.get("bytes") is not None:
                    from io import BytesIO
                    pil = Image.open(BytesIO(img["bytes"])).convert("RGB")
                elif img.get("path"):
                    pil = Image.open(img["path"]).convert("RGB")
                else:
                    continue
            else:
                # fallback: string path
                if isinstance(img, str) and os.path.exists(img):
                    pil = Image.open(img).convert("RGB")
                else:
                    continue
        except Exception:
            continue

        pairs.append((pil, q))

    return pairs

def load_pairs(args) -> List[Tuple[Image.Image, str]]:
    if args.dataset == "synthdog":
        if not args.data_json:
            raise ValueError("--data_json is required for synthdog")
        return load_synthdog_pairs(args.data_json, args.num_samples)

    if args.dataset == "docvqa":
        return load_docvqa_parquet_pairs(
            docvqa_root=args.docvqa_root,
            subset=args.docvqa_subset,
            split=args.docvqa_split,
            num_samples=args.num_samples,
        )

    if args.dataset == "ocrbench":
        return load_ocrbench_pairs(
            ocrbench_root=args.ocrbench_root,
            split=args.ocrbench_split,
            num_samples=args.num_samples,
        )

    raise ValueError(f"Unknown dataset: {args.dataset}")
def load_ocrbench_pairs(
    ocrbench_root: str,
    split: str = "test",
    num_samples: Optional[int] = None,
) -> List[Tuple[Image.Image, str]]:
    """
    OCRBench local parquet loader.

    Expected structure (yours):
      /data/model/datasets--echo840--OCRBench/data/test-00000-of-00001.parquet

    Will load:
      data/{split}-*.parquet

    Tries common columns:
      - question/prompt/text/query
      - image (datasets.Image -> PIL), OR image_path/image/img_path/img (string path), OR dict with bytes/path
    """
    import glob
    from datasets import load_dataset

    data_dir = os.path.join(ocrbench_root, "data")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"OCRBench data dir not found: {data_dir}")

    data_files = sorted(glob.glob(os.path.join(data_dir, f"{split}-*.parquet")))
    if not data_files:
        raise FileNotFoundError(f"No parquet found: {data_dir}/{split}-*.parquet")

    ds = load_dataset("parquet", data_files={split: data_files})[split]

    pairs: List[Tuple[Image.Image, str]] = []
    n = len(ds) if num_samples is None else min(len(ds), num_samples)

    for i in range(n):
        row = ds[i]

        # ---- text ----
        q = row.get("question") or row.get("prompt") or row.get("text") or row.get("query")
        if not q:
            continue
        q = str(q)

        # ---- image ----
        img = row.get("image") or row.get("img") or row.get("image_path") or row.get("img_path")
        if img is None:
            continue

        try:
            # datasets.Image usually returns PIL.Image directly
            if isinstance(img, Image.Image):
                pil = img.convert("RGB")

            # sometimes dict {"bytes":..., "path":...}
            elif isinstance(img, dict):
                if img.get("bytes") is not None:
                    from io import BytesIO
                    pil = Image.open(BytesIO(img["bytes"])).convert("RGB")
                elif img.get("path"):
                    p = img["path"]
                    if not os.path.isabs(p):
                        p = os.path.join(ocrbench_root, p)
                    pil = Image.open(p).convert("RGB")
                else:
                    continue

            # sometimes string path
            elif isinstance(img, str):
                p = img
                if not os.path.isabs(p):
                    # try relative to root first
                    p1 = os.path.join(ocrbench_root, p)
                    # then relative to data_dir
                    p2 = os.path.join(data_dir, p)
                    if os.path.exists(p1):
                        p = p1
                    elif os.path.exists(p2):
                        p = p2
                if not os.path.exists(p):
                    continue
                pil = Image.open(p).convert("RGB")
            else:
                continue

        except Exception:
            continue

        pairs.append((pil, q))

    return pairs


# ---------------------------
# Statistics builder (LLaVA-NeXT)
# ---------------------------
class LlavaShiftStats:
    def __init__(self, pretrained: str, conv_mode: str = "vicuna_v1",
                 attn_impl: str = "eager", device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.pretrained = pretrained
        self.conv_mode = conv_mode
        self.device = device
        self.dtype = dtype

        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            pretrained, None, get_model_name_from_path(pretrained), attn_implementation=attn_impl
        )
        self.model.eval()

        self.config = self.model.config
        self.num_layers = int(getattr(self.config, "num_hidden_layers", 32))

        self.layer_stats: Dict[int, Dict[str, List[float]]] = {
            i: {"prefill_means": [], "prefill_stds": [], "decode_means": [], "decode_stds": []}
            for i in range(self.num_layers)
        }

    @torch.no_grad()
    def collect_one(self, image: Image.Image, text: str, max_decode_tokens: int = 64) -> bool:
        try:
            image_size = image.size  # (W, H)

            # image tensor
            image_tensor = process_images([image], self.image_processor, self.config)
            image_tensor = image_tensor.to(dtype=self.dtype, device=self.device, non_blocking=True)

            # prompt (你可以换成固定 question，比如 "Describe the image."，这里用数据集 question 更自然)
            question = text.strip() if text.strip() else "Describe the image."
            prompt = build_prompt(question, self.conv_mode)

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            input_ids = input_ids.to(self.device, non_blocking=True).unsqueeze(0)  # [1, L]

            # -----------------------
            # Prefill
            # -----------------------
            outputs = self.model(
                input_ids=input_ids[:, :-1],
                images=image_tensor,
                image_sizes=[image_size],
                use_cache=True,
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = outputs.hidden_states
            past_kv = outputs.past_key_values

            prefill_stats = {}
            for l in range(self.num_layers):
                hs = hidden_states[l + 1]
                m, s = compute_global_statistics(hs)
                prefill_stats[l] = (m, s)

            # -----------------------
            # Decode
            # -----------------------
            decode_means = {l: [] for l in range(self.num_layers)}
            decode_stds = {l: [] for l in range(self.num_layers)}

            cur = input_ids[:, -1]  # [1]

            for _ in range(max_decode_tokens):
                out = self.model(
                    input_ids=cur.view(1, 1),
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True
                )
                past_kv = out.past_key_values

                hs_tuple = out.hidden_states
                for l in range(self.num_layers):
                    last_h = hs_tuple[l + 1]  # [1,1,h]
                    m, s = compute_global_statistics(last_h)
                    decode_means[l].append(m)
                    decode_stds[l].append(s)

                next_token = out.logits[:, -1, :].argmax(dim=-1)  # [1]
                cur = next_token

                if self.tokenizer.eos_token_id is not None and next_token.item() == self.tokenizer.eos_token_id:
                    break

            # aggregate
            for l in range(self.num_layers):
                if len(decode_means[l]) == 0:
                    continue
                pm, ps = prefill_stats[l]
                dm = float(sum(decode_means[l]) / len(decode_means[l]))
                ds = float(sum(decode_stds[l]) / len(decode_stds[l]))

                self.layer_stats[l]["prefill_means"].append(pm)
                self.layer_stats[l]["prefill_stds"].append(ps)
                self.layer_stats[l]["decode_means"].append(dm)
                self.layer_stats[l]["decode_stds"].append(ds)

            return True

        except Exception as e:
            print(f"[WARN] sample failed: {e}")
            return False

    def finalize(self) -> Dict[int, Dict[str, Any]]:
        table = {}
        for l in range(self.num_layers):
            s = self.layer_stats[l]
            if len(s["prefill_means"]) == 0:
                continue

            pmean = float(sum(s["prefill_means"]) / len(s["prefill_means"]))
            pstd = float(sum(s["prefill_stds"]) / len(s["prefill_stds"]))
            dmean = float(sum(s["decode_means"]) / len(s["decode_means"]))
            dstd = float(sum(s["decode_stds"]) / len(s["decode_stds"]))

            table[l] = {
                "mean_shift": dmean - pmean,
                "std_ratio": dstd / (pstd + 1e-8),
                "prefill_mean_avg": pmean,
                "prefill_std_avg": pstd,
                "decode_mean_avg": dmean,
                "decode_std_avg": dstd,
                "num_samples": len(s["prefill_means"])
            }
        return table


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser("Build LLaVA-NeXT prefill/decode shift stats (multi-dataset)")
    parser.add_argument("--pretrained", type=str, default="liuhaotian/llava-v1.6-vicuna-7b")
    parser.add_argument("--dataset", type=str, required=True, choices=["synthdog", "docvqa","ocrbench"])

    # synthdog
    parser.add_argument("--data_json", type=str, default="", help="SynthDog json path (required if --dataset synthdog)")
    parser.add_argument("--ocrbench_root", type=str, default="/data/model/datasets--lmms-lab--DocVQA")
    # docvqa parquet
    parser.add_argument("--docvqa_root", type=str, default="/data/model/datasets--lmms-lab--DocVQA")
    parser.add_argument("--docvqa_subset", type=str, default="DocVQA", choices=["DocVQA", "InfographicVQA"])
    parser.add_argument("--docvqa_split", type=str, default="validation", help="train/validation/test")

    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--max_decode_tokens", type=int, default=64)
    parser.add_argument("--conv_mode", type=str, default="vicuna_v1")
    parser.add_argument("--attn_impl", type=str, default="eager")
    parser.add_argument("--output_dir", type=str, default="./statistics_table/llava_next")

    parser.add_argument(
    "--ocrbench_split",
    type=str,
    default="test",
    help="OCRBench split name (e.g., test/val/train depending on your local files)"
)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pairs = load_pairs(args)
    print(f"Loaded {len(pairs)} samples. dataset={args.dataset}")

    stats = LlavaShiftStats(
        pretrained=args.pretrained,
        conv_mode=args.conv_mode,
        attn_impl=args.attn_impl,
        device="cuda",
        dtype=torch.float16
    )
    print(f"Model loaded: {args.pretrained}")
    print(f"num_layers = {stats.num_layers}")

    ok = 0
    for image, txt in tqdm(pairs, desc="samples"):
        ok += int(stats.collect_one(image, txt, max_decode_tokens=args.max_decode_tokens))

    print(f"successful: {ok} / {len(pairs)}")

    table = stats.finalize()

    out_json = os.path.join(args.output_dir, "layer_statistics_table.json")
    with open(out_json, "w") as f:
        json.dump(table, f, indent=2)
    print(f"Saved: {out_json}")

    rep = os.path.join(args.output_dir, "statistics_report.txt")
    with open(rep, "w") as f:
        f.write("LLaVA-NeXT LAYER SHIFT STATISTICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"pretrained: {args.pretrained}\n")
        f.write(f"dataset: {args.dataset}\n")
        if args.dataset == "synthdog":
            f.write(f"data_json: {args.data_json}\n")
        else:
            f.write(f"docvqa_root: {args.docvqa_root}\n")
            f.write(f"docvqa_subset: {args.docvqa_subset}\n")
            f.write(f"docvqa_split: {args.docvqa_split}\n")
        f.write(f"num_samples: {args.num_samples}\n")
        f.write(f"max_decode_tokens: {args.max_decode_tokens}\n")
        f.write(f"successful: {ok} / {len(pairs)}\n\n")

        f.write("USAGE:\n")
        f.write("  decode_mean = prefill_mean + mean_shift\n")
        f.write("  decode_std  = prefill_std  * std_ratio\n\n")

        f.write(f"{'Layer':<6} | {'Mean Shift':<12} | {'Std Ratio':<12} | {'Samples':<8}\n")
        f.write("-" * 60 + "\n")
        for l in sorted(table.keys()):
            s = table[l]
            f.write(f"{l:<6} | {s['mean_shift']:>12.6f} | {s['std_ratio']:>12.6f} | {s['num_samples']:<8}\n")

    print(f"Saved: {rep}")


if __name__ == "__main__":
    main()
