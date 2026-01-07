

"""
Build layer-wise prefill/decode hidden_states distribution shift statistics
for Qwen2-VL on multiple datasets.

✅ 已按你要求修改：
- PREFILL 阶段：不再统计全序列；改为 **只取最后 tail_tokens 个 token** 统计均值/方差（默认 32）
  统计方式：flatten B×tail×H
- DECODE 阶段：保持原来逻辑，只统计每步“最后一个 token”的 hidden_states（并对步数求平均）

Supported datasets:
  - synthdog (json)
  - docvqa / infographicvqa (local parquet)
  - ocrbench (local parquet)

Outputs:
  - layer_statistics_table.json
  - statistics_report.txt

Examples:
python build_stats_qwen2vl_multi_tail.py \
  --dataset docvqa \
  --parquet_dir /data/model/datasets--lmms-lab--DocVQA/DocVQA \
  --split validation \
  --model_path /data/model/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/xxx \
  --num_samples 200 --max_decode_tokens 64 \
  --tail_tokens 32 \
  --output_dir ./statistics_table/qwen2vl_docvqa_tail32
"""

import os
import json
import glob
import argparse
from typing import List, Tuple, Optional, Dict, Any, Union

import torch
from tqdm import tqdm
from PIL import Image

from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer


# -----------------------------
# Dataset loaders
# -----------------------------
def load_synthdog_dataset(json_path: str, num_samples: Optional[int] = None) -> List[Tuple[str, str]]:
    """
    SynthDog format:
      item['image_name']
      item['2_coord'] is list of dicts with 'chunk'
    Return: List[(image_path, text)]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data[:num_samples] if num_samples else data
    pairs: List[Tuple[str, str]] = []

    for item in items:
        if "image_name" not in item:
            continue

        image_name = item["image_name"]
        image_path = image_name if os.path.isabs(image_name) else os.path.join(os.path.dirname(json_path), image_name)

        text = ""
        if "2_coord" in item and isinstance(item["2_coord"], list) and len(item["2_coord"]) > 0:
            chunks = [c.get("chunk", "") for c in item["2_coord"] if isinstance(c, dict) and c.get("chunk")]
            if chunks:
                text = " ".join(chunks[:5])

        if text:
            pairs.append((image_path, text))

    return pairs


def load_parquet_vqa_dataset(
    parquet_dir: str,
    split: str = "validation",
    num_samples: int = 500
) -> List[Tuple[Image.Image, str]]:
    """
    Load DocVQA / InfographicVQA from local parquet files.

    Return: List[(PIL.Image, question)]
    """
    from datasets import load_dataset

    if not os.path.isdir(parquet_dir):
        raise FileNotFoundError(f"Parquet dir not found: {parquet_dir}")

    pattern = os.path.join(parquet_dir, f"{split}-*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet found: {pattern}")

    ds = load_dataset("parquet", data_files={split: files})[split]

    pairs: List[Tuple[Image.Image, str]] = []
    n = min(num_samples, len(ds))
    for i in range(n):
        item = ds[i]

        img = item.get("image", None)
        if img is None:
            continue

        q = item.get("question") or item.get("query") or item.get("text") or item.get("prompt")
        if not q:
            continue

        if not isinstance(img, Image.Image):
            try:
                img = Image.fromarray(img)
            except Exception:
                continue

        pairs.append((img.convert("RGB"), str(q)))

    return pairs


def load_ocrbench_parquet_dataset(
    ocrbench_root: str,
    split: str = "test",
    num_samples: int = 500,
) -> List[Tuple[Union[str, Image.Image], str]]:
    """
    OCRBench local parquet loader.
    Expected structure:
      {ocrbench_root}/data/{split}-*.parquet
    """
    from datasets import load_dataset

    data_dir = os.path.join(ocrbench_root, "data")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"OCRBench data dir not found: {data_dir}")

    data_files = sorted(glob.glob(os.path.join(data_dir, f"{split}-*.parquet")))
    if not data_files:
        raise FileNotFoundError(f"No parquet found: {data_dir}/{split}-*.parquet")

    ds = load_dataset("parquet", data_files={split: data_files})[split]

    pairs: List[Tuple[Union[str, Image.Image], str]] = []
    n = min(num_samples, len(ds))

    for i in range(n):
        row = ds[i]

        q = row.get("question") or row.get("prompt") or row.get("text") or row.get("query")
        if not q:
            continue
        q = str(q)

        img = row.get("image") or row.get("img") or row.get("image_path") or row.get("img_path")
        if img is None:
            continue

        if isinstance(img, Image.Image):
            pairs.append((img.convert("RGB"), q))
            continue

        if isinstance(img, dict):
            try:
                if img.get("bytes") is not None:
                    from io import BytesIO
                    pil = Image.open(BytesIO(img["bytes"])).convert("RGB")
                    pairs.append((pil, q))
                    continue
                if img.get("path"):
                    p = img["path"]
                    if not os.path.isabs(p):
                        p1 = os.path.join(ocrbench_root, p)
                        p2 = os.path.join(data_dir, p)
                        if os.path.exists(p1):
                            p = p1
                        elif os.path.exists(p2):
                            p = p2
                    if os.path.exists(p):
                        pairs.append((p, q))
                    continue
            except Exception:
                continue

        if isinstance(img, str):
            p = img
            if not os.path.isabs(p):
                p1 = os.path.join(ocrbench_root, p)
                p2 = os.path.join(data_dir, p)
                if os.path.exists(p1):
                    p = p1
                elif os.path.exists(p2):
                    p = p2
            if os.path.exists(p):
                pairs.append((p, q))
            continue

    return pairs


# -----------------------------
# Statistics builder
# -----------------------------
class StatisticsTableBuilder:
    """统计表构建器（prefill tail-window, decode last-token）"""

    def __init__(
        self,
        model,
        processor,
        tokenizer,
        num_layers: int = 28,
        device: str = "cuda",
        tail_tokens: int = 32,
    ):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        self.device = device
        self.tail_tokens = int(tail_tokens)

        self.layer_stats = {
            i: {
                "prefill_means": [],
                "prefill_stds": [],
                "decode_means": [],
                "decode_stds": [],
            }
            for i in range(num_layers)
        }

    @staticmethod
    def _mean_std_flat(x: torch.Tensor) -> Tuple[float, float]:
        flat = x.reshape(-1)
        return float(flat.mean().item()), float(flat.std().item())

    def _prefill_tail_stats(self, hs: torch.Tensor) -> Tuple[float, float]:
        """
        hs: [B, T, H] -> take last tail_tokens tokens: [B, tail, H] then flatten
        """
        T = hs.shape[1]
        t0 = max(0, T - self.tail_tokens)
        tail = hs[:, t0:, :]
        return self._mean_std_flat(tail)

    def collect_sample_statistics(
        self,
        image_input: Union[str, Image.Image],
        text: str,
        max_decode_tokens: int = 64,
    ) -> bool:
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_input},
                        {"type": "text", "text": text},
                    ],
                }
            ]

            text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            # ---------------------------
            # Prefill (tail window only)
            # ---------------------------
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=True,
                )

            hidden_states_tuple = outputs.hidden_states  # (num_layers+1,), each [B, seq, hidden]
            if hidden_states_tuple is None:
                raise RuntimeError("Prefill did not return hidden_states. Check output_hidden_states=True.")

            prefill_stats_per_layer: Dict[int, Tuple[float, float]] = {}
            for layer_idx in range(self.num_layers):
                hs = hidden_states_tuple[layer_idx + 1]
                mean, std = self._prefill_tail_stats(hs)  # ✅ tail
                prefill_stats_per_layer[layer_idx] = (mean, std)

            # ---------------------------
            # Decode (step-by-step, last-token)
            # ---------------------------
            decode_stats_per_layer = {i: {"means": [], "stds": []} for i in range(self.num_layers)}

            generated_ids = inputs.input_ids.clone()
            past_key_values = outputs.past_key_values

            for step in range(max_decode_tokens):
                with torch.no_grad():
                    if step == 0:
                        step_out = self.model(
                            input_ids=generated_ids,
                            attention_mask=inputs.attention_mask,
                            past_key_values=past_key_values,
                            output_hidden_states=True,
                            return_dict=True,
                            use_cache=True,
                        )
                    else:
                        attn_mask = torch.cat(
                            [
                                inputs.attention_mask,
                                torch.ones(
                                    1,
                                    step,
                                    device=self.device,
                                    dtype=inputs.attention_mask.dtype,
                                ),
                            ],
                            dim=1,
                        )
                        step_out = self.model(
                            input_ids=generated_ids[:, -1:],
                            attention_mask=attn_mask,
                            past_key_values=past_key_values,
                            output_hidden_states=True,
                            return_dict=True,
                            use_cache=True,
                        )

                past_key_values = step_out.past_key_values
                hs_tuple = step_out.hidden_states

                for layer_idx in range(self.num_layers):
                    hs = hs_tuple[layer_idx + 1]
                    last_hidden = hs[:, -1:, :]  # [B,1,H]
                    mean, std = self._mean_std_flat(last_hidden)
                    decode_stats_per_layer[layer_idx]["means"].append(mean)
                    decode_stats_per_layer[layer_idx]["stds"].append(std)

                logits = step_out.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                if self.tokenizer.eos_token_id is not None and next_token.item() == self.tokenizer.eos_token_id:
                    break

            # ---------------------------
            # Save into buffers
            # ---------------------------
            for layer_idx in range(self.num_layers):
                if layer_idx not in prefill_stats_per_layer:
                    continue
                if len(decode_stats_per_layer[layer_idx]["means"]) == 0:
                    continue

                prefill_mean, prefill_std = prefill_stats_per_layer[layer_idx]
                avg_decode_mean = sum(decode_stats_per_layer[layer_idx]["means"]) / len(decode_stats_per_layer[layer_idx]["means"])
                avg_decode_std = sum(decode_stats_per_layer[layer_idx]["stds"]) / len(decode_stats_per_layer[layer_idx]["stds"])

                self.layer_stats[layer_idx]["prefill_means"].append(prefill_mean)
                self.layer_stats[layer_idx]["prefill_stds"].append(prefill_std)
                self.layer_stats[layer_idx]["decode_means"].append(avg_decode_mean)
                self.layer_stats[layer_idx]["decode_stds"].append(avg_decode_std)

            return True

        except Exception as e:
            print(f"[WARN] sample failed: {e}")
            return False

    def compute_final_statistics(self) -> Dict[int, Dict[str, Any]]:
        statistics_table: Dict[int, Dict[str, Any]] = {}

        for layer_idx in range(self.num_layers):
            stats = self.layer_stats[layer_idx]
            if len(stats["prefill_means"]) == 0 or len(stats["decode_means"]) == 0:
                continue

            prefill_mean_avg = sum(stats["prefill_means"]) / len(stats["prefill_means"])
            prefill_std_avg = sum(stats["prefill_stds"]) / len(stats["prefill_stds"])
            decode_mean_avg = sum(stats["decode_means"]) / len(stats["decode_means"])
            decode_std_avg = sum(stats["decode_stds"]) / len(stats["decode_stds"])

            mean_shift = decode_mean_avg - prefill_mean_avg
            std_ratio = decode_std_avg / (prefill_std_avg + 1e-8)

            statistics_table[layer_idx] = {
                "mean_shift": float(mean_shift),
                "std_ratio": float(std_ratio),
                "prefill_mean_avg": float(prefill_mean_avg),
                "prefill_std_avg": float(prefill_std_avg),
                "decode_mean_avg": float(decode_mean_avg),
                "decode_std_avg": float(decode_std_avg),
                "num_samples": int(min(len(stats["prefill_means"]), len(stats["decode_means"]))),
            }

        return statistics_table


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Build statistics table for Qwen2-VL (multi-dataset, prefill tail-window)")

    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/model/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac",
        help="Path to Qwen2-VL model",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["synthdog", "docvqa", "infographicvqa", "ocrbench"],
        help="Which dataset loader to use",
    )

    parser.add_argument("--synthdog_json", type=str, default="/data/dataset/synthdog-en/synthdog-en.json")
    parser.add_argument("--parquet_dir", type=str, default="")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])

    parser.add_argument("--ocrbench_root", type=str, default="/data/model/datasets--echo840--OCRBench")
    parser.add_argument("--ocrbench_split", type=str, default="test")

    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--max_decode_tokens", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=28)
    parser.add_argument("--tail_tokens", type=int, default=32, help="Prefill tail window length (default: 32)")
    parser.add_argument("--output_dir", type=str, default="./statistics_table/qwen2vl_tail32")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Building Statistics Table (Qwen2-VL, Prefill Tail-Window)")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Tail tokens: {args.tail_tokens}")
    print(f"Num samples: {args.num_samples}")
    print(f"Max decode tokens: {args.max_decode_tokens}")
    print(f"Num layers: {args.num_layers}")
    print(f"Output dir: {args.output_dir}")
    print()

    print("Loading model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print("Model loaded.\n")

    print("Loading dataset...")
    if args.dataset == "synthdog":
        pairs = load_synthdog_dataset(args.synthdog_json, args.num_samples)
        print(f"Loaded {len(pairs)} samples from SynthDog.\n")

    elif args.dataset in ["docvqa", "infographicvqa"]:
        if not args.parquet_dir:
            raise ValueError("--parquet_dir is required for docvqa/infographicvqa.")
        pairs = load_parquet_vqa_dataset(args.parquet_dir, split=args.split, num_samples=args.num_samples)
        print(f"Loaded {len(pairs)} samples from parquet ({args.split}).\n")

    elif args.dataset == "ocrbench":
        pairs = load_ocrbench_parquet_dataset(
            ocrbench_root=args.ocrbench_root,
            split=args.ocrbench_split,
            num_samples=args.num_samples,
        )
        print(f"Loaded {len(pairs)} samples from OCRBench ({args.ocrbench_split}).\n")

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    builder = StatisticsTableBuilder(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        num_layers=args.num_layers,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tail_tokens=args.tail_tokens,
    )

    print("Collecting statistics...")
    successful = 0
    for i, (img, text) in enumerate(tqdm(pairs, desc="samples")):
        ok = builder.collect_sample_statistics(img, text, max_decode_tokens=args.max_decode_tokens)
        successful += int(ok)

        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(pairs)} | successful: {successful}")

    print(f"\nTotal successful samples: {successful}/{len(pairs)}\n")

    print("Computing final statistics table...")
    statistics_table = builder.compute_final_statistics()

    out_json = os.path.join(args.output_dir, "layer_statistics_table.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(statistics_table, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_json}\n")

    report_file = os.path.join(args.output_dir, "statistics_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("QWEN2-VL LAYER STATISTICS TABLE (PREFILL TAIL-WINDOW, DECODE LAST-TOKEN)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Tail tokens: {args.tail_tokens}\n")
        if args.dataset == "synthdog":
            f.write(f"SynthDog JSON: {args.synthdog_json}\n")
        elif args.dataset in ["docvqa", "infographicvqa"]:
            f.write(f"Parquet dir: {args.parquet_dir}\n")
            f.write(f"Split: {args.split}\n")
        elif args.dataset == "ocrbench":
            f.write(f"OCRBench root: {args.ocrbench_root}\n")
            f.write(f"OCRBench split: {args.ocrbench_split}\n")
        f.write(f"Successful samples: {successful}/{len(pairs)}\n")
        f.write(f"Max decode tokens: {args.max_decode_tokens}\n\n")
        f.write("USAGE:\n")
        f.write("  decode_mean = prefill_mean + mean_shift\n")
        f.write("  decode_std  = prefill_std  * std_ratio\n\n")
        f.write("TABLE:\n")
        f.write("-" * 80 + "\n")
        for layer_idx in sorted(statistics_table.keys()):
            s = statistics_table[layer_idx]
            f.write(
                f"{layer_idx}\tmean_shift={s['mean_shift']:.6f}\tstd_ratio={s['std_ratio']:.6f}\t"
                f"prefill_mean={s['prefill_mean_avg']:.6f}\tprefill_std={s['prefill_std_avg']:.6f}\t"
                f"decode_mean={s['decode_mean_avg']:.6f}\tdecode_std={s['decode_std_avg']:.6f}\t"
                f"samples={s['num_samples']}\n"
            )
        f.write("=" * 80 + "\n")

    print(f"Saved: {report_file}")
    print("Done!")


if __name__ == "__main__":
    main()


# python build_stats_qwen2vl_multi.py \
#   --dataset synthdog \
#   --synthdog_json /data/dataset/synthdog-en/synthdog-en.json \
#   --model_path /data/model/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac \
#   --num_samples 200 --max_decode_tokens 64 \
#   --output_dir ./statistics_table/qwen2vl_synthdoglast32


# # python build_stats_qwen2vl_multi.py \
# #   --dataset infographicvqa \
# #   --parquet_dir /data/model/datasets--lmms-lab--DocVQA/InfographicVQA \
# #   --split validation \
# #   --model_path /data/model/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac \
# #   --num_samples 200 --max_decode_tokens 64 \
# #   --output_dir ./statistics_table/qwen2vl_infovqa

# # python build_stats_qwen2vl_multi.py \
# #   --dataset docvqa \
# #   --parquet_dir /data/model/datasets--lmms-lab--DocVQA/DocVQA \
# #   --split validation \
# #   --model_path /data/model/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac \
# #   --num_samples 200 --max_decode_tokens 64 \
# #   --output_dir ./statistics_table/qwen2vl_docvqa
