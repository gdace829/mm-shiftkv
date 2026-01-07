"""
Build layer-wise prefill/decode hidden_states distribution shift statistics
for Qwen2-VL on multiple datasets.

Outputs:
  - layer_statistics_table.json   (layer-wise mean_shift/std_ratio + averages)
  - statistics_report.txt         (human-readable)

Usage examples:

# SynthDog (your current format)
python build_stats_multi_dataset.py \
  --dataset synthdog \
  --data_json /data/dataset/synthdog-en/synthdog-en.json \
  --num_samples 500 \
  --max_decode_tokens 64 \
  --output_dir ./stats/synthdog

# DocVQA-like
python build_stats_multi_dataset.py \
  --dataset docvqa \
  --data_json /data/dataset/docvqa/val.json \
  --image_root /data/dataset/docvqa/images \
  --num_samples 500 \
  --output_dir ./stats/docvqa

# TextVQA-like
python build_stats_multi_dataset.py \
  --dataset textvqa \
  --data_json /data/dataset/textvqa/val_questions.json \
  --image_root /data/dataset/textvqa/images \
  --num_samples 500 \
  --output_dir ./stats/textvqa
"""

import os
import json
import argparse
from typing import List, Tuple, Dict, Any, Optional

import torch
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer


# -----------------------------
# Dataset loaders
# -----------------------------
def resolve_image_path(image_name: str, image_root: str, json_path: str) -> str:
    """Resolve relative image paths."""
    if os.path.isabs(image_name):
        return image_name
    if image_root:
        return os.path.join(image_root, image_name)
    return os.path.join(os.path.dirname(json_path), image_name)


def load_synthdog_dataset(json_path: str, num_samples: Optional[int] = None, image_root: str = "") -> List[Tuple[str, str]]:
    """
    SynthDog format (your current):
      item['image_name']
      item['2_coord'] is list of dicts with 'chunk'
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data[:num_samples] if num_samples else data
    pairs: List[Tuple[str, str]] = []

    for item in items:
        image_name = item.get("image_name")
        if not image_name:
            continue
        image_path = resolve_image_path(image_name, image_root, json_path)

        # text from 2_coord blocks
        text = ""
        coord = item.get("2_coord", [])
        if isinstance(coord, list) and len(coord) > 0:
            chunks = [c.get("chunk", "") for c in coord if isinstance(c, dict) and c.get("chunk")]
            if chunks:
                text = " ".join(chunks[:5])  # take first 5 blocks

        # fallback to other fields if needed
        if not text:
            text = item.get("text") or item.get("question") or item.get("prompt") or ""

        if text:
            pairs.append((image_path, text))

    return pairs


def load_docvqa_like_dataset(json_path: str, num_samples: Optional[int] = None, image_root: str = "") -> List[Tuple[str, str]]:
    """
    DocVQA-like (common):
      item['image'] or item['image_path'] or item['image_name']
      item['question'] or item['query'] or item['text']
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # can be {"data": [...]} or list
    items = data.get("data", data)
    if not isinstance(items, list):
        raise ValueError("DocVQA-like json must be a list or contain key 'data' as a list.")

    items = items[:num_samples] if num_samples else items
    pairs: List[Tuple[str, str]] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        img = item.get("image") or item.get("image_path") or item.get("image_name")
        if not img:
            continue
        image_path = resolve_image_path(str(img), image_root, json_path)

        q = item.get("question") or item.get("query") or item.get("text") or item.get("prompt")
        if not q:
            continue

        pairs.append((image_path, str(q)))

    return pairs


def load_textvqa_dataset(json_path: str, num_samples: Optional[int] = None, image_root: str = "") -> List[Tuple[str, str]]:
    """
    TextVQA-like (common):
      data: {"data":[{"question":..., "image_id":...}, ...]} or list
    Image naming varies by local files. This loader uses:
      - if image_id is int -> f"{image_id}.jpg"
      - else if image_id string with extension -> use as is
      - else append ".jpg"
    You may need to modify the naming rule to match your local files.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("data", data)
    if not isinstance(items, list):
        raise ValueError("TextVQA json must be a list or contain key 'data' as a list.")

    items = items[:num_samples] if num_samples else items
    pairs: List[Tuple[str, str]] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        q = item.get("question")
        if not q:
            continue

        image_id = item.get("image_id") or item.get("image") or item.get("img_id")
        if image_id is None:
            continue

        # ---- image file naming rule (adjust if needed) ----
        if isinstance(image_id, int):
            image_name = f"{image_id}.jpg"
        else:
            image_name = str(image_id)
            if not (image_name.endswith(".jpg") or image_name.endswith(".png") or image_name.endswith(".jpeg")):
                image_name += ".jpg"
        # --------------------------------------------------

        image_path = resolve_image_path(image_name, image_root, json_path)
        pairs.append((image_path, str(q)))

    return pairs


def load_dataset(dataset: str, json_path: str, num_samples: Optional[int], image_root: str) -> List[Tuple[str, str]]:
    if dataset == "synthdog":
        return load_synthdog_dataset(json_path, num_samples, image_root)
    if dataset == "docvqa":
        return load_docvqa_like_dataset(json_path, num_samples, image_root)
    if dataset == "textvqa":
        return load_textvqa_dataset(json_path, num_samples, image_root)
    raise ValueError(f"Unknown dataset: {dataset}")


# -----------------------------
# Statistics builder
# -----------------------------
class StatisticsTableBuilder:
    """Layer-wise stats collector."""

    def __init__(self, model, processor, tokenizer, num_layers: int = 28, device: str = "cuda"):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        self.device = device

        self.layer_stats: Dict[int, Dict[str, List[float]]] = {
            i: {"prefill_means": [], "prefill_stds": [], "decode_means": [], "decode_stds": []}
            for i in range(num_layers)
        }

    @staticmethod
    def compute_global_statistics(hidden_states: torch.Tensor) -> Tuple[float, float]:
        """
        hidden_states: [bs, seq, hidden]
        Return: global mean/std over all entries
        """
        flat = hidden_states.reshape(-1)
        mean = flat.mean().item()
        std = flat.std().item()
        return mean, std

    def collect_sample_statistics(self, image_path: str, text: str, max_decode_tokens: int = 64) -> bool:
        try:
            # -----------------------------
            # Prefill
            # -----------------------------
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": text}
                ]
            }]

            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )

            hidden_states_tuple = outputs.hidden_states  # (num_layers+1,), each [1, seq, hidden]

            prefill_stats_per_layer: Dict[int, Tuple[float, float]] = {}
            for layer_idx in range(self.num_layers):
                hs = hidden_states_tuple[layer_idx + 1]
                mean, std = self.compute_global_statistics(hs)
                prefill_stats_per_layer[layer_idx] = (mean, std)

            # -----------------------------
            # Decode
            # -----------------------------
            decode_stats_per_layer = {i: {"means": [], "stds": []} for i in range(self.num_layers)}
            generated_ids = inputs.input_ids.clone()
            past_key_values = None

            for step in range(max_decode_tokens):
                with torch.no_grad():
                    if step == 0:
                        outputs = self.model(
                            input_ids=generated_ids,
                            attention_mask=inputs.attention_mask,
                            past_key_values=past_key_values,
                            output_hidden_states=True,
                            return_dict=True,
                            use_cache=True
                        )
                    else:
                        # grow attention mask to include generated tokens
                        attn_mask = torch.cat([
                            inputs.attention_mask,
                            torch.ones(1, step, device=self.device, dtype=inputs.attention_mask.dtype)
                        ], dim=1)

                        outputs = self.model(
                            input_ids=generated_ids[:, -1:],
                            attention_mask=attn_mask,
                            past_key_values=past_key_values,
                            output_hidden_states=True,
                            return_dict=True,
                            use_cache=True
                        )

                past_key_values = outputs.past_key_values
                hidden_states_tuple = outputs.hidden_states

                for layer_idx in range(self.num_layers):
                    hs = hidden_states_tuple[layer_idx + 1]
                    last_hidden = hs[:, -1:, :]  # [1, 1, hidden]
                    mean, std = self.compute_global_statistics(last_hidden)
                    decode_stats_per_layer[layer_idx]["means"].append(mean)
                    decode_stats_per_layer[layer_idx]["stds"].append(std)

                logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

            # -----------------------------
            # Save per-sample averages into global buffers
            # -----------------------------
            for layer_idx in range(self.num_layers):
                if layer_idx not in prefill_stats_per_layer:
                    continue
                if len(decode_stats_per_layer[layer_idx]["means"]) == 0:
                    continue

                prefill_mean, prefill_std = prefill_stats_per_layer[layer_idx]

                avg_decode_mean = float(sum(decode_stats_per_layer[layer_idx]["means"]) /
                                        len(decode_stats_per_layer[layer_idx]["means"]))
                avg_decode_std = float(sum(decode_stats_per_layer[layer_idx]["stds"]) /
                                       len(decode_stats_per_layer[layer_idx]["stds"]))

                self.layer_stats[layer_idx]["prefill_means"].append(prefill_mean)
                self.layer_stats[layer_idx]["prefill_stds"].append(prefill_std)
                self.layer_stats[layer_idx]["decode_means"].append(avg_decode_mean)
                self.layer_stats[layer_idx]["decode_stds"].append(avg_decode_std)

            return True

        except Exception as e:
            print(f"[WARN] Error processing sample: {e}")
            return False

    def compute_final_statistics(self) -> Dict[int, Dict[str, Any]]:
        statistics_table: Dict[int, Dict[str, Any]] = {}

        for layer_idx in range(self.num_layers):
            stats = self.layer_stats[layer_idx]
            if len(stats["prefill_means"]) == 0:
                continue

            prefill_mean_avg = float(sum(stats["prefill_means"]) / len(stats["prefill_means"]))
            prefill_std_avg = float(sum(stats["prefill_stds"]) / len(stats["prefill_stds"]))
            decode_mean_avg = float(sum(stats["decode_means"]) / len(stats["decode_means"]))
            decode_std_avg = float(sum(stats["decode_stds"]) / len(stats["decode_stds"]))

            mean_shift = decode_mean_avg - prefill_mean_avg
            std_ratio = decode_std_avg / (prefill_std_avg + 1e-8)

            statistics_table[layer_idx] = {
                "mean_shift": mean_shift,
                "std_ratio": std_ratio,
                "prefill_mean_avg": prefill_mean_avg,
                "prefill_std_avg": prefill_std_avg,
                "decode_mean_avg": decode_mean_avg,
                "decode_std_avg": decode_std_avg,
                "num_samples": len(stats["prefill_means"])
            }

        return statistics_table


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Build statistics table for Qwen2-VL (multi-dataset loaders)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Qwen2-VL model")
    parser.add_argument("--dataset", type=str, default="synthdog",
                        choices=["synthdog", "docvqa", "textvqa"],
                        help="Dataset loader to use")
    parser.add_argument("--data_json", type=str, required=True, help="Path to dataset json/annotation file")
    parser.add_argument("--image_root", type=str, default="",
                        help="Optional image root directory (if paths in json are relative)")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples")
    parser.add_argument("--max_decode_tokens", type=int, default=64, help="Max decode tokens per sample")
    parser.add_argument("--num_layers", type=int, default=28, help="Number of transformer layers")
    parser.add_argument("--output_dir", type=str, default="./statistics_table", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Building Statistics Table for Qwen2-VL")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Dataset loader: {args.dataset}")
    print(f"Data json: {args.data_json}")
    print(f"Image root: {args.image_root or '(auto)'}")
    print(f"Num samples: {args.num_samples}")
    print(f"Max decode tokens: {args.max_decode_tokens}")
    print(f"Num layers: {args.num_layers}")
    print(f"Output: {args.output_dir}")
    print()

    print("Loading model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print("Model loaded.\n")

    print("Loading dataset...")
    image_text_pairs = load_dataset(args.dataset, args.data_json, args.num_samples, args.image_root)
    print(f"Loaded {len(image_text_pairs)} samples\n")

    builder = StatisticsTableBuilder(model, processor, tokenizer, num_layers=args.num_layers)

    print("Collecting statistics...")
    successful = 0
    for idx, (image_path, text) in enumerate(tqdm(image_text_pairs, desc="Processing samples")):
        ok = builder.collect_sample_statistics(image_path, text, max_decode_tokens=args.max_decode_tokens)
        successful += int(ok)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/{len(image_text_pairs)} | successful: {successful}")

    print(f"\nTotal successful samples: {successful}/{len(image_text_pairs)}\n")

    print("Computing final statistics table...")
    statistics_table = builder.compute_final_statistics()

    out_json = os.path.join(args.output_dir, "layer_statistics_table.json")
    with open(out_json, "w") as f:
        json.dump(statistics_table, f, indent=2)
    print(f"Saved: {out_json}\n")

    # Print a short summary
    print("=" * 80)
    print("STATISTICS TABLE SUMMARY")
    print("=" * 80)
    print(f"{'Layer':<6} | {'Mean Shift':<12} | {'Std Ratio':<12} | {'Samples':<8}")
    print("-" * 60)
    for layer_idx in sorted(statistics_table.keys()):
        s = statistics_table[layer_idx]
        print(f"{layer_idx:<6} | {s['mean_shift']:>12.6f} | {s['std_ratio']:>12.6f} | {s['num_samples']:<8}")
    print("=" * 80)
    print()

    # Save report
    report_file = os.path.join(args.output_dir, "statistics_report.txt")
    with open(report_file, "w") as f:
        f.write("QWEN2-VL LAYER STATISTICS TABLE\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset loader: {args.dataset}\n")
        f.write(f"Data json: {args.data_json}\n")
        f.write(f"Image root: {args.image_root or '(auto)'}\n")
        f.write(f"Successful samples: {successful}/{len(image_text_pairs)}\n")
        f.write(f"Max decode tokens: {args.max_decode_tokens}\n\n")
        f.write("USAGE:\n")
        f.write("  decode_mean = prefill_mean + mean_shift\n")
        f.write("  decode_std  = prefill_std  * std_ratio\n\n")
        f.write("TABLE:\n")
        f.write(f"{'Layer':<6} | {'Mean Shift':<12} | {'Std Ratio':<12} | "
                f"{'Prefill Mean':<12} | {'Prefill Std':<12} | "
                f"{'Decode Mean':<12} | {'Decode Std':<12} | {'Samples':<8}\n")
        f.write("-" * 120 + "\n")
        for layer_idx in sorted(statistics_table.keys()):
            s = statistics_table[layer_idx]
            f.write(f"{layer_idx:<6} | "
                    f"{s['mean_shift']:>12.6f} | "
                    f"{s['std_ratio']:>12.6f} | "
                    f"{s['prefill_mean_avg']:>12.6f} | "
                    f"{s['prefill_std_avg']:>12.6f} | "
                    f"{s['decode_mean_avg']:>12.6f} | "
                    f"{s['decode_std_avg']:>12.6f} | "
                    f"{s['num_samples']:<8}\n")
        f.write("=" * 80 + "\n")

    print(f"Saved: {report_file}")
    print("Done!")


if __name__ == "__main__":
    main()
