import warnings
from time import time
from tqdm import tqdm
import json
import pprint
import os
import sys

import torch
from transformers import DynamicCache, AutoProcessor, Qwen2VLForConditionalGeneration

# ===== Optional: LLaVA imports (not used here) =====
try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.constants import (
        IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN, IGNORE_INDEX
    )
    from llava.conversation import conv_templates, SeparatorStyle
except ImportError:
    pass

# ===== SparseMM imports =====
try:
    from sparsemm.monkeypatch import replace_qwen
    from sparsemm.sparsemm_utils import DynamicCacheSplitHeadFlatten
except Exception as e:
    print(f"import sparsemm failed, error: {e}")
    replace_qwen = None
    DynamicCacheSplitHeadFlatten = None

warnings.filterwarnings("ignore")

# ===== environment =====
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = "cuda:0"

# ===== load Qwen2-VL =====
max_pixels: int = 16384 * 28 * 28
min_pixels: int = 32 * 28 * 28

pretrained = "/data/model/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    pretrained,
    torch_dtype="auto",
    attn_implementation="flash_attention_2",
).to(device).eval()

processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
tokenizer = processor.tokenizer

# Use a stable, valid token id instead of torch.arange (important for VL models)
HELLO_ID = tokenizer.encode("hello", add_special_tokens=False)[0]


# =========================
# Helpers
# =========================
def reset_kv_seq_len(model):
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "kv_seq_len"):
            layer.self_attn.kv_seq_len = 0


def clean_kv_cluster(model):
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "kv_cluster"):
            delattr(layer.self_attn, "kv_cluster")


def build_inputs(n_tokens: int):
    input_ids = torch.full((1, n_tokens), HELLO_ID, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def get_size_of_cache(cache):
    # Works for HF DynamicCache and SparseMM's DynamicCacheSplitHeadFlatten
    if isinstance(cache, DynamicCache) or (
        DynamicCacheSplitHeadFlatten is not None and isinstance(cache, DynamicCacheSplitHeadFlatten)
    ):
        size_in_bytes = 0
        for v in cache.value_cache:
            if v is None:
                continue
            size_in_bytes += v.element_size() * v.nelement()
        for k in cache.key_cache:
            if k is None:
                continue
            size_in_bytes += k.element_size() * k.nelement()
        return size_in_bytes

    raise NotImplementedError(f"{type(cache)} is not supported yet.")


def make_cache(method: str):
    if method in ["adakv", "sparsemm", "sparsemm_query"]:
        if DynamicCacheSplitHeadFlatten is None:
            raise RuntimeError("DynamicCacheSplitHeadFlatten not available but method requires it.")
        return DynamicCacheSplitHeadFlatten()
    return DynamicCache()


# =========================
# Stats
# =========================
def get_prefilling_stats(model, n_tokens, method="fullkv"):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    idle_peak_memory = torch.cuda.max_memory_allocated()

    model.to(device)
    initial_peak_memory = torch.cuda.max_memory_allocated()

    input_ids, attention_mask = build_inputs(n_tokens)

    # warmup
    reset_kv_seq_len(model)
    with torch.no_grad():
        _ = model(
            input_ids=input_ids[:, :123],
            attention_mask=attention_mask[:, :123],
            use_cache=True,
        )
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    reset_kv_seq_len(model)

    with torch.no_grad():
        cache = make_cache(method)
        start = time()
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
        prefilling_time = time() - start

        # IMPORTANT: use returned cache (may be a new object)
        cache_out = output.past_key_values
        cache_size = get_size_of_cache(cache_out)

        del cache
        del output

    peak_memory = torch.cuda.max_memory_allocated()

    model.cpu()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return {
        "Idle Peak memory": idle_peak_memory / 1024**3,
        "Initial Peak memory": initial_peak_memory / 1024**3,
        "Prefilling time": prefilling_time,
        "Peak memory usage": peak_memory / 1024**3,
        "Cache Size": cache_size / 1024**3,
        "Peak memory w/o weights and KV cache (GB)": (peak_memory - cache_size - initial_peak_memory) / 1024**3,
    }


def get_generation_stats(model, n_tokens, max_new_tokens=512, method="fullkv"):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    idle_peak_memory = torch.cuda.max_memory_allocated()

    model.to(device)
    reset_kv_seq_len(model)

    initial_peak_memory = torch.cuda.max_memory_allocated()

    input_ids, attention_mask = build_inputs(n_tokens)

    start = time()
    with torch.no_grad():
        _ = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=False,
            eos_token_id=None,  # disable early stopping for consistent timing
        )
    total_time = time() - start

    peak_memory = torch.cuda.max_memory_allocated()

    model.cpu()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return {
        "Idle Peak memory": idle_peak_memory / 1024**3,
        "Initial Peak memory": initial_peak_memory / 1024**3,
        "Total time": total_time,
        "Peak memory usage": peak_memory / 1024**3,
    }


def get_decoding_stats(model, n_tokens, max_new_tokens=100, method="fullkv"):
    """
    Returns:
      - Decoding latency (sec/token)
      - Decode peak memory usage (GB)   <-- THIS IS WHAT YOU ASKED FOR
    Measured by:
      1) Prefill to build cache
      2) reset_peak_memory_stats()
      3) run decode loop only, then read max_memory_allocated
    """
    torch.cuda.empty_cache()

    model.to(device)
    reset_kv_seq_len(model)

    input_ids, attention_mask = build_inputs(n_tokens)
    position_ids = torch.arange(0, n_tokens, device=device).unsqueeze(0)

    # ---- 1) Prefill ----
    with torch.no_grad():
        cache = make_cache(method)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
    cache = outputs.past_key_values

    next_token = outputs.logits[0, -1].argmax()
    position_ids = position_ids[:, -1:] + 1

    # ---- 2) Start decode-only measurement ----
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time()
    with torch.no_grad():
        for i in range(max_new_tokens):
            outputs = model(
                input_ids=next_token.view(1, 1),
                past_key_values=cache,
                position_ids=position_ids + i,
                use_cache=True,
            )
            cache = outputs.past_key_values
            next_token = outputs.logits[0, -1].argmax()
    torch.cuda.synchronize()
    total_time = time() - start

    decode_peak_memory = torch.cuda.max_memory_allocated() / 1024**3

    model.cpu()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return {
        "Decoding latency": total_time / max_new_tokens,
        "Decode peak memory usage": decode_peak_memory,
    }


def combine_stats(prefilling_stats, generation_stats, decoding_stats):
    return {
        "Peak memory usage": generation_stats["Peak memory usage"],  # end-to-end generate peak
        "Prefilling time": prefilling_stats["Prefilling time"],
        "Cache Size": prefilling_stats["Cache Size"],
        "Total time": generation_stats["Total time"],
        "Generation time": generation_stats["Total time"] - prefilling_stats["Prefilling time"],
        "Decoding latency": decoding_stats["Decoding latency"],
        "Decode peak memory usage": decoding_stats["Decode peak memory usage"],  # decode-only peak
    }


# =========================
# Main
# =========================
def main():
    stats = {}

    methods = ["full"]  # e.g. ["fullkv", "sparsemm", "sparsemm_query"]
    token_lengths = [2000, 4000, 8000, 16000, 32000]

    for method in methods:
        print(f"\n========== Method: {method} ==========")
        os.environ["METHOD"] = method

        clean_kv_cluster(model)

        if method != "fullkv":
            if replace_qwen is None:
                raise RuntimeError("replace_qwen not available (sparsemm import failed).")
            replace_qwen(method)

        for n_tokens in tqdm(token_lengths, desc=f"{method}"):
            prefilling_stats = get_prefilling_stats(model, n_tokens=n_tokens, method=method)
            generation_stats = get_generation_stats(model, n_tokens=n_tokens, method=method)
            decoding_stats = get_decoding_stats(model, n_tokens=n_tokens, method=method)

            stats[f"{method}-{n_tokens}"] = combine_stats(prefilling_stats, generation_stats, decoding_stats)

            print(method, n_tokens)
            pprint.pprint(stats[f"{method}-{n_tokens}"])

    out_path = "./results/qwen25/stats_full1.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
