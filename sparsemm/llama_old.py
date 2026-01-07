import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import warnings
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
    _flash_attention_forward
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import (
    logging,
)
from sparsemm.sparsemm_utils import init_snapkv, init_pyramidkv, init_adakv, init_sparsemm, init_mask
import math
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
from sparsemm.sparsemm_utils import DynamicCacheSplitHeadFlatten
import numpy as np
import csv
import os

import time


logger = logging.get_logger(__name__)

def llama_flash_attn2_forward_SnapKV(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
            "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
        )
    init_snapkv(self)

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    if past_key_value is not None:
        # NOTE: decoding update
        if q_len == 1:# decode阶段不管
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        else:# prefill阶段先根据注意力图更新了
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states)
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        position_ids=position_ids,
        dropout=dropout_rate,
        sliding_window=getattr(self, "sliding_window", None),
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
        is_causal=self.is_causal,
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def llama_flash_attn2_forward_PyramidKV(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
            "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
        )
    init_pyramidkv(self)

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    if past_key_value is not None:
        # NOTE: decoding update
        if q_len == 1:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        else:
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states)
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        position_ids=position_ids,
        dropout=dropout_rate,
        sliding_window=getattr(self, "sliding_window", None),
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
        is_causal=self.is_causal,
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def llama_flash_attn2_forward_AdaKV(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
            "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
        )
    init_adakv(self)
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += cache_position[0]

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    dropout_rate = self.attention_dropout if self.training else 0.0

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
        if (
            getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
            and cache_has_contents
        ):
            slicing_tokens = 1 - self.config.sliding_window

            past_key = past_key_value[self.layer_idx][0]
            past_value = past_key_value[self.layer_idx][1]

            past_key = past_key[:, :, slicing_tokens:, :].contiguous()
            past_value = past_value[:, :, slicing_tokens:, :].contiguous()

            if past_key.shape[-2] != self.config.sliding_window - 1:
                raise ValueError(
                    f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                    f" {past_key.shape}"
                )

            if attention_mask is not None:
                attention_mask = attention_mask[:, slicing_tokens:]
                attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)
    


    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    
    is_prefill = q_len != 1

    if is_prefill:
        key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states)
        past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    else:
        cache_kwargs["head_lens"] = self.kv_cluster.head_lens
        cache_kwargs["cu_klen"] = self.kv_cluster.cu_klen
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)


        # NOTE: update meta data
        self.kv_cluster.klen_sum += self.num_heads
        self.kv_cluster.max_seqlen_k += 1
        self.kv_cluster.cu_klen += self.kv_cluster.cu_offset
        self.kv_cluster.head_lens += 1

        query_states = query_states.view(-1, self.num_key_value_groups, self.head_dim)
        key_states = key_states.view(-1,1,self.head_dim)
        value_states = value_states.view(-1,1,self.head_dim)

        cu_seqlens_q = self.kv_cluster.cu_qlen
        cu_seqlens_k = self.kv_cluster.cu_klen
        max_seqlen_q = 1
        max_seqlen_k = self.kv_cluster.max_seqlen_k

        attn_output = flash_attn_varlen_func(query_states, key_states, value_states, cu_seqlens_q,
                                             cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=True)
        #  TODO: support batch size > 1
        assert bsz == 1
        attn_output = attn_output.reshape(bsz, self.num_heads, q_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def plot_attn_probs_this_step_bar(
    out,                      # attn_probs_this_step 的返回: list[Tensor]
    reduce_over_gqa=None,     # None | 'mean' | 'max'，要与你调用时一致
    save_dir="./attn_step_vis_bar",
    topk=None,                # 仅对柱状图有效（mean/max）
    cmap="viridis"            # 热力图 colormap
):
    """
    可视化 attn_probs_this_step 的输出。
      - None: 画 [G, Lh] 热力图
      - mean/max: 画 [Lh] 柱状图
    """
    import os, numpy as np, torch, matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)

    def _to_np(x: torch.Tensor): return x.detach().float().cpu().numpy()

    Hk = len(out)
    for h in range(Hk):
        x = out[h]
        fn = os.path.join(save_dir, f"head{h:02d}.png")

        if reduce_over_gqa is None:
            # [G, Lh] 热力图
            arr = _to_np(x)
            G, Lh = arr.shape
            plt.figure(figsize=(10, max(2.5, 2.5 * G / 2)))
            plt.imshow(arr, aspect="auto", interpolation="nearest", cmap=cmap)
            plt.colorbar(fraction=0.025, pad=0.04)
            plt.xlabel("Key index (Lh)")
            plt.ylabel("Group index (G)")
            plt.title(f"Head {h}  (shape: G={G}, Lh={Lh})")
            plt.tight_layout()
            plt.savefig(fn, dpi=200)
            plt.close()
        else:
            # [Lh] 柱状图
            vec = _to_np(x)
            Lh = vec.shape[0]
            plt.figure(figsize=(10, 3))
            xs = np.arange(Lh)
            plt.bar(xs, vec, width=0.8)
            plt.xlabel("Key index (Lh)")
            plt.ylabel("Attention prob")
            plt.title(f"Head {h}  (shape: Lh={Lh}, reduce={reduce_over_gqa})")

            # 标注 top-k
            if topk is not None and topk > 0:
                k = min(topk, Lh)
                idx = np.argpartition(vec, -k)[-k:]
                idx = idx[np.argsort(-vec[idx])]
                plt.scatter(idx, vec[idx], s=25, color="red", zorder=5)
                for r, (xi, yi) in enumerate(zip(idx, vec[idx])):
                    plt.text(xi, yi, f"#{r+1}", fontsize=8, ha="center", va="bottom")

            plt.tight_layout()
            plt.savefig(fn, dpi=200)
            plt.close()

    print(f"[OK] Saved {Hk} figures to: {save_dir}")



@torch.no_grad()
def attn_probs_this_step(key_states, query_states, cu_seqlens_k, cu_seqlens_q,
                         head_dim: int, num_key_value_groups: int,
                         reduce_over_gqa: str | None = None):
    """
    key_states:   [sum_k, 1, D]  （你前面 .view(-1, 1, D) 的形状）
    query_states: [H_k, G, D]    （你前面 .view(-1, G, D) 的形状；decode 每头 q_len=1）
    cu_seqlens_k: [H_k+1]        （每个 KV 头在展平 K 里的起止）
    cu_seqlens_q: [H_k+1]        （每个 '条目' 的 Q 起止；decode 时通常是 [0,1,2,...,H_k]）
    reduce_over_gqa: None | 'mean' | 'max'
        - None: 返回每个 KV 头下 **每个 G 头** 的注意力分布，形状 [H_k, G, len(K_h)]
        - 'mean' / 'max': 先在 G 维做聚合，返回 [H_k, len(K_h)]
    """
    assert key_states.ndim == 3 and key_states.size(1) == 1
    assert query_states.ndim == 3
    H_k = cu_seqlens_k.numel() - 1
    G   = num_key_value_groups
    D   = head_dim

    # 一致性检查（建议保留）
    assert query_states.size(0) == H_k and query_states.size(1) == G and query_states.size(2) == D
    assert cu_seqlens_q[-1].item() == H_k, "decode下每头1个query，cu_seqlens_q应为[0,1,...,H_k]"
    assert cu_seqlens_k.dtype == torch.int32 and cu_seqlens_q.dtype == torch.int32
    assert cu_seqlens_k.device == key_states.device == query_states.device

    out = []  # list of length H_k; 每项是 [G, len(K_h)] 或聚合后 [len(K_h)]
    scale = 1.0 / (D ** 0.5)

    for h in range(H_k):
        ks, ke = int(cu_seqlens_k[h].item()), int(cu_seqlens_k[h+1].item())
        # K_h: [len(K_h), D]; Q_h: [G, D]
        K_h = key_states[ks:ke, 0, :]                       # [Lh, D]
        Q_h = query_states[h, :, :]                         # [G,  D]

        # logits: [G, Lh]
        logits = (Q_h @ K_h.t()) * scale
        probs  = torch.softmax(logits.float(), dim=-1).to(Q_h.dtype)  # [G, Lh]

        if reduce_over_gqa is None:
            out.append(probs)                                # [G, Lh]
        elif reduce_over_gqa == 'mean':
            out.append(probs.mean(dim=0))                    # [Lh]
        elif reduce_over_gqa == 'max':
            out.append(probs.max(dim=0).values)              # [Lh]
        else:
            raise ValueError("reduce_over_gqa must be None|'mean'|'max'")

    return out  # list[Tensor]; 按需再拼接或导出

@torch.no_grad()
def attn_probs_this_step_fast(
    key_states,                 # [sum_k, 1, D]
    query_states,               # [H_k, G, D]（decode: 每头 q_len=1）
    cu_seqlens_k, cu_seqlens_q, # int32, CUDA
    head_dim: int,
    num_key_value_groups: int,
    reduce_over_gqa: str | None = None,
    acc_dtype: torch.dtype = torch.float32,
):
    H_k = cu_seqlens_k.numel() - 1
    G   = num_key_value_groups
    D   = head_dim
    device = key_states.device

    assert query_states.shape == (H_k, G, D)
    assert cu_seqlens_q[-1].item() == H_k
    assert cu_seqlens_k.dtype == torch.int32 and cu_seqlens_q.dtype == torch.int32

    # 1) 计算每个 KV 头的长度与 Lmax
    lengths = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to(torch.int64)  # [H_k]
    Lmax = int(lengths.max().item())

    # 2) pad K -> [H_k, Lmax, D]，并构造 mask: True 表示 padding
    ar = torch.arange(Lmax, device=device).view(1, Lmax)              # [1, Lmax]
    valid = ar < lengths.view(-1, 1)                                  # [H_k, Lmax]
    key_mask = ~valid                                                 # [H_k, Lmax], True=pad

    # 从展平的 K_flat 还原到 K_pad（避免 Python for）
    K_flat = key_states[:, 0, :]                                      # [sum_k, D]
    base   = cu_seqlens_k[:-1].to(torch.int64).view(-1, 1)            # [H_k,1]
    rel    = ar.expand(H_k, Lmax)                                     # [H_k,Lmax]
    src_idx= (base + torch.minimum(rel, lengths.view(-1,1)-1)).view(-1)  # [H_k*Lmax]
    gathered = K_flat.index_select(0, src_idx).view(H_k, Lmax, D)     # [H_k,Lmax,D]
    K_pad = torch.where(valid.unsqueeze(-1), gathered, gathered.new_zeros(()).expand_as(gathered))

    # 3) logits = Q @ K^T / sqrt(D)  → [H_k, G, Lmax]
    Q = query_states.to(acc_dtype)             # [H_k,G,D]
    K = K_pad.to(acc_dtype)                    # [H_k,Lmax,D]
    logits = torch.einsum('hgd,hld->hgl', Q, K) / (D ** 0.5)  # [H_k,G,Lmax]

    # 4) masked softmax 沿 Lmax
    neg_inf = torch.finfo(acc_dtype).min
    logits = logits.masked_fill(key_mask.unsqueeze(1), neg_inf)       # [H_k,G,Lmax]
    probs  = torch.softmax(logits, dim=-1).to(query_states.dtype)     # [H_k,G,Lmax]

    # 5) 输出与原函数兼容：list[Tensor]
    out = []
    if reduce_over_gqa is None:
        # 每头切回各自真实长度 → [G, Lh]
        for h in range(H_k):
            Lh = int(lengths[h].item())
            out.append(probs[h, :, :Lh].contiguous())
    elif reduce_over_gqa == 'mean':
        probs_mean = probs.mean(dim=1)                                 # [H_k,Lmax]
        for h in range(H_k):
            Lh = int(lengths[h].item())
            out.append(probs_mean[h, :Lh].contiguous())                # [Lh]
    elif reduce_over_gqa == 'max':
        probs_max = probs.max(dim=1).values                            # [H_k,Lmax]
        for h in range(H_k):
            Lh = int(lengths[h].item())
            out.append(probs_max[h, :Lh].contiguous())                 # [Lh]
    else:
        raise ValueError("reduce_over_gqa must be None|'mean'|'max'")

    return out

@torch.no_grad()
def attn_probs_this_step_fast(
    key_states,                 # [sum_k, 1, D]
    query_states,               # [H_k, G, D]（decode: 每头 q_len=1）
    cu_seqlens_k, cu_seqlens_q, # int32, CUDA
    head_dim: int,
    num_key_value_groups: int,
    reduce_over_gqa: str | None = None,
    acc_dtype: torch.dtype = torch.float32,
    *,
    prefill_len: int | None = None,   # ★ 新增：只计算前 L0 段
):
    H_k = cu_seqlens_k.numel() - 1
    G   = num_key_value_groups
    D   = head_dim
    device = key_states.device
    

    
    assert query_states.shape == (H_k, G, D)
    assert cu_seqlens_q[-1].item() == H_k
    assert cu_seqlens_k.dtype == torch.int32 and cu_seqlens_q.dtype == torch.int32

    # torch.cuda.synchronize()
    t0 = time.time()
    # 1) 真实长度（保持 int64 以便做 index_select）
    # lengths = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to(torch.int64)  # [H_k]
    H_k = 32
    Lcap = int((cu_seqlens_k[1] - cu_seqlens_k[0]).item())#0.02ms 0.12ms
    
    K_flat = key_states[:, 0, :]                          # [sum_k, D] (contiguous along dim=0)
    t0 = time.time()
    gathered = K_flat.narrow(0, 0, H_k * Lcap).view(H_k, Lcap, D)  # [H_k, Lcap, D]
    # 5) QK^T / sqrt(D)
    Q = query_states.to(acc_dtype)           # [H_k, G, D]
    K = gathered.to(acc_dtype)               # [H_k, Lcap, D]
    logits = torch.einsum('hgd,hld->hgl', Q, K) / (D ** 0.5)   # [H_k, G, Lcap] 0.1

    # torch.cuda.synchronize()
    # t1 = time.time()
    # print(f"[Timer] gather+matmul took {(t1 - t0)*1000:.3f} ms")# [Timer] gather+matmul took 0.787 ms


    probs = torch.softmax(logits, dim=-1).to(query_states.dtype)   # [H_k, G, Lcap]0.02ms
        

    # 7) reduce_over_gqa 兼容输出
    if reduce_over_gqa is None:
        out = probs[:, :, :Lcap].contiguous()                 # [H_k, G, Lcap]
    elif reduce_over_gqa == 'mean':
        out = probs.mean(dim=1)[:, :Lcap].contiguous()        # [H_k, Lcap]
    elif reduce_over_gqa == 'max':
        out = probs.max(dim=1).values[:, :Lcap].contiguous()  # [H_k, Lcap]
    else:
        raise ValueError("reduce_over_gqa must be None|'mean'|'max'")

    # 0.3ms
    return out
# @dataclass(frozen=True)
class PrunedKV:
    def __init__(self, key_states, value_states, cu_seqlens_k, max_seqlen_k, kept_idx_per_seq):
        self.key_states = key_states
        self.value_states = value_states
        self.cu_seqlens_k = cu_seqlens_k
        self.max_seqlen_k = max_seqlen_k
        self.kept_idx_per_seq = kept_idx_per_seq

# 测一下局部token的注意力分布
@torch.no_grad()
def lse_local_edges_only_this_step(
    key_states, query_states, cu_seqlens_k, cu_seqlens_q,
    head_dim: int, num_key_value_groups: int,
    top_k: int = 50,
    reduce_over_gqa: str | None = None,
):
    """
    仅在每个 head 的 K 序列的“前 top_k + 后 top_k”位置上计算局部 LSE：
      log_A_local = logsumexp( (q k^T) / sqrt(d) , over kept positions )

    返回：
      lse_list: list[Tensor]，每个元素对应一个 head，形状为：
          - reduce=None      -> [G]       （每个 GQA 组一个 LSE）
          - reduce='mean'    -> [] 标量   （G 维上平均）
          - reduce='max'     -> [] 标量   （G 维上最大）
      kept_local_idx: list[Tensor]，每个 head 的局部索引（相对该 head 的 [0..L_h-1]）
    """
    assert key_states.ndim == 3 and key_states.size(1) == 1
    assert query_states.ndim == 3
    H_k = cu_seqlens_k.numel() - 1
    G   = num_key_value_groups
    D   = head_dim

    assert query_states.size(0) == H_k and query_states.size(1) == G and query_states.size(2) == D
    assert cu_seqlens_q[-1].item() == H_k
    assert cu_seqlens_k.device == key_states.device == query_states.device
    assert reduce_over_gqa in (None, "mean", "max")

    device = key_states.device
    scale = 1.0 / math.sqrt(D)

    lse_list = []
    kept_local_idx = []

    for h in range(H_k):
        ks, ke = int(cu_seqlens_k[h].item()), int(cu_seqlens_k[h+1].item())
        K_h_full = key_states[ks:ke, 0, :]     # [L_h, D]
        Q_h      = query_states[h, :, :]       # [G,   D]
        L_h      = K_h_full.size(0)

        # 选择仅两端的局部索引
        if L_h >= 2 * top_k:
            idx_front = torch.arange(0,         top_k, device=device, dtype=torch.long)
            idx_back  = torch.arange(L_h-top_k, L_h,   device=device, dtype=torch.long)
            keep_idx  = torch.cat([idx_front, idx_back], dim=0)   # [2K]
        else:
            keep_idx  = torch.arange(0, L_h, device=device, dtype=torch.long)  # 不足 2K -> 用全长

        K_h = K_h_full.index_select(0, keep_idx)   # [2K' or L_h, D]
        kept_local_idx.append(keep_idx.detach().cpu())

        # 在保留子集上计算 logits，并对最后一维做 logsumexp
        # 使用 float32 计算以提升数值稳定性
        logits = (Q_h @ K_h.t()).float() * scale     # [G, 2K']
        lse = torch.logsumexp(logits, dim=-1)        # [G]，每个 G 一个 LSE
        lse = lse.to(Q_h.dtype)

        if reduce_over_gqa is None:
            lse_list.append(lse)                 # [G]
        elif reduce_over_gqa == "mean":
            lse_list.append(lse.mean())          # 标量（该 head 的 G 上均值）
        else:  # 'max'
            lse_list.append(lse.max())           # 标量（该 head 的 G 上最大）

    return lse_list, kept_local_idx


# 根据是否是检索头删除token
@torch.no_grad()
def prune_kv_per_sequence(
    key_states: torch.Tensor,        # [total_k, Hkv, Dh]
    value_states: torch.Tensor,      # [total_k, Hkv, Dh]
    cu_seqlens_k: torch.Tensor,      # [B+1], int32/int64
    top_k: int,
    prune_seq_flags: torch.Tensor,   # [B] bool
):
    # ---- 0) 设备与 dtype 规范化（健壮版）----
    assert key_states.shape == value_states.shape and key_states.dim() == 3, "K/V 形状不一致或维度!=3"
    device = key_states.device

    # 大张量：不隐式搬运，只要求在同一 device（避免在你不知情时拷贝巨量数据）
    if value_states.device != device:
        raise RuntimeError(
            f"value_states.device={value_states.device} 与 key_states.device={device} 不一致；"
            "请在调用前把两者放到同一张卡上（例如 .to(key_states.device)）。"
        )

    # 小张量：自动搬到正确 device，且规范 dtype
    if cu_seqlens_k.device != device:
        cu_seqlens_k = cu_seqlens_k.to(device, non_blocking=True)
    if prune_seq_flags.device != device:
        prune_seq_flags = prune_seq_flags.to(device, non_blocking=True)

    # cu 必须是 int32 或 int64；flags 必须是 bool
    if cu_seqlens_k.dtype not in (torch.int32, torch.int64):
        # 与许多 CUDA kernel 兼容，优先 int32（也保留 int64 的可能下游）
        cu_seqlens_k = cu_seqlens_k.to(torch.int32)
    prune_seq_flags = prune_seq_flags.to(torch.bool)

    # 形状检查
    assert cu_seqlens_k.dim() == 1 and cu_seqlens_k.numel() >= 2, "cu_seqlens_k 需要形如 [B+1]"
    B = cu_seqlens_k.numel() - 1
    assert prune_seq_flags.dim() == 1 and prune_seq_flags.numel() == B, "prune_seq_flags 需要形如 [B]"

    # 其后沿用你的优化版主体逻辑……
    # ---- 1) 分段起点/长度（全在 device）----
    starts  = cu_seqlens_k[:-1]                      # [B]
    lengths = cu_seqlens_k[1:] - starts              # [B]

    need_prune = prune_seq_flags & (lengths >= 2 * top_k)
    seq_ids = torch.repeat_interleave(torch.arange(B, device=device), lengths)

    keep_mask = (~need_prune)[seq_ids].clone()

    if need_prune.any() and top_k > 0:
        K = top_k
        ar = torch.arange(K, device=device)
        head_rel = ar.unsqueeze(0).expand(B, -1)                  # [B,K]
        head_idx = (starts.unsqueeze(1) + head_rel)[need_prune]   # [N_h]
        keep_mask[head_idx] = True

        tail_rel = (lengths - K).unsqueeze(1) + head_rel          # [B,K]
        tail_idx = (starts.unsqueeze(1) + tail_rel)[need_prune]   # [N_t]
        keep_mask[tail_idx] = True

    kept_idx_global = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)

    new_lengths = torch.zeros_like(lengths)
    new_lengths.index_add_(0, seq_ids, keep_mask.to(new_lengths.dtype))
    new_cu = torch.empty_like(cu_seqlens_k)
    new_cu[0] = 0
    new_cu[1:] = new_lengths.cumsum(0)
    new_maxL = int(new_lengths.max().item()) if B > 0 else 0

    new_key_states   = key_states.index_select(0, kept_idx_global)
    new_value_states = value_states.index_select(0, kept_idx_global)

    # kept_idx_per_seq（相对索引, 放 CPU）
    lengths_cpu    = new_lengths.to("cpu", non_blocking=True).tolist()
    need_prune_cpu = need_prune.to("cpu", non_blocking=True).tolist()

    kept_idx_per_seq = []
    for L, npf in zip(lengths_cpu, need_prune_cpu):
        if npf and top_k > 0:
            head = torch.arange(0, top_k, dtype=torch.long)
            tail = torch.arange(L - top_k, L, dtype=torch.long)
            kept_idx_per_seq.append(torch.cat([head, tail], dim=0))
        else:
            kept_idx_per_seq.append(torch.arange(L, dtype=torch.long))

    return new_key_states, new_value_states, new_cu, new_maxL, kept_idx_per_seq

# 计算keystates的均值和方差
@torch.no_grad()
def key_stats_per_head(
    key_states, cu_seqlens_k,
    head_dim: int,
    compute_cov: bool = False,
):
    """
    计算每个 head 的 key 分布统计量：
      - mean_k[h]: [D]
      - var_k[h] 或 cov_k[h]: [D] 或 [D, D]
    参数：
      key_states: [Total_L, 1, D]，每个 head 的连续 key 向量拼接
      cu_seqlens_k: 累积序列长度前缀，len = H + 1
      head_dim: 每个 head 的维度 D
      compute_cov: 若 True 返回协方差矩阵，否则返回对角方差
    返回：
      mean_k_list: list[Tensor]，每个 [D]
      var_or_cov_list: list[Tensor]，每个 [D] 或 [D, D]
    """
    assert key_states.ndim == 3 and key_states.size(1) == 1
    H_k = cu_seqlens_k.numel() - 1
    D = head_dim
    device = key_states.device

    mean_k_list = []
    var_or_cov_list = []

    for h in range(H_k):
        ks, ke = int(cu_seqlens_k[h].item()), int(cu_seqlens_k[h + 1].item())
        K_h = key_states[ks:ke, 0, :]  # [L_h, D]
        L_h = K_h.size(0)
        if L_h == 0:
            mean_k_list.append(torch.zeros(D, device=device))
            if compute_cov:
                var_or_cov_list.append(torch.zeros(D, D, device=device))
            else:
                var_or_cov_list.append(torch.zeros(D, device=device))
            continue

        mean_k = K_h.mean(dim=0)  # [D]
        mean_k_list.append(mean_k)

        if compute_cov:
            # 协方差矩阵 Σ_K = E[(K - μ)(K - μ)^T]
            centered = K_h - mean_k
            cov_k = centered.t() @ centered / (L_h - 1)
            var_or_cov_list.append(cov_k)
        else:
            # 仅计算每维方差（对角）
            var_k = K_h.var(dim=0, unbiased=False)  # [D]
            var_or_cov_list.append(var_k)

    return mean_k_list, var_or_cov_list
# import torch
# import torch.nn.functional as F

# 求一下每个头的注意力shang
def compute_attention_entropy(query_states, key_states, causal=True, attn_mask=None):
    """
    query_states: [B, H, Tq, D]
    key_states:   [B, H, Tk, D]
    causal: 是否使用因果mask（prefill阶段通常是True）
    attn_mask: 形如 [B, 1, Tq, Tk] 的附加mask（可选），0为可见，-inf为遮挡
    return:
      ent_per_head: [H] 该层各头的平均熵（对 batch 与 query 取均值）
      ent_map: [B, H, Tq] 每个query位置的熵（如需更细分析）
    """
    B, H, Tq, D = query_states.shape
    Tk = key_states.shape[-2]

    # 计算注意力 logits（用 fp32 更稳）
    q = query_states.to(torch.float32)
    k = key_states.to(torch.float32)
    scale = 1.0 / math.sqrt(D)
    # [B,H,Tq,Tk]
    attn_logits = torch.matmul(q, k.transpose(-1, -2)) * scale

    # 因果mask：禁止看到未来位置（prefill/解码都建议打开）
    if causal:
        # [Tq, Tk]
        causal_mask = torch.triu(
            torch.full((Tq, Tk), float('-inf'), device=attn_logits.device),
            diagonal=1
        )
        attn_logits = attn_logits + causal_mask  # broadcast到[B,H,Tq,Tk]

    # 额外的注意力mask（如padding），形如 [B,1,Tq,Tk]
    if attn_mask is not None:
        attn_logits = attn_logits + attn_mask

    # 概率分布
    attn_probs = torch.softmax(attn_logits, dim=-1)

    # 熵：-∑ p log p
    eps = 1e-12
    ent = -(attn_probs * (attn_probs.clamp_min(eps).log())).sum(dim=-1)  # [B,H,Tq]

    # 聚合：对 batch 与 Tq 求均值，得到每个头的熵
    ent_per_head = ent.mean(dim=(0, 2))  # [H]

    return ent_per_head, ent

def llama_flash_attn2_forward_SparseMM(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional["Cache"] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    # --- checks ---
    from transformers.cache_utils import StaticCache
    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
            "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
        )

    # --- init ---
    init_sparsemm(self)  # 你的 helper
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    # --- projections ---
    query_states = self.q_proj(hidden_states)
    key_states   = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # [B,H,T,D] / [B,H_kv,T,D]
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states   = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None and cache_position is not None:
        kv_seq_len += cache_position[0]

    # --- RoPE ---
    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # --- cache slice for sliding window ---
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
        if (
            getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
            and cache_has_contents
        ):
            slicing_tokens = 1 - self.config.sliding_window
            past_key   = past_key_value[self.layer_idx][0][:, :, slicing_tokens:, :].contiguous()
            past_value = past_key_value[self.layer_idx][1][:, :, slicing_tokens:, :].contiguous()

            if past_key.shape[-2] != self.config.sliding_window - 1:
                raise ValueError(
                    f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got {past_key.shape}"
                )
            if attention_mask is not None:
                attention_mask = attention_mask[:, slicing_tokens:]
                attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

    # --- dtype fix (fp32 -> target) ---
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to "
            f"the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in "
            f"{target_dtype}."
        )
        query_states = query_states.to(target_dtype)
        key_states   = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    is_prefill = q_len != 1

    # ==========================================================
    # QAdA: Gaussian Approx（等长 local 段，完全向量化）
    # ==========================================================
    @torch.no_grad()
    def _estimate_bulk_stats_diag_from_prefill(
        K_bhTd: torch.Tensor, T: int, *, t_sink: int, t_local: int
    ):
        """
        输入:
          - K_bhTd: [B, H, T, D]（按 head 展开的 full keys）
          - T: 当前 prefill 序列长度
          - t_sink: 前部 sink 长度
          - t_local: 局部窗口总长度（含 sink）

        输出:
          - mu_K:  [H,D]  bulk 的均值
          - var_K: [H,D]  bulk 的对角方差
        """
        device = K_bhTd.device
        B, H, _, D = K_bhTd.shape

        t_sink_ = max(int(t_sink), 0)
        t_loc_  = max(int(t_local), 0)
        t_loc_  = min(t_loc_, T)
        right_len = max(t_loc_ - t_sink_, 0)

        keep = torch.ones(T, dtype=torch.bool, device=device)
        if t_sink_ > 0:
            keep[:t_sink_] = False
        if right_len > 0:
            keep[T - right_len:] = False
        # bulk = ~I
        bulk_mask = keep  # True 表示 bulk

        Ks = K_bhTd[:, :, bulk_mask, :]        # [B,H,T_bulk,D]
        if Ks.numel() == 0:
            mu_K  = torch.zeros(H, D, device=device, dtype=K_bhTd.dtype)
            var_K = torch.ones (H, D, device=device, dtype=K_bhTd.dtype)
            return mu_K, var_K

        Ks = Ks.reshape(B, H, -1, D)           # [B,H,*,D]
        Ks = Ks.mean(dim=0, keepdim=False)     # [H,*,D]（先对 batch 平均）
        mu_K  = Ks.mean(dim=1)                 # [H,D]
        var_K = Ks.var (dim=1, unbiased=False) # [H,D]（有偏）
        var_K = torch.clamp(var_K, min=1e-6)
        return mu_K, var_K
    # import torch



    @torch.no_grad()
    def _qada_head_mask_gaussian_approx_vec_equalL(
        q_states: torch.Tensor,                 # [B,H,1,D]（当前 decode 步; B=1）
        *,
        K_local_var: torch.Tensor,              # [sum_k_local, 1, D]（varlen 拼接）
        cu_seqlens_local: torch.Tensor,         # [H+1]（local 的前缀和）——等长！
        T_full_per_head: torch.Tensor,          # [H]
        T_local_per_head: torch.Tensor,         # [H]
        mu_K: torch.Tensor,                     # [H,D]
        var_K: torch.Tensor,                    # [H,D]  对角协方差
        tau: float = 0.4,
        acc_dtype: torch.dtype = torch.float32,
        eps_T: float = 1e-8,
        eps_var: float = 1e-6
    ) -> torch.Tensor:
        """
        QAdA 近似判定（式(7)），假设所有 head 的 local 段长度完全一致。
        完全向量化实现。

        返回 mask_h: [1,H]，True=local，False=full
        """
        device = q_states.device
        B, H, _, D = q_states.shape
        assert B == 1, "Only support B==1 in decode."

        # ---- local 等长断言并提取 Lf
        Ls = (cu_seqlens_local[1:] - cu_seqlens_local[:-1])
        assert torch.all(Ls == Ls[0]), "Expected equal local lengths for all heads."
        Lf = int(Ls[0].item())

        # ---- 取 q: [H,D]
        q_h = q_states[:, :, -1, :].reshape(H, D).to(device=device, dtype=acc_dtype)
        inv_sqrt_D = 1.0 / math.sqrt(D)

        # ---- 组织 K_local: [H, Lf, D]
        K_loc = K_local_var[:, 0, :].to(device=device, dtype=acc_dtype).view(H, Lf, D)

        # ---- 向量化 logits_local: [H,Lf] = (q_h · K_loc)/sqrt(D)
        logits_local = torch.einsum('hd,hld->hl', q_h, K_loc) * inv_sqrt_D  # [H,Lf]
        log_A_local = torch.logsumexp(logits_local, dim=-1)                 # [H]

        # ---- 解析近似的 A_bulk
        mu_K   = mu_K.to(device=device, dtype=acc_dtype)       # [H,D]
        var_K  = var_K.to(device=device, dtype=acc_dtype)       # [H,D]
        var_K  = torch.clamp(var_K, min=eps_var)

        # 每个 head 的 bulk 长度
        T_visual = (T_full_per_head.to(device) - T_local_per_head.to(device)).clamp(min=0)  # [H]

        mu_s     = (q_h * mu_K).sum(dim=-1) * inv_sqrt_D                                    # [H]
        sigma_s2 = (q_h * q_h * var_K).sum(dim=-1) / D                                      # [H]
        log_A_bulk = torch.log(torch.clamp(T_visual.to(acc_dtype), min=eps_T)) \
                     + mu_s + 0.5 * sigma_s2                                                # [H]
        # print(log_A_bulk,log_A_local,"sjs",T_visual)
        # ---- log p_local 与阈值
        denom = torch.logaddexp(log_A_local, log_A_bulk)        # [H]
        log_p_local = log_A_local - denom                       # [H]
        mask_h = (log_p_local > math.log(float(tau)))           # [H]
       
        return mask_h.unsqueeze(0)                               # [1,H]

    # =========================
    # FORWARD
    # =========================
    if is_prefill:
        # --- 你的 KV 压缩与写 cache（保持不变） ---
        key_states_compress, value_states_compress = self.kv_cluster.update_kv(
            key_states, query_states, value_states
        )
        key_states_compress50, value_states_compress50 = self.kv_cluster.update_kv50(
            key_states, query_states, value_states
        )

        if self.kv_cluster.layer_avg_cos != {}:
            self.kv_cluster.layer_avg_cos = {}
            self.kv_cluster.layer_ref_unit = {}

        # 写两套 cache 槽（full & local）
        past_key_value.update(
            key_states_compress,  value_states_compress,  self.layer_idx,
            {"sin": sin, "cos": cos, "cache_position": cache_position}
        )
        past_key_value.update(
            key_states_compress50, value_states_compress50, self.layer_idx + 32,
            {"sin": sin, "cos": cos, "cache_position": cache_position}
        )

        # ----- 常规 flash-attn prefill -----
        q_ = query_states.transpose(1, 2)
        k_ = key_states.transpose(1, 2)
        v_ = value_states.transpose(1, 2)
        attn_output = _flash_attention_forward(
            q_, k_, v_, attention_mask, q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        self.kv_cluster.prefill_len = q_len

        # ----- 估计 bulk 统计量（高斯近似所需） -----
        # 展开到 per-head（若采用 GQA）
        if self.num_key_value_heads == self.num_heads:
            K_bhTd = key_states          # [B,H,T,D]
        else:
            # 将 kv heads 复制到 query heads
            K_bhTd = key_states.repeat_interleave(self.num_key_value_groups, dim=1)  # [B,H,T,D]

        # 读取/默认超参
        t_sink  = getattr(self.kv_cluster, "t_sink",  32)
        t_local = getattr(self.kv_cluster, "t_local", 128)

        mu_K, var_K = _estimate_bulk_stats_diag_from_prefill(
            K_bhTd=K_bhTd, T=q_len, t_sink=t_sink, t_local=t_local
        )

        # 缓存到 kv_cluster，供 decode 使用
        self.kv_cluster.qada_mu_K  = mu_K       # [H,D]
        self.kv_cluster.qada_var_K = var_K      # [H,D]
        self.kv_cluster.qada_tau   = getattr(self.kv_cluster, "qada_tau", 0.6)
        self.kv_cluster.qada_stats_ready = True

    else:
        # ===== DECODE =====

        # 更新两个 cache 槽：local & full
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        cache_kwargs["head_lens"] = self.kv_cluster.alt_head_lens
        cache_kwargs["cu_klen"]   = self.kv_cluster.alt_cu_klen
        key_states50, value_states50 = past_key_value.update(
            key_states, value_states, self.layer_idx + 32, cache_kwargs
        )

        cache_kwargs["head_lens"] = self.kv_cluster.head_lens
        cache_kwargs["cu_klen"]   = self.kv_cluster.cu_klen
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

        # meta updates（保持原样）
        self.kv_cluster.alt_klen_sum   += self.num_heads
        self.kv_cluster.alt_max_seqlen_k += 1
        self.kv_cluster.alt_cu_klen    += self.kv_cluster.alt_cu_offset
        self.kv_cluster.alt_head_lens  += 1

        self.kv_cluster.klen_sum       += self.num_heads
        self.kv_cluster.max_seqlen_k   += 1
        self.kv_cluster.cu_klen        += self.kv_cluster.cu_offset
        self.kv_cluster.head_lens      += 1

        # varlen 形状
        query_states_var = query_states.view(-1, self.num_key_value_groups, self.head_dim)  # [H,G,D]
        key_states_var   = key_states.view(-1, 1, self.head_dim)                            # [sum_k_full,1,D]
        value_states_var = value_states.view(-1, 1, self.head_dim)

        cu_seqlens_q = self.kv_cluster.cu_qlen
        cu_seqlens_k = self.kv_cluster.cu_klen
        max_seqlen_q = 1
        max_seqlen_k = self.kv_cluster.max_seqlen_k

        key_states50_var   = key_states50.view(-1, 1, self.head_dim)                        # [sum_k_local,1,D]
        value_states50_var = value_states50.view(-1, 1, self.head_dim)

        alt_cu_seqlens_q = self.kv_cluster.alt_cu_qlen
        alt_cu_seqlens_k = self.kv_cluster.alt_cu_klen
        alt_max_seqlen_q = 1
        alt_max_seqlen_k = self.kv_cluster.alt_max_seqlen_k

        # --- two-path attention ---
        attn_output_local = flash_attn_varlen_func(
            query_states_var, key_states50_var, value_states50_var,
            alt_cu_seqlens_q, alt_cu_seqlens_k, alt_max_seqlen_q, alt_max_seqlen_k,
            causal=True
        )
        attn_output_full = flash_attn_varlen_func(
            query_states_var, key_states_var, value_states_var,
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            causal=True, return_attn_probs=False
        )
        attn_output_local = attn_output_local.reshape(bsz, self.num_heads, q_len, self.head_dim)
        attn_output_full  = attn_output_full.reshape( bsz, self.num_heads, q_len, self.head_dim)

        # --- QAdA Gaussian Approx（等长 local 段）并融合 ---
        # 前两层 + 第31层 强制 local
        force_local = (self.layer_idx < 2) or (self.layer_idx == 31)

        if force_local:
            head_mask = torch.ones(1, self.num_heads, dtype=torch.bool, device=attn_output_full.device)
        elif not getattr(self.kv_cluster, "qada_stats_ready", False):
            # 统计未就绪：全部走 full
            head_mask = torch.zeros(1, self.num_heads, dtype=torch.bool, device=attn_output_full.device)
        else:
            # 每个 head 的当前长度
            T_full_per_head  = (cu_seqlens_k[1:]     - cu_seqlens_k[:-1]).to(torch.long)     # [H]
            T_local_per_head = (alt_cu_seqlens_k[1:] - alt_cu_seqlens_k[:-1]).to(torch.long) # [H]

            head_mask = _qada_head_mask_gaussian_approx_vec_equalL(
                q_states=query_states.view(bsz, self.num_heads, q_len, self.head_dim)[:, :, -1:, :],
                K_local_var=key_states50_var,
                cu_seqlens_local=alt_cu_seqlens_k,
                T_full_per_head=T_full_per_head,
                T_local_per_head=T_local_per_head,
                mu_K=self.kv_cluster.qada_mu_K,
                var_K=self.kv_cluster.qada_var_K,
                tau=getattr(self.kv_cluster, "qada_tau"),
                acc_dtype=attn_output_full.dtype,
            )  # [1,H] bool

        # 展开到 [B,H,Q,1]
        head_mask_4d = head_mask.view(bsz, self.num_heads, 1, 1).expand(bsz, self.num_heads, q_len, 1)

        # 融合输出：True=local, False=full
        attn_output = torch.where(head_mask_4d, attn_output_local, attn_output_full) \
                        .transpose(1, 2).reshape(bsz, q_len, self.hidden_size)

        # attn_output = torch.where(head_mask_4d, attn_output_local, attn_output_local) \
        #                 .transpose(1, 2).reshape(bsz, q_len, self.hidden_size)

    # --- output proj ---
    attn_output = self.o_proj(attn_output)
    attn_weights = None if not output_attentions else None
    return attn_output, attn_weights, past_key_value




def llama_flash_attn2_forward_Mask(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
            "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
        )
    # get head list
    init_mask(self)


    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)



    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.

    # if self.head_list:
    for h in self.head_list:
        if self.layer_idx==h[0]:
            query_states[:,h[1], :, :] = 0

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        position_ids=position_ids,
        dropout=dropout_rate,
        sliding_window=getattr(self, "sliding_window", None),
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
        is_causal=self.is_causal,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value




def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask

def prepare_inputs_for_generation_llama_new(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        if not isinstance(past_key_values, tuple):
            if len(past_key_values.key_cache) == 0:
                for layer in self.model.layers:
                    layer.self_attn.kv_seq_len = 0

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        # import pdb;pdb.set_trace()

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

def adaptive_LlamaModel_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    return_legacy_cache = False
    if (
        use_cache and not (type(past_key_values) == DynamicCacheSplitHeadFlatten) and not self.training
    ):  # kept for BC (non `Cache` `past_key_values` inputs)
        # return_legacy_cache = True  #! For 4.41 version.
        # 使用这个进行初始化
        past_key_values = DynamicCacheSplitHeadFlatten.from_legacy_cache(past_key_values)
        logger.warning_once(
            "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
            "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
        )

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )
    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
