import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import warnings
from transformers.cache_utils import Cache, DynamicCache, StaticCache
print("sjs555")
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
    _flash_attention_forward
)# 这里也用了
print("sjs666")
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import (
    logging,
)
from sparsemm.sparsemm_utils import init_snapkv, init_pyramidkv, init_adakv, init_sparsemm, init_mask,init_sparsemm_query
import math
from flash_attn import  flash_attn_varlen_func
# from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
from sparsemm.sparsemm_utils import DynamicCacheSplitHeadFlatten
import numpy as np
torch.set_printoptions(sci_mode=False)
import time
print("sjs555")
# from sparsemm.qada_fused_triton import qada_mask_fused_triton
# from sparsemm.qada_fused_triton import qada_mask_fused_triton_match_python

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

# import torch
# import torch.nn.functional as F


def llama_flash_attn2_forward_SparseMM(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
     **kwargs,  # 👈 新增：接收任意额外参数
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
            "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
        )
    init_sparsemm(self)
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    debug_topk_positions = kwargs.get("debug_topk_positions", None)
    debug_layer_idx = kwargs.get("debug_layer_idx", None)
    if q_len>1:# prefill
        # debug_topk_positions = kwargs.get("debug_topk_positions", None)
        self.kv_cluster.pp=debug_topk_positions
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
        key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states,debug_topk_positions)
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








import math
from typing import Optional, Tuple
import torch
from torch import nn

@torch.no_grad()
def llama_flash_attn2_forward_shift(
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
    init_sparsemm_query(self)
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
        # 新操作：
        if self.layer_idx==0 or self.layer_idx==1:
            num_groups = 1
            groups_size = 16

            # 复制最后一个 token 16 次：在序列维度重复，特征维不变
            hidden_samples = hidden_states[:, -1:, :].repeat(1, groups_size, 1)
            query_samples = self.q_proj(hidden_samples)
            key_samples = self.k_proj(hidden_samples)
            value_samples = self.v_proj(hidden_samples)
            bsz_s, q_len_s, _ = hidden_samples.size()

            query_samples = query_samples.view(bsz_s, q_len_s, self.num_heads, self.head_dim).transpose(1, 2)
            key_samples = key_samples.view(bsz_s, q_len_s, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_samples = value_samples.view(bsz_s, q_len_s, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            offset = torch.arange(1, groups_size+1,device = position_ids.device)[None,:]
            future_pos = position_ids[:,-1:] + offset * num_groups  #[bs,groups*num_groups]
            cos_f,sin_f = self.rotary_emb(value_samples[:,:,:groups_size,:],future_pos)

            cos_f = cos_f.repeat(1,num_groups,1)
            sin_f = sin_f.repeat(1,num_groups,1)
            query_samples, key_samples = apply_rotary_pos_emb(query_samples, key_samples, cos_f, sin_f) 
        else:
            ctx = 512; num_samples = 512
            hs_tail = hidden_states[:,:128,:]
            hs_tail = hs_tail.view(hidden_states.shape[0],-1)
            hs_mean = hs_tail.mean(dim=-1)
            hs_std = hs_tail.std(dim=-1)
            num_groups = 16
            groups_size = num_samples//num_groups

            if hasattr(self.kv_cluster, 'use_statistical_predictor') and self.kv_cluster.use_statistical_predictor:
                # 使用统计表预测器（最新方法）统计表\
                # print(111)
                try:
                    hidden_samples = self.kv_cluster.statistical_predictor.generate_query_samples(
                        layer_idx=self.layer_idx,
                        hidden_states=hidden_states[:,-512:,:],  # [bsz, seq_len, hidden_size]
                        num_samples=num_samples       # 512
                    )
                    # hidden_samples: [bsz, num_samples, hidden_size]

                    # 投影生成 query/key/value samples
                    query_samples = self.q_proj(hidden_samples)
                    key_samples = self.k_proj(hidden_samples)
                    value_samples = self.v_proj(hidden_samples)
                    # import torch

                except Exception as e:
                    print(f"[SparseMM] Statistical predictor failed at layer {self.layer_idx}: {e}")
                    print(f"[SparseMM] Falling back to original Gaussian sampling")
                    # Fallback to original method
                    hs_tail = hidden_states[:,-ctx:,:]
                    hs_tail = hs_tail.view(hidden_states.shape[0],-1)
                    hs_mean = hs_tail.mean(dim=-1)
                    hs_std = hs_tail.std(dim=-1)

                    hidden_samples = self.kv_cluster.diagonal_gaussian_sampling(hs_mean,hs_std,num_samples,hidden_states.shape[2])
                    query_samples = self.q_proj(hidden_samples)
                    key_samples = self.k_proj(hidden_samples)
                    value_samples = self.v_proj(hidden_samples)

            else:
                # 不使用预测器，使用原始的高斯采样方法
       
                hs_tail = hidden_states[:,:,:]
                hs_tail = hs_tail.view(hidden_states.shape[0],-1)
                hs_mean = hs_tail.mean(dim=-1)
                hs_std = hs_tail.std(dim=-1)
                hidden_samples = self.kv_cluster.diagonal_gaussian_sampling(hs_mean,hs_std,num_samples,hidden_states.shape[2])
                
                

                # ===== 1) 计算 hidden_states 的“整体均值方向” mu_all =====
                # (建议先按 token 归一化再均值，减少幅值偏置)
                hs_norm = hidden_states / (hidden_states.norm(dim=-1, keepdim=True) + 1e-6)  # [bsz, seq_len, hidden_dim]
                mu_all = hs_norm.mean(dim=1)                                                 # [bsz, hidden_dim]
                mu_all = mu_all / (mu_all.norm(dim=-1, keepdim=True) + 1e-6)                 # [bsz, hidden_dim]


                # ===== 2) 确保 hidden_samples 形状正确 =====
                # 有些实现可能返回 [bsz*num_samples, hidden_dim] 或 [num_samples, bsz, hidden_dim]
                # 下面做一个最小的 shape 兼容处理
                if hidden_samples.dim() == 2:
                    # [bsz*num_samples, hidden_dim] -> [bsz, num_samples, hidden_dim]
                    bsz = hidden_states.shape[0]
                    hidden_dim = hidden_states.shape[2]
                    hidden_samples = hidden_samples.view(bsz, num_samples, hidden_dim)
                elif hidden_samples.dim() == 3 and hidden_samples.shape[0] != hidden_states.shape[0]:
                    # 例如 [num_samples, bsz, hidden_dim] -> [bsz, num_samples, hidden_dim]
                    hidden_samples = hidden_samples.transpose(0, 1).contiguous()


                # ===== 3) 计算每个 probe 与 mu_all 的 cosine similarity =====
                probes_norm = hidden_samples / (hidden_samples.norm(dim=-1, keepdim=True) + 1e-6)  # [bsz, num_samples, hidden_dim]

                # cos_sim: [bsz, num_samples]
                cos_sim = (probes_norm * mu_all.unsqueeze(1)).sum(dim=-1)

                # ===== 4) 聚合（可用于 logging / 表格）=====
                cos_sim_mean = cos_sim.mean()                 # scalar
                # print(cos_sim_mean)


                hs_mean = hidden_states[:,-ctx:,:].mean(dim=1)              # [bsz, hidden_size]
                hs_std  = hidden_states.std(dim=1, unbiased=False)  # [bsz, hidden_size]
                hidden_samples = self.kv_cluster.diagonal_gaussian_sampling_vec(
                    hs_mean, hs_std, num_samples
                )  # [
                # ===== 2) 确保 hidden_samples 形状正确 =====
                # 有些实现可能返回 [bsz*num_samples, hidden_dim] 或 [num_samples, bsz, hidden_dim]
                # 下面做一个最小的 shape 兼容处理
                if hidden_samples.dim() == 2:
                    # [bsz*num_samples, hidden_dim] -> [bsz, num_samples, hidden_dim]
                    bsz = hidden_states.shape[0]
                    hidden_dim = hidden_states.shape[2]
                    hidden_samples = hidden_samples.view(bsz, num_samples, hidden_dim)
                elif hidden_samples.dim() == 3 and hidden_samples.shape[0] != hidden_states.shape[0]:
                    # 例如 [num_samples, bsz, hidden_dim] -> [bsz, num_samples, hidden_dim]
                    hidden_samples = hidden_samples.transpose(0, 1).contiguous()


                # ===== 3) 计算每个 probe 与 mu_all 的 cosine similarity =====
                probes_norm = hidden_samples / (hidden_samples.norm(dim=-1, keepdim=True) + 1e-6)  # [bsz, num_samples, hidden_dim]

                # cos_sim: [bsz, num_samples]
                cos_sim = (probes_norm * mu_all.unsqueeze(1)).sum(dim=-1)

                # ===== 4) 聚合（可用于 logging / 表格）=====
                cos_sim_mean = cos_sim.mean()                 # scalar
                print(cos_sim_mean)

              
                query_samples = self.q_proj(hidden_samples)
                key_samples = self.k_proj(hidden_samples)
                value_samples = self.v_proj(hidden_samples)
            bsz_s, q_len_s, _ = hidden_samples.size()
            query_samples = query_samples.view(bsz_s, q_len_s, self.num_heads, self.head_dim).transpose(1, 2)
            key_samples = key_samples.view(bsz_s, q_len_s, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_samples = value_samples.view(bsz_s, q_len_s, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            offset = torch.arange(1, groups_size+1,device = position_ids.device)[None,:]
            future_pos = position_ids[:,-1:] + offset * num_groups  #[bs,groups*num_groups]
            cos_f,sin_f = self.rotary_emb(value_samples[:,:,:groups_size,:],future_pos)

            cos_f = cos_f.repeat(1,num_groups,1)
            sin_f = sin_f.repeat(1,num_groups,1)
            query_samples, key_samples = apply_rotary_pos_emb(query_samples, key_samples, cos_f, sin_f) 

        # 调用SparseMM的update_kv，传入hidden_states、future RoPE和Wq投影
        key_states_compress, value_states_compress = self.kv_cluster.update_kv(
            key_states,
            query_states,
            value_states,
            query_samples,
            groups_size,
            num_groups
        )

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
        # Append 1 token per KV head at decode step (GQA aware)
        self.kv_cluster.klen_sum += self.num_heads
        self.kv_cluster.max_seqlen_k += 1
        self.kv_cluster.cu_klen += self.kv_cluster.cu_offset
        self.kv_cluster.head_lens += 1

        # Reshape for varlen FlashAttention (per-KV-head sequences)
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







import math
import torch
import triton
import triton.language as tl

# 需要确保你已在模块顶部有这个导入（或等价的别名）
# from flash_attn.flash_attn_interface import _flash_attn_varlen_forward as flash_attn_varlen_func

# import math
from typing import Optional, Tuple
# import torch
# from torch import nn
@triton.jit
def _qada_fused_single_kernel(
    q_ptr, mu_ptr, var_ptr,
    logTvis_ptr, lse_ptr,
    out_ptr,                 # uint8[H]
    H, D,
    c_thresh_f32: tl.constexpr, scale_f32: tl.constexpr,
    # q 的 strides（元素为单位）
    sq_h, sq_t, sq_d,
    # mu/var 的 strides（元素为单位）
    sm_h, sm_d,
    # 配置
    BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr,
    FULL_TILE: tl.constexpr,           # True => 无掩码快路径
    invD_f32: tl.constexpr,            # 1.0 / D
    q_t_offset: tl.int32,              # 固定时间步偏移（单位：元素）
):
    pid  = tl.program_id(0)
    h0   = pid * BLOCK_H + tl.arange(0, BLOCK_H)
    hmsk = h0 < H
    d    = tl.arange(0, BLOCK_D)

    # 偏移
    offs_q = h0[:, None] * sq_h + d[None, :] * sq_d + q_t_offset
    offs_m = h0[:, None] * sm_h + d[None, :] * sm_d

    if FULL_TILE:
        tl.max_contiguous(d, 16)
        tl.multiple_of(d, 16)
        q   = tl.load(q_ptr   + offs_q, cache_modifier=".cg")
        mu  = tl.load(mu_ptr  + offs_m, cache_modifier=".cg")
        var = tl.load(var_ptr + offs_m, cache_modifier=".cg")
    else:
        dmsk = d < D
        m = hmsk[:, None] & dmsk[None, :]
        q   = tl.load(q_ptr   + offs_q, mask=m, other=0.0)
        mu  = tl.load(mu_ptr  + offs_m, mask=m, other=0.0)
        var = tl.load(var_ptr + offs_m, mask=m, other=0.0)

    # 累计（fp32）
    qf  = q.to(tl.float32)
    muf = mu.to(tl.float32)
    vf  = tl.maximum(var.to(tl.float32), 1e-6)

    mu_acc = tl.sum(qf * muf, axis=1)
    s2_acc = tl.sum((qf * qf) * vf, axis=1)
    s2_mean = s2_acc * invD_f32

    logTv = tl.load(logTvis_ptr + h0, mask=hmsk, other=0.0)
    a     = tl.load(lse_ptr     + h0, mask=hmsk, other=-1e30)

    th = a + c_thresh_f32
    log_A_bulk = logTv + mu_acc * scale_f32 + 0.5 * s2_mean
    keep = log_A_bulk < th

    tl.store(out_ptr + h0, keep.to(tl.uint8), mask=hmsk)

# ===============================
# 多 chunk：沿 D 分块 + 原子累加 + 收尾
# ===============================
@triton.jit
def _qada_atomic_chunk_kernel(
    q_ptr, mu_ptr, var_ptr,
    H, D,
    acc_mu_ptr, acc_s2_ptr,        # fp32[H]，外部需 zero_()
    # strides
    sq_h, sq_t, sq_d, sm_h, sm_d,
    BLOCK_H: tl.constexpr, D_TILE: tl.constexpr,
    q_t_offset: tl.int32,
):
    pid_h = tl.program_id(0)
    pid_c = tl.program_id(1)

    h0   = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    hmsk = h0 < H

    d0   = pid_c * D_TILE + tl.arange(0, D_TILE)
    dmsk = d0 < D
    m    = hmsk[:, None] & dmsk[None, :]

    tl.max_contiguous(d0, 16)

    oq = h0[:, None] * sq_h + d0[None, :] * sq_d + q_t_offset
    om = h0[:, None] * sm_h + d0[None, :] * sm_d

    q   = tl.load(q_ptr   + oq, mask=m, other=0.0)
    mu  = tl.load(mu_ptr  + om, mask=m, other=0.0)
    var = tl.load(var_ptr + om, mask=m, other=0.0)

    qf  = q.to(tl.float32)
    muf = mu.to(tl.float32)
    vf  = tl.maximum(var.to(tl.float32), 1e-6)

    mu_acc = tl.sum(qf * muf, axis=1)
    s2_acc = tl.sum((qf * qf) * vf, axis=1)

    tl.atomic_add(acc_mu_ptr + h0, mu_acc, mask=hmsk)
    tl.atomic_add(acc_s2_ptr + h0, s2_acc, mask=hmsk)

@triton.jit
def _qada_finalize_kernel(
    acc_mu_ptr, acc_s2_ptr, logTvis_ptr, lse_ptr,
    out_ptr, H, D,
    c_thresh_f32: tl.constexpr, scale_f32: tl.constexpr,
    invD_f32: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid  = tl.program_id(0)
    h0   = pid * BLOCK_H + tl.arange(0, BLOCK_H)
    hmsk = h0 < H

    mu_acc = tl.load(acc_mu_ptr  + h0, mask=hmsk, other=0.0)
    s2_acc = tl.load(acc_s2_ptr  + h0, mask=hmsk, other=0.0)
    logTv  = tl.load(logTvis_ptr + h0, mask=hmsk, other=0.0)
    a      = tl.load(lse_ptr     + h0, mask=hmsk, other=-1e30)

    s2_mean = s2_acc * invD_f32
    th = a + c_thresh_f32
    log_A_bulk = logTv + mu_acc * scale_f32 + 0.5 * s2_mean
    keep = log_A_bulk < th

    tl.store(out_ptr + h0, keep.to(tl.uint8), mask=hmsk)

# ===============================
# 入口函数（带 prep 分段打印；消除 host 侧 q 的 view/cast）
# ===============================
# ===============================
# 入口函数（prep 仅记录总时间；TOTAL 仍为纯 kernel 时间）
# ===============================
@torch.no_grad()
def qada_mask_fused_triton_packed_fast_adaptive(
    q_states: torch.Tensor,           # [1,H,1,D]（只读）
    *,
    mu_K: torch.Tensor,               # [H,D]
    var_K: torch.Tensor,              # [H,D]
    log_Tvis_per_head: torch.Tensor,  # [H] fp32
    softmax_lse_local: torch.Tensor,  # [H]/[H,1,*] -> [H] fp32 更好
    tau: float = 0.6,
    softmax_scale: float | None = None,
    acc_dtype: torch.dtype | None = None,   # 建议 None：跟随 q_states，避免 cast

    BLOCK_H: int = 32,
    D_TILE: int = 256,
    num_warps: int = 8,
    num_stages: int = 4,

    out_buf: torch.Tensor | None = None,             # 复用更快
    acc_buf: tuple[torch.Tensor, torch.Tensor] | None = None,  # (acc_mu, acc_s2) 仅多chunk用
    profile: bool = True,
    print_profile: bool = True,
    self_obj=None,  # 访问 self.kv_cluster 的缓存（可选）
):
    import math
    device = q_states.device
    _, H, _, D = q_states.shape

    def _evt(): return torch.cuda.Event(enable_timing=True)

    # === 唯一 E2E 计时（含 prep） ===
    if profile:
        torch.cuda.synchronize()
        Ee2e0 = _evt(); Ee2e1 = _evt()
        Eprep1 = _evt()  # prep 结束标记
        Ee2e0.record()

    # ====== PREP ======
    if acc_dtype is None:
        acc_dtype = q_states.dtype

    sq_h = q_states.stride(1)
    sq_t = q_states.stride(2)
    sq_d = q_states.stride(3)
    t_idx = q_states.size(2) - 1
    q_t_offset = int(t_idx * sq_t)

    if self_obj is not None and hasattr(self_obj, "kv_cluster"):
        mu = getattr(self_obj.kv_cluster, "qada_mu_K", mu_K)
        vr = getattr(self_obj.kv_cluster, "qada_var_K", var_K)
    else:
        mu, vr = mu_K, var_K

    if (self_obj is not None) and hasattr(self_obj.kv_cluster, "qada_lse_fp32"):
        lse = self_obj.kv_cluster.qada_lse_fp32
    else:
        lse = softmax_lse_local
        if lse.dim()==3: lse = lse[:,0,0]
        elif lse.dim()==2: lse = lse[:,0]
        if (lse.dtype != torch.float32) or (not lse.is_contiguous()):
            lse = lse.contiguous().to(torch.float32)

    if (self_obj is not None) and hasattr(self_obj.kv_cluster, "qada_log_Tvis_per_head"):
        logTvis = self_obj.kv_cluster.qada_log_Tvis_per_head
    else:
        logTvis = log_Tvis_per_head
        if (logTvis.dtype != torch.float32) or (not logTvis.is_contiguous()):
            logTvis = logTvis.contiguous().to(torch.float32)

    kc = self_obj.kv_cluster if self_obj is not None else None
    if kc is not None:
        if (getattr(kc, "_qada_last_D", None) != D or
            getattr(kc, "_qada_last_tau", None) != tau or
            getattr(kc, "_qada_last_scale", None) != softmax_scale):
            kc._qada_scale_f32 = float(1.0 / math.sqrt(D)) if softmax_scale is None else float(softmax_scale)
            kc._qada_invD_f32  = float(1.0 / D)
            kc._qada_cth_f32   = float(math.log1p(-float(tau)) - math.log(float(tau)))
            kc._qada_last_D     = D
            kc._qada_last_tau   = tau
            kc._qada_last_scale = softmax_scale
        scale_f32    = kc._qada_scale_f32
        invD_f32     = kc._qada_invD_f32
        c_thresh_f32 = kc._qada_cth_f32
    else:
        scale_f32    = float(1.0 / math.sqrt(D)) if softmax_scale is None else float(softmax_scale)
        invD_f32     = float(1.0 / D)
        c_thresh_f32 = float(math.log1p(-float(tau)) - math.log(float(tau)))

    nchunks = (D + D_TILE - 1) // D_TILE

    if (out_buf is None or out_buf.numel()!=H or out_buf.dtype!=torch.uint8 or out_buf.device!=device):
        out_buf = torch.empty(H, dtype=torch.uint8, device=device)

    if profile:
        Eprep1.record()

    # ====== 执行 Kernel ======
    kernel_ms = 0.0
    k1_ms = 0.0
    k2_ms = 0.0

    if nchunks == 1:
        grid = ((H + BLOCK_H - 1) // BLOCK_H, )
        BD = D
        FULL_TILE = (q_states.stride(3) == 1) and (mu.stride(1) == 1) and (vr.stride(1) == 1)

        if profile:
            torch.cuda.synchronize()
            k0 = _evt(); k1 = _evt(); k0.record()

        _qada_fused_single_kernel[grid](
            q_states, mu, vr, logTvis, lse, out_buf,
            H, D, c_thresh_f32, scale_f32,
            sq_h, sq_t, sq_d,
            mu.stride(0), mu.stride(1),
            BLOCK_H=BLOCK_H, BLOCK_D=BD,
            FULL_TILE=FULL_TILE,
            invD_f32=invD_f32,
            q_t_offset=q_t_offset,
            num_warps=num_warps, num_stages=num_stages,
        )

        if profile:
            k1.record(); torch.cuda.synchronize()
            kernel_ms = k0.elapsed_time(k1)

        # E2E 结束并打印
        if profile:
            Ee2e1.record(); torch.cuda.synchronize()
            prep_ms  = Ee2e0.elapsed_time(Eprep1)
            total_ms = Ee2e0.elapsed_time(Ee2e1)
            if print_profile:
                print(f"[QAdA TIME] prep={prep_ms:.3f} | kernel={kernel_ms:.3f} | E2E={total_ms:.3f} ms")
        return out_buf.view(1, H).bool(), None

    # 多 chunk：atomic + finalize 两核之和
    need_new_acc = (
        acc_buf is None or
        acc_buf[0].numel()!=H or acc_buf[1].numel()!=H or
        acc_buf[0].dtype!=torch.float32 or acc_buf[1].dtype!=torch.float32 or
        acc_buf[0].device!=device or acc_buf[1].device!=device
    )
    if need_new_acc:
        acc_mu = torch.empty(H, dtype=torch.float32, device=device)
        acc_s2 = torch.empty(H, dtype=torch.float32, device=device)
    else:
        acc_mu, acc_s2 = acc_buf
    acc_mu.zero_(); acc_s2.zero_()

    if profile:
        torch.cuda.synchronize()
        k1a=_evt(); k1b=_evt(); k1a.record()
    grid1 = ((H + BLOCK_H - 1) // BLOCK_H, nchunks)
    _qada_atomic_chunk_kernel[grid1](
        q_states, mu, vr, H, D, acc_mu, acc_s2,
        sq_h, sq_t, sq_d, mu.stride(0), mu.stride(1),
        BLOCK_H=BLOCK_H, D_TILE=D_TILE,
        q_t_offset=q_t_offset,
        num_warps=num_warps, num_stages=num_stages,
    )
    if profile:
        k1b.record(); torch.cuda.synchronize()
        k1_ms = k1a.elapsed_time(k1b)

    if profile:
        k2a=_evt(); k2b=_evt(); k2a.record()
    grid2 = ((H + BLOCK_H - 1) // BLOCK_H, )
    _qada_finalize_kernel[grid2](
        acc_mu, acc_s2, logTvis, lse, out_buf,
        H, D, c_thresh_f32, scale_f32,
        invD_f32=invD_f32,
        BLOCK_H=BLOCK_H,
        num_warps=4, num_stages=2,
    )
    if profile:
        k2b.record(); torch.cuda.synchronize()
        k2_ms = k2a.elapsed_time(k2b)
        kernel_ms = k1_ms + k2_ms

    # E2E 结束并打印
    if profile:
        Ee2e1.record(); torch.cuda.synchronize()
        prep_ms  = Ee2e0.elapsed_time(Eprep1)
        total_ms = Ee2e0.elapsed_time(Ee2e1)
        if print_profile:
            print(f"[QAdA TIME] prep={prep_ms:.3f} | kernel={kernel_ms:.3f} (k1={k1_ms:.3f}, k2={k2_ms:.3f}) | E2E={total_ms:.3f} ms")

    return out_buf.view(1, H).bool(), (acc_mu, acc_s2)

# ===============================
# LLaMA 前向（prefill 一次就绪，decode 零拷贝）
# ===============================
@torch.no_grad()
def llama_flash_attn2_forward_SparseMM_query1(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional["Cache"] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # v4.46+
):
    if isinstance(past_key_value, StaticCache):
        raise ValueError("`static` cache not compatible with flash_attention_2")

    init_sparsemm_query(self)
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device

    # --- projections ---
    query_states = self.q_proj(hidden_states)
    key_states   = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states   = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None and cache_position is not None:
        kv_seq_len += cache_position[0]

    # --- RoPE ---
    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # --- dtype fix (fp32 -> target) ---
    dropout_rate = self.attention_dropout if self.training else 0.0
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype
        query_states = query_states.to(target_dtype)
        key_states   = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # GQA：KV 复制到 query 头数
    key_states   = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    is_prefill = q_len != 1
    compute_dtype = query_states.dtype
    Hh = self.num_heads
    Dd = self.head_dim

    # ------------------------- helper：bulk mean/var 估计 -------------------------
    def _estimate_bulk_stats_diag_from_prefill(K_bhTd: torch.Tensor, T: int, *, t_sink: int, t_local: int):
        B, Hh_, _, Dd_ = K_bhTd.shape
        t_sink_ = max(int(t_sink), 0)
        t_loc_  = max(min(int(t_local), T), 0)
        right_len = max(t_loc_ - t_sink_, 0)

        keep = torch.ones(T, dtype=torch.bool, device=K_bhTd.device)
        if t_sink_ > 0: keep[:t_sink_] = False
        if right_len > 0: keep[T - right_len:] = False
        Ks = K_bhTd[:, :, keep, :]
        if Ks.numel() == 0:
            mu_K  = torch.zeros(Hh_, Dd_, device=K_bhTd.device, dtype=compute_dtype)
            var_K = torch.ones (Hh_, Dd_, device=K_bhTd.device, dtype=compute_dtype)
            return mu_K, var_K
        Ks   = Ks.reshape(B, Hh_, -1, Dd_).mean(dim=0, keepdim=False)
        mu_K = Ks.mean(dim=1)
        var_K= Ks.var(dim=1, unbiased=False).clamp_min(1e-6)
        return mu_K.to(compute_dtype), var_K.to(compute_dtype)

    # ========================= PREFILL =========================
    if is_prefill and past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        # 生成 query-samples（与 SparseMM 同策略）
        if self.layer_idx in (0, 1):
            num_groups  = 1
            groups_size = 16
            hidden_samples = hidden_states[:, -1:, :].repeat(1, groups_size, 1)  # [B,16,H]
        else:
            ctx         = 512
            num_samples = 512
            num_groups  = 16
            groups_size = num_samples // num_groups
            hs_tail      = hidden_states[:, -ctx:, :]
            hs_tail_flat = hs_tail.reshape(hidden_states.shape[0], -1)
            hs_mean = hs_tail_flat.mean(dim=-1)
            hs_std  = hs_tail_flat.std(dim=-1)
            hidden_samples = self.kv_cluster.diagonal_gaussian_sampling(
                hs_mean, hs_std, num_samples, hidden_states.shape[2]
            )  # [B, num_samples, H]

        # 通过 Wq/Wk/Wv
        q_samp = self.q_proj(hidden_samples)
        k_samp = self.k_proj(hidden_samples)
        v_samp = self.v_proj(hidden_samples)

        bsz_s, q_len_s, _ = hidden_samples.size()
        q_samp = q_samp.view(bsz_s, q_len_s, self.num_heads, self.head_dim).transpose(1, 2)
        k_samp = k_samp.view(bsz_s, q_len_s, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v_samp = v_samp.view(bsz_s, q_len_s, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 未来 RoPE（需要 position_ids）
        assert position_ids is not None, "position_ids required for future RoPE during prefill."
        offset = torch.arange(1, groups_size + 1, device=position_ids.device)[None, :]
        future_pos = position_ids[:, -1:] + offset * num_groups
        cos_f, sin_f = self.rotary_emb(v_samp[:, :, :groups_size, :], future_pos)
        cos_f = cos_f.repeat(1, num_groups, 1); sin_f = sin_f.repeat(1, num_groups, 1)
        q_samp, k_samp = apply_rotary_pos_emb(q_samp, k_samp, cos_f, sin_f)

        # KV 复制到 query 头数
        k_samp = repeat_kv(k_samp, self.num_key_value_groups)
        v_samp = repeat_kv(v_samp, self.num_key_value_groups)

        # 写两套缓存：full 路 + local 路
        key_states_main, value_states_main = self.kv_cluster.update_kv(
            key_states, query_states, value_states, q_samp, groups_size, num_groups
        )
        past_key_value.update(
            key_states_main, value_states_main, self.layer_idx,
            {"sin": sin, "cos": cos, "cache_position": cache_position}
        )
        key_states_local, value_states_local = self.kv_cluster.update_kv50(
            key_states, query_states, value_states, q_samp, groups_size, num_groups
        )
        past_key_value.update(
            key_states_local, value_states_local, self.layer_idx + 32,
            {"sin": sin, "cos": cos, "cache_position": cache_position}
        )

        # 常规致密 FA2（无窗口）
        q_fa = query_states.transpose(1, 2)
        k_fa = key_states.transpose(1, 2)
        v_fa = value_states.transpose(1, 2)
        attn_output = _flash_attention_forward(
            q_fa, k_fa, v_fa,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=None,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        self.kv_cluster.prefill_len = q_len

        # 均值/方差统计（prefill 固定）
        if self.num_key_value_heads == self.num_heads:
            K_bhTd = key_states                                # [B,H,T,D]
        else:
            K_bhTd = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        t_sink  = getattr(self.kv_cluster, "t_sink",  32)
        t_local = getattr(self.kv_cluster, "t_local", 128)
        mu_K, var_K = _estimate_bulk_stats_diag_from_prefill(
            K_bhTd=K_bhTd, T=q_len, t_sink=t_sink, t_local=t_local
        )

        # —— 关键：prefill 一次性准备 QAdA 的缓存 —— #
        # a) mu/var：到 compute dtype + contiguous
        self.kv_cluster.qada_mu_K  = mu_K.contiguous()
        self.kv_cluster.qada_var_K = var_K.contiguous()
        self.kv_cluster.qada_tau   = getattr(self.kv_cluster, "qada_tau", 0.6)
        self.kv_cluster.qada_stats_ready = True

        # b) log_Tvis_per_head：fp32 + contiguous
        T_full_scalar  = int(kv_seq_len)
        T_local_scalar = min(T_full_scalar, int(t_local))
        T_vis_scalar   = max(float(T_full_scalar - T_local_scalar), 1e-8)
        log_Tvis_scalar = math.log(T_vis_scalar)
        log_Tvis_per_head = torch.full((Hh,), float(log_Tvis_scalar),
                                       device=device, dtype=torch.float32).contiguous()
        self.kv_cluster.qada_log_Tvis_per_head = log_Tvis_per_head   # [H], fp32

        # c) （可选）local LSE 存成 [H] fp32 contiguous，供 decode 复用
        # self.kv_cluster.qada_lse_fp32 = ...

        # d)（可选）持久化输出缓冲
        if not hasattr(self.kv_cluster, "_qada_out_u8") or self.kv_cluster._qada_out_u8.numel() != self.num_heads:
            self.kv_cluster._qada_out_u8 = torch.empty(self.num_heads, dtype=torch.uint8, device=device)

        # e) （可选）流同步：避免 decode 第一次触碰缓存引起隐式等待
        self.kv_cluster._prefill_stream = torch.cuda.current_stream(device)
        if not hasattr(self.kv_cluster, "_decode_stream"):
            self.kv_cluster._decode_stream = torch.cuda.current_stream(device)
        self.kv_cluster._decode_stream.wait_stream(self.kv_cluster._prefill_stream)

    # ========================= DECODE =========================
    else:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt   = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        # local 槽
        cache_kwargs["head_lens"] = self.kv_cluster.alt_head_lens
        cache_kwargs["cu_klen"]   = self.kv_cluster.alt_cu_klen
        key_states_local, value_states_local = past_key_value.update(
            key_states, value_states, self.layer_idx + 32, cache_kwargs
        )

        # full 槽
        cache_kwargs["head_lens"] = self.kv_cluster.head_lens
        cache_kwargs["cu_klen"]   = self.kv_cluster.cu_klen
        key_states_full, value_states_full = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

        # 元数据推进
        self.kv_cluster.alt_klen_sum     += self.num_heads
        self.kv_cluster.alt_max_seqlen_k += 1
        self.kv_cluster.alt_cu_klen      += self.kv_cluster.alt_cu_offset
        self.kv_cluster.alt_head_lens    += 1

        self.kv_cluster.klen_sum         += self.num_heads
        self.kv_cluster.max_seqlen_k     += 1
        self.kv_cluster.cu_klen          += self.kv_cluster.cu_offset
        self.kv_cluster.head_lens        += 1

        # varlen 输入
        q_var        = query_states.view(-1, self.num_key_value_groups, self.head_dim)
        k_var_full   = key_states_full.view(-1, 1, self.head_dim)
        v_var_full   = value_states_full.view(-1, 1, self.head_dim)
        k_var_local  = key_states_local.view(-1, 1, self.head_dim)
        v_var_local  = value_states_local.view(-1, 1, self.head_dim)

        cu_seqlens_q    = self.kv_cluster.cu_qlen
        cu_seqlens_k    = self.kv_cluster.cu_klen
        max_seqlen_q    = 1
        max_seqlen_k    = self.kv_cluster.max_seqlen_k

        alt_cu_seqlens_q = self.kv_cluster.alt_cu_qlen
        alt_cu_seqlens_k = self.kv_cluster.alt_cu_klen
        alt_max_seqlen_q = 1
        alt_max_seqlen_k = self.kv_cluster.alt_max_seqlen_k



        # 两路 varlen FA2
        attn_output_local, softmax_lse_local, _ = flash_attn_varlen_func(
            q_var, k_var_local, v_var_local,
            alt_cu_seqlens_q, alt_cu_seqlens_k,
            alt_max_seqlen_q, alt_max_seqlen_k,
            causal=True,
            return_attn_probs=True,   # 取 LSE
            softmax_scale=None
        )


        attn_output_full = flash_attn_varlen_func(
            q_var, k_var_full, v_var_full,
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            causal=True, return_attn_probs=False
        )


        attn_output_local = attn_output_local.reshape(bsz, self.num_heads, q_len, self.head_dim)
        attn_output_full  = attn_output_full.reshape( bsz, self.num_heads, q_len, self.head_dim)

        # --- QAdA 路由 ---
        force_local = (self.layer_idx < 2) or (self.layer_idx == 31)
        if force_local:
            head_mask = torch.ones(1, self.num_heads, dtype=torch.bool, device=attn_output_full.device)
        elif not getattr(self.kv_cluster, "qada_stats_ready", False):
            head_mask = torch.zeros(1, self.num_heads, dtype=torch.bool, device=attn_output_full.device)
        else:
            if hasattr(self.kv_cluster, "qada_log_Tvis_per_head"):
                log_Tvis_per_head = self.kv_cluster.qada_log_Tvis_per_head.contiguous()
            else:
                T_full_per_head  = (cu_seqlens_k[1:]     - cu_seqlens_k[:-1]).to(torch.float32)
                T_local_per_head = (alt_cu_seqlens_k[1:] - alt_cu_seqlens_k[:-1]).to(torch.float32)
                T_vis_per_head   = torch.clamp(T_full_per_head - T_local_per_head, min=1e-8)
                log_Tvis_per_head = torch.log(T_vis_per_head).contiguous()
                self.kv_cluster.qada_log_Tvis_per_head = log_Tvis_per_head

            # 可选：把 local LSE 存成 [H] fp32 contiguous，供下步 decode 直接复用
            lse_for_kernel = softmax_lse_local.squeeze(0).transpose(0, 1).contiguous()
            if lse_for_kernel.dim() == 2:
                lse_for_kernel = lse_for_kernel[:, 0]
            lse_for_kernel = lse_for_kernel.to(torch.float32).contiguous()
            self.kv_cluster.qada_lse_fp32 = lse_for_kernel

            # 复用输出缓冲
            out_buf = getattr(self.kv_cluster, "_qada_out_u8", None)
            if (out_buf is None) or (out_buf.numel() != self.num_heads) or (out_buf.device != device):
                out_buf = torch.empty(self.num_heads, dtype=torch.uint8, device=device)
                self.kv_cluster._qada_out_u8 = out_buf
            # print(self.kv_cluster.qada_mu_K.shape,self.kv_cluster.qada_var_K.shape,query_states.shape)
            # 调用 Triton QAdA（传 4D query_states，kernel 内按 t_offset 取最后 token）
                       # 调用 Triton QAdA（传 4D query_states，kernel 内按 t_offset 取最后 token）
            head_mask, acc_buf = qada_mask_fused_triton_packed_fast_adaptive(
                q_states=query_states,                         # [B,H,T,D]（B=1，T=1）
                mu_K=self.kv_cluster.qada_mu_K,               # [H,D]（compute dtype, contiguous）
                var_K=self.kv_cluster.qada_var_K,             # [H,D]
                log_Tvis_per_head=log_Tvis_per_head,          # [H] fp32 contiguous
                softmax_lse_local=lse_for_kernel,             # [H] fp32 contiguous
                tau=getattr(self.kv_cluster, "qada_tau", 0.6),
                acc_dtype=None,                               # 跟随输入，避免 cast
                BLOCK_H=32, D_TILE=256, num_warps=8, num_stages=4,
                out_buf=out_buf,
                acc_buf=getattr(self.kv_cluster, "_qada_accbuf", None),
                profile=False, print_profile=False,
                self_obj=self,
            )
            self.kv_cluster._qada_out_u8 = head_mask.view(-1).to(torch.uint8)
            if acc_buf is not None:
                self.kv_cluster._qada_accbuf = acc_buf

            # ========= 下面是 CUDA 版 mask 的对比 =========
            # 注意：这里假设你的 CUDA binding 叫 qada_local
            #       并且签名是：
            #       qada_local(q, k, v, cu_q, cu_k, max_q, max_k,
            #                  mu, var, logTvis, tau, causal, softmax_scale)



            # varlen 索引转换成 int32 contiguous
            alt_cu_q_i = alt_cu_seqlens_q.to(torch.int32).contiguous()
            alt_cu_k_i = alt_cu_seqlens_k.to(torch.int32).contiguous()

            # 调 CUDA kernel 得到 CUDA 版的 head_mask_u8 / lse_local_h
            attn_output_local_cuda, lse_local_h, head_mask_u8 = qada_local(
                q_var,                    # [Nq, Hh, D]，上面已经算好
                k_var_local,              # [Nk, 1, D]
                v_var_local,              # [Nk, 1, D]
                alt_cu_q_i, alt_cu_k_i,
                alt_max_seqlen_q, alt_max_seqlen_k,
                self.kv_cluster.qada_mu_K,      # [H,D]，此处 B*Hh == H
                self.kv_cluster.qada_var_K,     # [H,D]
                log_Tvis_per_head,              # [H]
                getattr(self.kv_cluster, "qada_tau", 0.6),
                True,                            # causal=True
                0.0,                             # softmax_scale=0 => 内部用 1/sqrt(D)
            )

            # import math

            # Triton mask: [1,H] -> [H]
            hm_triton = head_mask.view(-1).to(torch.uint8)          # [H]
            hm_cuda   = head_mask_u8.view(-1)                       # [H]

            print("head_mask_triton:", hm_triton)
            print("head_mask_cuda:   ", hm_cuda)
            print("head_mask_diff_max:", (hm_triton - hm_cuda).abs().max().item())

            # LSE 对比
            lse_triton = self.kv_cluster.qada_lse_fp32.view(-1).to(torch.float32)  # [H]
            lse_cuda   = lse_local_h.view(-1).to(torch.float32)                    # [H]
            print("lse_diff_max:", (lse_triton - lse_cuda).abs().max().item())

            # ---- 找出被翻转的 head index ----
                       # ---- 找出被翻转的 head index ----
            diff_idx = torch.nonzero(hm_triton != hm_cuda).view(-1)
            if diff_idx.numel() == 0:
                print("No head mismatch between Triton and CUDA.")
            else:
                diff_idx = diff_idx.tolist()
                print("different head index:", diff_idx)

    
                # Triton 这边：直接从原始 query_states 里拿最后一个 token 的 per-head q
                # query_states: [B, H, T, D]，decode-only 一般 B=1, T=1
                q_tri = query_states[:, :, -1, :].to(torch.float32)   # [B, H, D]
                if q_tri.size(0) != 1:
                    print(f"[WARN] q_tri batch size = {q_tri.size(0)}, 下面只看第 0 个 batch")
                q_tri = q_tri[0]                                      # [H, D]

                # CUDA 这边：沿用你之前的约定，用 q_var[:, 0, :] 当作 [H, D]
                # q_var: [Nq, Hh, D]，在 decode-only + 你现在的 reshape 下，
                #       Nq 应该等于 num_heads，Hh=num_key_value_groups
                q_cuda = q_var[:, 0, :].to(torch.float32)            # [H, D]

                mu   = self.kv_cluster.qada_mu_K.to(torch.float32)   # [H, D]
                var  = self.kv_cluster.qada_var_K.to(torch.float32).clamp_min(1e-6)
                logT = log_Tvis_per_head.to(torch.float32).view(-1)  # [H]
                lse_t = lse_triton                                   # [H]
                lse_c = lse_cuda                                     # [H]

                D_f = float(self.head_dim)
                invD = 1.0 / D_f
                scale = 1.0 / math.sqrt(D_f)
                tau = float(getattr(self.kv_cluster, "qada_tau", 0.6))
                c_thresh = math.log1p(-tau) - math.log(tau)

                H_debug = hm_triton.numel()

                for h in diff_idx:
                    if h >= H_debug or h >= q_tri.size(0) or h >= q_cuda.size(0):
                        print(f"[WARN] head {h} 越界，H_debug={H_debug}, "
                              f"q_tri_H={q_tri.size(0)}, q_cuda_H={q_cuda.size(0)}")
                        continue

                    h_int = int(h)

                    # 1) q 向量是否一致？
                    q_diff = (q_tri[h_int] - q_cuda[h_int]).abs().max().item()

                    # 2) 用 Triton 的 q_tri 以及 CUDA 的 q_cuda，分别按同一套公式算 logA/th
                    # Triton 视角：用 q_tri[h]
                    mu_acc_t = (q_tri[h_int] * mu[h_int]).sum().item()
                    s2_acc_t = ((q_tri[h_int] * q_tri[h_int]) * var[h_int]).sum().item()
                    s2_mean_t = s2_acc_t * invD
                    logA_t = logT[h_int].item() + mu_acc_t * scale + 0.5 * s2_mean_t
                    th_t   = lse_t[h_int].item() + c_thresh

                    # CUDA 视角：用 q_cuda[h]
                    mu_acc_c = (q_cuda[h_int] * mu[h_int]).sum().item()
                    s2_acc_c = ((q_cuda[h_int] * q_cuda[h_int]) * var[h_int]).sum().item()
                    s2_mean_c = s2_acc_c * invD
                    logA_c = logT[h_int].item() + mu_acc_c * scale + 0.5 * s2_mean_c
                    th_c   = lse_c[h_int].item() + c_thresh

                    print(f"\n====== Head {h_int} debug ======")
                    print(f"q diff max        = {q_diff:.6e}")
                    print(f"lse_triton        = {lse_t[h_int].item():.6f}")
                    print(f"lse_cuda          = {lse_c[h_int].item():.6f}")
                    print(f"Triton logA       = {logA_t:.6f}")
                    print(f"Triton th         = {th_t:.6f}")
                    print(f"Triton logA - th  = {logA_t - th_t:.6e}")
                    print(f"CUDA   logA       = {logA_c:.6f}")
                    print(f"CUDA   th         = {th_c:.6f}")
                    print(f"CUDA   logA - th  = {logA_c - th_c:.6e}")
                # ================= 额外 debug 结束 =================



            # ========= 对比结束 =========

            self.kv_cluster._qada_out_u8 = head_mask.view(-1).to(torch.uint8)
            if acc_buf is not None:
                self.kv_cluster._qada_accbuf = acc_buf

        head_mask_4d = head_mask.view(bsz, self.num_heads, 1, 1).expand(bsz, self.num_heads, q_len, 1)
        attn_output = torch.where(head_mask_4d, attn_output_local, attn_output_full) \
                        .transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        end_evt.record()
        torch.cuda.synchronize()
        elapsed_ms = start_evt.elapsed_time(end_evt)   # 毫秒 float 值
        print(f"[QAdA fused kernel] time = {elapsed_ms:.4f} ms")

    # --- output proj ---
    attn_output = self.o_proj(attn_output)
    attn_weights = None if not output_attentions else None

    return attn_output, attn_weights, past_key_value
import torch

DEBUG_DIMS = [2533,1415]  # 示例维度，你可以根据需要调整
THRESHOLD = 20  # 激活值的阈值     Dim  2533: 892.5000 llava的最大激活值 Dim  1415: 530.5000

def custom_LlamaDecoderLayer_forward(
    self,
    idx: int,  # 新增索引参数，用于标识当前层
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> Tuple[torch.FloatTensor, ...]:
    """
    Custom forward pass for a LlamaDecoderLayer with optional debug logging.
    
    Adds activation analysis for specific layers during prefill:
      - Computes φ_d = |x_d| / RMS(x) for selected dimensions
      - Prints top-20 token positions with highest φ_d (first sample only)
    """
    bsz, q_len, hidden_dim = hidden_states.shape
    

    bsz, q_len, hidden_dim = hidden_states.shape

    if q_len > 1:  # Prefill phase only
        # 💡 关键修复：转为 float32 避免 x**2 上溢（float16 最大约 65504，sqrt≈256）
        x = hidden_states[0, :, :].to(torch.float32)  # (q_len, hidden_dim)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-8)  # (q_len, 1)

        valid_phis = []
        valid_dims = []

        for dim in DEBUG_DIMS:
            if dim >= hidden_dim:
                continue
            x_d = x[:, dim]  # (q_len,)
            phi_d = torch.abs(x_d) / rms.squeeze(-1)  # (q_len,)
            
            # 可选：打印调试信息（现在会显示正确 φ 值）
            # print(f"Layer {idx} | Dim {dim}: x[0]={x_d[0].item():8.1f}, φ[0]={phi_d[0].item():6.2f}")

            valid_phis.append(phi_d)
            valid_dims.append(dim)

        if valid_phis:
            phi_matrix = torch.stack(valid_phis, dim=0)      # (D, q_len)
            max_phi, _ = torch.max(phi_matrix, dim=0)        # (q_len,)

            # ✅ 修复2：使用合理阈值（φ > 3 已很显著）
            THRESHOLD = 10.0  # ← 改这里！20 几乎永远不会触发
            
            mask = max_phi > THRESHOLD                       # (q_len,)
            selected_indices = torch.nonzero(mask, as_tuple=True)[0]  # (N,)

            kwargs["debug_topk_positions"] = selected_indices
            kwargs["debug_layer_idx"] = idx

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def adaptive_LlamaModel_forward1(
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

    for idx,decoder_layer in enumerate(self.layers):
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
            layer_outputs = decoder_layer(# 传入层
                idx,
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