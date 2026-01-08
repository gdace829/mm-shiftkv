import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
import os
from typing import List
import random
import numpy as np
import json
import warnings
from typing import List, Optional, Tuple
from transformers.cache_utils import Cache
import matplotlib.pyplot as plt
from mmshift.statistical_query_predictor import StatisticalQueryPredictor
STATISTICAL_PREDICTOR_AVAILABLE = True
# 导入统计表预测器模块
# try:
#     # import sys
#     # sys.path.insert(0, '/data')
#     from statistical_query_predictor import StatisticalQueryPredictor
#     STATISTICAL_PREDICTOR_AVAILABLE = True
# except ImportError:
#     STATISTICAL_PREDICTOR_AVAILABLE = False
#     print("[SparseMM] Statistical query predictor module not available")

def plot_attention_head(attn_score: torch.Tensor, layer_idx: int, head_idx: int, topk: int = 20, save_path: str = None):
    """
    可视化单个 head 的注意力分数。
    支持 attn_score 形状：
      - [1, H, S]  （例如对 T 已做平均）
      - [1, H, T, S]
    参数：
      head_idx: 要可视化的 head
      topk: 同时打印 top-k 的位置与得分，便于调试
      save_path: 如传入则保存图片到该路径
    """
    assert attn_score.dim() in (3, 4), f"Unsupported shape: {attn_score.shape}"
    dev = attn_score.device

    if attn_score.dim() == 3:
        # [1, H, S]
        scores_1d = attn_score[0, head_idx]                      # [S]
        scores_np = scores_1d.detach().float().cpu().numpy()

        plt.figure(figsize=(10, 3))
        plt.plot(scores_np)
        plt.title(f'Layer {layer_idx} - Head {head_idx} attention scores (S)')
        plt.xlabel('Key index')
        plt.ylabel('Score')
        plt.tight_layout()

        # 打印 top-k
        if topk is not None and topk > 0:
            vals, idxs = torch.topk(scores_1d, k=min(topk, scores_1d.numel()))
            print(f"[Top-{len(idxs)}] indices:", idxs.tolist())
            print(f"[Top-{len(idxs)}] values :", vals.tolist())

    else:
        # [1, H, T, S]
        scores_2d = attn_score[0, head_idx]                      # [T, S]
        scores_np = scores_2d.detach().float().cpu().numpy()

        plt.figure(figsize=(8, 6))
        plt.imshow(scores_np, aspect='auto', origin='lower')
        plt.title(f'Layer {layer_idx} - Head {head_idx} attention map (T x S)')
        plt.xlabel('Key index (S)')
        plt.ylabel('Query index (T)')
        plt.colorbar(label='Score')
        plt.tight_layout()

        # 打印每个 query 行的 top-k（可按需关闭/精简）
        if topk is not None and topk > 0:
            t = scores_2d.size(0)
            k = min(topk, scores_2d.size(1))
            vals, idxs = torch.topk(scores_2d, k=k, dim=-1)     # [T, k]
            print(f"Per-query Top-{k} key indices (showing first few rows):")
            max_rows = min(3, t)
            for qi in range(max_rows):
                print(f"  q={qi:02d} idx={idxs[qi].tolist()} val={vals[qi].tolist()}")

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"Saved attention figure to {save_path}")

def load_head_score(model_type):
    if 'llava' in model_type:
        if 'mistral' not in model_type:
            head_score_path = './visual_head/head_score/llava-v1.6.json'
        else:
            head_score_path = './visual_head/head_score/llava-mistral-v1.6.json'
    elif 'qwen' in model_type:
        head_score_path = './visual_head/head_score/qwen.json'
    else:
        raise NotImplementedError
    with open(head_score_path, 'r') as f:
        head_score = json.load(f)
    return head_score

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def merge_kv(key_states, value_states, indices, window_size, merge):
    # merge methods in LOOK-M

    bsz, num_heads, k_len, head_dim = key_states.shape

    # kv-selected
    selected_keys = key_states.gather(dim=2, index=indices)  # [bsz, num_heads, topk_len, head_dim]
    selected_values = value_states.gather(dim=2, index=indices)  # [bsz, num_heads, topk_len, head_dim]

    # kv-drop
    all_indices = torch.arange(k_len, device=key_states.device).unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, k_len)
    all_indices_flattened = all_indices.flatten()  # [bsz * num_heads * (k_len-window_size)]
    selected_indices_flattened = indices.flatten()  # [bsz * num_heads * topk_len]
    is_selected = torch.isin(all_indices_flattened, selected_indices_flattened)
    drop_indices_flattened = all_indices_flattened[~is_selected]
    drop_len = drop_indices_flattened.shape[0] // (all_indices.shape[0] * all_indices.shape[1])
    drop_indices = drop_indices_flattened.reshape(all_indices.shape[0], all_indices.shape[1], drop_len) # [bsz * num_heads * (k_len-window_size-topk_len)]
    drop_indices = drop_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # [bsz, num_heads, (k_len-window_size-topk_len), head_dim]
    drop_keys = key_states.gather(dim=2, index=drop_indices)
    drop_values = value_states.gather(dim=2, index=drop_indices)

    # kv-recent
    recent_keys = key_states[:, :, -window_size:, :]

    ##### apply merge #####
    # prepare for merge
    k_hh_pruned = drop_keys  # [bsz, num_heads, k_len-topk_len-window_size, head_dim]
    k_hh_recent = torch.cat([recent_keys, selected_keys], dim=2)  # [bsz, num_heads, topk_len+window_size, head_dim]
    v_hh_pruned = drop_values  # [bsz, num_heads, k_len-topk_len-window_size, head_dim]
    v_hh_recent = torch.cat([selected_values, value_states[:, :, -window_size:, :]], dim=2)  # [bsz, num_heads, topk_len+window_size, head_dim]
    # similarity matrix
    similarity = (k_hh_pruned / torch.norm(k_hh_pruned, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128)) @ ((k_hh_recent / (torch.norm(k_hh_recent, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128))).transpose(-1, -2)) # cosin
    max_values, max_indices = similarity.max(dim=-1)

    # pivot merge
    if merge=="pivot":
        print("Pivot merge")
        merged_indices = max_indices.unsqueeze(-1).repeat(1, 1, 1, 128)
        k_hh_selected = torch.gather(input=k_hh_recent, dim=2, index=merged_indices)
        k_hh_merged = (k_hh_pruned + k_hh_selected)/2
        k_hh_recent = torch.scatter_reduce(input=k_hh_recent, dim=2, index=merged_indices, src=k_hh_merged, reduce='mean', include_self=True) # include_self=True seems decrease the performance
        v_hh_selected = torch.gather(input=v_hh_recent, dim=2, index=merged_indices)
        v_hh_merged = (v_hh_pruned + v_hh_selected)/2
        v_hh_recent = torch.scatter_reduce(input=v_hh_recent, dim=2, index=merged_indices, src=v_hh_merged, reduce='mean', include_self=True)
    else:
        raise ValueError('Merge method not supported')

    # TODO: other merge strategies
    # average merge
    # weight merge

    return k_hh_recent, v_hh_recent

class DynamicCacheSplitHeadFlatten(Cache):
    '''
    adapt from https://github.com/FFY0/AdaKV.
    '''
    def __init__(self) ->None:
        # Token wise List[]  Head wise KV List[torch.Tensor]
        super().__init__()
        # print(self) qwen 改为28 llava 改为32
        self.key_cache: List[Optional[torch.Tensor]] = [None] * 32 # 一开始将这个初始化为None
        self.value_cache: List[Optional[torch.Tensor]] = [None] * 32

        self._seen_tokens = 0

    def __len__(self):
        return len(self.key_cache)

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))

    def __getitem__(self, layer_idx: int) -> Tuple[Tuple[torch.Tensor],Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")
    # 最后更新到这里
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None,up=False):
        kc = self.key_cache[layer_idx]
        if kc is None or (isinstance(kc, list) and len(kc) == 0):
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        else:
            kc = self.key_cache[layer_idx]
            # print("DEBUG cache type:", layer_idx, type(kc), "len" if isinstance(kc, list) else "", (len(kc) if isinstance(kc, list) else ""))

            assert self.key_cache[layer_idx].dim() == 2
            bs, head, seqlen, dim = key_states.shape
            assert bs == 1 and seqlen == 1
            head_lens = cache_kwargs["head_lens"]
            cu_klen = cache_kwargs["cu_klen"]

            # import nvtx
            # copy_old_rng = nvtx.start_range("copy old")
            from tiny_api_cuda import update_flatten_view
            new_key_cache = update_flatten_view(self.key_cache[layer_idx].view(-1,dim), key_states.view(-1, dim), head_lens, cu_klen)
            new_value_cache = update_flatten_view(self.value_cache[layer_idx].view(-1,dim), value_states.view(-1, dim), head_lens, cu_klen)

            # nvtx.end_range(copy_old_rng)

            self.key_cache[layer_idx] = new_key_cache
            self.value_cache[layer_idx] = new_value_cache

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        # TODO: return 1 to means has content for now
        return 1
        # return max(map(lambda states: states.shape[-2], self.key_cache[layer_idx]))

    def get_max_length(self) -> Optional[int]:
        return None

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. DynamicCache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCacheEachHead":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
              
                cache.update(key_states, value_states, layer_idx)
        return cache

class SnapKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', layer_idx = None, num_hidden_layers = None, 
                 pyram_mode = False, pyram_beta = 20,num_key_value_groups = 1, gqa_func='mean'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

        self.pyram_init = False
        self.pyram_mode = pyram_mode
        self.pyram_beta = pyram_beta
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        self.num_key_value_groups = num_key_value_groups
        self.gqa_func = gqa_func

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, origin_key_states, query_states, origin_value_states):
        
        # support gqa
        key_states = repeat_kv(origin_key_states, self.num_key_value_groups)
        value_states = repeat_kv(origin_value_states, self.num_key_value_groups)
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        # compute pyramidal capacity
        if self.pyram_mode and not self.pyram_init:
            # NOTE: (max_num + min_num) / 2 == base_capacity to restrict the total capacity
            base_capacity = self.max_capacity_prompt - self.window_size
            min_num = base_capacity // self.pyram_beta
            max_num = base_capacity * 2 - min_num
                
            # if the max_num is larger than the query length, we need to adjust the max_num
            if max_num >= q_len - self.window_size:
                max_num = q_len - self.window_size
                min_num = base_capacity * 2 - max_num
        
            # NOTE: compute interval
            steps = (max_num - min_num) // (self.num_hidden_layers - 1)

            self.max_capacity_prompt = max_num - self.layer_idx * steps + self.window_size
            self.pyram_init = True
            print(f"Pyram mode adaptive capacity, layer: {self.layer_idx}, max_capacity_prompt: {self.max_capacity_prompt}, base_capacity: {self.max_capacity_prompt - self.window_size}", flush=True)

        if q_len < self.max_capacity_prompt:
            return origin_key_states, origin_value_states
        else:# 手动计算注意力
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim = -2)
            
            attn_weights_mean = attn_weights_mean.view(attn_weights_mean.shape[0], -1, self.num_key_value_groups, attn_weights_mean.shape[-1])
            if self.gqa_func == 'max':
                attn_weights_mean = attn_weights_mean.max(dim=-2).values
            elif self.gqa_func == 'mean':
                attn_weights_mean = attn_weights_mean.mean(dim=-2)
            else:
                raise ValueError('gqa_func not supported')
                
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_mean, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_mean, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')

            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            
            k_past_compress = origin_key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = origin_value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = origin_key_states[:, :, -self.window_size:, :]
            v_cur = origin_value_states[:, :, -self.window_size:, :]

            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states

class AdaKVCluster():
    def __init__(self, window_size = 32, kernel_size = 7, pooling = 'maxpool',base_capacity=None,floor_alpha = None,skip = None,normalize=None, 
                 layer_idx = None, num_hidden_layers = None, pyram_mode = False, pyram_beta = 20, num_key_value_groups=1, gqa_func='mean'):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.base_capacity = base_capacity - window_size
        self.floor_ratio = floor_alpha
        self.floor_capacity = int(self.base_capacity * self.floor_ratio)
        self.adaptive_capacity = self.base_capacity - self.floor_capacity
        self.skip = skip

        self.normalize = normalize
        self.pyram_init = False
        self.pyram_mode = pyram_mode
        self.pyram_beta = pyram_beta
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        # NOTE: layer-wise meta-data
        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_headlens = None

        self.num_key_value_groups = num_key_value_groups
        self.gqa_func = gqa_func

    def calcul_attn_sore(self, key_states, query_states):
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(
            head_dim)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim=-2)

        attn_weights_mean = attn_weights_mean.view(attn_weights_mean.shape[0],num_heads//self.num_key_value_groups,self.num_key_value_groups,-1)
        if self.gqa_func == 'max':
            attn_weights_mean = attn_weights_mean.max(dim=-2).values
        elif self.gqa_func == 'mean':
            attn_weights_mean = attn_weights_mean.mean(dim=-2)
        else:
            raise ValueError('gqa_func not supported')

        if self.pooling == 'avgpool':
            attn_weights_mean_pooling = F.avg_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        elif self.pooling == 'maxpool':
            attn_weights_mean_pooling = F.max_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        else:
            raise ValueError('Pooling method not supported')
        return attn_weights_mean_pooling

    def update_kv(self, origin_key_states, query_states, origin_value_states):
        key_states = repeat_kv(origin_key_states, self.num_key_value_groups)
        # value_states = repeat_kv(origin_value_states, self.num_key_value_groups)

        # check if prefix phase        assert key_states.shape[-2] == query_states.shape[-2]
        _device = key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_score= self.calcul_attn_sore(key_states,query_states)
        # import pdb; pdb.set_trace()
        origin_heads_key_states = torch.split(origin_key_states, 1, dim=1)
        origin_heads_value_states = torch.split(origin_value_states, 1, dim=1)

        # compute pyramidal capacity
        if self.pyram_mode and not self.pyram_init:
            # NOTE: (max_num + min_num) / 2 == base_capacity to restrict the total capacity
            min_num = self.base_capacity // self.pyram_beta
            max_num = self.base_capacity * 2 - min_num
                
            # if the max_num is larger than the query length, we need to adjust the max_num
            if max_num >= q_len - self.window_size:
                max_num = q_len - self.window_size
                min_num = self.base_capacity * 2 - max_num
        
            # NOTE: compute interval
            steps = (max_num - min_num) // (self.num_hidden_layers - 1)

            # renew adaptive capacity
            self.base_capacity = max_num - self.layer_idx * steps
            self.floor_capacity = int(self.base_capacity * self.floor_ratio)
            self.adaptive_capacity = self.base_capacity - self.floor_capacity
            self.pyram_init = True
            print(f"Pyram mode adaptive capacity, layer: {self.layer_idx}, acap: {self.adaptive_capacity}, bcap: {self.base_capacity}, fcap: {self.floor_capacity}",  flush=True)

        def init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k):
            # init metadata
            self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=_device)
            self.klen_sum = klen_sum
            self.max_seqlen_k = max_seqlen_k
            self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
            # init varlen flash attention metadata
            self.cu_klen = self.cu_headlens - self.head_lens
            self.cu_klen = torch.cat(
                [self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=_device)], dim=0)
            # check bug
            self.layer_qlens = torch.ones(num_heads//self.num_key_value_groups, dtype=torch.int32,device=_device)
            self.qlen_sum = num_heads//self.num_key_value_groups
            self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
            self.cu_qlen = torch.cat(
                [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=_device)], dim=0)
            
            
            self.cu_offset = torch.arange(0, num_heads//self.num_key_value_groups + 1, dtype=torch.int32, device=_device)
            self.cu_head_offset = torch.arange(1, num_heads//self.num_key_value_groups +1, dtype=torch.int32, device=_device)

        if self.base_capacity > attn_score.size(-1):
            init_metadata(num_heads, [q_len] * (num_heads//self.num_key_value_groups), q_len * (num_heads//self.num_key_value_groups), q_len)
            # not compress
            return origin_key_states.reshape(-1, head_dim), origin_value_states.reshape(-1, head_dim)

        sorted_attn_score,sorted_attn_score_indices = attn_score.sort(dim=-1,descending=True)
        if self.layer_idx >= self.skip:
            adaptive_attn_score = sorted_attn_score
            length = adaptive_attn_score.size(dim=-1)
            if self.normalize:
                ratio_weight = sorted_attn_score[...,:self.base_capacity].sum(dim=-1,keepdim=True)/sorted_attn_score.sum(dim=-1,keepdim=True)
                adaptive_attn_score = adaptive_attn_score*ratio_weight
            adaptive_attn_score = adaptive_attn_score.reshape(bsz,length*num_heads//self.num_key_value_groups)
            sorted_indices = torch.topk(adaptive_attn_score,k=num_heads*self.base_capacity//self.num_key_value_groups,dim=-1).indices
            sorted_indices = sorted_indices//length

            # floor_alpha capacity set
            head_adaptive_capacity = torch.zeros((bsz,num_heads//self.num_key_value_groups),device=_device,dtype = sorted_indices.dtype)
            head_adaptive_capacity.scatter_add_(-1,sorted_indices,torch.ones_like(sorted_indices,dtype=head_adaptive_capacity.dtype),)
            assert head_adaptive_capacity.sum().item() == num_heads*self.base_capacity//self.num_key_value_groups
            head_adaptive_capacity = torch.round(head_adaptive_capacity * (1-self.floor_ratio) + self.floor_capacity).int()
        else:
            head_adaptive_capacity = torch.ones((bsz,num_heads),device=_device,dtype = sorted_attn_score_indices.dtype) * self.base_capacity
        sorted_attn_score_indices = sorted_attn_score_indices.split(1,dim=1)

        heads_key_states = []
        heads_value_states = []
        assert bsz == 1
        # per head

        # reinit varlen metadata
        k_lens = []
        klen_sum = 0
        max_seqlen_k = 0
        self.cu_klen = 0

        for head_idx in range(num_heads//self.num_key_value_groups):
            cache_index = sorted_attn_score_indices[head_idx][...,:head_adaptive_capacity[0][head_idx]]

            l = cache_index.shape[-1] + self.window_size
            k_lens.append(l)
            max_seqlen_k = max(max_seqlen_k, l)
            klen_sum += l

            cache_index = cache_index.view(1, 1, -1, 1).expand(-1, -1, -1, head_dim)
            top_Kcache = origin_heads_key_states[head_idx].gather(dim=2,index=cache_index)
            top_Vcache = origin_heads_value_states[head_idx].gather(dim=2,index=cache_index)
            selected_k = torch.cat([top_Kcache,origin_heads_key_states[head_idx][:, :, -self.window_size:, :]],dim=2)
            selected_v = torch.cat([top_Vcache,origin_heads_value_states[head_idx][:, :, -self.window_size:, :]],dim=2)

            # NOTE: flatten view
            heads_key_states.append(selected_k.view(-1, head_dim))
            heads_value_states.append(selected_v.view(-1, head_dim))

        init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k)

        # NOTE: compose as flatten view
        heads_key_states = torch.cat(heads_key_states, dim=0)
        heads_value_states = torch.cat(heads_value_states, dim=0)

        return heads_key_states, heads_value_states

class SparseMM():
    def __init__(self, window_size = 32, kernel_size = 7, pooling = 'maxpool', base_capacity=None, ratio=None, normalize=None, 
                 layer_idx = None, num_hidden_layers = None, head_score=None, num_attention_heads=32, num_key_value_groups=1, gqa_func='mean', model_type=None):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.base_capacity = base_capacity - window_size# 在这里减去了
        self.ratio = ratio

        self.normalize = normalize
        self.layer_idx = layer_idx
        self.num_attention_heads = num_attention_heads  
        self.num_hidden_layers = num_hidden_layers

        # NOTE: layer-wise meta-data
        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_headlens = None

        self.num_key_value_groups = num_key_value_groups
        self.gqa_func = gqa_func

        if head_score == 'random':
            head_score_list = np.array([random.random() for _ in range(self.num_hidden_layers * self.num_attention_heads)])
        elif head_score == 'visual':
            head_score = load_head_score(model_type)
            head_score_list = [np.mean(l[1]) for l in head_score.items()]
        head_score_list = torch.tensor(head_score_list / sum(head_score_list))
        # GQA support
        self.score = head_score_list.view(self.num_hidden_layers, self.num_attention_heads//self.num_key_value_groups, self.num_key_value_groups)
        self.score = self.score.sum(dim=-1)

        min_cache = int(self.base_capacity * self.ratio)
        remain_capacity = (self.base_capacity - min_cache) * self.num_hidden_layers * self.num_attention_heads // self.num_key_value_groups
        self.head_adaptive_capacity = torch.round(self.score * remain_capacity + min_cache).int()
        self.head_adaptive_capacity = torch.full(
            (self.num_hidden_layers, self.num_attention_heads // self.num_key_value_groups),
            fill_value=self.base_capacity,
            dtype=torch.int32,
            device=self.score.device
        )

                # self.head_adaptive_capacity = self.base_capacity
        self.head_adaptive_capacity_copy=torch.round(self.score * remain_capacity + min_cache).int()
        # print(self.head_adaptive_capacity)
    def calcul_attn_sore(self, key_states, query_states):
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(
            head_dim)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]
        
        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim=-2)

        attn_weights_mean = attn_weights_mean.view(attn_weights_mean.shape[0],num_heads//self.num_key_value_groups,self.num_key_value_groups,-1)
        if self.gqa_func == 'max':
            attn_weights_mean = attn_weights_mean.max(dim=-2).values
        elif self.gqa_func == 'mean':
            attn_weights_mean = attn_weights_mean.mean(dim=-2)
        else:
            raise ValueError('gqa_func not supported')

        if self.pooling == 'avgpool':
            attn_weights_mean_pooling = F.avg_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        elif self.pooling == 'maxpool':
            attn_weights_mean_pooling = F.max_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        else:
            raise ValueError('Pooling method not supported')
        return attn_weights_mean_pooling
   
    def update_kv(self, origin_key_states, query_states, origin_value_states,debug_topk_positions=None):
        key_states = repeat_kv(origin_key_states, self.num_key_value_groups)
        # value_states = repeat_kv(origin_value_states, self.num_key_value_groups)
        _device = key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_score= self.calcul_attn_sore(key_states,query_states)

        origin_heads_key_states = torch.split(origin_key_states, 1, dim=1)
        origin_heads_value_states = torch.split(origin_value_states, 1, dim=1)

        def init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k):
            # init metadata
            self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=_device)
            self.klen_sum = klen_sum
            self.max_seqlen_k = max_seqlen_k
            self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
            # init varlen flash attention metadata
            self.cu_klen = self.cu_headlens - self.head_lens
            self.cu_klen = torch.cat(
                [self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=_device)], dim=0)
            # check bug
            self.layer_qlens = torch.ones(num_heads//self.num_key_value_groups, dtype=torch.int32,device=_device)
            self.qlen_sum = num_heads//self.num_key_value_groups
            self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
            self.cu_qlen = torch.cat(
                [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=_device)], dim=0)
            
            
            self.cu_offset = torch.arange(0, num_heads//self.num_key_value_groups + 1, dtype=torch.int32, device=_device)
            self.cu_head_offset = torch.arange(1, num_heads//self.num_key_value_groups +1, dtype=torch.int32, device=_device)

        if self.base_capacity > attn_score.size(-1):
            init_metadata(num_heads, [q_len] * (num_heads//self.num_key_value_groups), q_len * (num_heads//self.num_key_value_groups), q_len)
            # not compress
            return origin_key_states.reshape(-1, head_dim), origin_value_states.reshape(-1, head_dim)

        _,indices = attn_score.sort(dim=-1,descending=True)

        indices = indices.split(1,dim=1)

        heads_key_states = []
        heads_value_states = []
        assert bsz == 1
        # per head

        # reinit varlen metadata
        k_lens = []
        klen_sum = 0
        max_seqlen_k = 0
        self.cu_klen = 0

        for head_idx in range(num_heads//self.num_key_value_groups):

            if self.layer_idx!=32 :# 
                # self.head_adaptive_capacity[self.layer_idx][head_idx]=self.base_capacity
                cache_index = indices[head_idx][...,:self.head_adaptive_capacity[self.layer_idx][head_idx]]

                # cache_index = indices[head_idx][...,:]
            else:
              
                cache_index = indices[head_idx][...,:]
   
            l = cache_index.shape[-1] + self.window_size
            k_lens.append(l)
            max_seqlen_k = max(max_seqlen_k, l)
            klen_sum += l
            cache_index, original_indices = cache_index.sort(dim=2)
            # print(cache_index.shape)
            cache_index = cache_index.view(1, 1, -1, 1).expand(-1, -1, -1, head_dim)
            top_Kcache = origin_heads_key_states[head_idx].gather(dim=2,index=cache_index)
            top_Vcache = origin_heads_value_states[head_idx].gather(dim=2,index=cache_index)
            selected_k = torch.cat([top_Kcache,origin_heads_key_states[head_idx][:, :, -self.window_size:, :]],dim=2)
            selected_v = torch.cat([top_Vcache,origin_heads_value_states[head_idx][:, :, -self.window_size:, :]],dim=2)

            # NOTE: flatten view
            heads_key_states.append(selected_k.view(-1, head_dim))
            heads_value_states.append(selected_v.view(-1, head_dim))

        init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k)

        # NOTE: compose as flatten view
        heads_key_states = torch.cat(heads_key_states, dim=0)
        heads_value_states = torch.cat(heads_value_states, dim=0)

        return heads_key_states, heads_value_states

class ShiftKVCluster():
    def __init__(self, window_size = 32, kernel_size = 7, pooling = 'maxpool', base_capacity=None, ratio=None, normalize=None,
                 layer_idx = None, num_hidden_layers = None, head_score=None, num_attention_heads=32, num_key_value_groups=1, gqa_func='mean', model_type=None):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.base_capacity = base_capacity - window_size
        self.ratio = ratio
        # self.pp = None
        # self.ii = None
        self.normalize = normalize
        self.layer_idx = layer_idx
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers

        # NOTE: layer-wise meta-data
        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_headlens = None
        self.decode_full_head_count=0
        self.decode_full_head_count1=0
        self.num_key_value_groups = num_key_value_groups
        self.gqa_func = gqa_func

        if head_score == 'random':
            head_score_list = torch.tensor(
                [random.random() for _ in range(self.num_hidden_layers * self.num_attention_heads)],
                dtype=torch.float32,
            )
        elif head_score == 'visual':
            head_score_dict = load_head_score(model_type)
            head_score_list = torch.tensor(
                [np.mean(l[1]) for l in head_score_dict.items()],
                dtype=torch.float32,
            )
        else:
            raise ValueError('head_score should be either "random" or "visual"')
        head_score_list = head_score_list / head_score_list.sum()
        # GQA support
        self.score = head_score_list.view(self.num_hidden_layers, self.num_attention_heads//self.num_key_value_groups, self.num_key_value_groups)
        self.score = self.score.sum(dim=-1)

        min_cache = int(self.base_capacity * self.ratio)
        self.remain_capacity = (self.base_capacity - min_cache) * self.num_hidden_layers * self.num_attention_heads // self.num_key_value_groups
        self.head_adaptive_capacity = torch.round(self.score * self.remain_capacity + min_cache).int()
        self.head_adaptive_capacity = torch.full(
            (self.num_hidden_layers, self.num_attention_heads // self.num_key_value_groups),
            fill_value=self.base_capacity,
            dtype=torch.int32,
            device=self.score.device
        )
        # self.head_adaptive_capacity = self.base_capacity

        # ============================================
        # 集成统计表预测器（新方法）
        # ============================================
        self.statistical_predictor = None
        self.use_statistical_predictor = True

        # 从环境变量读取配置
       
        statistics_table_path ='./statistics_table/layer_statistics_table.json'
        enable_statistical_predictor = True

        if enable_statistical_predictor and STATISTICAL_PREDICTOR_AVAILABLE and statistics_table_path:
            self.statistical_predictor = StatisticalQueryPredictor(statistics_table_path)
            self.use_statistical_predictor =True
            # except Exception as e:
            #     self.statistical_predictor = None
            #     self.use_statistical_predictor = False
            #     print(self.use_statistical_predictor)
        else:
            if not enable_statistical_predictor:
                print(f"[SparseMM] Statistical predictor disabled (ENABLE_STATISTICAL_PREDICTOR=false)")
            elif not STATISTICAL_PREDICTOR_AVAILABLE:
                print(f"[SparseMM] Statistical predictor module not available")
            elif not statistics_table_path:
                print(f"[SparseMM] STATISTICS_TABLE_PATH not set")

    def calcul_attn_sore(self, key_states, query_states):
        bsz, num_heads, q_len, head_dim = query_states.shape
        # print(num_heads)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            head_dim)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights[:,:,:,:-1]
        attn_weights = attn_weights.view(attn_weights.shape[0],num_heads//self.num_key_value_groups,self.num_key_value_groups,q_len,-1)

        if self.gqa_func == 'max':# 每七个头聚合
            attn_weights_mean = attn_weights.max(dim=-3).values
        elif self.gqa_func == 'mean':
            attn_weights_mean = attn_weights.mean(dim=-3)
        else:
            raise ValueError('gqa_func not supported')
        
        return attn_weights_mean

    def diagonal_gaussian_sampling_vec(self, mean, std, num_samples):
        """
        mean: [bsz, dim]
        std : [bsz, dim]
        return: [bsz, num_samples, dim]
        """
        bsz, dim = mean.shape
        mean_expanded = mean.unsqueeze(1).expand(bsz, num_samples, dim)
        std_expanded  = std.unsqueeze(1).expand(bsz, num_samples, dim)

        eps = torch.randn_like(std_expanded)
        return mean_expanded + eps * std_expanded

   
    def diagonal_gaussian_sampling(self,mean, std, num_samples,dim): #[bsz]
        # 扩展维度以便广播    
        bsz = mean.shape[0]
        mean_expanded = mean.unsqueeze(-1).unsqueeze(-1).expand(bsz, num_samples, dim) 
        std_expanded = std.unsqueeze(-1).unsqueeze(-1).expand(bsz, num_samples, dim)
        
        # 对角高斯采样 - 各维度独立采样
        eps = torch.randn_like(std_expanded)
        samples = mean_expanded + eps * std_expanded
        return samples


    
    
    def update_kv(self, origin_key_states, query_states, origin_value_states, query_samples, groups_size, num_groups):

            key_states = repeat_kv(origin_key_states, self.num_key_value_groups)
            _device = key_states.device
            bsz, num_heads, q_len, head_dim = query_states.shape

            origin_heads_key_states = torch.split(origin_key_states, 1, dim=1)
            origin_heads_value_states = torch.split(origin_value_states, 1, dim=1)

            heads_key_states = []
            heads_value_states = []
            assert bsz == 1
            k_lens = []
            klen_sum = 0
            max_seqlen_k = 0
            self.cu_klen = 0

            # 初始化metadata
            def init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k):
                self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=_device)
                self.klen_sum = klen_sum
                self.max_seqlen_k = max_seqlen_k
                self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
                self.cu_klen = self.cu_headlens - self.head_lens
                self.cu_klen = torch.cat(
                    [self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=_device)], dim=0)
                self.layer_qlens = torch.ones(num_heads//self.num_key_value_groups, dtype=torch.int32, device=_device)
                self.qlen_sum = num_heads//self.num_key_value_groups
                self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
                self.cu_qlen = torch.cat(
                    [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=_device)], dim=0)
                self.cu_offset = torch.arange(0, num_heads//self.num_key_value_groups + 1, dtype=torch.int32, device=_device)
                self.cu_head_offset = torch.arange(1, num_heads//self.num_key_value_groups + 1, dtype=torch.int32, device=_device)


            # 非0或1 层的。
            attn_score = self.calcul_attn_sore(key_states,query_samples)  #[bsz,head,q_len,s_len]
            #print(attn_score)
            last_attn_score = self.calcul_attn_sore(key_states,query_states[...,-1:,:]) #[bsz,head,1,s_len]
            

            #优化代码-w
            score_bz, score_h, score_len,score_s = attn_score.shape
            attn_score = attn_score.view(score_bz,score_h,score_len//groups_size,groups_size,score_s).sum(dim=-2)
            attn_score = attn_score + last_attn_score
            attn_score = attn_score/(1.0+groups_size)
            sorted_scores, sorted_indices = attn_score.sort(dim=-1, descending=True)
            cum = sorted_scores.to(torch.float32).cumsum(dim=-1)
            key_len = attn_score.shape[-1]
            thr = 0.95 * cum[..., -1]                                            # [bsz,h,g]
            k = torch.searchsorted(cum, thr.unsqueeze(-1), right=True).squeeze(-1)  # [bsz,h,g]
            k = k.clamp(min=0, max=key_len)

        
            B = bsz;G = num_groups;H = attn_score.shape[1];L = key_len
            R = B * H * G
            selected_num = torch.zeros((bsz,H,key_len),device=query_states.device, dtype=torch.float32)
            idx_f = sorted_indices.reshape(R, L).contiguous().long()             # [R, L]
            k_f = k.reshape(R).contiguous()                                      # [R]
            if (k_f > 0).any():
                rangeL = torch.arange(L, device=idx_f.device).unsqueeze(0)       # [1, L]
                mask_first_k = rangeL < k_f.unsqueeze(1)                         # [R, L]
                sel = idx_f[mask_first_k]                                        # [sum_k]
                row_offsets = (torch.arange(R, device=idx_f.device) * L)
                off = torch.repeat_interleave(row_offsets, k_f)                   # [sum_k]
                keys = sel + off                                                 # [sum_k]
                counts = torch.bincount(keys, minlength=R * L).view(R, L)        # [R, L]
                counts = counts.view(B, H, G, L).sum(dim=2).to(selected_num.dtype)  # [bsz,H,L]
                selected_num += counts
            # 若所有 k 为 0，则不增加 selected_num
            #=============
            # 优化代码
            #=============

            selected_num = selected_num + last_attn_score.squeeze(-2)
            _, sorted_indices = selected_num.sort(dim=-1,descending=True)
            final_sorted_indices = torch.split(sorted_indices, 1, dim=1)


            # 对每个头进行操作（CUDA 怎么解决？，）
            for head_idx in range(num_heads // self.num_key_value_groups):

                head_key = origin_heads_key_states[head_idx]    # [bsz, 1, seq_len, head_dim]
                head_value = origin_heads_value_states[head_idx]

                # 1. 修改窗口处理：只保留最后一个token
                recent_key = head_key[:, :, -1:, :]  # 只保留最后一个token
                recent_value = head_value[:, :, -1:, :]

                # 2. 计算可用预算：原预算 + (window_size-1)
                original_budget = self.head_adaptive_capacity[self.layer_idx][head_idx].item()
                additional_budget = self.window_size - 1 
                key_len = head_key.shape[-2] - 1
                total_budget = min(max(original_budget + additional_budget, 0), key_len)
                # total_budget = min(max(64, 0), key_len)
                l = total_budget+1
                k_lens.append(l)
                max_seqlen_k = max(max_seqlen_k, l)
                klen_sum += l

                # 选择前total_budget个token 已经优化完成。
                selected_token_indices = final_sorted_indices[head_idx][...,:total_budget]
                
                # 5. 组装最终的key和value
                selected_token_indices = selected_token_indices.sort()[0]  # 恢复原始顺序      #[bsz,1,slen]
                selected_token_indices = selected_token_indices.view(1, 1, -1, 1).expand(-1, -1, -1, head_dim)

                selected_keys = head_key[:, :, :-1, :].gather(dim=2, index=selected_token_indices)
                selected_values = head_value[:, :, :-1, :].gather(dim=2, index=selected_token_indices)

                #6. 拼接选中的token和最后一个token
                final_keys = torch.cat([selected_keys, recent_key], dim=2)
                final_values = torch.cat([selected_values, recent_value], dim=2)
                heads_key_states.append(final_keys.view(-1, head_dim))
                heads_value_states.append(final_values.view(-1, head_dim))


            init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k)

            # 拼接所有头的结果
            heads_key_states = torch.cat(heads_key_states, dim=0)
            heads_value_states = torch.cat(heads_value_states, dim=0)

            return heads_key_states, heads_value_states
class KeyDiffCluster():
    def __init__(self, window_size = 15, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', layer_idx = None, num_hidden_layers = None,
                 pyram_mode = False, pyram_beta = 20,num_key_value_groups = 1, gqa_func='mean'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

        self.pyram_init = False
        self.pyram_mode = pyram_mode
        self.pyram_beta = pyram_beta
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        self.num_key_value_groups = num_key_value_groups
        self.gqa_func = gqa_func


    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, origin_key_states, query_states, origin_value_states):
    # origin_key_states: [B, num_kv_heads, L, D]
    # query_states:      [B, num_heads,   L, D]  (prefill)
    # origin_value_states same as origin_key_states

        # 只检查长度一致
        assert origin_key_states.shape[-2] == query_states.shape[-2]
        bsz, num_kv_heads, q_len, head_dim = origin_key_states.shape

        if q_len < self.max_capacity_prompt:
            return origin_key_states, origin_value_states

        # ✅ window/prefix 全部在 KV heads 上做
        k_window = origin_key_states[:, :, -self.window_size:, :]     # [B, kvH, W, D]
        v_window = origin_value_states[:, :, -self.window_size:, :]
        k_prefix = origin_key_states[:, :, :-self.window_size, :]     # [B, kvH, L-W, D]
        v_prefix = origin_value_states[:, :, :-self.window_size, :]

        anchor = F.normalize(k_prefix, p=2, dim=-1).mean(dim=-2, keepdim=True)
        scores = -F.cosine_similarity(k_prefix, anchor, dim=-1)       # [B, kvH, L-W]

        nums = min(self.max_capacity_prompt, k_prefix.size(-2))
        _, indices = torch.topk(scores, k=nums, dim=-1, largest=True)

        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        k_selected = k_prefix.gather(dim=-2, index=indices_expanded)  # [B, kvH, nums, D]
        v_selected = v_prefix.gather(dim=-2, index=indices_expanded)

        # ✅ 现在两边都是 kvH 头，cat 不会炸
        k_selected = torch.cat([k_selected, k_window], dim=-2)
        v_selected = torch.cat([v_selected, v_window], dim=-2)

        return k_selected, v_selected

def init_keydiff(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = int(os.getenv('BUDGET'))
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config,'num_hidden_layers'):
            raise ValueError('num_hidden_layers should be set')
        if not hasattr(self.config,'gqa_func'):
            if 'llama' in self.config.model_type or 'mistral' in self.config.model_type or \
                'llava' in self.config.model_type or 'qwen' in self.config.model_type:
                self.config.gqa_func = 'mean'

    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = KeyDiffCluster(
            window_size = self.config.window_size,
            max_capacity_prompt = self.config.max_capacity_prompt,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads,
            gqa_func=self.config.gqa_func
        )
class StreamingLLMCluster():
    def __init__(self, window_size = 15, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', layer_idx = None, num_hidden_layers = None,
                 pyram_mode = False, pyram_beta = 20,num_key_value_groups = 1, gqa_func='mean'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

        self.pyram_init = False
        self.pyram_mode = pyram_mode
        self.pyram_beta = pyram_beta
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        self.num_key_value_groups = num_key_value_groups
        self.gqa_func = gqa_func


    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, origin_key_states, query_states, origin_value_states):
        
        # support gqa
        key_states = repeat_kv(origin_key_states, self.num_key_value_groups)
        value_states = repeat_kv(origin_value_states, self.num_key_value_groups)
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        # compute pyramidal capacity
        if self.pyram_mode and not self.pyram_init:
            # NOTE: (max_num + min_num) / 2 == base_capacity to restrict the total capacity
            base_capacity = self.max_capacity_prompt - self.window_size
            min_num = base_capacity // self.pyram_beta
            max_num = base_capacity * 2 - min_num
                
            # if the max_num is larger than the query length, we need to adjust the max_num
            if max_num >= q_len - self.window_size:
                max_num = q_len - self.window_size
                min_num = base_capacity * 2 - max_num
        
            # NOTE: compute interval
            steps = (max_num - min_num) // (self.num_hidden_layers - 1)

            self.max_capacity_prompt = max_num - self.layer_idx * steps + self.window_size
            self.pyram_init = True
            print(f"Pyram mode adaptive capacity, layer: {self.layer_idx}, max_capacity_prompt: {self.max_capacity_prompt}, base_capacity: {self.max_capacity_prompt - self.window_size}", flush=True)

        if q_len < self.max_capacity_prompt:
            return origin_key_states, origin_value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim = -2)
            
            attn_weights_mean = attn_weights_mean.view(attn_weights_mean.shape[0], -1, self.num_key_value_groups, attn_weights_mean.shape[-1])
            if self.gqa_func == 'max':
                attn_weights_mean = attn_weights_mean.max(dim=-2).values
            elif self.gqa_func == 'mean':
                attn_weights_mean = attn_weights_mean.mean(dim=-2)
            else:
                raise ValueError('gqa_func not supported')
                
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_mean, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_mean, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            
            k_attention_sink = origin_key_states[:, :, :self.window_size, :]
            v_attention_sink = origin_value_states[:, :, :self.window_size, :]
            num_windows = self.max_capacity_prompt - self.window_size
            k_window = origin_key_states[:, :, -num_windows:, :]
            v_window = origin_value_states[:, :, -num_windows:, :]

            key_states = torch.cat([k_attention_sink, k_window], dim = 2)
            value_states = torch.cat([v_attention_sink, v_window], dim = 2)
            return key_states, value_states
        
def init_streamingllm(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 4
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = int(os.getenv('BUDGET'))
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config,'num_hidden_layers'):
            raise ValueError('num_hidden_layers should be set')
        if not hasattr(self.config,'gqa_func'):
            if 'llama' in self.config.model_type or 'mistral' in self.config.model_type or \
                'llava' in self.config.model_type or 'qwen' in self.config.model_type:
                self.config.gqa_func = 'mean'

    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = StreamingLLMCluster(
            window_size = self.config.window_size,
            max_capacity_prompt = self.config.max_capacity_prompt,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads,
            gqa_func=self.config.gqa_func
        )
def init_pyramidkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = int(os.getenv('BUDGET'))
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config,'num_hidden_layers'):
            raise ValueError('num_hidden_layers should be set')
        if not hasattr(self.config,'gqa_func'):
            if 'llama' in self.config.model_type or 'mistral' in self.config.model_type or \
                'llava' in self.config.model_type or 'qwen' in self.config.model_type:
                self.config.gqa_func = 'mean'

    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = SnapKVCluster(
            window_size = self.config.window_size,
            max_capacity_prompt = self.config.max_capacity_prompt,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            layer_idx = self.layer_idx,
            num_hidden_layers = self.config.num_hidden_layers,
            pyram_mode=True,
            pyram_beta=20,
            num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads,
            gqa_func=self.config.gqa_func
        )

def init_snapkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 1
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = int(os.getenv('BUDGET'))
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config,'num_hidden_layers'):
            raise ValueError('num_hidden_layers should be set')
        if not hasattr(self.config,'gqa_func'):
            if 'llama' in self.config.model_type or 'mistral' in self.config.model_type or \
                'llava' in self.config.model_type or 'qwen' in self.config.model_type:
                self.config.gqa_func = 'mean'

    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = SnapKVCluster(
            window_size = self.config.window_size,
            max_capacity_prompt = self.config.max_capacity_prompt,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads,
            gqa_func=self.config.gqa_func
        )

def init_adakv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = int(os.getenv('BUDGET'))
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'floor_ratio'):
            self.config.floor_ratio = 0.2
        if not hasattr(self.config, 'normalize'):
            self.config.normalize = True
        if not hasattr(self.config, 'num_hidden_layers'):
            raise ValueError('num_hidden_layers should be set')
        if not hasattr(self.config, 'skip'):
            self.config.skip = 0
        if not hasattr(self.config,'gqa_func'):
            if 'llama' in self.config.model_type or 'mistral' in self.config.model_type or \
                'llava' in self.config.model_type or 'qwen' in self.config.model_type:
                self.config.gqa_func = 'mean'

    # init only once
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = AdaKVCluster(
            window_size = self.config.window_size,
            base_capacity=self.config.max_capacity_prompt,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            floor_alpha= self.config.floor_ratio,
            skip = self.config.skip,
            layer_idx = self.layer_idx,
            normalize = self.config.normalize,
            num_hidden_layers = self.config.num_hidden_layers,
            num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads,
            gqa_func = self.config.gqa_func
        )

def init_sparsemm(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = int(os.getenv('BUDGET'))
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'head_score'):
            method = os.getenv('METHOD', None)
            if method == 'sparsemm':
                self.config.head_score = 'visual' 
            elif method == 'random':
                self.config.head_score = 'random'
            else:
                raise ValueError('head_score should be set')
        if not hasattr(self.config, 'ratio'):
            self.config.ratio = float(os.getenv('RATIO'))
        if not hasattr(self.config,'gqa_func'):
            if 'llama' in self.config.model_type or 'mistral' in self.config.model_type or \
                'llava' in self.config.model_type or 'qwen' in self.config.model_type:
                self.config.gqa_func = 'mean'

    # init only once 只初始化一次
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = SparseMM(
            window_size = self.config.window_size,
            base_capacity=self.config.max_capacity_prompt,
            head_score=self.config.head_score,
            ratio=self.config.ratio,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            layer_idx = self.layer_idx,
            num_hidden_layers = self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads,
            gqa_func = self.config.gqa_func,
            model_type=self.config.model_type
        )

def init_mask(self):
    if not hasattr(self, "head_list"):
        method = os.getenv('METHOD', None)

        head_score = load_head_score(self.config.model_type)
        head_list = [(l[0], np.mean(l[1])) for l in head_score.items()]
        head_list = sorted(head_list, key=lambda x: x[1], reverse=True) 

        if method == 'mask':
            ratio = float(os.getenv('MASK_RATIO'))
            num = int(ratio * len(head_list))
            print(f"mask ratio: {ratio}, num: {num}")
            head_list = [[int(ll) for ll in l[0].split("-")] for l in head_list][:num]
            self.head_list = head_list
        else:
            ratio = float(os.getenv('MASK_RATIO'))
            layer_num = 32 if 'llava' in self.config.model_type else 28
            head_num = 32 if 'llava' in self.config.model_type else 32
            num = int(ratio * layer_num * head_num)
            print(f"mask random ratio: {ratio}, num: {num}")
            head_list = [[int(ll) for ll in l[0].split("-")] for l in head_list][:num]
            self.head_list = []
            seed_list = [i  for i in range(layer_num)]
            random.shuffle(seed_list)
            while len(self.head_list) < num:
                l, h = random.choices(seed_list, k=2)
                if (l, h) in self.head_list or (h, l) in head_list:
                    continue
                else:
                    self.head_list.append((l, h))
def init_shiftkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = int(os.getenv('BUDGET'))
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'head_score'):
            method = os.getenv('METHOD', None)
            print(method)
            if method == 'shiftkv':
                self.config.head_score = 'visual'
            elif method == 'sparsemm_cpu':
                self.config.head_score = 'visual'
            elif method == 'random':
                self.config.head_score = 'random'
            else:
                raise ValueError('head_score should be set')
        if not hasattr(self.config, 'ratio'):
            self.config.ratio = float(os.getenv('RATIO'))
        if not hasattr(self.config,'gqa_func'):
            if 'llama' in self.config.model_type or 'mistral' in self.config.model_type or \
                'llava' in self.config.model_type or 'qwen' in self.config.model_type:
                self.config.gqa_func = 'mean'

    # init only once
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = ShiftKVCluster(
            window_size = self.config.window_size,
            base_capacity=self.config.max_capacity_prompt,
            head_score=self.config.head_score,
            ratio=self.config.ratio,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            layer_idx = self.layer_idx,
            num_hidden_layers = self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads,
            gqa_func = self.config.gqa_func,
            model_type=self.config.model_type
        )


class ExpectedAttentionCluster:
    

    def __init__(
        self,
        max_capacity_prompt: int,
        attn_module,
        n_future_positions: int = 512,
        n_sink: int = 4,
        stats_window: int = 128,
        use_covariance: bool = True,
        use_vnorm: bool = True,
        epsilon: float = 0.0,
        num_key_value_groups: int = 1,
    ):
        self.max_capacity_prompt = max_capacity_prompt
        self.attn_module = attn_module
        self.n_future_positions = n_future_positions
        self.n_sink = n_sink
        self.stats_window = stats_window
        self.use_covariance = use_covariance
        self.use_vnorm = use_vnorm
        self.epsilon = epsilon
        self.num_key_value_groups = num_key_value_groups

    # ----------- 内部工具：从 hidden_states 取 pre-RoPE query -----------

    def _get_prerope_query_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        从 attention 模块的 q_proj 得到 RoPE 之前的 query。

        hidden_states: [bsz, q_len, hidden_dim]
        返回: [bsz, num_heads, q_len, head_dim]
        """
        module = self.attn_module
        bsz, q_len, _ = hidden_states.shape
        num_heads = module.num_heads
        head_dim = module.head_dim

        # LLaMA 风格：直接用 q_proj
        query = module.q_proj(hidden_states)  # [bsz, q_len, num_heads * head_dim]
        query = query.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        return query  # [bsz, num_heads, q_len, head_dim]

    def _apply_avg_rope(self, mu: torch.Tensor, cov: torch.Tensor, q_len: int):
        """
        完整版 ExpectedAttention 里对均值/协方差做 RoPE 平均旋转。
        mu: [bsz, num_heads, head_dim]
        cov: [bsz, num_heads, head_dim, head_dim] 或 None
        """
        module = self.attn_module
        device = mu.device
        head_dim = module.head_dim

        # 在未来的 n_future_positions 上生成 RoPE 旋转矩阵
        position_ids = torch.arange(q_len, q_len + self.n_future_positions, device=device).unsqueeze(0)
        # 这里仿照 expected_attention_press 直接喂 mu 进去，只是为了拿 cos/sin
        cos, sin = module.rotary_emb(mu, position_ids)  # 形状与实现有关，只用最后两个维度
        cos, sin = cos[0], sin[0]  # [n_future_positions, head_dim]

        Id = torch.eye(head_dim, device=device, dtype=cos.dtype)
        P = torch.zeros((head_dim, head_dim), device=device, dtype=cos.dtype)
        half = head_dim // 2
        P[half:, :half] = torch.eye(half, device=device, dtype=cos.dtype)
        P[:half, half:] = -torch.eye(half, device=device, dtype=cos.dtype)

        # R(t) = cos_t * I + sin_t * P
        R = cos.unsqueeze(1) * Id + sin.unsqueeze(1) * P  # [n_future_positions, head_dim, head_dim]
        R = R.mean(dim=0).to(device)                      # [head_dim, head_dim]

        mu = torch.matmul(mu, R.T)                        # [bsz, num_heads, head_dim]
        if cov is not None:
            # cov: [bsz, num_heads, d, d]
            cov = torch.matmul(R, torch.matmul(cov, R.T))  # [bsz, num_heads, d, d]

        return mu, cov

    def _get_query_statistics(self, q_pre: torch.Tensor, q_len: int):
        """
        按 expected_attention_press 的公式，从 pre-RoPE query 估计均值/协方差，
        再做 RoPE 平均旋转。
        q_pre: [bsz, num_heads, q_len, head_dim]
        当前实现：在 sink 之后使用全局历史（不再使用 stats_window 截断）
        """
        bsz, num_heads, _, head_dim = q_pre.shape
        if q_len <= self.n_sink:
            return None, None

        # 在 sink 之后使用全局历史 token 统计均值与协方差
        h = q_pre[:, :, self.n_sink :, :]  # [bsz, num_heads, Lw, head_dim]

        # 均值
        mu = h.mean(dim=2)          # [bsz, num_heads, head_dim]

        # 协方差
        cov = None
        if self.use_covariance:
            centered = h - mu.unsqueeze(2)  # [bsz, num_heads, Lw, d]
            # bnsi,bnsj->bnij  (b=batch, n=head, s=seq, i/j=dim)
            cov = torch.einsum("bnsi,bnsj->bnij", centered, centered) / h.shape[2]  # [bsz, num_heads, d, d]

        # 平均 RoPE 到未来位置
        mu, cov = self._apply_avg_rope(mu, cov, q_len)
        return mu, cov

    # ----------- 核心接口：压缩 KV -----------

    def update_kv(self, origin_key_states, query_states, origin_value_states):
        """
        origin_key_states: [bsz, num_kv_heads, q_len, head_dim]  (RoPE 之后)
        query_states     : [bsz, num_heads,   q_len, head_dim]  (RoPE 之后) 这里不用
        origin_value_states 同 key_states
        """
        bsz, num_kv_heads, q_len, head_dim = origin_key_states.shape
        ratio_set = os.getenv('RATIO_SET')
        if ratio_set ==1:
            ratio = os.getenv('RATIO')
            self.max_capacity_prompt = max(q_len * ratio,1)
            self.window_size = 1
        if q_len <= self.max_capacity_prompt or q_len <= self.n_sink:
            return origin_key_states, origin_value_states

        # 从 attention 模块上取出刚刚存的 hidden_states（RoPE 之前）
        hidden_states = getattr(self.attn_module, "_expected_attention_hidden_states", None)
        if hidden_states is None:
            # 接口没被正确设置，直接退回不压缩
            return origin_key_states, origin_value_states

        # 1) 得到 pre-RoPE 的 query
        q_pre = self._get_prerope_query_states(hidden_states)  # [bsz, num_heads, q_len, head_dim]

        # 2) 估计未来 query 的均值/协方差
        mean_query, cov_query = self._get_query_statistics(q_pre, q_len)
        if mean_query is None:
            return origin_key_states, origin_value_states
        # mean_query: [bsz, num_heads, head_dim]
        # cov_query : [bsz, num_heads, head_dim, head_dim] 或 None

        # 3) 去掉 sink token，按 expected attention 打分
        keys_body = origin_key_states[:, :, self.n_sink :, :]   # [bsz, num_kv_heads, L, d]
        values_body = origin_value_states[:, :, self.n_sink :, :]
        bsz, num_kv_heads, L, d = keys_body.shape

        num_heads_total = self.attn_module.config.num_attention_heads
        num_groups = num_heads_total // num_kv_heads
        # 展开成 full heads
        keys_full = repeat_kv(keys_body, num_groups).transpose(2, 3)  # [bsz, num_heads_total, d, L]

        # 一阶项: k·mu / sqrt(d)
        scores = torch.matmul(mean_query.unsqueeze(2), keys_full).squeeze(2) / math.sqrt(d)  # [bsz, num_heads_total, L]

        # 二阶项: 0.5 * k^T Σ k / d
        if self.use_covariance and cov_query is not None:
            # keys_full: [b,h,i,n], cov_query: [b,h,i,j]
            scores = scores + torch.einsum(
                "bhin,bhij,bhjn->bhn", keys_full, cov_query, keys_full
            ) / d / 2.0

        scores = F.softmax(scores, dim=-1)  # [bsz, num_heads_total, L]

        # 4) 在 GQA 分组上平均，回到 num_kv_heads
        scores = scores.view(bsz, num_kv_heads, num_groups, L).mean(dim=2)  # [bsz, num_kv_heads, L]

        # 5) 可选：乘上对应 value 的 L2 范数
        if self.use_vnorm:
            scores = (scores + self.epsilon) * values_body.norm(dim=-1)

        # 6) 把 sink token 加回去，用最大分数保护
        max_score = scores.max().item()
        scores = F.pad(scores, (self.n_sink, 0), value=max_score)  # [bsz, num_kv_heads, q_len]
        scores[...,-1] = max_score 
        # 7) 按全局 budget 做 top‑k 选位置
        n_kept = min(self.max_capacity_prompt, q_len)
        topk_idx = scores.topk(n_kept, dim=-1).indices                 # [bsz, num_kv_heads, n_kept]
        topk_idx = topk_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # [bsz, num_kv_heads, n_kept, d]

        new_keys = origin_key_states.gather(2, topk_idx).contiguous()
        new_values = origin_value_states.gather(2, topk_idx).contiguous()
        return new_keys, new_values
def init_expected_attention(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, "max_capacity_prompt"):
            self.config.max_capacity_prompt = int(os.getenv("BUDGET"))
        if not hasattr(self.config, "n_sink"):
            self.config.n_sink = 4
        if not hasattr(self.config, "n_future_positions"):
            self.config.n_future_positions = 512
        if not hasattr(self.config, "stats_window"):
            self.config.stats_window = 128
        if not hasattr(self.config, "use_covariance"):
            self.config.use_covariance = True
        if not hasattr(self.config, "use_vnorm"):
            self.config.use_vnorm = True
        if not hasattr(self.config, "epsilon"):
            self.config.epsilon = 0.0
        if not hasattr(self.config, "num_hidden_layers"):
            raise ValueError("num_hidden_layers should be set")
        if not hasattr(self.config, "gqa_func"):
            if "llama" in self.config.model_type or "mistral" in self.config.model_type or \
               "llava" in self.config.model_type or "qwen" in self.config.model_type:
                self.config.gqa_func = "mean"

    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = ExpectedAttentionCluster(
            max_capacity_prompt=self.config.max_capacity_prompt,
            attn_module=self,
            n_future_positions=self.config.n_future_positions,
            n_sink=self.config.n_sink,
            stats_window=self.config.stats_window,
            use_covariance=self.config.use_covariance,
            use_vnorm=self.config.use_vnorm,
            epsilon=self.config.epsilon,
            num_key_value_groups=self.config.num_attention_heads // self.config.num_key_value_heads,
        )