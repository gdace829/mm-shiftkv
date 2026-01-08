"""
基于统计表的 Decode Query 预测模块

使用方法：
1. 先运行 build_statistics_table.py 生成统计表
2. 使用本模块加载统计表并进行预测
"""

import torch
import json
import os
from typing import Tuple, Optional


class StatisticalQueryPredictor:
    """
    基于统计表的 Query 预测器

    核心思路：
    - 使用预先统计的层级偏差表
    - 均值使用加减法：decode_mean = prefill_mean + mean_shift
    - 方差使用乘法：decode_std = prefill_std * std_ratio
    """

    def __init__(self, statistics_table_path: str):
        """
        Args:
            statistics_table_path: 统计表 JSON 文件路径
        """
        self.statistics_table_path = statistics_table_path
        self.stats_table = self._load_statistics_table()

        # print(f"[StatisticalQueryPredictor] Loaded statistics table from: {statistics_table_path}")
        # print(f"[StatisticalQueryPredictor] Available layers: {sorted(self.stats_table.keys())}")

    def _load_statistics_table(self):
        """加载统计表"""
        with open(self.statistics_table_path, 'r') as f:
            data = json.load(f)

        # 转换 key 为 int
        stats_table = {}
        for key, value in data.items():
            layer_idx = int(key)
            stats_table[layer_idx] = value

        return stats_table

    def compute_prefill_statistics(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 prefill 阶段的全局统计量

        Args:
            hidden_states: [bsz, seq_len, hidden_size]

        Returns:
            mean: [bsz, 1] 全局均值
            std: [bsz, 1] 全局标准差
        """
        bsz, seq_len, hidden_size = hidden_states.shape

        # Flatten 所有维度并计算全局统计
        flat_values = hidden_states.reshape(bsz, -1)  # [bsz, seq_len * hidden_size]

        mean = flat_values.mean(dim=-1, keepdim=True)  # [bsz, 1]
        std = flat_values.std(dim=-1, keepdim=True)    # [bsz, 1]

        return mean, std

    def predict_decode_statistics(
        self,
        layer_idx: int,
        prefill_mean: torch.Tensor,
        prefill_std: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于统计表预测 decode 阶段的分布参数

        Args:
            layer_idx: 层索引
            prefill_mean: [bsz, 1] prefill 均值
            prefill_std: [bsz, 1] prefill 标准差

        Returns:
            decode_mean: [bsz, 1] 预测的 decode 均值
            decode_std: [bsz, 1] 预测的 decode 标准差
        """
        if layer_idx not in self.stats_table:
            raise ValueError(f"Layer {layer_idx} not found in statistics table. "
                           f"Available layers: {sorted(self.stats_table.keys())}")

        # 获取该层的统计偏差
        layer_stats = self.stats_table[layer_idx]
        # mean_shift = layer_stats['mean_shift']
        # std_ratio = layer_stats['std_ratio']

        # 预测 decode 统计量
        # 均值：加减法
        # decode_mean = prefill_mean + mean_shift
        decode_mean = prefill_mean


        # 方差：乘法
        # decode_std = prefill_std * std_ratio
        decode_std = prefill_std * 10
        return decode_mean, decode_std

    def generate_query_samples(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        num_samples: int = 512
    ) -> torch.Tensor:
        """
        生成 decode 阶段的 query 样本（在 hidden_states 空间）

        Args:
            layer_idx: 层索引
            hidden_states: [bsz, seq_len, hidden_size] prefill 阶段的 hidden_states
            num_samples: 生成的样本数量

        Returns:
            hidden_samples: [bsz, num_samples, hidden_size] 生成的 hidden_states 样本
        """
        bsz, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # 计算 prefill 统计
        prefill_mean, prefill_std = self.compute_prefill_statistics(hidden_states)

        # 预测 decode 统计
        decode_mean, decode_std = self.predict_decode_statistics(
            layer_idx, prefill_mean, prefill_std
        )

        # 生成样本：从预测的高斯分布中采样
        # [bsz, num_samples, hidden_size]
        eps = torch.randn(bsz, num_samples, hidden_size, device=device, dtype=dtype)

        # decode_mean: [bsz, 1] -> [bsz, num_samples, hidden_size]
        # decode_std: [bsz, 1] -> [bsz, num_samples, hidden_size]
        mean_expanded = decode_mean.unsqueeze(-1).expand(bsz, num_samples, hidden_size)
        std_expanded = decode_std.unsqueeze(-1).expand(bsz, num_samples, hidden_size)

        hidden_samples = mean_expanded + eps * std_expanded

        return hidden_samples

    def get_layer_statistics(self, layer_idx: int) -> dict:
        """
        获取指定层的统计信息

        Args:
            layer_idx: 层索引

        Returns:
            统计信息字典
        """
        if layer_idx not in self.stats_table:
            raise ValueError(f"Layer {layer_idx} not found in statistics table")

        return self.stats_table[layer_idx]

    def print_statistics_summary(self):
        """打印统计表摘要"""
        print("\n" + "="*80)
        print("STATISTICS TABLE SUMMARY")
        print("="*80)
        print(f"{'Layer':<6} | {'Mean Shift':<12} | {'Std Ratio':<12} | {'Samples':<8}")
        print("-"*50)

        for layer_idx in sorted(self.stats_table.keys()):
            stats = self.stats_table[layer_idx]
            print(f"{layer_idx:<6} | "
                  f"{stats['mean_shift']:>12.6f} | "
                  f"{stats['std_ratio']:>12.6f} | "
                  f"{stats['num_samples']:<8}")

        print("="*80 + "\n")


def test_statistical_predictor():
    """测试统计预测器"""
    print("Testing Statistical Query Predictor")
    print("="*80)

    # 假设统计表路径
    stats_table_path = "./statistics_table/layer_statistics_table.json"

    if not os.path.exists(stats_table_path):
        print(f"Error: Statistics table not found at {stats_table_path}")
        print("Please run build_statistics_table.py first!")
        return

    # 创建预测器
    predictor = StatisticalQueryPredictor(stats_table_path)

    # 打印统计表摘要
    predictor.print_statistics_summary()

    # 测试预测
    print("Testing prediction...")

    # 创建假的 hidden_states
    bsz, seq_len, hidden_size = 1, 100, 3584
    hidden_states = torch.randn(bsz, seq_len, hidden_size) * 5.0

    # 测试几个层
    test_layers = [0, 7, 14, 21, 27]

    for layer_idx in test_layers:
        if layer_idx not in predictor.stats_table:
            continue

        print(f"\nLayer {layer_idx}:")

        # 计算 prefill 统计
        prefill_mean, prefill_std = predictor.compute_prefill_statistics(hidden_states)
        print(f"  Prefill - Mean: {prefill_mean.item():.6f}, Std: {prefill_std.item():.6f}")

        # 预测 decode 统计
        decode_mean, decode_std = predictor.predict_decode_statistics(
            layer_idx, prefill_mean, prefill_std
        )
        print(f"  Predicted Decode - Mean: {decode_mean.item():.6f}, Std: {decode_std.item():.6f}")

        # 生成样本
        samples = predictor.generate_query_samples(layer_idx, hidden_states, num_samples=512)
        print(f"  Generated samples shape: {samples.shape}")
        print(f"  Samples - Mean: {samples.mean().item():.6f}, Std: {samples.std().item():.6f}")

    print("\n" + "="*80)
    print("Test completed!")


if __name__ == "__main__":
    test_statistical_predictor()
