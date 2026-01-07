"""
统计表生成脚本

目的：
使用 SynthDog 数据集统计 Qwen2-VL 模型在 prefill 和 decode 阶段的 hidden_states 分布偏差
生成每层的统计偏差表，用于预测 decode 阶段的分布

统计方法：
- Prefill 阶段：统计所有 hidden_states 数值的均值和标准差
- Decode 阶段：统计所有 hidden_states 数值的均值和标准差
- 偏差计算：
  - 均值偏差 = decode_mean - prefill_mean（加减法）
  - 方差比率 = decode_std / prefill_std（乘法）
"""

import torch
import json
import os
import argparse
from tqdm import tqdm
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer


class StatisticsTableBuilder:
    """统计表构建器"""

    def __init__(self, model, processor, tokenizer, num_layers=28):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.num_layers = num_layers

        # 存储每层的统计数据
        self.layer_stats = {
            i: {
                'prefill_means': [],  # 存储每个样本的 prefill 均值
                'prefill_stds': [],   # 存储每个样本的 prefill 标准差
                'decode_means': [],   # 存储每个样本的 decode 均值
                'decode_stds': []     # 存储每个样本的 decode 标准差
            } for i in range(num_layers)
        }

    def compute_global_statistics(self, hidden_states):
        """
        计算 hidden_states 的全局统计量

        Args:
            hidden_states: [bsz, seq_len, hidden_size]

        Returns:
            mean: 标量，所有数值的均值
            std: 标量，所有数值的标准差
        """
        # Flatten 所有维度
        flat_values = hidden_states.reshape(-1)  # [bsz * seq_len * hidden_size]

        # 计算全局统计
        mean = flat_values.mean().item()
        std = flat_values.std().item()

        return mean, std

    def collect_sample_statistics(self, image_path, text, max_decode_tokens=64):
        """
        收集单个样本的统计数据

        Args:
            image_path: 图像路径
            text: 文本内容
            max_decode_tokens: decode 最大 token 数
        """
        try:
            # ============================================
            # Prefill 阶段
            # ============================================
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": text}
                    ]
                }
            ]

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
            ).to('cuda')

            # Forward - Prefill
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )

            hidden_states_tuple = outputs.hidden_states  # (num_layers+1,) 每个 [1, seq_len, hidden_size]

            # 为每层计算 prefill 统计
            prefill_stats_per_layer = {}
            for layer_idx in range(self.num_layers):
                # 跳过 embedding 层，从第1层开始（索引1对应layer 0）
                hidden_states = hidden_states_tuple[layer_idx + 1]  # [1, seq_len, hidden_size]

                # 计算全局统计
                mean, std = self.compute_global_statistics(hidden_states)
                prefill_stats_per_layer[layer_idx] = (mean, std)

            # ============================================
            # Decode 阶段
            # ============================================
            decode_stats_per_layer = {i: {'means': [], 'stds': []} for i in range(self.num_layers)}

            # 生成配置
            generated_ids = inputs.input_ids.clone()
            past_key_values = None

            for step in range(max_decode_tokens):
                with torch.no_grad():
                    if step == 0:
                        # 第一步使用完整输入
                        outputs = self.model(
                            input_ids=generated_ids,
                            attention_mask=inputs.attention_mask,
                            past_key_values=past_key_values,
                            output_hidden_states=True,
                            return_dict=True,
                            use_cache=True
                        )
                    else:
                        # 后续步骤使用最后生成的 token
                        outputs = self.model(
                            input_ids=generated_ids[:, -1:],
                            attention_mask=torch.cat([
                                inputs.attention_mask,
                                torch.ones(1, step, device='cuda', dtype=inputs.attention_mask.dtype)
                            ], dim=1),
                            past_key_values=past_key_values,
                            output_hidden_states=True,
                            return_dict=True,
                            use_cache=True
                        )

                past_key_values = outputs.past_key_values

                # 提取每层的 hidden_states
                hidden_states_tuple = outputs.hidden_states

                for layer_idx in range(self.num_layers):
                    hidden_states = hidden_states_tuple[layer_idx + 1]  # [1, 1, hidden_size] or [1, seq_len, hidden_size]

                    # 只取最后一个 token
                    last_hidden = hidden_states[:, -1:, :]  # [1, 1, hidden_size]

                    # 计算全局统计
                    mean, std = self.compute_global_statistics(last_hidden)
                    decode_stats_per_layer[layer_idx]['means'].append(mean)
                    decode_stats_per_layer[layer_idx]['stds'].append(std)

                # 采样下一个 token
                logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                # 检查是否生成了 EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

            # ============================================
            # 保存统计数据
            # ============================================
            for layer_idx in range(self.num_layers):
                if layer_idx not in prefill_stats_per_layer:
                    continue

                if len(decode_stats_per_layer[layer_idx]['means']) == 0:
                    continue

                # Prefill 统计
                prefill_mean, prefill_std = prefill_stats_per_layer[layer_idx]

                # Decode 统计（平均所有生成步骤）
                avg_decode_mean = sum(decode_stats_per_layer[layer_idx]['means']) / len(decode_stats_per_layer[layer_idx]['means'])
                avg_decode_std = sum(decode_stats_per_layer[layer_idx]['stds']) / len(decode_stats_per_layer[layer_idx]['stds'])

                # 保存到统计表
                self.layer_stats[layer_idx]['prefill_means'].append(prefill_mean)
                self.layer_stats[layer_idx]['prefill_stds'].append(prefill_std)
                self.layer_stats[layer_idx]['decode_means'].append(avg_decode_mean)
                self.layer_stats[layer_idx]['decode_stds'].append(avg_decode_std)

            return True

        except Exception as e:
            print(f"Error processing sample: {e}")
            return False

    def compute_final_statistics(self):
        """
        计算最终的统计偏差表

        Returns:
            statistics_table: {
                layer_idx: {
                    'mean_shift': decode_mean - prefill_mean,
                    'std_ratio': decode_std / prefill_std,
                    'prefill_mean_avg': prefill 均值的平均,
                    'prefill_std_avg': prefill 标准差的平均,
                    'decode_mean_avg': decode 均值的平均,
                    'decode_std_avg': decode 标准差的平均,
                    'num_samples': 样本数量
                }
            }
        """
        statistics_table = {}

        for layer_idx in range(self.num_layers):
            stats = self.layer_stats[layer_idx]

            if len(stats['prefill_means']) == 0:
                continue

            # 计算平均值
            prefill_mean_avg = sum(stats['prefill_means']) / len(stats['prefill_means'])
            prefill_std_avg = sum(stats['prefill_stds']) / len(stats['prefill_stds'])
            decode_mean_avg = sum(stats['decode_means']) / len(stats['decode_means'])
            decode_std_avg = sum(stats['decode_stds']) / len(stats['decode_stds'])

            # 计算偏差
            mean_shift = decode_mean_avg - prefill_mean_avg
            std_ratio = decode_std_avg / (prefill_std_avg + 1e-8)  # 避免除零

            statistics_table[layer_idx] = {
                'mean_shift': mean_shift,
                'std_ratio': std_ratio,
                'prefill_mean_avg': prefill_mean_avg,
                'prefill_std_avg': prefill_std_avg,
                'decode_mean_avg': decode_mean_avg,
                'decode_std_avg': decode_std_avg,
                'num_samples': len(stats['prefill_means'])
            }

        return statistics_table


def load_synthdog_dataset(json_path, num_samples=None):
    """
    加载 SynthDog 数据集

    Args:
        json_path: SynthDog JSON 文件路径
        num_samples: 使用的样本数量（None = 全部）

    Returns:
        List of (image_path, text) tuples
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    image_text_pairs = []

    # 数据是一个列表，直接遍历
    items = data[:num_samples] if num_samples else data

    for item in items:
        # 获取图像路径
        image_name = item['image_name']

        # 如果是绝对路径，直接使用；否则拼接
        if os.path.isabs(image_name):
            image_path = image_name
        else:
            image_path = os.path.join(os.path.dirname(json_path), image_name)

        # 提取文本（从 2_coord 块）
        if '2_coord' in item and len(item['2_coord']) > 0:
            text_blocks = [coord['chunk'] for coord in item['2_coord'] if 'chunk' in coord]
            if text_blocks:
                text = ' '.join(text_blocks[:5])  # 取前5个文本块
                image_text_pairs.append((image_path, text))

    return image_text_pairs


def main():
    parser = argparse.ArgumentParser(description='Build statistics table for Qwen2-VL')
    parser.add_argument('--model_path', type=str, default='/data/model/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac',
                        help='Path to Qwen2-VL model')
    parser.add_argument('--synthdog_json', type=str,
                        default='/data/dataset/synthdog-en/synthdog-en.json',
                        help='Path to SynthDog JSON file')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of samples to use for statistics')
    parser.add_argument('--max_decode_tokens', type=int, default=64,
                        help='Maximum decode tokens per sample')
    parser.add_argument('--output_dir', type=str, default='./statistics_table/llama',
                        help='Output directory for statistics table')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("Building Statistics Table for Qwen2-VL")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.synthdog_json}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Max decode tokens: {args.max_decode_tokens}")
    print()

    # 加载模型
    print("Loading model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print("Model loaded successfully!")
    print()

    # 加载数据集
    print("Loading dataset...")
    image_text_pairs = load_synthdog_dataset(args.synthdog_json, args.num_samples)
    print(f"Loaded {len(image_text_pairs)} samples")
    print()

    # 创建统计表构建器
    builder = StatisticsTableBuilder(model, processor, tokenizer, num_layers=28)

    # 收集统计数据
    print("Collecting statistics...")
    successful_samples = 0

    for idx, (image_path, text) in enumerate(tqdm(image_text_pairs, desc="Processing samples")):
        success = builder.collect_sample_statistics(
            image_path, text,
            max_decode_tokens=args.max_decode_tokens
        )
        if success:
            successful_samples += 1

        # 每处理10个样本输出一次进度
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(image_text_pairs)} samples, "
                  f"successful: {successful_samples}")

    print()
    print(f"Total successful samples: {successful_samples}/{len(image_text_pairs)}")
    print()

    # 计算最终统计表
    print("Computing final statistics table...")
    statistics_table = builder.compute_final_statistics()

    # 保存统计表
    output_file = os.path.join(args.output_dir, 'layer_statistics_table.json')
    with open(output_file, 'w') as f:
        json.dump(statistics_table, f, indent=2)

    print(f"Statistics table saved to: {output_file}")
    print()

    # 打印统计表摘要
    print("="*80)
    print("STATISTICS TABLE SUMMARY")
    print("="*80)
    print(f"{'Layer':<6} | {'Mean Shift':<12} | {'Std Ratio':<12} | "
          f"{'Prefill Mean':<12} | {'Prefill Std':<12} | "
          f"{'Decode Mean':<12} | {'Decode Std':<12} | {'Samples':<8}")
    print("-"*120)

    for layer_idx in sorted(statistics_table.keys()):
        stats = statistics_table[layer_idx]
        print(f"{layer_idx:<6} | "
              f"{stats['mean_shift']:>12.6f} | "
              f"{stats['std_ratio']:>12.6f} | "
              f"{stats['prefill_mean_avg']:>12.6f} | "
              f"{stats['prefill_std_avg']:>12.6f} | "
              f"{stats['decode_mean_avg']:>12.6f} | "
              f"{stats['decode_std_avg']:>12.6f} | "
              f"{stats['num_samples']:<8}")

    print("="*80)
    print()

    # 保存人类可读的报告
    report_file = os.path.join(args.output_dir, 'statistics_report.txt')
    with open(report_file, 'w') as f:
        f.write("QWEN2-VL LAYER STATISTICS TABLE\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated from {successful_samples} samples\n")
        f.write(f"Max decode tokens: {args.max_decode_tokens}\n\n")
        f.write("USAGE:\n")
        f.write("------\n")
        f.write("Use this table to predict decode-phase hidden_states distribution:\n")
        f.write("  decode_mean = prefill_mean + mean_shift\n")
        f.write("  decode_std = prefill_std * std_ratio\n\n")
        f.write("STATISTICS TABLE:\n")
        f.write("-"*120 + "\n")
        f.write(f"{'Layer':<6} | {'Mean Shift':<12} | {'Std Ratio':<12} | "
                f"{'Prefill Mean':<12} | {'Prefill Std':<12} | "
                f"{'Decode Mean':<12} | {'Decode Std':<12} | {'Samples':<8}\n")
        f.write("-"*120 + "\n")

        for layer_idx in sorted(statistics_table.keys()):
            stats = statistics_table[layer_idx]
            f.write(f"{layer_idx:<6} | "
                   f"{stats['mean_shift']:>12.6f} | "
                   f"{stats['std_ratio']:>12.6f} | "
                   f"{stats['prefill_mean_avg']:>12.6f} | "
                   f"{stats['prefill_std_avg']:>12.6f} | "
                   f"{stats['decode_mean_avg']:>12.6f} | "
                   f"{stats['decode_std_avg']:>12.6f} | "
                   f"{stats['num_samples']:<8}\n")

        f.write("="*80 + "\n")

    print(f"Report saved to: {report_file}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
