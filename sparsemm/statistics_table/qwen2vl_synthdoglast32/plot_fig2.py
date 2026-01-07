import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# -----------------------------
# Load statistics
# -----------------------------
def load_stats(json_path: Path):
    with open(json_path, "r") as f:
        data = json.load(f)

    layers = sorted(int(k) for k in data.keys())
    mean_shift = np.array([data[str(l)]["mean_shift"] for l in layers])
    std_ratio = np.array([data[str(l)]["std_ratio"] for l in layers])

    return np.array(layers), mean_shift, std_ratio


# -----------------------------
# Plot
# -----------------------------
def plot_fig2_single(layers, mean_shift, std_ratio, out_path: Path):
    fig, ax1 = plt.subplots(figsize=(6.5, 3.6), dpi=200)

    # Left y-axis: mean shift
    color1 = "tab:blue"
    ax1.plot(layers, mean_shift, color=color1, linewidth=2, label=r"Mean shift $\Delta\mu_l$")
    ax1.axhline(0.0, linestyle="--", linewidth=1, color=color1, alpha=0.6)
    ax1.set_xlabel("Layer index")
    ax1.set_ylabel(r"Mean shift $\Delta\mu_l$", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    # Right y-axis: std ratio
    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.plot(layers, std_ratio, color=color2, linewidth=2, label=r"Std ratio $\rho_l$")
    ax2.axhline(1.0, linestyle="--", linewidth=1, color=color2, alpha=0.6)
    ax2.set_ylabel(r"Std ratio $\rho_l$", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main (no arguments)
# -----------------------------
if __name__ == "__main__":
    # 当前脚本所在目录
    cwd = Path(__file__).resolve().parent

    json_path = cwd / "layer_statistics_table.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Cannot find layer_statistics_table.json in {cwd}")

    # 输出文件名 = 目录名
    exp_name = cwd.name
    out_path = cwd / f"fig2_{exp_name}.pdf"

    layers, mean_shift, std_ratio = load_stats(json_path)
    plot_fig2_single(layers, mean_shift, std_ratio, out_path)

    print(f"[OK] Saved Figure 2 to {out_path}")
# python build_stats_qwen25vl_multi.py   --dataset ocrbench   --ocrbench_root /data/model/datasets--echo840--OCRBench   --ocrbench_split test   --model_path Qwen/Qwen2.5-VL-7B-Instruct   --num_samples 500   --max_decode_tokens 64   --output_dir ./statistics_table/qwen25vl_ocrbench_test   --use_amp