import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_stats(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    layers = sorted(int(k) for k in data.keys())
    mean_shift = np.array([data[str(l)]["mean_shift"] for l in layers])
    std_ratio = np.array([data[str(l)]["std_ratio"] for l in layers])

    return np.array(layers), mean_shift, std_ratio


def plot_fig2_single(layers, mean_shift, std_ratio, out_path):
    fig, ax1 = plt.subplots(figsize=(6.5, 3.6), dpi=200)

    # 左 y 轴：mean shift
    color1 = "tab:blue"
    ax1.plot(layers, mean_shift, color=color1, linewidth=2, label=r"Mean shift $\Delta\mu_l$")
    ax1.axhline(0.0, linestyle="--", linewidth=1, color=color1, alpha=0.6)
    ax1.set_xlabel("Layer index")
    ax1.set_ylabel(r"Mean shift $\Delta\mu_l$", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    # 右 y 轴：std ratio
    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.plot(layers, std_ratio, color=color2, linewidth=2, label=r"Std ratio $\rho_l$")
    ax2.axhline(1.0, linestyle="--", linewidth=1, color=color2, alpha=0.6)
    ax2.set_ylabel(r"Std ratio $\rho_l$", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # 合并 legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path",default='./SparseMM/sparsemm/statistics_table/llava_next_v16/layer_statistics_table.json', type=str, required=True,
                        help="Path to layer_statistics_table.json")
    parser.add_argument("--out",default='./figs/fig2_llava_infographic.pdf', type=str,
                        help="Output figure path")
    args = parser.parse_args()

    layers, mean_shift, std_ratio = load_stats(args.json_path)
    plot_fig2_single(layers, mean_shift, std_ratio, args.out)

    print(f"Saved Figure 2 to {args.out}")
#  python plot_fig2.py \
#    --json_path ~/SparseMM/sparsemm/statistics_table/qwen2vl_docvqa600/layer_statistics_table.json \
#    --out .fig2_llava_infographic.pdf
