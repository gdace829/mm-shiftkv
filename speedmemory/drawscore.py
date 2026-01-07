import json
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, VPacker, DrawingArea

# =========================================================
# 0) Plot style (global fonts)
# =========================================================
def set_plot_style(
    base_font=15,
    title_scale=1.10,
    label_scale=1.10,
    tick_scale=0.95,
    lines_lw=2.4,
    marker_sz=7,
    axes_lw=1.2,
):
    plt.rcParams.update({
        "font.size": base_font,
        "axes.titlesize": int(base_font * title_scale),
        "axes.labelsize": int(base_font * label_scale),
        "xtick.labelsize": int(base_font * tick_scale),
        "ytick.labelsize": int(base_font * tick_scale),
        "lines.linewidth": lines_lw,
        "lines.markersize": marker_sz,
        "axes.linewidth": axes_lw,
    })

# You can tune global font sizes here:
set_plot_style(base_font=18)

# =========================================================
# 1) Load score.json
# =========================================================
with open("score.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# =========================================================
# 2) Parse keys: "{method}-{length}"
# =========================================================
pat = re.compile(r"^(.*)-(\d+)$")
store = defaultdict(lambda: defaultdict(dict))
lengths_set = set()

for k, v in data.items():
    m = pat.match(k)
    if not m:
        continue
    method, length = m.group(1), int(m.group(2))
    store[method][length] = v
    lengths_set.add(length)

lengths = sorted(lengths_set)

# =========================================================
# 3) Display names (legend)
# =========================================================
DISPLAY_NAME = {
    "fullkv": "FullKV",
    "expected_attention": "ExpAttn",
    "expected_attn": "ExpAttn",
    "expectedattn": "ExpAttn",
    "snapkv": "SnapKV",
    "keydiff": "KeyDiff",
    "streamingllm": "StreamingLLM",
    "tova": "TOVA",
    "ours": "MM-ShiftKV",
    "mm_shiftkv": "MM-ShiftKV",
    "mm-shiftkv": "MM-ShiftKV",
}

def show_name(m: str) -> str:
    return DISPLAY_NAME.get(m.lower(), m)

# =========================================================
# 4) Method order
# =========================================================
def sort_methods(ms):
    def key(m):
        n = m.lower()
        if n == "fullkv":
            return (0, n)
        if n in {"ours", "mm_shiftkv", "mm-shiftkv"}:
            return (2, n)
        return (1, n)
    return sorted(ms, key=key)

# =========================================================
# 5) Canonical names (for colors/offsets)
# =========================================================
ALIASES = {
    "mm_shiftkv": "ours",
    "mm-shiftkv": "ours",
    "expectedattn": "expected_attention",
    "expected_attn": "expected_attention",
}

def canon(m: str) -> str:
    return ALIASES.get(m.lower(), m.lower())

# =========================================================
# 4.5) Build methods list and REMOVE TOVA
# =========================================================
methods = [m for m in sort_methods(store.keys()) if canon(m) != "tova"]

# =========================================================
# 6) Colors
# =========================================================
LINE_COLOR = {
    "fullkv": "#0072B2",
    "expected_attention": "#000000",
    "keydiff": "#009E73",
    "snapkv": "#CC79A7",
    "streamingllm": "#E69F00",
    # "tova": "#56B4E9",  # removed
    "ours": "#D55E00",
}

def lighten(c, amount=0.65):
    rgb = np.array(mcolors.to_rgb(c))
    return mcolors.to_hex(rgb + (1 - rgb) * amount)

def line_color(m):
    cm = canon(m)
    if cm not in LINE_COLOR:
        raise KeyError(f"Missing line color for method '{m}' (canon='{cm}'). Add to LINE_COLOR.")
    return LINE_COLOR[cm]

def bar_color(m, amount=0.65):
    return lighten(line_color(m), amount=amount)

# =========================================================
# 7) Visualization-only memory offsets (optional)
# =========================================================
VIS_MEM_OFFSET = {
    "snapkv": -0.5,
    "keydiff": -0.8,
    # "tova": +0.7,  # removed
    "streamingllm": +0.3,
}

def apply_offset(m, vals):
    off = VIS_MEM_OFFSET.get(canon(m), 0.0)
    out = []
    for v in vals:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append(np.nan)
        else:
            out.append(v + off)
    return out

# =========================================================
# 8) Compact legend (3 columns: Method | bar sample | line sample)
# =========================================================
def add_compact_legend(
    ax,
    methods,
    loc="upper left",

    # ---- Legend tuning knobs ----
    legend_font=11.5,
    col1_width=90,       # Method column width
    col_pad=4,           # spacing between columns
    row_pad=1,           # spacing between rows

    bar_w=22,
    bar_h=10,
    line_w=32,
    line_h=12,

    bar_alpha=0.85,
    bar_lighten_amount=0.65,

    frame=True,
    frame_alpha=0.95,
    pad=0.25,
    borderpad=0.35,
):
    rows = []
    for m in methods:
        name = show_name(m)

        # ---- method name column
        col1 = DrawingArea(col1_width, max(bar_h, line_h), 0, 0)
        col1.add_artist(plt.Text(0, 0, name, fontsize=legend_font, va="bottom"))

        # ---- latency bar sample
        col2 = DrawingArea(bar_w, bar_h, 0, 0)
        col2.add_artist(Rectangle(
            (0, 0), bar_w, bar_h,
            facecolor=bar_color(m, amount=bar_lighten_amount),
            edgecolor="none",
            alpha=bar_alpha
        ))

        # ---- memory line sample
        col3 = DrawingArea(line_w, line_h, 0, 0)
        y = line_h / 2.0
        lw = 3.0 if canon(m) == "ours" else 2.4
        col3.add_artist(Line2D([0, line_w], [y, y], color=line_color(m), linewidth=lw))
        col3.add_artist(Line2D([line_w * 0.65], [y], color=line_color(m),
                               marker="o", markersize=legend_font * 0.6))

        rows.append(HPacker(children=[col1, col2, col3],
                            align="center", pad=0, sep=col_pad))

    box = VPacker(children=rows, align="left", pad=0, sep=row_pad)

    anchored = AnchoredOffsetbox(
        loc=loc,
        child=box,
        pad=pad,
        borderpad=borderpad,
        frameon=frame
    )
    if frame:
        anchored.patch.set_alpha(frame_alpha)

    ax.add_artist(anchored)

# =========================================================
# 9) Plot function
# =========================================================
def plot_latency_bar_and_memory_line(
    score_path="score.json",
    savepath="latency_bar_memory_line.pdf",
    title="Decode Latency (bar) and Peak Memory (line) vs Input length",

    fig_w=9.2,
    fig_h=5.2,
    dpi=200,

    latency_scale=1000.0,     # s -> ms
    bar_alpha=0.85,
    bar_lighten_amount=0.65,

    group_width=0.84,

    legend_loc="upper left",
    legend_font=11.5,
    legend_col1_width=90,
    legend_col_pad=4,
    legend_row_pad=1,
):
    fig, ax_lat = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax_mem = ax_lat.twinx()

    x = np.arange(len(lengths))
    n_methods = len(methods)
    bar_w = group_width / n_methods

    # ---- latency bars
    for i, m in enumerate(methods):
        lat = [store[m].get(L, {}).get("Decoding latency", np.nan) for L in lengths]
        lat_ms = [v * latency_scale for v in lat]
        offs = x - group_width / 2 + (i + 0.5) * bar_w
        ax_lat.bar(
            offs, lat_ms, bar_w,
            color=bar_color(m, amount=bar_lighten_amount),
            alpha=bar_alpha,
            edgecolor="none",
            zorder=1
        )

    # ---- memory lines
    for m in methods:
        mem = [store[m].get(L, {}).get("Peak memory usage", np.nan) for L in lengths]
        mem = apply_offset(m, mem)
        ax_mem.plot(
            x, mem,
            marker="o",
            color=line_color(m),
            linewidth=3.0 if canon(m) == "ours" else 2.4,
            zorder=3
        )

    # ---- axes labels
    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels([str(L) for L in lengths])
    ax_lat.set_xlabel("Input length")
    ax_lat.set_ylabel("Decode Latency (ms/token)")
    ax_mem.set_ylabel("Peak Memory (GB)")
    ax_lat.set_title(title)
    ax_lat.grid(axis="y", linestyle="--", alpha=0.55, zorder=0)

    # ---- compact legend (tight spacing)
    add_compact_legend(
        ax_lat,
        methods,
        loc=legend_loc,
        legend_font=legend_font,
        col1_width=legend_col1_width,
        col_pad=legend_col_pad,
        row_pad=legend_row_pad,
        bar_alpha=bar_alpha,
        bar_lighten_amount=bar_lighten_amount,
    )

    fig.tight_layout()
    plt.savefig(savepath, dpi=450, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {savepath}")

# =========================================================
# 10) Run (tune legend spacing here)
# =========================================================
plot_latency_bar_and_memory_line(
    legend_font=16,
    legend_col1_width=120,
    legend_col_pad=5,
    legend_row_pad=5,
)
