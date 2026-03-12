#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MEMORY_REGEX = re.compile(r"CRAFT allocates\s*([0-9]*\.?[0-9]+)\s*x less memory", re.IGNORECASE)
PLACEMENT_REGEX = re.compile(r"placement-only balancedness:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
EPLB_REGEX = re.compile(r"EPLB balancedness:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
CRAFT_REGEX = re.compile(r"CRAFT balancedness:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


def parse_opt_file(path: Path) -> tuple[float, float, float, float]:
    text = path.read_text(encoding="utf-8")
    memory_match = MEMORY_REGEX.search(text)
    if memory_match is None:
        raise ValueError(f"Could not parse memory saving from: {path}")
    memory_saving = float(memory_match.group(1))

    lines = [line.strip() for line in text.splitlines()]
    non_empty_lines = [line for line in lines if line]
    if not non_empty_lines:
        raise ValueError(f"File is empty: {path}")
    last_line = non_empty_lines[-1]

    placement_match = PLACEMENT_REGEX.search(last_line)
    eplb_match = EPLB_REGEX.search(last_line)
    craft_match = CRAFT_REGEX.search(last_line)
    if placement_match is None or eplb_match is None or craft_match is None:
        raise ValueError(f"Could not parse balancedness values from last line in: {path}")

    placement_bal = float(placement_match.group(1))
    eplb_bal = float(eplb_match.group(1))
    craft_bal = float(craft_match.group(1))
    return memory_saving, placement_bal, eplb_bal, craft_bal


def make_plot(input_dir: Path, output_path: Path) -> None:
    opt_files = sorted(input_dir.glob("*.opt"))
    if not opt_files:
        raise ValueError(f"No .opt files found in {input_dir}")

    labels: list[str] = []
    memory_vals: list[float] = []
    placement_vals: list[float] = []
    eplb_vals: list[float] = []
    craft_vals: list[float] = []

    for opt_file in opt_files:
        memory, placement, eplb, craft = parse_opt_file(opt_file)
        labels.append(opt_file.stem)
        memory_vals.append(memory)
        placement_vals.append(placement)
        eplb_vals.append(eplb)
        craft_vals.append(craft)

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 10,
        }
    )

    x = np.arange(len(labels), dtype=float)
    bar_w = 0.2
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_w

    fig_w = max(10.0, len(labels) * 1.3)
    fig, ax_left = plt.subplots(figsize=(fig_w, 5.8), dpi=150)
    fig.patch.set_facecolor("#fbfcfe")
    ax_left.set_facecolor("#fbfcfe")
    ax_right = ax_left.twinx()
    ax_right.set_facecolor("none")

    bars_mem = ax_left.bar(
        x + offsets[0],
        memory_vals,
        width=bar_w,
        color="#8fd18f",
        edgecolor="#000000",
        linewidth=1.0,
        label="CRAFT Memory Saving (compared to EPLB)",
        zorder=3,
    )
    bars_place = ax_right.bar(
        x + offsets[1],
        placement_vals,
        width=bar_w,
        color="#9ecae1",
        edgecolor="#000000",
        linewidth=1.0,
        label="EPLB (placement-only) Balancedness",
        zorder=3,
    )
    bars_eplb = ax_right.bar(
        x + offsets[2],
        eplb_vals,
        width=bar_w,
        color="#4292c6",
        edgecolor="#000000",
        linewidth=1.0,
        label="EPLB Balancedness",
        zorder=3,
    )
    bars_craft = ax_right.bar(
        x + offsets[3],
        craft_vals,
        width=bar_w,
        color="#08519c",
        edgecolor="#000000",
        linewidth=1.0,
        label="CRAFT Balancedness",
        zorder=3,
    )

    ax_left.set_title("Memory Saving and Load Balancedness", pad=14, weight="bold")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels)
    ax_left.set_ylabel("CRAFT Memory Saving (compared to EPLB) ↑")
    ax_right.set_ylabel("Load Balancedness ↑")

    left_max = max(memory_vals)
    ax_left.set_ylim(0, left_max * 1.2)
    right_max = max(placement_vals + eplb_vals + craft_vals)
    ax_right.set_ylim(0, min(1.0, right_max * 1.2) if right_max <= 1.0 else right_max * 1.2)

    ax_left.grid(axis="y", linestyle="--", color="#d7dfe8", alpha=0.8, zorder=0)
    for spine in ax_left.spines.values():
        spine.set_color("#9aa4b2")
    for spine in ax_right.spines.values():
        spine.set_color("#9aa4b2")

    handles = [bars_mem, bars_place, bars_eplb, bars_craft]
    labels_leg = [h.get_label() for h in handles]
    legend = ax_left.legend(handles, labels_leg, loc="upper left", frameon=True)
    legend.get_frame().set_facecolor("#ffffff")
    legend.get_frame().set_edgecolor("#ccd6e3")
    legend.get_frame().set_alpha(0.75)

    fig.tight_layout()
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    print(f"Saved figure to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a dual-axis clustered bar chart from .opt files."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="results",
        help="Directory containing .opt files (default: results).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output image path (default: <input_dir>/ae_fig.png).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    output_path = Path(args.output) if args.output else input_dir / "ae_fig.png"
    make_plot(input_dir, output_path)


if __name__ == "__main__":
    main()
