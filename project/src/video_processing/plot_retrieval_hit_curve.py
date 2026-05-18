from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from .io_utils import read_json
from .report_paths import latest_log_path, timestamped_log_path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot cumulative TopK retrieval hit-rate curve.")
    parser.add_argument("--profile", help="Named storage profile for side-by-side indexes, e.g. seg4s.")
    parser.add_argument("--report-json")
    parser.add_argument("--export-png")
    parser.add_argument("--export-svg")
    return parser.parse_args()


def load_curve_points(report: dict) -> list[tuple[int, float]]:
    cumulative_hit_rates = report.get("cumulative_hit_rates", {})
    points: list[tuple[int, float]] = []
    for key, value in cumulative_hit_rates.items():
        if not key.startswith("top") or not key.endswith("_hit_rate"):
            continue
        rank_text = key[3:-9]
        if rank_text.isdigit():
            points.append((int(rank_text), float(value)))
    points.sort(key=lambda item: item[0])
    return points


def plot_curve(report: dict, png_path, svg_path) -> None:
    points = load_curve_points(report)
    if not points:
        raise ValueError("No cumulative_hit_rates found in report. Re-run analyze_retrieval_gaps with --top-k >= 2.")

    x_values = [rank for rank, _ in points]
    y_values = [rate for _, rate in points]

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=160)
    ax.plot(x_values, y_values, color="#2563eb", linewidth=2.5, marker="o", markersize=6)
    ax.fill_between(x_values, y_values, [0.0] * len(y_values), color="#93c5fd", alpha=0.2)

    for x_value, y_value in points:
        ax.annotate(
            f"{y_value:.1%}",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
            color="#1d4ed8",
        )

    ax.set_title("Retrieval TopK Hit Rate Curve", fontsize=16, pad=14)
    ax.set_xlabel("TopK threshold", fontsize=12)
    ax.set_ylabel("Hit rate", fontsize=12)
    ax.set_xticks(x_values)
    ax.set_ylim(0.0, max(max(y_values) + 0.05, 0.1))
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.grid(True, axis="x", linestyle=":", alpha=0.15)

    query_count = report.get("query_count", 0)
    top_k_analyzed = report.get("top_k_analyzed", 0)
    ax.text(
        0.99,
        0.02,
        f"Queries: {query_count}   TopK analyzed: {top_k_analyzed}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="#6b7280",
    )

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight")
    if svg_path:
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    report_path = latest_log_path(args.profile, args.report_json, "retrieval_gap_report.json")
    png_path = timestamped_log_path(args.profile, args.export_png, "retrieval_hit_curve.png")
    run_timestamp = png_path.stem.removeprefix("retrieval_hit_curve_")
    svg_path = timestamped_log_path(
        args.profile,
        args.export_svg,
        "retrieval_hit_curve.svg",
        timestamp=run_timestamp,
    )
    report = read_json(report_path)
    plot_curve(report, png_path=png_path, svg_path=svg_path)
    print(f"Saved retrieval hit-rate curve PNG: {png_path}", flush=True)
    print(f"Saved retrieval hit-rate curve SVG: {svg_path}", flush=True)


if __name__ == "__main__":
    main()
