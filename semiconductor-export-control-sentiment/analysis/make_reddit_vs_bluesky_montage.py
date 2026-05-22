#!/usr/bin/env python3
"""
Create side-by-side comparison montages: Reddit (left) vs Bluesky (right).

Inputs (expected to exist):
  analysis/analysis_output/figures/**.png
  analysis/analysis_output_bluesky/figures/**.png

Outputs:
  analysis/analysis_output_platform_compare/montages/*.png

This is purely a visualization helper (no re-computation of metrics).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _load_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency Pillow. Install with:\n  pip install pillow"
        ) from e
    return Image, ImageDraw, ImageFont


def _title_bar(img, left_title: str, right_title: str):
    Image, ImageDraw, ImageFont = _load_pillow()
    w, h = img.size
    bar_h = max(48, int(h * 0.05))
    out = Image.new("RGB", (w, h + bar_h), (255, 255, 255))
    out.paste(img, (0, bar_h))

    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, int(bar_h * 0.35)))
    except Exception:
        font = ImageFont.load_default()

    # left label
    draw.text((12, int(bar_h * 0.25)), left_title, fill=(0, 0, 0), font=font)
    # right label
    rt_w = draw.textlength(right_title, font=font) if hasattr(draw, "textlength") else None
    x_rt = w - (rt_w + 12 if rt_w is not None else 12)
    draw.text((x_rt, int(bar_h * 0.25)), right_title, fill=(0, 0, 0), font=font)
    # divider line
    draw.line([(w // 2, 0), (w // 2, bar_h - 1)], fill=(180, 180, 180), width=2)
    draw.line([(0, bar_h - 1), (w, bar_h - 1)], fill=(200, 200, 200), width=1)
    return out


def stitch_lr(left: Path, right: Path, out_path: Path) -> bool:
    Image, _, _ = _load_pillow()
    if not left.exists() or not right.exists():
        return False
    a = Image.open(left).convert("RGB")
    b = Image.open(right).convert("RGB")

    # Resize to same height (preserve aspect).
    target_h = max(a.size[1], b.size[1])

    def resize_to_h(im):
        w, h = im.size
        if h == target_h:
            return im
        new_w = int(round(w * (target_h / h)))
        return im.resize((new_w, target_h), Image.LANCZOS)

    a2 = resize_to_h(a)
    b2 = resize_to_h(b)
    gap = 14
    merged = Image.new("RGB", (a2.size[0] + gap + b2.size[0], target_h), (255, 255, 255))
    merged.paste(a2, (0, 0))
    merged.paste(b2, (a2.size[0] + gap, 0))
    merged = _title_bar(merged, "Reddit", "Bluesky")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.save(out_path, quality=95)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Stitch Reddit vs Bluesky figure montages.")
    ap.add_argument(
        "--reddit-fig-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "analysis_output" / "figures",
    )
    ap.add_argument(
        "--bluesky-fig-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "analysis_output_bluesky" / "figures",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "analysis_output_platform_compare" / "montages",
    )
    args = ap.parse_args()

    reddit_dir = args.reddit_fig_dir.resolve()
    bluesky_dir = args.bluesky_fig_dir.resolve()
    out_dir = args.out_dir.resolve()

    pairs = []
    for rel in [
        # Trend comparisons (most useful for narrative)
        Path("sentiment_return_trends/compare_ALL_cumret_vs_models.png"),
        Path("sentiment_return_trends/compare_NVDA_overlay_zscore_split.png"),
        Path("sentiment_return_trends/compare_AMD_overlay_zscore_split.png"),
        Path("sentiment_return_trends/compare_INTC_overlay_zscore_split.png"),
        Path("sentiment_return_trends/compare_NVDA_emotional_change_5d_split.png"),
        Path("sentiment_return_trends/compare_AMD_emotional_change_5d_split.png"),
        Path("sentiment_return_trends/compare_INTC_emotional_change_5d_split.png"),
        # Diagnostics (model behavior / correlation structure)
        Path("sentiment_diagnostics/corr_fwd_return_heatmap_nrc.png"),
        Path("sentiment_diagnostics/corr_fwd_return_heatmap_hf.png"),
        Path("sentiment_diagnostics/corr_fwd_return_heatmap_finbert.png"),
        Path("sentiment_diagnostics/corr_fwd_return_nets_bar.png"),
        Path("sentiment_diagnostics/ts_NVDA_nets.png"),
        Path("sentiment_diagnostics/ts_AMD_nets.png"),
        Path("sentiment_diagnostics/ts_INTC_nets.png"),
        Path("sentiment_diagnostics/corr_NVDA_cross_net.png"),
        Path("sentiment_diagnostics/corr_AMD_cross_net.png"),
        Path("sentiment_diagnostics/corr_INTC_cross_net.png"),
    ]:
        pairs.append(rel)

    wrote = 0
    missing = 0
    for rel in pairs:
        left = reddit_dir / rel
        right = bluesky_dir / rel
        out = out_dir / rel.with_suffix("").name
        out = out.with_name(out.name + "_REDDIT_vs_BLUESKY.png")
        ok = stitch_lr(left, right, out)
        if ok:
            wrote += 1
        else:
            missing += 1

    print(f"Done. Wrote {wrote} montage(s). Missing pairs: {missing}.")
    print(f"Output folder: {out_dir}")


if __name__ == "__main__":
    main()


