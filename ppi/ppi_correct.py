#!/usr/bin/env python3
"""
PPI (Prediction-Powered Inference) correction for TikTok hashtag frequencies,
using a hand-rolled PPI mean estimator (lambda clipped to [0, 1]).

Data:
  - 100 videos with BOTH human annotations (GT) and LLM annotations (std2*8)
  - 900 videos with ONLY LLM annotations (std2*8_900)

For each hashtag h, we form binary indicators:
  Y_h[i]               = 1 if human says video i is about h  (n=100)
  Yhat_h[i]            = 1 if LLM  says video i is about h  (n=100)
  Yhat_unlabeled_h[i]  = 1 if LLM  says video i is about h  (N=900)

PPI estimate:
  theta = lam * mean(Yhat_unlabeled) + mean(Y - lam * Yhat)
        = LLM frequency on big set (tuned) + bias correction from small set

Usage:
  python -m ppi.ppi_correct [--labeled_csv sampled_videos.csv] \
      [--ann_labeled "data/annotations_std2*8.json"] [--ann_unlabeled "data/annotations_std2*8_900.json"]
"""

import argparse
import warnings
from collections import Counter

import numpy as np
from scipy import stats

from ppi.common import (INVALID_HASHTAGS, load_tiktok_split, normalize_ppi, one_hot,
                        plot_frequencies, read_csv_rows)

warnings.filterwarnings("ignore")


def ppi_mean_ci(Y, Yhat, Yhat_unlabeled, alpha=0.1):
    """
    PPI confidence interval for the mean of a binary indicator.
    Returns: (point_est, ci_lower, ci_upper, lam_opt)
    """
    n = len(Y)
    N = len(Yhat_unlabeled)

    # Optimal lambda: lam* = Cov(Y, Yhat) / Var(Yhat), clipped to [0, 1]
    var_yhat = np.var(Yhat, ddof=1)
    if var_yhat > 1e-10:
        cov_yyhat = np.cov(Y, Yhat, ddof=1)[0, 1]
        lam = np.clip(cov_yyhat / var_yhat, 0, 1)
    else:
        lam = 0.0  # no LLM variance → fall back to classical

    theta = lam * np.mean(Yhat_unlabeled) + np.mean(Y - lam * Yhat)

    # CLT-based standard error
    var_rect = np.var(Y - lam * Yhat, ddof=1) / n
    var_imp = lam**2 * np.var(Yhat_unlabeled, ddof=1) / N
    se = np.sqrt(var_rect + var_imp)

    z = stats.norm.ppf(1 - alpha / 2)
    return theta, theta - z * se, theta + z * se, lam


def classical_mean_ci(Y, alpha=0.1):
    """Classical CI using only human labels."""
    n = len(Y)
    theta = np.mean(Y)
    se = np.sqrt(np.var(Y, ddof=1) / n)
    z = stats.norm.ppf(1 - alpha / 2)
    return theta, theta - z * se, theta + z * se


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labeled_csv",   default="sampled_videos.csv", help="GT hashtag per video (all 1000)")
    p.add_argument("--ann_labeled",   default="data/annotations_std2*8.json")
    p.add_argument("--ann_unlabeled", default="data/annotations_std2*8_900.json")
    p.add_argument("--output",        default="ppi_correction_plot", help="output path stem (no extension)")
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()
    alpha = args.alpha

    all_rows = read_csv_rows(args.labeled_csv)
    gt_map = {r["relative_path"]: r["hashtag"] for r in all_rows}
    pred_all, pred_900, labeled_paths, unlabeled_paths = load_tiktok_split(
        args.ann_labeled, args.ann_unlabeled, keep_unannotated=False)

    print(f"Labeled (human + LLM): {len(labeled_paths)}")
    print(f"Unlabeled (LLM only):  {len(unlabeled_paths)}")

    HASHTAGS = sorted(h for h in set(gt_map.values()) if h not in INVALID_HASHTAGS)
    print(f"Hashtags ({len(HASHTAGS)}): {HASHTAGS}\n")

    n, N = len(labeled_paths), len(unlabeled_paths)
    Y = one_hot(labeled_paths, gt_map, HASHTAGS)                # human label, labeled set
    Yhat = one_hot(labeled_paths, pred_all, HASHTAGS)           # LLM label, labeled set
    Yhat_unlabeled = one_hot(unlabeled_paths, pred_900, HASHTAGS)
    gt_counts = Counter(r["hashtag"] for r in all_rows)

    # ── PPI per hashtag ──────────────────────────────────────────────────────
    print(f"{'Hashtag':17s} | {'GT freq':>8s} | {'LLM freq':>9s} | {'PPI est':>8s} | {'95% CI':>18s} | {'Classical':>8s} | {'Class. CI':>18s} | {'lam':>5s}")
    print("-" * 120)

    results = []
    for k, ht in enumerate(HASHTAGS):
        ppi_est, ppi_lo, ppi_hi, lam = ppi_mean_ci(Y[:, k], Yhat[:, k], Yhat_unlabeled[:, k], alpha=alpha)
        cl_est, cl_lo, cl_hi = classical_mean_ci(Y[:, k], alpha=alpha)
        gt_freq = gt_counts[ht] / len(all_rows)      # over all 1000 videos (for validation)
        llm_freq = np.mean(Yhat_unlabeled[:, k])     # naive LLM frequency on the 900
        results.append(dict(
            hashtag=ht, gt_freq=gt_freq, llm_freq=llm_freq,
            ppi_est=ppi_est, ppi_lo=ppi_lo, ppi_hi=ppi_hi,
            cl_est=cl_est, cl_lo=cl_lo, cl_hi=cl_hi, lam=lam,
        ))
        print(f"{ht:17s} | {gt_freq:7.4f}  | {llm_freq:8.4f}  | {ppi_est:7.4f}  | [{ppi_lo:.4f}, {ppi_hi:.4f}] | {cl_est:7.4f}  | [{cl_lo:.4f}, {cl_hi:.4f}] | {lam:5.3f}")

    # ── Normalized table ─────────────────────────────────────────────────────
    S = normalize_ppi(results)

    print("\n" + "=" * 120)
    print("NORMALIZED PPI ESTIMATES (sum to 1)")
    print("=" * 120)
    print(f"\n{'Hashtag':17s} | {'GT freq':>8s} | {'LLM freq':>9s} | {'PPI (raw)':>10s} | {'PPI (norm)':>10s} | {'95% CI (norm)':>22s}")
    print("-" * 100)
    for r in results:
        print(f"{r['hashtag']:17s} | {r['gt_freq']:7.4f}  | {r['llm_freq']:8.4f}  | {r['ppi_est']:9.4f}  | {r['ppi_norm']:9.4f}  | [{r['ppi_norm_lo']:.4f}, {r['ppi_norm_hi']:.4f}]")
    ppi_norm_sum = sum(r["ppi_norm"] for r in results)
    print(f"{'TOTAL':17s} | {'':8s} | {'':9s} | {S:9.4f}  | {ppi_norm_sum:9.4f}  |")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)

    ppi_widths = [r["ppi_hi"] - r["ppi_lo"] for r in results]
    cl_widths = [r["cl_hi"] - r["cl_lo"] for r in results]
    print(f"\nAvg CI width (raw)  — PPI: {np.mean(ppi_widths):.4f}  Classical: {np.mean(cl_widths):.4f}  "
          f"(PPI {np.mean(cl_widths)/np.mean(ppi_widths):.1f}x narrower)")

    ppi_covers = sum(1 for r in results if r["ppi_norm_lo"] <= r["gt_freq"] <= r["ppi_norm_hi"])
    cl_covers = sum(1 for r in results if r["cl_lo"] <= r["gt_freq"] <= r["cl_hi"])
    print(f"Coverage (GT in CI) — PPI norm: {ppi_covers}/{len(results)}  Classical: {cl_covers}/{len(results)}")

    ppi_mae = np.mean([abs(r["ppi_norm"] - r["gt_freq"]) for r in results])
    llm_mae = np.mean([abs(r["llm_freq"] - r["gt_freq"]) for r in results])
    cl_mae = np.mean([abs(r["cl_est"] - r["gt_freq"]) for r in results])
    print(f"MAE vs GT freq  — PPI norm: {ppi_mae:.4f}  Naive LLM: {llm_mae:.4f}  Classical: {cl_mae:.4f}")

    print(f"\nn={n} labeled, N={N} unlabeled, alpha={alpha} (95% CI)")
    print(f"Raw PPI sum before normalization: {S:.4f}")

    plot_frequencies(results, args.output, llm_label="LLM (raw std2×8)",
                     title="Hashtag Frequency: Ground Truth vs LLM (raw) vs PPI corrected")
    print(f"\nSaved: {args.output}.png / .pdf")


if __name__ == "__main__":
    main()
