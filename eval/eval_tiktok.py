#!/usr/bin/env python3
"""
Per-class P/R/F1, accuracy and Cohen's kappa for TikTok hashtag predictions.

Usage:
  python -m eval.eval_tiktok --pred data/annotations.json --samples sampled_videos.csv
"""
import argparse
from collections import Counter
from pathlib import Path

from eval.metrics import kappa_from_confusion, load_ground_truth, load_predictions, per_class_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="Combined annotations JSON")
    ap.add_argument("--samples", required=True, help="sampled_videos.csv with ground truth hashtags")
    args = ap.parse_args()

    gt = load_ground_truth(Path(args.samples))
    preds = load_predictions(Path(args.pred))

    conf = Counter((true_tag, preds[rel]) for rel, true_tag in gt.items() if rel in preds)
    misses = [rel for rel in gt if rel not in preds]
    total = sum(conf.values())
    correct = sum(c for (t, p), c in conf.items() if t == p)

    print(f"Total evaluated: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {correct / total if total else 0.0:.4f}")
    print(f"Cohen's kappa: {kappa_from_confusion(conf):.4f}")
    print("\nTop confusions (true -> pred):")
    for (t, p), c in conf.most_common(20):
        print(f"{t} -> {p}: {c}")

    print("\nPer-class metrics (tab-separated):")
    print("label\tsupport\tprecision\trecall\tf1\taccuracy\tkappa")
    for m in sorted(per_class_metrics(conf), key=lambda m: (-m["support"], m["label"])):
        print(f"{m['label']}\t{m['support']}\t{m['precision']:.3f}\t{m['recall']:.3f}\t{m['f1']:.3f}"
              f"\t{m['accuracy']:.3f}\t{m['kappa']:.3f}")

    if misses:
        print(f"\nMissing predictions for {len(misses)} videos (not found in pred file).")


if __name__ == "__main__":
    main()
