#!/usr/bin/env python3
"""
Evaluate Gemini annotation results (CSV output of annotation/annotate_gemini.py).

Usage:
  python -m eval.eval_gemini annotated_gemini_100.csv
"""
import argparse
import csv
from collections import Counter

from eval.metrics import kappa_from_confusion, per_class_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Annotated CSV with true_hashtag and predicted_hashtag columns")
    args = ap.parse_args()

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["predicted_hashtag"]]
    if not rows:
        print("No predictions found.")
        return

    conf = Counter((r["true_hashtag"], r["predicted_hashtag"]) for r in rows)
    total = len(rows)
    correct = sum(c for (t, p), c in conf.items() if t == p)
    per_class = per_class_metrics(conf)
    macro = {k: sum(m[k] for m in per_class) / len(per_class) for k in ("precision", "recall", "f1")}

    print(f"Total: {total}  Correct: {correct}  Accuracy: {correct / total:.1%}  "
          f"Cohen's kappa: {kappa_from_confusion(conf):.4f}")
    print(f"Macro avg — P: {macro['precision']:.3f}  R: {macro['recall']:.3f}  F1: {macro['f1']:.3f}\n")

    print(f"{'hashtag':20s} {'support':>7s} {'prec':>6s} {'recall':>6s} {'f1':>6s}")
    print("-" * 47)
    for m in sorted(per_class, key=lambda m: -m["support"]):
        print(f"{m['label']:20s} {m['support']:7d} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f}")

    print(f"\nTop confusions (true -> pred):")
    for (t, p), c in conf.most_common(20):
        if t != p:
            print(f"  {t:20s} -> {p:20s}: {c}")


if __name__ == "__main__":
    main()
