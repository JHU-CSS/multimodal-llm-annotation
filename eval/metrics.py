"""Loaders and metrics shared by the TikTok eval scripts."""
import csv
import json
from collections import Counter
from pathlib import Path


def load_ground_truth(samples_csv: Path) -> dict:
    """Map relative_path -> ground-truth hashtag."""
    gt = {}
    with Path(samples_csv).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel = row.get("relative_path", "").strip()
            tag = row.get("hashtag", "").strip()
            if rel:
                gt[rel] = tag
    return gt


def load_predictions(pred_json: Path, fix_json: Path | None = None) -> dict:
    """Map relative_path -> predicted hashtag. `fix_json` entries override `pred_json`."""
    preds = {}
    for path in filter(None, [pred_json, fix_json]):
        with Path(path).open(encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            rel = entry.get("video")
            segments = entry.get("segments", [])
            if not rel or not segments:
                continue
            ann = segments[0].get("annotations", {}) or {}
            preds[rel] = ann.get("hashtag") or ann.get("label") or ""
    return preds


def kappa_from_confusion(conf: Counter) -> float:
    """Cohen's kappa from a Counter of (true, pred) -> count."""
    total = sum(conf.values())
    if not total:
        return 0.0
    correct = sum(c for (t, p), c in conf.items() if t == p)
    true_marg, pred_marg = Counter(), Counter()
    for (t, p), c in conf.items():
        true_marg[t] += c
        pred_marg[p] += c
    pe = sum(true_marg[l] * pred_marg[l] for l in true_marg) / (total * total)
    return (correct / total - pe) / (1 - pe) if (1 - pe) != 0 else 0.0


def per_class_metrics(conf: Counter) -> list[dict]:
    """One-vs-rest P/R/F1/accuracy/kappa per label."""
    total = sum(conf.values())
    labels = sorted({t for t, _ in conf} | {p for _, p in conf})
    out = []
    for lbl in labels:
        tp = conf.get((lbl, lbl), 0)
        fp = sum(conf.get((t, lbl), 0) for t in labels if t != lbl)
        fn = sum(conf.get((lbl, p), 0) for p in labels if p != lbl)
        tn = total - tp - fp - fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        acc = (tp + tn) / total if total else 0.0
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total * total) if total else 0.0
        kappa = (acc - pe) / (1 - pe) if total and (1 - pe) != 0 else 0.0
        out.append(dict(label=lbl, support=tp + fn, precision=precision, recall=recall,
                        f1=f1, accuracy=acc, kappa=kappa))
    return out
