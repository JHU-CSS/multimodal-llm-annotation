"""
Compare MELD LLM annotations to human ground truth.

Computes Cohen's kappa and accuracy for speaker, emotion, and sentiment,
both on all data and excluding N/A predictions.

Usage:
  python -m eval.eval_meld <human_csv> <llm_json>
  python -m eval.eval_meld <human_csv>             # reads LLM JSON from stdin
"""

import argparse
import json
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score, classification_report


NA_TOKENS = {"N/A", "NA", "NONE", ""}


def normalize_label(val) -> str:
    if val is None:
        return "n/a"
    s = str(val).strip()
    return "n/a" if s.upper() in NA_TOKENS else s.lower()


def _label_and_conf(val) -> tuple:
    """A field is either {"label": ..., "confidence": ...} or a bare label."""
    if isinstance(val, dict):
        return val.get("label", "N/A"), val.get("confidence")
    return val, None


def extract_pred_row(item: dict) -> dict:
    """Pull (speaker, emotion, sentiment) from one annotation record."""
    ann = item.get("annotations", {}) or {}
    if not ann:
        speaker, emo_val, sent_val = "Other", "N/A", "N/A"
        emo_conf = sent_conf = None
    else:
        speaker = min(ann)          # alphabetically first key if the model returned several
        vals = ann[speaker]
        emo_val, emo_conf = _label_and_conf(vals.get("emotion", "N/A"))
        sent_val, sent_conf = _label_and_conf(vals.get("sentiment", "N/A"))

    return {
        "Dialogue_ID": item.get("dialogue_id"),
        "Utterance_ID": item.get("utterance_id"),
        "Pred_Speaker": (speaker or "Unknown").strip(),
        "Pred_Emotion": normalize_label(emo_val),
        "Pred_Sentiment": normalize_label(sent_val),
        "Emotion_Confidence": emo_conf,
        "Sentiment_Confidence": sent_conf,
    }


def per_class(df: pd.DataFrame, col: str) -> None:
    """Per-class accuracy (recall) plus precision/F1/support for one field."""
    y_true, y_pred = df[col], df[f"Pred_{col}"]
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    classes = sorted(set(y_true) | set(y_pred))
    print(f"    {'class':12s} {'acc(recall)':>12s} {'precision':>10s} {'f1':>8s} {'n':>5s}")
    for c in classes:
        r = rep.get(c)
        if not r:
            continue
        n = int(r["support"])
        print(f"    {c:12s} {r['recall']:>12.2%} {r['precision']:>10.2%} {r['f1-score']:>8.2f} {n:>5d}")


def report(label: str, df: pd.DataFrame) -> None:
    if df.empty:
        print(f"\n=== {label}: empty ===")
        return
    print(f"\n=== {label} ({len(df)} utterances) ===")
    for col in ("Speaker", "Emotion", "Sentiment"):
        kappa = cohen_kappa_score(df[col], df[f"Pred_{col}"])
        acc = (df[col] == df[f"Pred_{col}"]).mean()
        print(f"  {col:9s}  kappa={kappa:.4f}  accuracy={acc:.2%}")
    for col in ("Emotion", "Sentiment"):
        print(f"\n  -- per-class {col} --")
        per_class(df, col)


def main():
    p = argparse.ArgumentParser(description="Cohen's kappa for MELD predictions vs human labels.")
    p.add_argument("human_csv", help="Human annotation CSV (Dialogue_ID, Utterance_ID, Speaker, Emotion, Sentiment)")
    p.add_argument("llm_json", nargs="?", default=None,
                   help="LLM annotation JSON (omit to read from stdin)")
    p.add_argument("--out", default="comparison_results.csv",
                   help="Where to write the merged comparison CSV")
    p.add_argument("--out_no_na", default="comparison_results_no_na.csv",
                   help="Where to write the merged comparison CSV with N/A rows excluded")
    args = p.parse_args()

    print("Loading human annotations...", file=sys.stderr)
    human = pd.read_csv(args.human_csv)[
        ["Dialogue_ID", "Utterance_ID", "Speaker", "Emotion", "Sentiment"]
    ].copy()
    human["Speaker"] = human["Speaker"].str.strip()
    human["Emotion"] = human["Emotion"].str.strip().str.lower()
    human["Sentiment"] = human["Sentiment"].str.strip().str.lower()

    print("Loading LLM annotations...", file=sys.stderr)
    if args.llm_json:
        with open(args.llm_json, encoding="utf-8") as f:
            llm_items = json.load(f)
    else:
        llm_items = json.load(sys.stdin)

    pred = pd.DataFrame(extract_pred_row(item) for item in llm_items)
    print(f"Loaded {len(human)} human / {len(pred)} LLM annotations", file=sys.stderr)

    merged = pd.merge(human, pred, on=["Dialogue_ID", "Utterance_ID"], how="inner")
    print(f"Matched {len(merged)} utterances", file=sys.stderr)
    if len(merged) < len(human):
        print(f"  ({len(human) - len(merged)} human rows had no matching LLM prediction)", file=sys.stderr)
    if len(merged) < len(pred):
        print(f"  ({len(pred) - len(merged)} LLM rows had no matching human label)", file=sys.stderr)

    for col in ("Speaker", "Emotion", "Sentiment"):
        merged[f"{col}_Match"] = merged[col] == merged[f"Pred_{col}"]

    merged.to_csv(args.out, index=False)

    no_na = merged[
        (merged["Emotion"] != "n/a") & (merged["Pred_Emotion"] != "n/a")
        & (merged["Sentiment"] != "n/a") & (merged["Pred_Sentiment"] != "n/a")
    ]
    no_na.to_csv(args.out_no_na, index=False)

    report("ALL DATA", merged)
    report("EXCLUDING N/A", no_na)

    if not merged.empty:
        excl = len(merged) - len(no_na)
        print(f"\nExcluded {excl} utterances with N/A emotion/sentiment "
              f"({excl / len(merged):.1%} of total)")


if __name__ == "__main__":
    main()
