"""Helpers shared by the PPI scripts."""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

MELD_DATA = Path(__file__).resolve().parent.parent / "meld_data"

INVALID_HASHTAGS = {"", "none", "fashion", "health", "motivational", "selfcare"}


# ── TikTok ───────────────────────────────────────────────────────────────────

def read_csv_rows(path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def extract_hashtag_preds(entries: list[dict], keep_unannotated: bool) -> dict:
    """relative_path -> predicted hashtag; keep_unannotated maps entries without segments to ""."""
    preds = {}
    for e in entries:
        segs = e.get("segments") or []
        if segs:
            preds[e["video"]] = segs[0].get("annotations", {}).get("hashtag", "")
        elif keep_unannotated:
            preds[e["video"]] = ""
    return preds


def load_tiktok_split(ann_labeled, ann_unlabeled, keep_unannotated: bool):
    """Labeled videos = those in `ann_labeled` but not in `ann_unlabeled`."""
    with open(ann_labeled) as f:
        ann_all = json.load(f)
    with open(ann_unlabeled) as f:
        ann_unl = json.load(f)
    pred_all = extract_hashtag_preds(ann_all, keep_unannotated)
    pred_unl = extract_hashtag_preds(ann_unl, keep_unannotated)
    unl_videos = {e["video"] for e in ann_unl}
    labeled_paths = sorted(v for v in pred_all if v not in unl_videos)
    unlabeled_paths = sorted(pred_unl)
    return pred_all, pred_unl, labeled_paths, unlabeled_paths


def one_hot(paths: list[str], label_of: dict, classes: list[str]) -> np.ndarray:
    """Indicator matrix; labels outside `classes` stay all-zero."""
    idx = {c: k for k, c in enumerate(classes)}
    out = np.zeros((len(paths), len(classes)), dtype=float)
    for i, p in enumerate(paths):
        k = idx.get(label_of.get(p, ""))
        if k is not None:
            out[i, k] = 1.0
    return out


def normalize_ppi(results: list[dict]) -> float:
    """Clip at 0 and rescale to sum to 1 (CIs by the same factor), in place. Returns the sum S."""
    raw = np.clip(np.array([r["ppi_est"] for r in results]), 0, None)
    S = raw.sum()
    for r, v in zip(results, raw):
        r["ppi_norm"] = v / S
        r["ppi_norm_lo"] = max(0, r["ppi_lo"]) / S
        r["ppi_norm_hi"] = r["ppi_hi"] / S
    return S


def plot_frequencies(results: list[dict], out_stem: str, llm_label: str, title: str) -> None:
    """GT vs LLM vs normalized PPI frequencies; writes .png and .pdf."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    gt = np.array([r["gt_freq"] for r in results])
    order = np.argsort(-gt)
    col = lambda key: np.array([r[key] for r in results])[order]
    hashtags = [results[i]["hashtag"] for i in order]
    x = np.arange(len(hashtags))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x, col("gt_freq") * 100, "o-", color="black", linewidth=2, markersize=7,
            label="Ground Truth", zorder=3)
    ax.plot(x, col("llm_freq") * 100, "s--", color="#1a3a5c", linewidth=1.8, markersize=6,
            label=llm_label, zorder=3)
    ax.plot(x, col("ppi_norm") * 100, "D-", color="#2ca02c", linewidth=2, markersize=7,
            label="PPI (corrected)", zorder=3)
    ax.fill_between(x, col("ppi_norm_lo") * 100, col("ppi_norm_hi") * 100,
                    color="#2ca02c", alpha=0.18, label="95% CI (PPI)")
    ax.set_ylabel("Frequency (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(hashtags, rotation=45, ha="right", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_ylim(-5, 20)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{out_stem}.png", dpi=200, bbox_inches="tight")
    plt.savefig(f"{out_stem}.pdf", bbox_inches="tight")


# ── MELD ─────────────────────────────────────────────────────────────────────

def design(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(df)), df["X"].to_numpy()])


def load_meld_pool(check_order: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1000-video pool (100_HUMAN + 900_LLM; columns utterance, X, GT, Yhat) and ALL_HUMAN."""
    human100 = pd.read_csv(MELD_DATA / "100_HUMAN.csv")
    llm100 = pd.read_csv(MELD_DATA / "100_LLM.csv")
    llm900 = pd.read_csv(MELD_DATA / "900_LLM.csv")
    all_human = pd.read_csv(MELD_DATA / "ALL_HUMAN.csv")
    gt_map = dict(zip(all_human["utterance"], all_human["Y"]))

    if check_order:
        assert (human100["utterance"].to_numpy() == llm100["video_path"].to_numpy()).all(), \
            "100_HUMAN and 100_LLM are not in the same row order"

    labeled = pd.DataFrame({
        "utterance": human100["utterance"],
        "X": human100["X"].to_numpy(),
        "GT": human100["Y"].to_numpy(),
        "Yhat": llm100["Yhat"].to_numpy(),
    })
    unlabeled = pd.DataFrame({
        "utterance": llm900["utterance"],
        "X": llm900["X"].to_numpy(),
        "GT": llm900["utterance"].map(gt_map).to_numpy(),
        "Yhat": llm900["Yhat"].to_numpy(),
    })
    assert len(labeled) == 100
    assert len(unlabeled) == 900
    assert unlabeled["GT"].notna().all(), "Missing GT in 900_LLM pool"

    pool = pd.concat([labeled, unlabeled], ignore_index=True)
    assert len(pool) == 1000
    return pool, all_human


def print_gt_betas(all_human: pd.DataFrame, labels: dict, alpha: float) -> dict:
    """Logistic fit on ALL_HUMAN per class; prints and returns {label: beta}."""
    from ppi_py import classical_logistic_ci, logistic

    X_gold = design(all_human)
    gt_beta = {}
    for label, name in labels.items():
        Y_gold = (all_human["Y"] == label).astype(int).to_numpy()
        theta = logistic(X_gold, Y_gold)
        lo, hi = classical_logistic_ci(X_gold, Y_gold, alpha=alpha)
        gt_beta[label] = float(theta[1])
        print(f"  GT {name}    β={theta[1]:+.3f}  [{lo[1]:.3f}, {hi[1]:.3f}]")
    return gt_beta


def run_logistic_ppi(lab_df: pd.DataFrame, unl_df: pd.DataFrame, label: int, alpha: float):
    """(beta, ci_lo, ci_hi, p) for one class, or NaNs if singular."""
    from ppi_py import ppi_logistic_ci, ppi_logistic_pointestimate, ppi_logistic_pval

    X_lab, X_unl = design(lab_df), design(unl_df)
    Y_lab = (lab_df["GT"] == label).astype(int).to_numpy()
    Yhat_lab = (lab_df["Yhat"] == label).astype(int).to_numpy()
    Yhat_unl = (unl_df["Yhat"] == label).astype(int).to_numpy()
    try:
        theta = ppi_logistic_pointestimate(X_lab, Y_lab, Yhat_lab, X_unl, Yhat_unl)
        lo, hi = ppi_logistic_ci(X_lab, Y_lab, Yhat_lab, X_unl, Yhat_unl, alpha=alpha)
        pval = ppi_logistic_pval(X_lab, Y_lab, Yhat_lab, X_unl, Yhat_unl)
        return float(theta[1]), float(lo[1]), float(hi[1]), float(pval[1])
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan


def stratified_sample_first(df: pd.DataFrame, n: int, rng) -> list[int]:
    """n rows with >=1 of each GT class and X value, using the first matching row (v1)."""
    must_include = set()
    for c in df["GT"].unique():
        must_include.add(df[df["GT"] == c].index[0])
    for v in df["X"].unique():
        must_include.add(df[df["X"] == v].index[0])
    must = list(must_include)[:n]
    must_set = set(must)
    remaining = [i for i in df.index if i not in must_set]
    fill = rng.choice(remaining, n - len(must), replace=False).tolist()
    return sorted(must + fill)


def stratified_sample_random(df: pd.DataFrame, n: int, rng) -> list[int]:
    """n rows with >=1 of each GT class and X value, picked at random (v2)."""
    must = set()
    for c in df["GT"].unique():
        must.add(int(rng.choice(df.index[df["GT"] == c].to_numpy())))
    for v in df["X"].unique():
        must.add(int(rng.choice(df.index[df["X"] == v].to_numpy())))
    must = sorted(must)[:n]
    must_set = set(must)
    remaining = np.array([i for i in df.index if i not in must_set])
    fill = rng.choice(remaining, n - len(must), replace=False).tolist()
    return sorted(must + [int(x) for x in fill])
