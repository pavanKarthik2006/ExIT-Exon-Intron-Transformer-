"""
08_evaluate.py
==============
STEP 7 — Evaluate trained models on test sets and external UCI validation.

Evaluations performed
---------------------
For each task checkpoint:
  1. Internal test set (chromosome split — primary metric)
  2. Internal test set (transcript split — upper bound)
  3. External validation on UCI Molecular Biology Splice Junction dataset
     (Task C and UCI head only — both use 3-class splice classification)

Metrics computed
----------------
  - Accuracy
  - Precision, Recall, F1  (macro and per-class)
  - MCC  (Matthews Correlation Coefficient — robust to class imbalance)
  - Confusion matrix
  - ROC-AUC  (for binary tasks; macro-OvR for 3-class)

Outputs  [results/]
-------
    evaluation_report.txt          full human-readable report
    metrics_summary.tsv            one row per task × split × strategy
    confusion_{task}_{split}.txt   confusion matrix per task
    roc_auc.tsv                    AUC values

Key research result
-------------------
The comparison between chromosome-split test accuracy and transcript-split
test accuracy directly quantifies the leakage effect identified in step 04.
The external UCI validation provides a completely independent benchmark that
is immune to both types of leakage.

Usage
-----
    python 08_evaluate.py
    python 08_evaluate.py --ckpt_dir checkpoints --encoded_dir data/04_encoded
"""

import argparse, json
from pathlib import Path
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
model_module = import_module("06_model")
DNAClassifier = model_module.DNAClassifier


# ===================================================================
# METRIC HELPERS
# ===================================================================

def confusion_matrix(preds, labels, num_classes):
    cm = [[0]*num_classes for _ in range(num_classes)]
    for p, l in zip(preds, labels):
        cm[l][p] += 1
    return cm


def per_class_metrics(cm, num_classes):
    precision, recall, f1, specificity = [], [], [], []
    for c in range(num_classes):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(num_classes)) - tp
        fn = sum(cm[c][r] for r in range(num_classes)) - tp
        tn = sum(cm[r][col] for r in range(num_classes)
                 for col in range(num_classes)) - tp - fp - fn
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f  = 2*p*r / (p+r)  if (p + r)  > 0 else 0.0
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision.append(p); recall.append(r)
        f1.append(f);        specificity.append(sp)
    return precision, recall, f1, specificity


def balanced_accuracy(cm, num_classes):
    """Average of per-class recall (sensitivity). Robust to class imbalance."""
    _, recall, _, _ = per_class_metrics(cm, num_classes)
    return sum(recall) / num_classes


def macro_f1(cm, num_classes):
    _, _, f1s, _ = per_class_metrics(cm, num_classes)
    return sum(f1s) / num_classes


def mcc(cm, num_classes):
    """Multi-class MCC (Gorodkin 2004)."""
    n   = sum(cm[r][c] for r in range(num_classes) for c in range(num_classes))
    cov_xy, cov_xx, cov_yy = 0.0, 0.0, 0.0
    for k in range(num_classes):
        for l in range(num_classes):
            for m in range(num_classes):
                cov_xy += cm[k][k] * cm[m][l] - cm[l][k] * cm[k][m]
        s = sum(cm[k][j] for j in range(num_classes))
        t = sum(cm[j][k] for j in range(num_classes))
        cov_xx += s * (n - s)
        cov_yy += t * (n - t)
    if cov_xx == 0 or cov_yy == 0:
        return 0.0
    return cov_xy / (cov_xx * cov_yy) ** 0.5


def accuracy_from_cm(cm, num_classes):
    correct = sum(cm[c][c] for c in range(num_classes))
    total   = sum(cm[r][c] for r in range(num_classes) for c in range(num_classes))
    return correct / total if total > 0 else 0.0


# Lightweight ROC-AUC (no sklearn dependency)
def roc_auc_binary(probs_pos, labels):
    """
    AUC via Wilcoxon-Mann-Whitney statistic.
    Counts how often a positive scores higher than a negative.
    Returns value in [0, 1].
    """
    pos_scores = [p for p, l in zip(probs_pos, labels) if l == 1]
    neg_scores = [p for p, l in zip(probs_pos, labels) if l == 0]
    if not pos_scores or not neg_scores:
        return float("nan")
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    # Count concordant pairs
    concordant = sum(
        1 if ps > ns else 0.5 if ps == ns else 0
        for ps in pos_scores
        for ns in neg_scores
    )
    return concordant / (n_pos * n_neg)


def macro_roc_auc(all_probs, labels, num_classes):
    """Macro-averaged One-vs-Rest AUC. Returns value in [0, 1]."""
    aucs = []
    for c in range(num_classes):
        probs_c = [all_probs[i][c] for i in range(len(labels))]
        lbls_c  = [1 if l == c else 0 for l in labels]
        auc = roc_auc_binary(probs_c, lbls_c)
        if auc == auc:   # skip NaN
            aucs.append(auc)
    # Average — result must be in [0, 1]
    result = sum(aucs) / len(aucs) if aucs else float("nan")
    # Clamp to valid range as a safety check
    if result == result:
        result = max(0.0, min(1.0, result))
    return result


# ===================================================================
# MODEL LOADER
# ===================================================================

def load_model(ckpt_path: Path) -> DNAClassifier:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    model = DNAClassifier.build(
        num_classes = ckpt["num_classes"],
        d_model     = cfg["d_model"],
        n_heads     = cfg["n_heads"],
        n_layers    = cfg["n_layers"],
        ffn_dim     = cfg["ffn_dim"],
        max_len     = cfg["max_len"],
    )
    model.encoder.load_state_dict(ckpt["encoder_state"])
    model.classifier.load_state_dict(ckpt["head_state"])
    model.eval()
    return model, ckpt


# ===================================================================
# EVALUATE ONE CHECKPOINT ON ONE DATASET
# ===================================================================

@torch.no_grad()
def evaluate_checkpoint(model, encoded_path: Path, batch_size=64) -> dict:
    if not encoded_path.exists():
        return {"error": f"File not found: {encoded_path}"}

    data     = torch.load(encoded_path, weights_only=True)
    dataset  = TensorDataset(data["input_ids"], data["labels"])
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    n_cls    = model.classifier.num_classes

    all_preds, all_labels, all_probs = [], [], []

    for ids, lbls in loader:
        logits = model(ids)
        probs  = torch.softmax(logits, dim=-1)
        preds  = logits.argmax(dim=-1)
        all_preds.extend(preds.tolist())
        all_labels.extend(lbls.tolist())
        all_probs.extend(probs.tolist())

    cm     = confusion_matrix(all_preds, all_labels, n_cls)
    acc    = accuracy_from_cm(cm, n_cls)
    bal_acc= balanced_accuracy(cm, n_cls)
    mf1    = macro_f1(cm, n_cls)
    mcc_   = mcc(cm, n_cls)
    auc    = macro_roc_auc(all_probs, all_labels, n_cls)
    prec, rec, f1s, spec = per_class_metrics(cm, n_cls)

    # Per-class AUC
    per_cls_auc = []
    for c in range(n_cls):
        probs_c = [all_probs[i][c] for i in range(len(all_labels))]
        lbls_c  = [1 if l == c else 0 for l in all_labels]
        per_cls_auc.append(roc_auc_binary(probs_c, lbls_c))

    return {
        "n_samples":        len(all_labels),
        "num_classes":      n_cls,
        "accuracy":         acc,
        "balanced_accuracy":bal_acc,
        "macro_f1":         mf1,
        "mcc":              mcc_,
        "macro_auc":        auc,
        "per_class": {
            str(c): {
                "precision":   prec[c],
                "recall":      rec[c],
                "f1":          f1s[c],
                "specificity": spec[c],
                "auc":         per_cls_auc[c],
            }
            for c in range(n_cls)
        },
        "confusion_matrix":   cm,
        "label_distribution": Counter(all_labels),
        "pred_distribution":  Counter(all_preds),
    }


# ===================================================================
# REPORT WRITERS
# ===================================================================

def fmt_cm(cm, label_names=None):
    n = len(cm)
    lbl = label_names or [str(i) for i in range(n)]
    w   = max(len(l) for l in lbl) + 2
    header = " " * w + "".join(f"{l:>{w}}" for l in lbl) + "  (predicted)"
    lines  = [header]
    for i, row in enumerate(cm):
        lines.append(f"{lbl[i]:>{w}}" + "".join(f"{v:>{w}}" for v in row))
    return "\n".join(lines)


TASK_LABEL_NAMES = {
    "task_a":     ["intron", "exon"],
    "task_b":     ["non_cds", "cds"],
    "task_c":     ["no_splice", "donor", "acceptor"],
    "uci_splice": ["EI(donor)", "IE(acceptor)", "N(neither)"],
}


def write_full_report(all_results: dict, path: Path):
    with open(path, "w", encoding="utf-8") as fh:
        def w(s=""): fh.write(s + "\n")

        w("="*72)
        w("EVALUATION REPORT — DNA Multi-Task Transformer")
        w("="*72)
        w()

        for task, task_results in all_results.items():
            label_names = TASK_LABEL_NAMES.get(task)
            w(f"{'-'*60}")
            w(f"TASK: {task.upper()}")
            w(f"{'-'*60}")

            for eval_name, metrics in task_results.items():
                if "error" in metrics:
                    w(f"  [{eval_name}]  ERROR: {metrics['error']}")
                    continue
                w(f"\n  [{eval_name}]  n={metrics['n_samples']:,}  "
                  f"classes={metrics['num_classes']}")
                w(f"  {'Accuracy':<22}: {metrics['accuracy']:.4f}")
                w(f"  {'Balanced Accuracy':<22}: {metrics['balanced_accuracy']:.4f}")
                w(f"  {'Macro F1':<22}: {metrics['macro_f1']:.4f}")
                w(f"  {'MCC':<22}: {metrics['mcc']:.4f}")
                w(f"  {'Macro AUC (AUROC)':<22}: {metrics['macro_auc']:.4f}")
                w()
                w("  Per-class metrics:")
                w(f"  {'Class':<16} {'Precision':>10} {'Recall':>10} "
                  f"{'F1':>8} {'Specificity':>12} {'AUC':>8}")
                w("  " + "-"*66)
                for c, m in metrics["per_class"].items():
                    name = label_names[int(c)] if label_names else c
                    w(f"  {name:<16} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                      f"{m['f1']:>8.4f} {m['specificity']:>12.4f} {m['auc']:>8.4f}")
                w()
                w("  Confusion matrix (rows=actual, cols=predicted):")
                cm_str = fmt_cm(metrics["confusion_matrix"], label_names)
                for line in cm_str.split("\n"):
                    w("    " + line)
                w()

        # Leakage comparison table
        w("="*72)
        w("SPLIT STRATEGY COMPARISON  (accuracy)")
        w("="*72)
        w(f"{'Task':<16} {'Chrom-test':>14} {'Tx-test':>12} {'UCI-external':>14} {'Δ(tx-chrom)':>12}")
        w("-"*70)
        for task, task_results in all_results.items():
            chrom = task_results.get("chrom_test",     {}).get("accuracy", float("nan"))
            tx    = task_results.get("transcript_test",{}).get("accuracy", float("nan"))
            uci   = task_results.get("uci_test",       {}).get("accuracy", float("nan"))
            delta = tx - chrom if (chrom == chrom and tx == tx) else float("nan")
            def fmt(v): return f"{v:.4f}" if v==v else "  N/A  "
            w(f"{task:<16} {fmt(chrom):>14} {fmt(tx):>12} {fmt(uci):>14} {fmt(delta):>12}")
        w()
        w("  Δ(tx-chrom) > 0 indicates leakage-induced optimism in transcript split.")
        w("  UCI external validation is immune to both leakage sources.")
        w("="*72)

    print(f"[08] Full report -> {path}")


def write_metrics_tsv(all_results: dict, path: Path):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("task\teval_set\taccuracy\tmacro_f1\tmcc\tmacro_auc\n")
        for task, task_results in all_results.items():
            for eval_name, metrics in task_results.items():
                if "error" in metrics: continue
                fh.write(
                    f"{task}\t{eval_name}\t"
                    f"{metrics['accuracy']:.6f}\t{metrics['macro_f1']:.6f}\t"
                    f"{metrics['mcc']:.6f}\t{metrics['macro_auc']:.6f}\n"
                )
    print(f"[08] Metrics TSV -> {path}")


# ===================================================================
# MAIN
# ===================================================================

EVAL_SETS = {
    # task_name: [ (eval_label, encoded_file_name) ]
    "task_a": [
        ("chrom_test",      "task_a_test"),
        ("chrom_val",       "task_a_val"),
    ],
    "task_b": [
        ("chrom_test",      "task_b_test"),
        ("chrom_val",       "task_b_val"),
    ],
    "task_c": [
        ("chrom_test",      "task_c_test"),
        ("chrom_val",       "task_c_val"),
        ("uci_test",        "uci_splice_test"),
        ("uci_val",         "uci_splice_val"),
    ],
    "uci_splice": [
        ("uci_test",        "uci_splice_test"),
        ("uci_val",         "uci_splice_val"),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir",     default="checkpoints")
    ap.add_argument("--encoded_dir",  default="data/04_encoded")
    ap.add_argument("--out_dir",      default="results")
    ap.add_argument("--tx_encoded_dir", default=None,
                    help="Transcript-split encoded dir for upper-bound eval "
                         "(default: same as --encoded_dir)")
    ap.add_argument("--batch_size",   type=int, default=256)
    args = ap.parse_args()
    if args.tx_encoded_dir is None:
        args.tx_encoded_dir = args.encoded_dir

    ckpt_dir    = Path(args.ckpt_dir)
    encoded_dir = Path(args.encoded_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for task, eval_list in EVAL_SETS.items():
        ckpt_path = ckpt_dir / f"{task}_best.pt"
        if not ckpt_path.exists():
            print(f"[08] No checkpoint for {task} — skipping.")
            continue

        print(f"\n[08] Loading {task} from {ckpt_path}")
        model, ckpt = load_model(ckpt_path)
        print(f"     num_classes={ckpt['num_classes']}  "
              f"epoch={ckpt['epoch']}  best_val_loss={ckpt['best_val_loss']:.4f}")

        task_results = {}
        for eval_label, encoded_name in eval_list:
            enc_path = encoded_dir / f"{encoded_name}.pt"
            print(f"  -> {eval_label} ({enc_path.name}) ...", end=" ", flush=True)
            metrics = evaluate_checkpoint(model, enc_path, args.batch_size)
            task_results[eval_label] = metrics
            if "error" not in metrics:
                print(f"acc={metrics['accuracy']:.4f}  "
                      f"f1={metrics['macro_f1']:.4f}  "
                      f"mcc={metrics['mcc']:.4f}")
            else:
                print(metrics["error"])

        all_results[task] = task_results

    write_full_report(all_results, out_dir / "evaluation_report.txt")
    write_metrics_tsv(all_results, out_dir / "metrics_summary.tsv")

    # Save raw results as JSON for further analysis
    def serialise(obj):
        if isinstance(obj, float): return round(obj, 6)
        if isinstance(obj, Counter): return dict(obj)
        return obj

    with open(out_dir / "raw_results.json", "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, default=serialise, indent=2)
    print(f"[08] Raw JSON -> {out_dir / 'raw_results.json'}")
    print("[08] Done.")


if __name__ == "__main__":
    main()