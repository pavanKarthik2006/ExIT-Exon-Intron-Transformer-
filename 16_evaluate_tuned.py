"""
17_evaluate_tuned.py
=====================
Evaluates the tuned checkpoints from Script 15 on the test set
and compares MCC against the original experiment baselines.

Each task uses the correct encoded directory automatically:
  task_a -> data/05_encoded       (512nt, 170 tokens)
  task_b -> data/05_encoded_bio   (333nt, 111 tokens)
  task_c -> data/05_encoded_bio   (200nt,  66 tokens)

Outputs  [results_tuned/]
--------------------------
  tuned_eval_report.txt    human-readable MCC comparison
  tuned_eval_summary.tsv   one row per task with all metrics

Usage
-----
  
  python scripts\\16_evaluate_tuned.py --ckpt_dir checkpoints_tuned
"""

import argparse, sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
model_module   = import_module("06_model")
DNAClassifier  = model_module.DNAClassifier


# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

# Which encoded dir and test file each task uses
TASK_CFG = {
    "task_a": {
        "encoded_dir": "data/05_encoded",
        "test_file":   "task_a_test.pt",
        "n_classes":   2,
        "labels":      {0: "intron", 1: "exon"},
    },
    "task_b": {
        "encoded_dir": "data/05_encoded_bio",
        "test_file":   "task_b_test.pt",
        "n_classes":   2,
        "labels":      {0: "non_cds", 1: "cds"},
    },
    "task_c": {
        "encoded_dir": "data/05_encoded_bio",
        "test_file":   "task_c_test.pt",
        "n_classes":   3,
        "labels":      {0: "no_splice", 1: "donor", 2: "acceptor"},
    },
}

# Baselines from best previous experiments
BASELINES = {
    "task_a": {"mcc": 0.8719, "f1": 0.9343, "acc": 0.9343, "exp": "Exp1"},
    "task_b": {"mcc": 0.4452, "f1": 0.5566, "acc": 0.5892, "exp": "Exp2"},
    "task_c": {"mcc": 0.4972, "f1": 0.6354, "acc": 0.6535, "exp": "Exp1"},
}


# ═══════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════

def confusion_matrix(preds, labels, n_cls):
    cm = [[0]*n_cls for _ in range(n_cls)]
    for p, l in zip(preds, labels):
        cm[l][p] += 1
    return cm

def accuracy(cm, n_cls):
    correct = sum(cm[i][i] for i in range(n_cls))
    total   = sum(cm[i][j] for i in range(n_cls) for j in range(n_cls))
    return correct / total if total > 0 else 0.0

def balanced_accuracy(cm, n_cls):
    recalls = []
    for c in range(n_cls):
        tp = cm[c][c]
        fn = sum(cm[c][j] for j in range(n_cls)) - tp
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    return sum(recalls) / n_cls

def macro_f1(cm, n_cls):
    f1s = []
    for c in range(n_cls):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(n_cls)) - tp
        fn = sum(cm[c][r] for r in range(n_cls)) - tp
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2*p*r / (p+r) if (p+r) > 0 else 0.0)
    return sum(f1s) / n_cls

def mcc(cm, n_cls):
    total = sum(cm[i][j] for i in range(n_cls) for j in range(n_cls))
    xy = xx = yy = 0.0
    for k in range(n_cls):
        for l in range(n_cls):
            for m in range(n_cls):
                xy += cm[k][k]*cm[m][l] - cm[l][k]*cm[k][m]
        s = sum(cm[k][j] for j in range(n_cls))
        t = sum(cm[j][k] for j in range(n_cls))
        xx += s * (total - s)
        yy += t * (total - t)
    return xy / (xx * yy) ** 0.5 if xx > 0 and yy > 0 else 0.0

def per_class_metrics(cm, n_cls):
    results = {}
    for c in range(n_cls):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(n_cls)) - tp
        fn = sum(cm[c][r] for r in range(n_cls)) - tp
        tn = sum(cm[i][j] for i in range(n_cls)
                 for j in range(n_cls)) - tp - fp - fn
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2*p*r / (p+r) if (p+r) > 0 else 0.0
        results[c] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                      "precision": p, "recall": r, "f1": f1}
    return results


# ═══════════════════════════════════════════════════════════════════
# MODEL LOADER
# ═══════════════════════════════════════════════════════════════════

def load_model(ckpt_path):
    ckpt  = torch.load(str(ckpt_path), map_location="cpu",
                       weights_only=False)
    cfg   = ckpt["config"]
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


# ═══════════════════════════════════════════════════════════════════
# EVALUATE
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, test_path, batch_size, n_cls):
    data    = torch.load(str(test_path), weights_only=True)
    dataset = TensorDataset(data["input_ids"], data["labels"])
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds, labels = [], []
    for ids, lbl in loader:
        logits = model(ids)
        preds.extend(logits.argmax(-1).tolist())
        labels.extend(lbl.tolist())

    cm_  = confusion_matrix(preds, labels, n_cls)
    return {
        "n_samples":        len(labels),
        "accuracy":         accuracy(cm_, n_cls),
        "balanced_accuracy":balanced_accuracy(cm_, n_cls),
        "macro_f1":         macro_f1(cm_, n_cls),
        "mcc":              mcc(cm_, n_cls),
        "confusion_matrix": cm_,
        "per_class":        per_class_metrics(cm_, n_cls),
    }


# ═══════════════════════════════════════════════════════════════════
# WRITERS
# ═══════════════════════════════════════════════════════════════════

def write_tsv(all_metrics, out_dir):
    path = out_dir / "tuned_eval_summary.tsv"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("task\texp\tbaseline_mcc\ttuned_mcc\tdelta_mcc\t"
                 "baseline_f1\ttuned_f1\tdelta_f1\t"
                 "tuned_acc\ttuned_bal_acc\tn_samples\n")
        for task, m in all_metrics.items():
            base  = BASELINES[task]
            delta_mcc = m["mcc"] - base["mcc"]
            delta_f1  = m["macro_f1"] - base["f1"]
            fh.write(
                f"{task}\t{base['exp']}\t"
                f"{base['mcc']:.6f}\t{m['mcc']:.6f}\t{delta_mcc:+.6f}\t"
                f"{base['f1']:.6f}\t{m['macro_f1']:.6f}\t{delta_f1:+.6f}\t"
                f"{m['accuracy']:.6f}\t{m['balanced_accuracy']:.6f}\t"
                f"{m['n_samples']}\n"
            )
    print(f"[17] tuned_eval_summary.tsv -> {path}")


def write_report(all_metrics, out_dir):
    path = out_dir / "tuned_eval_report.txt"
    with open(path, "w", encoding="utf-8") as fh:
        def w(s=""): fh.write(s + "\n")

        w("=" * 68)
        w("SCRIPT 17 — TUNED CHECKPOINT EVALUATION ON TEST SET")
        w("Checkpoints from: checkpoints_tuned/")
        w("Split: Chromosome-based test set (chr21, chr22, chrX, chrY)")
        w("=" * 68)

        # MCC comparison table
        w()
        w("MCC COMPARISON — Baseline vs Tuned")
        w("-" * 68)
        w(f"  {'Task':<12} {'Exp':>6} {'Baseline MCC':>14} "
          f"{'Tuned MCC':>12} {'Delta':>8}  Result")
        w("  " + "-" * 60)
        for task, m in all_metrics.items():
            base  = BASELINES[task]
            delta = m["mcc"] - base["mcc"]
            if delta > 0.005:
                result = "IMPROVED ▲"
            elif delta >= -0.005:
                result = "SIMILAR  ~"
            else:
                result = "WORSE    ▼"
            w(f"  {task:<12} {base['exp']:>6} {base['mcc']:>14.4f} "
              f"{m['mcc']:>12.4f} {delta:>+8.4f}  {result}")

        # Full metrics table
        w()
        w("FULL METRICS — Tuned Models")
        w("-" * 68)
        w(f"  {'Task':<12} {'MCC':>8} {'Macro F1':>10} "
          f"{'Bal Acc':>10} {'Accuracy':>10} {'N Test':>8}")
        w("  " + "-" * 60)
        for task, m in all_metrics.items():
            w(f"  {task:<12} {m['mcc']:>8.4f} {m['macro_f1']:>10.4f} "
              f"{m['balanced_accuracy']:>10.4f} {m['accuracy']:>10.4f} "
              f"{m['n_samples']:>8,}")

        # Per-class breakdown
        w()
        w("PER-CLASS BREAKDOWN")
        w("-" * 68)
        for task, m in all_metrics.items():
            labels = TASK_CFG[task]["labels"]
            w()
            w(f"  {task.upper()}")
            w(f"  {'Class':<16} {'Precision':>10} {'Recall':>10} "
              f"{'F1':>8} {'TP':>8} {'FP':>8} {'FN':>8}")
            w("  " + "-" * 62)
            for c, pc in m["per_class"].items():
                name = labels.get(c, str(c))
                w(f"  {name:<16} {pc['precision']:>10.4f} "
                  f"{pc['recall']:>10.4f} {pc['f1']:>8.4f} "
                  f"{pc['tp']:>8} {pc['fp']:>8} {pc['fn']:>8}")

        # Confusion matrices
        w()
        w("CONFUSION MATRICES")
        w("-" * 68)
        for task, m in all_metrics.items():
            labels = TASK_CFG[task]["labels"]
            n_cls  = TASK_CFG[task]["n_classes"]
            cm_    = m["confusion_matrix"]
            w()
            w(f"  {task.upper()}  (rows=actual, cols=predicted)")
            header = "  " + " "*14 + "".join(
                f"{labels[c]:>12}" for c in range(n_cls))
            w(header)
            for i in range(n_cls):
                row = f"  {labels[i]:<14}" + "".join(
                    f"{cm_[i][j]:>12,}" for j in range(n_cls))
                w(row)

        w()
        w("=" * 68)

    print(f"[17] tuned_eval_report.txt  -> {path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate tuned checkpoints on test set"
    )
    ap.add_argument("--ckpt_dir",   default="checkpoints_tuned",
                    help="Directory with tuned checkpoints (default: checkpoints_tuned)")
    ap.add_argument("--out_dir",    default="results_tuned",
                    help="Where to save evaluation outputs (default: results_tuned)")
    ap.add_argument("--tasks",      nargs="+",
                    default=["task_a", "task_b", "task_c"],
                    choices=["task_a", "task_b", "task_c"])
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[17] Evaluating Tuned Checkpoints on Test Set")
    print(f"     ckpt_dir  : {ckpt_dir}")
    print(f"     out_dir   : {out_dir}")
    print(f"     tasks     : {args.tasks}")

    all_metrics = {}

    for task in args.tasks:
        cfg       = TASK_CFG[task]
        ckpt_path = ckpt_dir / f"{task}_best.pt"
        test_path = Path(cfg["encoded_dir"]) / cfg["test_file"]

        print(f"\n  [{task}]")
        print(f"    checkpoint : {ckpt_path}")
        print(f"    test data  : {test_path}")

        if not ckpt_path.exists():
            print(f"    [SKIP] checkpoint not found: {ckpt_path}")
            continue
        if not test_path.exists():
            print(f"    [SKIP] test file not found: {test_path}")
            continue

        model, ckpt_info = load_model(ckpt_path)
        print(f"    model      : d_model={ckpt_info['config']['d_model']}  "
              f"n_layers={ckpt_info['config']['n_layers']}  "
              f"n_heads={ckpt_info['config']['n_heads']}  "
              f"epoch={ckpt_info.get('epoch','?')}")

        m = evaluate(model, test_path, args.batch_size, cfg["n_classes"])
        all_metrics[task] = m

        base  = BASELINES[task]
        delta = m["mcc"] - base["mcc"]
        print(f"    baseline MCC : {base['mcc']:.4f}  ({base['exp']})")
        print(f"    tuned MCC    : {m['mcc']:.4f}  (delta {delta:+.4f})")
        print(f"    macro F1     : {m['macro_f1']:.4f}")
        print(f"    accuracy     : {m['accuracy']:.4f}")
        print(f"    n_test       : {m['n_samples']:,}")

    if not all_metrics:
        print("\n[ERROR] No tasks were evaluated. Check checkpoint paths.")
        sys.exit(1)

    print("\n[17] Writing outputs ...")
    write_tsv(all_metrics, out_dir)
    write_report(all_metrics, out_dir)

    # Final summary
    print(f"\n{'='*55}")
    print(f"[17] FINAL SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Task':<12} {'Baseline':>10} {'Tuned':>10} "
          f"{'Delta':>8}  Status")
    print(f"  {'-'*52}")
    for task, m in all_metrics.items():
        base  = BASELINES[task]["mcc"]
        delta = m["mcc"] - base
        status = ("IMPROVED ▲" if delta >  0.005 else
                  "SIMILAR  ~" if delta >= -0.005 else
                  "WORSE    ▼")
        print(f"  {task:<12} {base:>10.4f} {m['mcc']:>10.4f} "
              f"{delta:>+8.4f}  {status}")

    print(f"\n[17] Done. Results -> {out_dir}/")


if __name__ == "__main__":
    main()