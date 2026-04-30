"""
11_evaluate_compare.py
=======================
Compare Experiment 1 vs Experiment 2 on Task B and Task C only.

  Experiment 1 : checkpoints/      uniform 512nt, frozen encoder
  Experiment 2 : checkpoints_bio/  biological lengths, full end-to-end

Tasks evaluated
---------------
  task_b  (CDS/non-CDS)   — Exp1 uses data/04_encoded/
                             Exp2 uses data/05_encoded_bio/
  task_c  (splice sites)  — same dirs as above

Tasks NOT evaluated here
------------------------
  task_a     : Exp1 MCC=0.877 sufficient, no Exp2 checkpoint exists
  uci_splice : replaced by cross-species generalisation

Outputs  [results_compare/]
-------
  comparison_report.txt    full human-readable side-by-side report
  comparison_summary.tsv   one row per task — import to Excel
  raw_comparison.json      all numbers for custom analysis
"""

import argparse, json
from pathlib import Path
from collections import Counter
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
model_module = import_module("06_model")
DNAClassifier = model_module.DNAClassifier


# ═══════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════

def confusion_matrix(preds, labels, n_cls):
    cm = [[0]*n_cls for _ in range(n_cls)]
    for p, l in zip(preds, labels):
        cm[l][p] += 1
    return cm

def accuracy_from_cm(cm, n):
    return sum(cm[c][c] for c in range(n)) / max(
        sum(cm[r][c] for r in range(n) for c in range(n)), 1)

def per_class_metrics(cm, n):
    precision, recall, f1, specificity = [], [], [], []
    total = sum(cm[r][c] for r in range(n) for c in range(n))
    for c in range(n):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(n)) - tp
        fn = sum(cm[c][r] for r in range(n)) - tp
        tn = total - tp - fp - fn
        p  = tp/(tp+fp) if (tp+fp) > 0 else 0.0
        r  = tp/(tp+fn) if (tp+fn) > 0 else 0.0
        f  = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        sp = tn/(tn+fp)  if (tn+fp) > 0 else 0.0
        precision.append(p); recall.append(r)
        f1.append(f);        specificity.append(sp)
    return precision, recall, f1, specificity

def macro_f1(cm, n):
    _, _, f1s, _ = per_class_metrics(cm, n)
    return sum(f1s) / n

def balanced_accuracy(cm, n):
    _, rec, _, _ = per_class_metrics(cm, n)
    return sum(rec) / n

def mcc(cm, n):
    total = sum(cm[r][c] for r in range(n) for c in range(n))
    cov_xy = cov_xx = cov_yy = 0.0
    for k in range(n):
        for l in range(n):
            for m in range(n):
                cov_xy += cm[k][k]*cm[m][l] - cm[l][k]*cm[k][m]
        s = sum(cm[k][j] for j in range(n))
        t = sum(cm[j][k] for j in range(n))
        cov_xx += s*(total-s)
        cov_yy += t*(total-t)
    if cov_xx == 0 or cov_yy == 0:
        return 0.0
    return cov_xy / (cov_xx * cov_yy)**0.5

def roc_auc_binary(probs_pos, labels):
    pos = [p for p,l in zip(probs_pos,labels) if l==1]
    neg = [p for p,l in zip(probs_pos,labels) if l==0]
    if not pos or not neg:
        return float("nan")
    concordant = sum(
        1 if ps>ns else 0.5 if ps==ns else 0
        for ps in pos for ns in neg
    )
    return concordant / (len(pos)*len(neg))

def macro_auc(all_probs, labels, n):
    aucs = []
    for c in range(n):
        pc = [all_probs[i][c] for i in range(len(labels))]
        lc = [1 if l==c else 0 for l in labels]
        a  = roc_auc_binary(pc, lc)
        if a == a:
            aucs.append(a)
    return sum(aucs)/len(aucs) if aucs else float("nan")


# ═══════════════════════════════════════════════════════════════════
# MODEL LOADER
# ═══════════════════════════════════════════════════════════════════

def load_model(ckpt_path):
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
def evaluate_model(model, encoded_path, batch_size=256):
    if not encoded_path.exists():
        return {"error": f"Not found: {encoded_path}"}

    data    = torch.load(encoded_path, weights_only=True)
    dataset = TensorDataset(data["input_ids"], data["labels"])
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=False, num_workers=0)
    n_cls   = model.classifier.num_classes

    all_preds, all_labels, all_probs = [], [], []
    for ids, lbls in loader:
        logits = model(ids)
        probs  = torch.softmax(logits, dim=-1)
        all_preds.extend(logits.argmax(dim=-1).tolist())
        all_labels.extend(lbls.tolist())
        all_probs.extend(probs.tolist())

    cm             = confusion_matrix(all_preds, all_labels, n_cls)
    prec, rec, f1s, spec = per_class_metrics(cm, n_cls)

    return {
        "n_samples":         len(all_labels),
        "num_classes":       n_cls,
        "accuracy":          accuracy_from_cm(cm, n_cls),
        "balanced_accuracy": balanced_accuracy(cm, n_cls),
        "macro_f1":          macro_f1(cm, n_cls),
        "mcc":               mcc(cm, n_cls),
        "macro_auc":         macro_auc(all_probs, all_labels, n_cls),
        "per_class": {
            str(c): {
                "precision":   prec[c],
                "recall":      rec[c],
                "f1":          f1s[c],
                "specificity": spec[c],
            }
            for c in range(n_cls)
        },
        "confusion_matrix": cm,
        "label_dist": dict(Counter(all_labels)),
        "pred_dist":  dict(Counter(all_preds)),
    }


# ═══════════════════════════════════════════════════════════════════
# EVAL CONFIG — task_b and task_c only
# ═══════════════════════════════════════════════════════════════════

LABEL_NAMES = {
    "task_b": ["non_cds", "cds"],
    "task_c": ["no_splice", "donor", "acceptor"],
}

# Structure: task -> eval_set -> exp_name -> (ckpt_dir, enc_dir, enc_file)
EVAL_CONFIG = {
    "task_b": {
        "chrom_test": {
            "exp1": ("checkpoints",     "data/05_encoded",     "task_b_test"),
            "exp2": ("checkpoints_bio", "data/05_encoded_bio", "task_b_test"),
        },
        "chrom_val": {
            "exp1": ("checkpoints",     "data/05_encoded",     "task_b_val"),
            "exp2": ("checkpoints_bio", "data/05_encoded_bio", "task_b_val"),
        },
    },
    "task_c": {
        "chrom_test": {
            "exp1": ("checkpoints",     "data/05_encoded",     "task_c_test"),
            "exp2": ("checkpoints_bio", "data/05_encoded_bio", "task_c_test"),
        },
        "chrom_val": {
            "exp1": ("checkpoints",     "data/05_encoded",     "task_c_val"),
            "exp2": ("checkpoints_bio", "data/05_encoded_bio", "task_c_val"),
        },
    },
}


# ═══════════════════════════════════════════════════════════════════
# REPORT WRITERS
# ═══════════════════════════════════════════════════════════════════

def fmt_metric(v):
    return "  N/A  " if v != v else f"{v:.4f}"

def write_comparison_report(results, path):
    with open(path, "w", encoding="utf-8") as fh:
        def w(s=""): fh.write(s + "\n")

        w("=" * 76)
        w("EXPERIMENT COMPARISON REPORT — Task B and Task C")
        w("Experiment 1 : uniform 512nt, frozen encoder       -> checkpoints/")
        w("Experiment 2 : biological lengths, full end-to-end -> checkpoints_bio/")
        w("=" * 76)

        for task, task_data in results.items():
            lnames = LABEL_NAMES.get(task)
            w()
            w("-" * 70)
            w(f"TASK : {task.upper()}")
            w("-" * 70)
            w(f"  {'Metric':<24} {'Exp1 (frozen)':>16} {'Exp2 (bio)':>16} {'Delta':>10}")
            w("  " + "-" * 66)

            for eval_set, eval_data in task_data.items():
                e1 = eval_data.get("exp1", {})
                e2 = eval_data.get("exp2", {})
                if "error" in e1 and "error" in e2:
                    continue

                w(f"\n  [{eval_set}]")
                for metric in ["accuracy", "balanced_accuracy",
                               "macro_f1", "mcc", "macro_auc"]:
                    v1    = e1.get(metric, float("nan"))
                    v2    = e2.get(metric, float("nan"))
                    delta = v2 - v1 if (v1==v1 and v2==v2) else float("nan")
                    arrow = ""
                    if delta == delta:
                        arrow = " +" if delta > 0.01 else (" -" if delta < -0.01 else "  ~")
                    w(f"  {metric:<24} {fmt_metric(v1):>16} "
                      f"{fmt_metric(v2):>16} {fmt_metric(delta):>10}{arrow}")

                # Per-class F1
                n_cls = max(len(e1.get("per_class",{})),
                            len(e2.get("per_class",{})))
                if n_cls:
                    w(f"\n  Per-class F1:")
                    for c in range(n_cls):
                        name  = lnames[c] if lnames and c < len(lnames) else str(c)
                        f1_1  = e1.get("per_class",{}).get(str(c),{}).get("f1", float("nan"))
                        f1_2  = e2.get("per_class",{}).get(str(c),{}).get("f1", float("nan"))
                        delta = f1_2 - f1_1 if (f1_1==f1_1 and f1_2==f1_2) else float("nan")
                        arrow = ""
                        if delta == delta:
                            arrow = " +" if delta > 0.01 else (" -" if delta < -0.01 else "  ~")
                        w(f"    {name:<22} {fmt_metric(f1_1):>16} "
                          f"{fmt_metric(f1_2):>16} {fmt_metric(delta):>10}{arrow}")

                # Confusion matrices
                for exp_label, exp_data in [("Exp1", e1), ("Exp2", e2)]:
                    if "confusion_matrix" in exp_data:
                        w(f"\n  Confusion matrix [{exp_label}] "
                          f"(rows=actual, cols=predicted):")
                        cm_data   = exp_data["confusion_matrix"]
                        col_names = lnames[:len(cm_data)] if lnames else \
                                    [str(i) for i in range(len(cm_data))]
                        cw = max(len(x) for x in col_names) + 2
                        w("    " + " "*cw +
                          "".join(f"{x:>{cw}}" for x in col_names))
                        for i, row in enumerate(cm_data):
                            w("    " + f"{col_names[i]:>{cw}}" +
                              "".join(f"{v:>{cw}}" for v in row))

        # Summary MCC table
        w()
        w("=" * 76)
        w("SUMMARY - MCC comparison")
        w("=" * 76)
        w(f"  {'Task':<12} {'Eval set':<16} {'Exp1 MCC':>12} "
          f"{'Exp2 MCC':>12} {'Delta':>10} {'Winner':>8}")
        w("  " + "-" * 72)
        for task, task_data in results.items():
            for eval_set, eval_data in task_data.items():
                e1 = eval_data.get("exp1", {})
                e2 = eval_data.get("exp2", {})
                m1 = e1.get("mcc", float("nan"))
                m2 = e2.get("mcc", float("nan"))
                if m1 != m1 and m2 != m2:
                    continue
                delta  = m2 - m1 if (m1==m1 and m2==m2) else float("nan")
                winner = "Exp2" if (delta==delta and delta > 0.01) else \
                         "Exp1" if (delta==delta and delta < -0.01) else "Tie"
                w(f"  {task:<12} {eval_set:<16} {fmt_metric(m1):>12} "
                  f"{fmt_metric(m2):>12} {fmt_metric(delta):>10} {winner:>8}")
        w()
        w("  + = Experiment 2 better by > 0.01")
        w("  - = Experiment 1 better by > 0.01")
        w("  ~ = Essentially equal (within 0.01)")
        w("=" * 76)

    print(f"[11] Comparison report -> {path}")


def write_summary_tsv(results, path):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("task\teval_set\t"
                 "exp1_accuracy\texp1_bal_acc\texp1_f1\texp1_mcc\texp1_auc\t"
                 "exp2_accuracy\texp2_bal_acc\texp2_f1\texp2_mcc\texp2_auc\t"
                 "delta_mcc\twinner\n")
        for task, task_data in results.items():
            for eval_set, eval_data in task_data.items():
                e1 = eval_data.get("exp1", {})
                e2 = eval_data.get("exp2", {})
                def g(d, k): return d.get(k, float("nan"))
                m1    = g(e1, "mcc")
                m2    = g(e2, "mcc")
                delta = m2 - m1 if (m1==m1 and m2==m2) else float("nan")
                winner = "Exp2" if (delta==delta and delta > 0.01) else \
                         "Exp1" if (delta==delta and delta < -0.01) else "Tie"
                def fmt(v): return f"{v:.6f}" if v==v else "nan"
                fh.write(
                    f"{task}\t{eval_set}\t"
                    f"{fmt(g(e1,'accuracy'))}\t{fmt(g(e1,'balanced_accuracy'))}\t"
                    f"{fmt(g(e1,'macro_f1'))}\t{fmt(g(e1,'mcc'))}\t"
                    f"{fmt(g(e1,'macro_auc'))}\t"
                    f"{fmt(g(e2,'accuracy'))}\t{fmt(g(e2,'balanced_accuracy'))}\t"
                    f"{fmt(g(e2,'macro_f1'))}\t{fmt(g(e2,'mcc'))}\t"
                    f"{fmt(g(e2,'macro_auc'))}\t"
                    f"{fmt(delta)}\t{winner}\n"
                )
    print(f"[11] Summary TSV -> {path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Compare Exp1 vs Exp2 on Task B and Task C"
    )
    ap.add_argument("--out_dir",    default="results_compare")
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[11] Evaluating Task B and Task C — Exp1 vs Exp2")
    print("     task_a     : SKIPPED (Exp1 MCC=0.877 sufficient)")
    print("     uci_splice : SKIPPED (cross-species generalisation planned)")

    all_results = {}

    for task, eval_sets in EVAL_CONFIG.items():
        print(f"\n[11] Task: {task}")
        task_results = {}

        for eval_set, exp_configs in eval_sets.items():
            eval_results = {}

            for exp_name, (ckpt_folder, enc_folder, enc_file) in exp_configs.items():
                ckpt_path = Path(ckpt_folder) / f"{task}_best.pt"
                enc_path  = Path(enc_folder)  / f"{enc_file}.pt"

                if not ckpt_path.exists():
                    print(f"  [{exp_name}] No checkpoint at {ckpt_path} — skipping")
                    eval_results[exp_name] = {"error": f"No checkpoint: {ckpt_path}"}
                    continue

                print(f"  [{exp_name}] {eval_set} ...", end=" ", flush=True)
                model, _    = load_model(ckpt_path)
                metrics     = evaluate_model(model, enc_path, args.batch_size)

                if "error" not in metrics:
                    print(f"acc={metrics['accuracy']:.4f}  "
                          f"f1={metrics['macro_f1']:.4f}  "
                          f"mcc={metrics['mcc']:.4f}")
                else:
                    print(metrics["error"])

                eval_results[exp_name] = metrics

            task_results[eval_set] = eval_results
        all_results[task] = task_results

    write_comparison_report(all_results, out_dir / "comparison_report.txt")
    write_summary_tsv(all_results,       out_dir / "comparison_summary.tsv")

    def serialise(obj):
        if isinstance(obj, float): return round(obj, 6)
        if isinstance(obj, dict) and all(isinstance(k, int) for k in obj):
            return {str(k): v for k, v in obj.items()}
        return obj

    with open(out_dir / "raw_comparison.json", "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, default=serialise, indent=2)

    print(f"\n[11] Raw JSON -> {out_dir}/raw_comparison.json")
    print("[11] Done.")


if __name__ == "__main__":
    main()