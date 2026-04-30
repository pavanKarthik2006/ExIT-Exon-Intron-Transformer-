"""
save_exp1_results.py
====================
Saves Experiment 1 results directly from known values
without rerunning the model. Fixes the encoding error in 08_evaluate.py
by writing all files with explicit utf-8 encoding.
"""

import json
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 1 RESULTS  — paste your numbers here
# ═══════════════════════════════════════════════════════════════════

RESULTS = {
    "task_a": {
        "chrom_test": {
            "n_samples": 208405,
            "accuracy":          0.9343,
            "balanced_accuracy": 0.9364,
            "macro_f1":          0.9343,
            "mcc":               0.8719,
            "macro_auc":         "pending_fix",   # AUC bug fixed in 08b
            "epoch":             5,
            "best_val_loss":     0.1527,
        },
        "chrom_val": {
            "n_samples": 270054,
            "accuracy":          0.9381,
            "balanced_accuracy": 0.9401,
            "macro_f1":          0.9381,
            "mcc":               0.8790,
            "macro_auc":         "pending_fix",
            "epoch":             5,
            "best_val_loss":     0.1527,
        },
    },
    "task_b": {
        "chrom_test": {
            "n_samples": 115993,
            "accuracy":          0.5892,
            "balanced_accuracy": 0.6029,
            "macro_f1":          0.5566,
            "mcc":               0.2541,
            "macro_auc":         "pending_fix",
            "epoch":             3,
            "best_val_loss":     0.6420,
            "note": "Poor — uniform 512nt + frozen encoder. Exp2 expected to improve."
        },
        "chrom_val": {
            "n_samples": 150937,
            "accuracy":          0.6100,
            "balanced_accuracy": 0.6135,
            "macro_f1":          0.5761,
            "mcc":               0.2783,
            "macro_auc":         "pending_fix",
            "epoch":             3,
            "best_val_loss":     0.6420,
        },
    },
    "task_c": {
        "chrom_test": {
            "n_samples": 295535,
            "accuracy":          0.6535,
            "balanced_accuracy": 0.6536,
            "macro_f1":          0.6354,
            "mcc":               0.4972,
            "macro_auc":         "pending_fix",
            "epoch":             4,
            "best_val_loss":     0.5175,
            "note": "Moderate — 512nt padding dominated 200nt splice windows. Exp2 corrects this."
        },
        "chrom_val": {
            "n_samples": 381848,
            "accuracy":          0.6544,
            "balanced_accuracy": 0.6544,
            "macro_f1":          0.6330,
            "mcc":               0.5015,
            "macro_auc":         "pending_fix",
        },
        "uci_test": {
            "n_samples": 319,
            "accuracy":          0.2665,
            "balanced_accuracy": 0.3328,
            "macro_f1":          0.1829,
            "mcc":               0.0096,
            "macro_auc":         "pending_fix",
            "note": "Failed — domain mismatch: 512nt encoder vs 60nt UCI sequences."
        },
        "uci_val": {
            "n_samples": 319,
            "accuracy":          0.3103,
            "balanced_accuracy": 0.3436,
            "macro_f1":          0.2020,
            "mcc":               0.0309,
            "macro_auc":         "pending_fix",
        },
    },
}

LABEL_NAMES = {
    "task_a": ["intron", "exon"],
    "task_b": ["non_cds", "cds"],
    "task_c": ["no_splice", "donor", "acceptor"],
}

TASK_NOTES = {
    "task_a": "Encoder + head trained together. 512nt -> 170 codon tokens.",
    "task_b": "Encoder FROZEN after Task A. 512nt -> 170 tokens. "
              "Excessive padding harmed CDS signal detection.",
    "task_c": "Encoder FROZEN. 200nt splice windows padded to 512nt. "
              "GT/AG signals diluted. UCI validation failed due to length mismatch.",
}


# ═══════════════════════════════════════════════════════════════════
# WRITERS
# ═══════════════════════════════════════════════════════════════════

def write_report(results, path):
    with open(path, "w", encoding="utf-8") as fh:
        def w(s=""): fh.write(s + "\n")

        w("=" * 72)
        w("EXPERIMENT 1 RESULTS — DNA Multi-Task Transformer")
        w("Strategy: uniform 512nt windows, encoder frozen after Task A")
        w("=" * 72)

        for task, task_data in results.items():
            w()
            w("─" * 60)
            w(f"TASK: {task.upper()}")
            w(f"Note: {TASK_NOTES.get(task, '')}")
            w("─" * 60)

            for eval_set, m in task_data.items():
                w(f"\n  [{eval_set}]  n={m['n_samples']:,}")
                w(f"  {'Accuracy':<22}: {m['accuracy']:.4f}")
                w(f"  {'Balanced Accuracy':<22}: {m['balanced_accuracy']:.4f}")
                w(f"  {'Macro F1':<22}: {m['macro_f1']:.4f}")
                w(f"  {'MCC':<22}: {m['mcc']:.4f}")
                if "note" in m:
                    w(f"  Note: {m['note']}")

        w()
        w("=" * 72)
        w("SUMMARY TABLE")
        w("=" * 72)
        w(f"  {'Task':<12} {'Eval set':<16} {'Accuracy':>10} "
          f"{'Bal.Acc':>10} {'F1':>8} {'MCC':>8}")
        w("  " + "-" * 66)
        for task, task_data in results.items():
            for eval_set, m in task_data.items():
                w(f"  {task:<12} {eval_set:<16} "
                  f"{m['accuracy']:>10.4f} {m['balanced_accuracy']:>10.4f} "
                  f"{m['macro_f1']:>8.4f} {m['mcc']:>8.4f}")

        w()
        w("=" * 72)
        w("INTERPRETATION")
        w("=" * 72)
        w("  Task A  MCC=0.8719  EXCELLENT — strong exon/intron discrimination")
        w("  Task B  MCC=0.2541  POOR      — frozen encoder + 512nt padding")
        w("                                  Experiment 2 targets improvement here")
        w("  Task C  MCC=0.4972  MODERATE  — splice windows padded to 512nt")
        w("                                  Experiment 2 uses correct 200nt windows")
        w("  UCI     MCC=0.0096  FAILED    — 60nt sequences incompatible with")
        w("                                  512nt trained encoder")
        w()
        w("  These results motivate Experiment 2:")
        w("  - Task B: 333nt biological window + unfrozen encoder")
        w("  - Task C: correct 200nt window + unfrozen encoder")
        w("=" * 72)

    print(f"[save] Report saved -> {path}")


def write_tsv(results, path):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("experiment\ttask\teval_set\tn_samples\t"
                 "accuracy\tbalanced_accuracy\tmacro_f1\tmcc\n")
        for task, task_data in results.items():
            for eval_set, m in task_data.items():
                fh.write(
                    f"Exp1\t{task}\t{eval_set}\t{m['n_samples']}\t"
                    f"{m['accuracy']:.6f}\t{m['balanced_accuracy']:.6f}\t"
                    f"{m['macro_f1']:.6f}\t{m['mcc']:.6f}\n"
                )
    print(f"[save] TSV saved    -> {path}")


def write_json(results, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"experiment": 1,
                   "strategy":   "uniform 512nt, frozen encoder",
                   "results":    results}, fh, indent=2)
    print(f"[save] JSON saved   -> {path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    write_report(RESULTS, out_dir / "exp1_results_report.txt")
    write_tsv(RESULTS,    out_dir / "exp1_results_summary.tsv")
    write_json(RESULTS,   out_dir / "exp1_results_raw.json")

    print("\n[save] All Experiment 1 results saved to results/")
    print("       exp1_results_report.txt  — human readable")
    print("       exp1_results_summary.tsv — open in Excel")
    print("       exp1_results_raw.json    — for comparison script")

if __name__ == "__main__":
    main()