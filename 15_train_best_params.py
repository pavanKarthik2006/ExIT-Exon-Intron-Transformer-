"""
15_train_best_params.py
========================
Reads best_params.json produced by Script 14 and runs a FRESH full
training for each task using those parameters — no retraining from
existing checkpoints, everything starts from scratch.

All outputs (checkpoints, training logs, results) are saved to
separate folders so nothing from previous experiments is overwritten.

Folder layout
-------------
  checkpoints_tuned/
      task_a_best.pt      <- fresh model trained with best params
      task_b_best.pt
      task_c_best.pt
      task_a_train_log.json
      task_b_train_log.json
      task_c_train_log.json

  results_tuned/
      tuned_summary.tsv   <- before vs after MCC for all tasks
      tuned_report.txt    <- full human-readable comparison

How it works
------------
Calls 07_train.py for task_a  (Exp1 encoded data, 512nt)
Calls 10_train_bio.py for task_b and task_c  (Exp2 encoded data)

All hyperparameters (lr, batch_size, d_model, n_layers, n_heads,
dropout, ffn_dim) are passed as command-line arguments — the training
scripts themselves are NOT modified in any way.

Usage
-----
  python scripts\\15_train_best_params.py

Optional flags
--------------
  --epochs 10              override training epochs (default 10)
  --tasks task_b task_c    train only specific tasks
  --dry_run                print commands without running them
"""

import argparse, json, subprocess, sys, time
from pathlib import Path
import torch


# ═══════════════════════════════════════════════════════════════════
# TASK CONFIG — which script handles which task
# ═══════════════════════════════════════════════════════════════════

TASK_SCRIPT = {
    "task_a": "07_train.py",
    "task_b": "10_train_bio.py",
    "task_c": "10_train_bio.py",
}

TASK_ENCODED_DIR = {
    "task_a": "data/05_encoded",
    "task_b": "data/05_encoded_bio",
    "task_c": "data/05_encoded_bio",
}

# Previous best results for comparison
BASELINES = {
    "task_a": {"mcc": 0.8719, "f1": 0.9343, "exp": "Exp1"},
    "task_b": {"mcc": 0.4452, "f1": 0.5566, "exp": "Exp2"},
    "task_c": {"mcc": 0.4972, "f1": 0.6354, "exp": "Exp1"},
}

# Current default hyperparameters — used as fallback if best_params
# is missing a value or if the search never sampled a parameter
CURRENT_DEFAULTS = {
    "lr":       3e-4,
    "batch":    256,
    "d_model":  128,
    "n_layers": 4,
    "n_heads":  4,
    "dropout":  0.1,
    "ffn_dim":  256,
}


# ═══════════════════════════════════════════════════════════════════
# BUILD COMMAND
# ═══════════════════════════════════════════════════════════════════

def build_command(task, best_params, scripts_dir,
                  ckpt_dir, epochs):
    """
    Build the subprocess command to run a fresh training
    with the best parameters passed as CLI arguments.
    Returns list of strings for subprocess.run().
    """
    script  = Path(scripts_dir) / TASK_SCRIPT[task]
    enc_dir = TASK_ENCODED_DIR[task]
    params  = best_params.get(task, {})

    # Safety: n_heads must divide d_model
    d_model = params.get("d_model", 128)
    n_heads = params.get("n_heads", 4)
    if d_model % n_heads != 0:
        # Find largest valid n_heads that divides d_model
        valid   = [h for h in [8,4,2,1] if d_model % h == 0]
        n_heads = valid[0]
        print(f"  [fix] {task}: n_heads adjusted {params['n_heads']}"
              f" -> {n_heads} (must divide d_model={d_model})")

    task_letter = task  # pass full name e.g. "task_a", not just "a"

    cmd = [
        sys.executable, str(script),
        "--task",        task_letter,
        "--encoded_dir", enc_dir,
        "--ckpt_dir",    str(ckpt_dir),
        "--epochs",      str(epochs),
        "--lr",          str(params.get("lr",       3e-4)),
        "--batch_size",  str(params.get("batch",    256)),
        "--d_model",     str(d_model),
        "--n_layers",    str(params.get("n_layers", 4)),
        "--n_heads",     str(n_heads),
        "--ffn_dim",     str(params.get("ffn_dim",  256)),
    ]

    return cmd


# ═══════════════════════════════════════════════════════════════════
# READ CHECKPOINT MCC
# ═══════════════════════════════════════════════════════════════════

def read_ckpt_metrics(ckpt_path):
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu",
                          weights_only=False)
        return {
            "mcc":    ckpt.get("best_val_mcc",  None),
            "epoch":  ckpt.get("epoch",          None),
            "config": ckpt.get("config",         {}),
        }
    except Exception as e:
        return {"mcc": None, "epoch": None, "config": {}, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# REPORT WRITERS
# ═══════════════════════════════════════════════════════════════════

def write_summary_tsv(results, out_dir):
    path = out_dir / "tuned_summary.tsv"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("task\tprev_exp\tbaseline_mcc\ttuned_mcc\tdelta\t"
                 "retention_pct\tepochs\tcheckpoint\t"
                 "lr\tbatch\td_model\tn_layers\tn_heads\tffn_dim\n")
        for task, res in results.items():
            base   = BASELINES[task]["mcc"]
            tuned  = res.get("tuned_mcc")
            ckpt   = res.get("ckpt_path", "")
            params = res.get("params",    {})
            ep     = res.get("epochs",    "?")

            if tuned is None:
                fh.write(f"{task}\t{BASELINES[task]['exp']}\t"
                         f"{base:.6f}\tN/A\tN/A\tN/A\t{ep}\t{ckpt}\n")
            else:
                delta = tuned - base
                ret   = tuned/base*100 if base > 0 else 0
                fh.write(
                    f"{task}\t{BASELINES[task]['exp']}\t"
                    f"{base:.6f}\t{tuned:.6f}\t{delta:+.6f}\t"
                    f"{ret:.1f}\t{ep}\t{ckpt}\t"
                    f"{params.get('lr','')}\t"
                    f"{params.get('batch','')}\t"
                    f"{params.get('d_model','')}\t"
                    f"{params.get('n_layers','')}\t"
                    f"{params.get('n_heads','')}\t"
                    f"{params.get('ffn_dim','')}\n"
                )
    print(f"[15] tuned_summary.tsv    -> {path}")


def write_full_report(results, best_params, out_dir):
    path = out_dir / "tuned_report.txt"
    with open(path, "w", encoding="utf-8") as fh:
        def w(s=""): fh.write(s+"\n")
        w("="*70)
        w("SCRIPT 15 — FRESH TRAINING WITH TUNED HYPERPARAMETERS")
        w("Training scripts used AS-IS (no file modification)")
        w("Parameters passed as command-line arguments")
        w("="*70)

        # Parameters used
        w()
        w("HYPERPARAMETERS USED (from best_params.json)")
        w("-"*70)
        params_header = (f"  {'Task':<12} {'LR':>8} {'Batch':>6} "
                         f"{'d_model':>8} {'Layers':>7} "
                         f"{'Heads':>6} {'FFN':>6}")
        w(params_header)
        w("  " + "-"*53)
        for task in results:
            p = best_params.get(task, {})
            w(f"  {task:<12} "
              f"{str(p.get('lr','?')):>8} "
              f"{str(p.get('batch','?')):>6} "
              f"{str(p.get('d_model','?')):>8} "
              f"{str(p.get('n_layers','?')):>7} "
              f"{str(p.get('n_heads','?')):>6} "
              f"{str(p.get('ffn_dim','?')):>6}")

        # Results comparison
        w()
        w("BEFORE vs AFTER COMPARISON")
        w("-"*70)
        w(f"  {'Task':<12} {'Prev Exp':>8} {'Baseline MCC':>14} "
          f"{'Tuned MCC':>12} {'Delta':>8} {'Retention':>10}")
        w("  " + "-"*64)
        for task, res in results.items():
            base  = BASELINES[task]["mcc"]
            tuned = res.get("tuned_mcc")
            exp   = BASELINES[task]["exp"]
            if tuned is None:
                w(f"  {task:<12} {exp:>8} {base:>14.4f} {'FAILED':>12}")
            else:
                delta = tuned - base
                ret   = tuned/base*100 if base > 0 else 0
                arrow = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else "~")
                w(f"  {task:<12} {exp:>8} {base:>14.4f} "
                  f"{tuned:>12.4f} {delta:>+8.4f} "
                  f"{ret:>9.1f}%  {arrow}")

        # Per-task detail
        w()
        w("PER-TASK DETAIL")
        w("-"*70)
        for task, res in results.items():
            w()
            w(f"  {task.upper()}")
            w(f"    elapsed     : {res.get('elapsed_min', '?'):.1f} min")
            w(f"    checkpoint  : {res.get('ckpt_path', 'N/A')}")
            w(f"    tuned MCC   : {res.get('tuned_mcc', 'N/A')}")
            w(f"    baseline MCC: {BASELINES[task]['mcc']}")
            cmd = res.get("command", [])
            if cmd:
                w(f"    command     : {' '.join(cmd)}")

        w()
        w("="*70)
        w("NOTE: Checkpoints saved to checkpoints_tuned/")
        w("      Use these for final evaluation in Script 11.")
        w("="*70)

    print(f"[15] tuned_report.txt     -> {path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Fresh training with best params from Script 14"
    )
    ap.add_argument("--hparam_dir",  default="results_hparam",
                    help="Directory containing best_params.json (Script 14 output)")
    ap.add_argument("--scripts_dir", default="scripts",
                    help="Directory containing 07_train.py, 10_train_bio.py")
    ap.add_argument("--ckpt_dir",    default="checkpoints_tuned",
                    help="Where to save fresh trained checkpoints")
    ap.add_argument("--out_dir",     default="results_tuned",
                    help="Where to save comparison reports")
    ap.add_argument("--epochs",      type=int, default=10,
                    help="Training epochs (default 10)")
    ap.add_argument("--tasks",       nargs="+",
                    default=["task_a","task_b","task_c"],
                    choices=["task_a","task_b","task_c"])
    ap.add_argument("--dry_run",     action="store_true",
                    help="Print commands without running them")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    out_dir  = Path(args.out_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[15] Fresh Training with Tuned Hyperparameters")
    print(f"     hparam_dir   : {args.hparam_dir}")
    print(f"     scripts_dir  : {args.scripts_dir}")
    print(f"     ckpt_dir     : {args.ckpt_dir}   <- separate folder")
    print(f"     out_dir      : {args.out_dir}    <- separate folder")
    print(f"     epochs       : {args.epochs}")
    print(f"     tasks        : {args.tasks}")

    # Load best params
    best_params_path = Path(args.hparam_dir) / "best_params.json"
    if not best_params_path.exists():
        print(f"\n[ERROR] best_params.json not found at {best_params_path}")
        print(f"  Run Script 14 first:")
        print(f"  python scripts\\14_hyperparam_search.py")
        sys.exit(1)

    with open(best_params_path) as fh:
        best_params = json.load(fh)

    print(f"\n[15] Loaded best_params.json")
    for task, params in best_params.items():
        print(f"  {task}: {params}")

    # Safety: for any parameter where the search found no improvement
    # over the current default, keep the default instead.
    # This prevents accidentally degrading performance.
    print(f"\n[15] Checking best params against current defaults ...")
    for task in list(best_params.keys()):
        for param, default_val in CURRENT_DEFAULTS.items():
            found_val = best_params[task].get(param)
            if found_val is None:
                print(f"  [fix] {task}.{param}: missing in best_params, "
                      f"using default = {default_val}")
                best_params[task][param] = default_val

    # Run training for each task
    all_results = {}

    for task in args.tasks:
        if task not in best_params:
            print(f"\n[15] SKIP {task} — not found in best_params.json")
            all_results[task] = {"tuned_mcc": None,
                                 "ckpt_path": None,
                                 "params": {},
                                 "elapsed_min": 0}
            continue

        cmd = build_command(
            task        = task,
            best_params = best_params,
            scripts_dir = args.scripts_dir,
            ckpt_dir    = ckpt_dir,
            epochs      = args.epochs,
        )

        print(f"\n{'='*65}")
        print(f"[15] Task: {task.upper()}")
        print(f"     Params : {best_params[task]}")
        print(f"     Command: {' '.join(cmd)}")
        print(f"{'='*65}")

        if args.dry_run:
            print("  [dry_run] skipping execution")
            all_results[task] = {
                "tuned_mcc": None, "ckpt_path": None,
                "params": best_params[task],
                "elapsed_min": 0, "command": cmd,
                "epochs": args.epochs,
            }
            continue

        t0 = time.time()
        print(f"  Running ... (output below)\n")
        result = subprocess.run(
            cmd,
            text=True,
            stderr=subprocess.STDOUT,
        )
        elapsed_min = (time.time() - t0) / 60

        if result.returncode != 0:
            print(f"\n  [ERROR] {task} training FAILED "
                  f"(returncode={result.returncode})")
            print(f"  [ERROR] Check the output above for the error message.")
            all_results[task] = {
                "tuned_mcc": None, "ckpt_path": None,
                "params": best_params[task],
                "elapsed_min": elapsed_min, "command": cmd,
                "epochs": args.epochs,
            }
            continue

        # Read MCC from saved checkpoint
        ckpt_path = ckpt_dir / f"{task}_best.pt"
        metrics   = read_ckpt_metrics(ckpt_path)

        print(f"\n  [OK] {task} complete  ({elapsed_min:.1f} min)")
        print(f"       tuned MCC  = {metrics['mcc']}")
        print(f"       baseline   = {BASELINES[task]['mcc']:.4f}")
        if metrics["mcc"] is not None:
            delta = metrics["mcc"] - BASELINES[task]["mcc"]
            print(f"       delta      = {delta:+.4f}")

        all_results[task] = {
            "tuned_mcc":   metrics["mcc"],
            "ckpt_path":   str(ckpt_path),
            "params":      best_params[task],
            "elapsed_min": elapsed_min,
            "command":     cmd,
            "epochs":      args.epochs,
        }

    # Write reports
    print("\n[15] Writing reports ...")
    write_summary_tsv(all_results, out_dir)
    write_full_report(all_results, best_params, out_dir)

    # Final summary print
    print(f"\n{'='*50}")
    print(f"[15] RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"  {'Task':<12} {'Baseline':>10} {'Tuned':>10} {'Delta':>8}  Status")
    print(f"  {'-'*50}")
    for task, res in all_results.items():
        base  = BASELINES[task]["mcc"]
        tuned = res.get("tuned_mcc")
        if tuned is None:
            print(f"  {task:<12} {base:>10.4f} {'FAILED':>10}")
        else:
            delta  = tuned - base
            status = "OK - improved" if delta > 0.005 else \
                     ("OK - similar" if delta >= -0.005 else
                      "WARNING: WORSE than baseline")
            print(f"  {task:<12} {base:>10.4f} {tuned:>10.4f} "
                  f"{delta:>+8.4f}  {status}")

    # Explicit warning if any task is worse
    worse = [t for t, r in all_results.items()
             if r.get("tuned_mcc") is not None
             and r["tuned_mcc"] < BASELINES[t]["mcc"] - 0.005]
    if worse:
        print(f"\n  [!] WARNING: tuned params gave LOWER MCC for: {worse}")
        print(f"  [!] Consider keeping the original checkpoints for those tasks.")
        print(f"  [!] Original checkpoints are in: checkpoints/ and checkpoints_bio/")
    else:
        print(f"\n  [OK] All tuned models meet or exceed baseline MCC.")

    print(f"\n[15] Checkpoints -> {ckpt_dir}/")
    print(f"[15] Reports     -> {out_dir}/")
    print(f"[15] Done.")


if __name__ == "__main__":
    main()