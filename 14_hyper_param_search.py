"""

========================
HYPERPARAMETER SEARCH — finds the best VALUE of each parameter
independently, then saves best_params.json for use by Script 15.

Outputs [results_hparam/]
--------------------------
  best_params.json          best value per parameter per task
                            → read by Script 15 to update training scripts
  param_importance.tsv      mean/max/min MCC per parameter value per task
  hparam_all_runs.tsv       every single run (paper appendix table)
  search_log.json           raw run log (reusable with --skip_search)

Search space
------------
  lr        : [1e-4, 3e-4, 5e-4, 1e-3]
  batch_size : [128, 256, 512]
  d_model    : [64, 128, 256]
  n_layers   : [2, 4, 6]
  n_heads    : [2, 4, 8]   (only valid when divides d_model)
  dropout    : [0.1, 0.2, 0.3]
  ffn_dim    : [128, 256, 512]

Current defaults (for comparison in report)
--------------------------------------------
  lr=3e-4  batch=256  d_model=128  n_layers=4  n_heads=4
  dropout=0.1  ffn_dim=256
"""

import argparse, json, time, random, itertools
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
model_module = import_module("06_model")
DNAClassifier = model_module.DNAClassifier



# SEARCH SPACE  — all values to try per parameter


SEARCH_SPACE = {
    "lr":       [1e-4, 3e-4, 5e-4, 1e-3],
    "batch":    [128, 256, 512],
    "d_model":  [64, 128, 256],
    "n_layers": [2, 4, 6],
    "n_heads":  [2, 4, 8],     # filtered: must divide d_model
    "dropout":  [0.1, 0.2, 0.3],
    "ffn_dim":  [128, 256, 512],
}

# Current defaults — used in comparison report
CURRENT_DEFAULTS = {
    "lr": 3e-4, "batch": 256, "d_model": 128,
    "n_layers": 4, "n_heads": 4, "dropout": 0.1, "ffn_dim": 256,
}

TASK_CONFIGS = {
    "task_a": {
        "encoded_dir":   "data/05_encoded",
        "search_epochs": 3,
        "baseline_mcc":  0.8719,
    },
    "task_b": {
        "encoded_dir":   "data/05_encoded_bio",
        "search_epochs": 3,
        "baseline_mcc":  0.4452,
    },
    "task_c": {
        "encoded_dir":   "data/05_encoded_bio",
        "search_epochs": 3,
        "baseline_mcc":  0.4972,
    },
}


# ═══════════════════════════════════════════════════════════════════
# COMBINATION GENERATOR
# ═══════════════════════════════════════════════════════════════════

def generate_valid_combinations(space, seed=42):
   
    keys   = list(space.keys())
    combos = []
    for vals in itertools.product(*[space[k] for k in keys]):
        cfg = dict(zip(keys, vals))
        if cfg["d_model"] % cfg["n_heads"] != 0:
            continue
        if cfg["ffn_dim"] < cfg["d_model"]:
            continue
        combos.append(cfg)
    random.Random(seed).shuffle(combos)

    # Always put current defaults first — guaranteed to be included
    # even when --max_combos limits the total number of runs
    defaults_cfg = dict(CURRENT_DEFAULTS)  # copy
    # Remove defaults from wherever they are in the list
    combos = [c for c in combos if c != defaults_cfg]
    combos.insert(0, defaults_cfg)

    print(f"  Total valid combinations : {len(combos):,}")
    print(f"  Run #1 (always)          : current defaults {defaults_cfg}")
    return combos


# ═══════════════════════════════════════════════════════════════════
# DATA UTILITIES
# ═══════════════════════════════════════════════════════════════════

def load_split(encoded_dir, task, split):
    path = Path(encoded_dir) / f"{task}_{split}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    data = torch.load(path, weights_only=True)
    return TensorDataset(data["input_ids"], data["labels"])

def get_num_classes(encoded_dir, task):
    data = torch.load(Path(encoded_dir) / f"{task}_train.pt",
                      weights_only=True)
    return int(data["labels"].max().item()) + 1

def class_weights(ds, n_cls):
    lbls = ds.tensors[1]
    cnt  = torch.bincount(lbls, minlength=n_cls).float().clamp(min=1)
    w    = 1.0 / cnt
    return w / w.sum() * n_cls

def cap(ds, n):
    if n and len(ds) > n:
        idx = torch.randperm(len(ds))[:n]
        return TensorDataset(*[t[idx] for t in ds.tensors])
    return ds


# ═══════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════

def mcc_score(preds, labels, n_cls):
    total  = len(preds)
    cm     = [[0]*n_cls for _ in range(n_cls)]
    for p, l in zip(preds, labels):
        cm[l][p] += 1
    xy = xx = yy = 0.0
    for k in range(n_cls):
        for l in range(n_cls):
            for m in range(n_cls):
                xy += cm[k][k]*cm[m][l] - cm[l][k]*cm[k][m]
        s = sum(cm[k][j] for j in range(n_cls))
        t = sum(cm[j][k] for j in range(n_cls))
        xx += s*(total-s); yy += t*(total-t)
    return xy/(xx*yy)**0.5 if xx > 0 and yy > 0 else 0.0

def macro_f1(preds, labels, n_cls):
    cm = [[0]*n_cls for _ in range(n_cls)]
    for p, l in zip(preds, labels):
        cm[l][p] += 1
    f1s = []
    for c in range(n_cls):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(n_cls)) - tp
        fn = sum(cm[c][r] for r in range(n_cls)) - tp
        p  = tp/(tp+fp) if (tp+fp) > 0 else 0.0
        r  = tp/(tp+fn) if (tp+fn) > 0 else 0.0
        f1s.append(2*p*r/(p+r) if (p+r) > 0 else 0.0)
    return sum(f1s)/n_cls

@torch.no_grad()
def evaluate(model, loader, loss_fn, device, n_cls):
    model.eval()
    tot_loss, tot_acc, n = 0.0, 0.0, 0
    preds, labels = [], []
    for ids, lbl in loader:
        ids, lbl = ids.to(device), lbl.to(device)
        logits    = model(ids)
        tot_loss += loss_fn(logits, lbl).item()
        tot_acc  += (logits.argmax(-1)==lbl).float().mean().item()
        preds.extend(logits.argmax(-1).tolist())
        labels.extend(lbl.tolist())
        n += 1
    return {
        "loss":     tot_loss/max(n,1),
        "accuracy": tot_acc/max(n,1),
        "mcc":      mcc_score(preds, labels, n_cls),
        "macro_f1": macro_f1(preds, labels, n_cls),
    }


# ═══════════════════════════════════════════════════════════════════
# SINGLE TRAINING RUN
# ═══════════════════════════════════════════════════════════════════

def run_single(cfg, task, encoded_dir, epochs, max_samples, device):
    try:
        train_ds = load_split(encoded_dir, task, "train")
        val_ds   = load_split(encoded_dir, task, "val")
        n_cls    = get_num_classes(encoded_dir, task)
        max_len  = int(train_ds.tensors[0].shape[1])

        train_ds = cap(train_ds, max_samples)
        val_ds   = cap(val_ds,   max(int(max_samples*0.15), 500))

        w        = class_weights(train_ds, n_cls).to(device)
        loss_fn  = nn.CrossEntropyLoss(weight=w)
        tr_load  = DataLoader(train_ds, batch_size=cfg["batch"],
                              shuffle=True,  num_workers=0)
        va_load  = DataLoader(val_ds,   batch_size=cfg["batch"],
                              shuffle=False, num_workers=0)

        model = DNAClassifier.build(
            num_classes=n_cls,   d_model=cfg["d_model"],
            n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
            ffn_dim=cfg["ffn_dim"], max_len=max_len,
            dropout=cfg["dropout"],
        )
        opt   = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
        sched = CosineAnnealingLR(opt, T_max=epochs)

        best_loss, best_m = float("inf"), None
        for _ in range(epochs):
            model.train()
            for ids, lbl in tr_load:
                ids, lbl = ids.to(device), lbl.to(device)
                opt.zero_grad()
                loss = loss_fn(model(ids), lbl)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            m = evaluate(model, va_load, loss_fn, device, n_cls)
            if m["loss"] < best_loss:
                best_loss, best_m = m["loss"], m
        return best_m

    except Exception as e:
        print(f"    ERROR: {e}")
        return {"loss":999.0, "accuracy":0.0, "mcc":-1.0,
                "macro_f1":0.0, "error":str(e)}


# ═══════════════════════════════════════════════════════════════════
# PARAMETER IMPORTANCE — best value per parameter
# ═══════════════════════════════════════════════════════════════════

def compute_param_importance(run_results, space):
    """
    For each parameter, group all run MCCs by the value used.
    Best value = highest mean MCC across all runs using that value.

    Returns:
      importance[param][value] = {mean_mcc, max_mcc, min_mcc, count}
      best_values[param]       = value with highest mean_mcc
    """
    param_mcc = {p: defaultdict(list) for p in space}
    for r in run_results:
        mcc = r["metrics"].get("mcc", -1)
        if mcc < 0:
            continue
        for p, v in r["config"].items():
            if p in param_mcc:
                param_mcc[p][v].append(mcc)

    importance  = {}
    best_values = {}
    for param in space:
        importance[param] = {}
        best_mean, best_val = -1.0, None
        for val in space[param]:
            mccs = param_mcc[param].get(val, [])
            if not mccs:
                importance[param][val] = {
                    "mean_mcc":0.0,"max_mcc":0.0,
                    "min_mcc":0.0, "count":0}
                continue
            mean_m = sum(mccs)/len(mccs)
            importance[param][val] = {
                "mean_mcc": round(mean_m,    6),
                "max_mcc":  round(max(mccs), 6),
                "min_mcc":  round(min(mccs), 6),
                "count":    len(mccs),
            }
            if mean_m > best_mean:
                best_mean, best_val = mean_m, val

        # Safety fallback: if no value was found (all runs failed or
        # a value never appeared in the limited combo set), fall back
        # to the current default so we never end up with None
        if best_val is None:
            best_val = CURRENT_DEFAULTS.get(param)
            print(f"  [fallback] {param}: no valid runs found, "
                  f"keeping default = {best_val}")

        best_values[param] = best_val

    return importance, best_values


# ═══════════════════════════════════════════════════════════════════
# WRITERS
# ═══════════════════════════════════════════════════════════════════

def write_param_importance_tsv(all_imp, all_best, out_dir):
    path = out_dir / "param_importance.tsv"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("task\tparameter\tvalue\tmean_mcc\tmax_mcc\t"
                 "min_mcc\tcount\tis_best\n")
        for task, imp in all_imp.items():
            for param, val_dict in imp.items():
                for val, s in val_dict.items():
                    is_best = (val == all_best[task].get(param))
                    fh.write(
                        f"{task}\t{param}\t{val}\t"
                        f"{s['mean_mcc']:.6f}\t{s['max_mcc']:.6f}\t"
                        f"{s['min_mcc']:.6f}\t{s['count']}\t"
                        f"{'YES' if is_best else 'no'}\n"
                    )
    print(f"[14] param_importance.tsv -> {path}")


def write_all_runs_tsv(all_runs, out_dir):
    path = out_dir / "hparam_all_runs.tsv"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("task\trun_id\tlr\tbatch\td_model\tn_layers\t"
                 "n_heads\tdropout\tffn_dim\t"
                 "val_mcc\tval_f1\tval_acc\tval_loss\telapsed_s\n")
        for task, runs in all_runs.items():
            for r in sorted(runs,
                            key=lambda x: x["metrics"].get("mcc",0),
                            reverse=True):
                c = r["config"]; m = r["metrics"]
                fh.write(
                    f"{task}\t{r['run_id']}\t"
                    f"{c['lr']}\t{c['batch']}\t{c['d_model']}\t"
                    f"{c['n_layers']}\t{c['n_heads']}\t"
                    f"{c['dropout']}\t{c['ffn_dim']}\t"
                    f"{m.get('mcc',0):.6f}\t{m.get('macro_f1',0):.6f}\t"
                    f"{m.get('accuracy',0):.6f}\t{m.get('loss',0):.6f}\t"
                    f"{r['elapsed']}\n"
                )
    print(f"[14] hparam_all_runs.tsv  -> {path}")


def write_best_params_json(all_best, out_dir):
    """
    Save best_params.json — this is what Script 15 reads
    to update the training scripts.
    """
    path = out_dir / "best_params.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(all_best, fh, indent=2)
    print(f"[14] best_params.json     -> {path}")
    print(f"     --> Script 15 reads this file to update "
          f"07_train.py and 10_train_bio.py")


def write_importance_report(all_imp, all_best, all_runs, baselines, out_dir):
    path = out_dir / "param_importance_report.txt"
    with open(path, "w", encoding="utf-8") as fh:
        def w(s=""): fh.write(s+"\n")
        w("="*70)
        w("HYPERPARAMETER SEARCH — PARAMETER IMPORTANCE REPORT")
        w("For each parameter: which value gives the highest mean MCC")
        w("across all random search runs that used that value.")
        w("="*70)

        for task in all_imp:
            baseline = baselines[task]
            runs     = all_runs.get(task, [])
            valid    = [r for r in runs if r["metrics"].get("mcc",-1) >= 0]
            w()
            w(f"{'─'*65}")
            w(f"  TASK: {task.upper()}   baseline MCC = {baseline:.4f}   "
              f"runs completed = {len(valid)}")
            w(f"{'─'*65}")

            for param in SEARCH_SPACE:
                val_dict = all_imp[task].get(param, {})
                best_val = all_best[task].get(param)
                current  = CURRENT_DEFAULTS.get(param)
                w()
                w(f"  {param}")
                w(f"  {'Value':<12} {'Mean MCC':>10} {'Max MCC':>10} "
                  f"{'Min MCC':>10} {'Runs':>6}  Note")
                w("  " + "-"*62)
                for val in SEARCH_SPACE[param]:
                    s      = val_dict.get(val, {})
                    mean_m = s.get("mean_mcc", 0)
                    max_m  = s.get("max_mcc",  0)
                    min_m  = s.get("min_mcc",  0)
                    count  = s.get("count",    0)
                    note   = ""
                    if val == best_val:
                        note += " <-- BEST"
                    if val == current:
                        note += " (current default)"
                    w(f"  {str(val):<12} {mean_m:>10.4f} {max_m:>10.4f} "
                      f"{min_m:>10.4f} {count:>6}  {note}")

            w()
            w(f"  BEST VALUES FOR {task.upper()} (saved to best_params.json):")
            for param, val in all_best[task].items():
                current = CURRENT_DEFAULTS.get(param)
                changed = " <- will be updated" if val != current else " (no change)"
                w(f"    {param:<14} {str(val):<10}  current={current}{changed}")

        w()
        w("="*70)
        w("NEXT STEP")
        w("="*70)
        w("  Run Script 15 to update training scripts with best params:")
        w()
        w("  python scripts\\15_apply_best_params.py")
        w()
        w("  Script 15 will:")
        w("  1. Read best_params.json")
        w("  2. Update defaults in 07_train.py and 10_train_bio.py")
        w("  3. Retrain all tasks with the updated scripts")
        w("  4. Save new checkpoints to checkpoints_tuned/")
        w("="*70)

    print(f"[14] importance report    -> {path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Hyperparameter search — best value per parameter"
    )
    ap.add_argument("--tasks",       nargs="+",
                    default=["task_a","task_b","task_c"],
                    choices=["task_a","task_b","task_c"])
    ap.add_argument("--out_dir",     default="results_hparam")
    ap.add_argument("--max_samples", type=int, default=20000,
                    help="Training samples per run (smaller = faster)")
    ap.add_argument("--max_combos",  type=int, default=None,
                    help="Limit combos (None = all valid ~160 per task)")
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--skip_search", action="store_true",
                    help="Skip search, load existing search_log.json")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cpu")

    print("[14] Hyperparameter Search — Best Value Per Parameter")
    print(f"     Tasks       : {args.tasks}")
    print(f"     Max samples : {args.max_samples:,} per run")
    print(f"     Seed        : {args.seed}")

    combos = generate_valid_combinations(SEARCH_SPACE, seed=args.seed)
    if args.max_combos:
        combos = combos[:args.max_combos]
        print(f"     Limited to  : {len(combos)} combinations")

    n_runs   = len(combos) * len(args.tasks)
    eta_hrs  = n_runs * 30 / 3600
    print(f"     Total runs  : {n_runs:,}  (~{eta_hrs:.1f} hrs)")
    print(f"     Tip         : use --max_combos 50 to limit to "
          f"{50*len(args.tasks)} runs (~{50*len(args.tasks)*30/3600:.1f} hrs)")

    # ── Search ────────────────────────────────────────────────────
    all_runs = {}

    if args.skip_search:
        log_path = out_dir / "search_log.json"
        print(f"\n[14] Loading existing search log: {log_path}")
        with open(log_path) as fh:
            all_runs = json.load(fh)
    else:
        for task in args.tasks:
            tcfg    = TASK_CONFIGS[task]
            runs    = []
            t_start = time.time()
            n       = len(combos)

            print(f"\n{'='*65}")
            print(f"  Task: {task.upper()}   baseline={tcfg['baseline_mcc']:.4f}  "
                  f"combos={n}  epochs={tcfg['search_epochs']}")
            print(f"{'='*65}")

            for i, cfg in enumerate(combos):
                t0 = time.time()
                m  = run_single(cfg, task, tcfg["encoded_dir"],
                                tcfg["search_epochs"],
                                args.max_samples, device)
                elapsed = time.time()-t0
                runs.append({"run_id":i+1,"task":task,
                             "config":cfg,"metrics":m,
                             "elapsed":round(elapsed,1)})
                eta = (time.time()-t_start)/(i+1)*(n-i-1)
                print(f"  [{i+1:>4}/{n}]  "
                      f"lr={cfg['lr']:.0e} bs={cfg['batch']:>3} "
                      f"d={cfg['d_model']:>3} L={cfg['n_layers']} "
                      f"h={cfg['n_heads']} drop={cfg['dropout']} "
                      f"ffn={cfg['ffn_dim']:>3}  "
                      f"mcc={m['mcc']:>7.4f} f1={m['macro_f1']:.4f}  "
                      f"({elapsed:.0f}s ETA {eta/60:.0f}m)",
                      flush=True)

            all_runs[task] = runs

        # Save search log so --skip_search can reuse it
        log_path = out_dir / "search_log.json"
        with open(log_path, "w", encoding="utf-8") as fh:
            json.dump(all_runs, fh,
                      default=lambda o: round(o,6) if isinstance(o,float) else o,
                      indent=2)
        print(f"\n[14] Search log -> {log_path}")

    # ── Parameter importance ──────────────────────────────────────
    print("\n[14] Computing parameter importance ...")
    all_importance  = {}
    all_best_values = {}
    for task in args.tasks:
        imp, best = compute_param_importance(
            all_runs[task], SEARCH_SPACE
        )
        all_importance[task]  = imp
        all_best_values[task] = best

        print(f"  {task} best values:")
        for p, v in best.items():
            current = CURRENT_DEFAULTS.get(p)
            flag    = " <- changed" if v != current else ""
            print(f"    {p:<14} {str(v):<10}  (current={current}){flag}")

    # ── Write outputs ─────────────────────────────────────────────
    baselines = {t: TASK_CONFIGS[t]["baseline_mcc"] for t in args.tasks}
    write_best_params_json(all_best_values, out_dir)
    write_param_importance_tsv(all_importance, all_best_values, out_dir)
    write_all_runs_tsv(all_runs, out_dir)
    write_importance_report(all_importance, all_best_values,
                            all_runs, baselines, out_dir)

    print(f"\n[14] Done. Run Script 15 next:")
    print(f"     python scripts\\15_apply_best_params.py")


if __name__ == "__main__":
    main()