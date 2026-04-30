"""
10_train_bio.py
================
EXPERIMENT 2 — Train Task B and Task C with biological lengths,
no encoder freezing.

Key differences from Experiment 1 (07_train.py)
------------------------------------------------
1. Reads from data/05_encoded_bio/ (biological length encoded files)
2. Encoder + head trained TOGETHER — no freezing
3. Each task uses its own max_len inferred from the .pt file shape:
     task_b : 111 tokens  (333nt, min 150nt filter applied)
     task_c :  66 tokens  (200nt)
4. Checkpoints saved to checkpoints_bio/ separate from Experiment 1

Tasks NOT trained here
----------------------
  task_a     : SKIPPED — Experiment 1 MCC=0.877 is sufficient
  uci_splice : SKIPPED — replaced by cross-species generalisation

Comparison with Experiment 1
------------------------------
  Experiment 1 : frozen encoder, uniform 512nt  -> checkpoints/
  Experiment 2 : full end-to-end, bio lengths   -> checkpoints_bio/
  Evaluation   : 11_evaluate_compare.py compares both side by side
"""

import argparse, time, json
from pathlib import Path
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


# ═══════════════════════════════════════════════════════════════════
# DATASET LOADER
# ═══════════════════════════════════════════════════════════════════

def load_split(encoded_dir, name, split):
    path = encoded_dir / f"{name}_{split}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    data = torch.load(path, weights_only=True)
    return TensorDataset(data["input_ids"], data["labels"])


def get_num_classes(encoded_dir, name):
    path = encoded_dir / f"{name}_train.pt"
    data = torch.load(path, weights_only=True)
    return int(data["labels"].max().item()) + 1


def compute_class_weights(dataset, num_classes):
    labels  = dataset.tensors[1]
    counts  = torch.bincount(labels, minlength=num_classes).float().clamp(min=1)
    weights = 1.0 / counts
    return weights / weights.sum() * num_classes


def accuracy(logits, labels):
    return (logits.argmax(dim=-1) == labels).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for ids, lbls in loader:
        ids, lbls  = ids.to(device), lbls.to(device)
        logits      = model(ids)
        total_loss += loss_fn(logits, lbls).item()
        total_acc  += accuracy(logits, lbls)
        n += 1
    return {"loss": total_loss / max(n, 1), "acc": total_acc / max(n, 1)}


# ═══════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

def train_task(
    task_name,   encoded_dir, ckpt_dir,
    epochs=5,    batch_size=256, lr=3e-4,
    d_model=128, n_heads=4, n_layers=4, ffn_dim=256,
    log_every=100, max_samples=50000,
):
    device = torch.device("cpu")
    print(f"\n{'='*60}")
    print(f"[Exp 2] Training: {task_name}  (encoder+head, no freezing)")
    print(f"{'='*60}")

    # ── Data ──────────────────────────────────────────────────────
    train_ds = load_split(encoded_dir, task_name, "train")
    val_ds   = load_split(encoded_dir, task_name, "val")
    n_cls    = get_num_classes(encoded_dir, task_name)

    # max_len inferred automatically from data shape
    # task_b -> 111 tokens (333nt), task_c -> 66 tokens (200nt)
    max_len = int(train_ds.tensors[0].shape[1])
    nt_len  = max_len * 3

    # Cap training samples for fast runs
    if max_samples and len(train_ds) > max_samples:
        idx      = torch.randperm(len(train_ds))[:max_samples]
        train_ds = TensorDataset(*[t[idx] for t in train_ds.tensors])

    # Cap val proportionally (15% of max_samples)
    val_cap = max(int(max_samples * 0.15), 1000) if max_samples else len(val_ds)
    if len(val_ds) > val_cap:
        idx    = torch.randperm(len(val_ds))[:val_cap]
        val_ds = TensorDataset(*[t[idx] for t in val_ds.tensors])

    print(f"  num_classes={n_cls}  train={len(train_ds):,}  val={len(val_ds):,}")
    print(f"  token_len={max_len}  (= {nt_len}nt biological window)")

    class_w      = compute_class_weights(train_ds, n_cls).to(device)
    loss_fn      = nn.CrossEntropyLoss(weight=class_w)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0)

    # ── Model — fresh per task, trained end-to-end ────────────────
    model = DNAClassifier.build(
        num_classes = n_cls,
        d_model     = d_model,
        n_heads     = n_heads,
        n_layers    = n_layers,
        ffn_dim     = ffn_dim,
        max_len     = max_len,
    )
    params    = model.count_parameters()
    trainable = params["encoder_trainable"] + params["head_total"]
    print(f"  Params — encoder={params['encoder_total']:,}  "
          f"head={params['head_total']:,}  trainable={trainable:,}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training loop ─────────────────────────────────────────────
    best_val_loss = float("inf")
    history       = []

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        ep_loss, ep_acc, n_steps = 0.0, 0.0, 0

        for step, (ids, lbls) in enumerate(train_loader):
            ids, lbls = ids.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(ids)
            loss   = loss_fn(logits, lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss += loss.item()
            ep_acc  += accuracy(logits, lbls)
            n_steps += 1

            if (step + 1) % log_every == 0:
                print(f"  epoch={epoch} step={step+1}/{len(train_loader)} "
                      f"loss={ep_loss/n_steps:.4f} acc={ep_acc/n_steps:.3f}",
                      flush=True)

        scheduler.step()
        val_m   = evaluate(model, val_loader, loss_fn, device)
        tr_m    = {"loss": ep_loss/max(n_steps, 1), "acc": ep_acc/max(n_steps, 1)}
        elapsed = time.time() - t0

        print(f"  -- epoch {epoch}/{epochs}  "
              f"train_loss={tr_m['loss']:.4f} train_acc={tr_m['acc']:.3f}  "
              f"val_loss={val_m['loss']:.4f} val_acc={val_m['acc']:.3f}  "
              f"({elapsed:.1f}s)", flush=True)

        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in tr_m.items()},
            **{f"val_{k}":   v for k, v in val_m.items()},
        })

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            ckpt_path     = ckpt_dir / f"{task_name}_best.pt"
            torch.save({
                "encoder_state": model.encoder.state_dict(),
                "head_state":    model.classifier.state_dict(),
                "num_classes":   n_cls,
                "config": {
                    "d_model":  d_model,  "n_heads":  n_heads,
                    "n_layers": n_layers, "ffn_dim":  ffn_dim,
                    "max_len":  max_len,
                },
                "best_val_loss": best_val_loss,
                "epoch":         epoch,
                "history":       history,
                "freeze_enc":    False,
                "experiment":    2,
                "nt_len":        nt_len,
            }, ckpt_path)
            print(f"  Checkpoint saved -> {ckpt_path}", flush=True)

    print(f"  Best val_loss={best_val_loss:.4f}")
    return {"task": task_name, "best_val_loss": best_val_loss,
            "history": history}


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

TASKS = ["task_b", "task_c"]   # task_a and uci_splice excluded


def main():
    ap = argparse.ArgumentParser(
        description="Experiment 2 — train Task B and C with biological lengths"
    )
    ap.add_argument("--task",        default="all",
                    choices=TASKS + ["all"])
    ap.add_argument("--encoded_dir", default="data/05_encoded_bio")
    ap.add_argument("--ckpt_dir",    default="checkpoints_bio")
    ap.add_argument("--epochs",      type=int,   default=5)
    ap.add_argument("--batch_size",  type=int,   default=256)
    ap.add_argument("--lr",          type=float, default=3e-4)
    ap.add_argument("--d_model",     type=int,   default=128)
    ap.add_argument("--n_heads",     type=int,   default=4)
    ap.add_argument("--n_layers",    type=int,   default=4)
    ap.add_argument("--ffn_dim",     type=int,   default=256)
    ap.add_argument("--max_samples", type=int,   default=50000)
    args = ap.parse_args()

    encoded_dir = Path(args.encoded_dir)
    ckpt_dir    = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tasks       = TASKS if args.task == "all" else [args.task]
    all_results = []

    print("[10] Experiment 2 training — Task B and Task C only")
    print(f"     encoded_dir : {encoded_dir}")
    print(f"     ckpt_dir    : {ckpt_dir}")
    print(f"     epochs      : {args.epochs}")
    print(f"     max_samples : {args.max_samples:,}")
    print(f"     task_a      : SKIPPED (Exp1 MCC=0.877)")
    print(f"     uci_splice  : SKIPPED (cross-species planned)")

    for task in tasks:
        result = train_task(
            task_name   = task,
            encoded_dir = encoded_dir,
            ckpt_dir    = ckpt_dir,
            epochs      = args.epochs,
            batch_size  = args.batch_size,
            lr          = args.lr,
            d_model     = args.d_model,
            n_heads     = args.n_heads,
            n_layers    = args.n_layers,
            ffn_dim     = args.ffn_dim,
            max_samples = args.max_samples,
        )
        all_results.append(result)

    print("\n" + "="*60)
    print("[Exp 2] Training summary")
    print("="*60)
    for r in all_results:
        print(f"  {r['task']:<16}  best_val_loss={r['best_val_loss']:.4f}")

    with open(ckpt_dir / "training_summary.json", "w", encoding="utf-8") as fh:
        json.dump([{"task": r["task"], "best_val_loss": r["best_val_loss"]}
                   for r in all_results], fh, indent=2)

    print(f"\nCheckpoints saved to: {ckpt_dir}/")
    print("Next step: run 11_evaluate_compare.py to compare Exp1 vs Exp2")


if __name__ == "__main__":
    main()