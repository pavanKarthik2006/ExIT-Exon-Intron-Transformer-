"""
07_train.py
===========
STEP 6 — Train the DNA classifier for each task.

Training strategy
-----------------
1. Task A (exon/intron, 2 classes)  — train encoder + head together
2. Task B (CDS/non-CDS, 2 classes)  — FREEZE encoder, train head only
3. Task C (splice site, 3 classes)  — FREEZE encoder, train head only

Checkpoints saved
-----------------
checkpoints/
    task_a_best.pt
    task_b_best.pt
    task_c_best.pt
    encoder_pretrained.pt
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

def load_split(encoded_dir: Path, name: str, split: str) -> TensorDataset:
    path = encoded_dir / f"{name}_{split}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Encoded file not found: {path}")
    data = torch.load(path, weights_only=True)
    return TensorDataset(data["input_ids"], data["labels"])


def get_num_classes(encoded_dir: Path, name: str) -> int:
    path = encoded_dir / f"{name}_train.pt"
    data = torch.load(path, weights_only=True)
    return int(data["labels"].max().item()) + 1


# ═══════════════════════════════════════════════════════════════════
# CLASS WEIGHTS
# ═══════════════════════════════════════════════════════════════════

def compute_class_weights(dataset: TensorDataset, num_classes: int) -> torch.Tensor:
    labels  = dataset.tensors[1]
    counts  = torch.bincount(labels, minlength=num_classes).float()
    counts  = counts.clamp(min=1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return weights


# ═══════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == labels).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> dict:
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for ids, lbls in loader:
        ids, lbls = ids.to(device), lbls.to(device)
        logits     = model(ids)
        total_loss += loss_fn(logits, lbls).item()
        total_acc  += accuracy(logits, lbls)
        n += 1
    return {"loss": total_loss / max(n, 1), "acc": total_acc / max(n, 1)}


# ═══════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

def train_task(
    task_name:    str,
    encoded_dir:  Path,
    ckpt_dir:     Path,
    encoder_ckpt: str   = None,
    freeze_enc:   bool  = True,
    epochs:       int   = 5,
    batch_size:   int   = 256,
    lr:           float = 3e-4,
    d_model:      int   = 128,
    n_heads:      int   = 4,
    n_layers:     int   = 4,
    ffn_dim:      int   = 256,
    log_every:    int   = 100,
    max_samples:  int   = None,
) -> dict:

    device = torch.device("cpu")
    print(f"\n{'='*60}")
    print(f"Training: {task_name}  (freeze_encoder={freeze_enc})")
    print(f"{'='*60}")

    # ── Data ──────────────────────────────────────────────────────
    train_ds = load_split(encoded_dir, task_name, "train")
    val_ds   = load_split(encoded_dir, task_name, "val")
    n_cls    = get_num_classes(encoded_dir, task_name)

    # Cap all splits proportionally to max_samples
    if max_samples is not None:
        # Train: cap to max_samples
        if len(train_ds) > max_samples:
            idx      = torch.randperm(len(train_ds))[:max_samples]
            train_ds = TensorDataset(*[t[idx] for t in train_ds.tensors])

        # Val: cap to 15% of max_samples (same ratio as original 70/15/15)
        val_cap = max(int(max_samples * 0.15), 1000)
        if len(val_ds) > val_cap:
            idx    = torch.randperm(len(val_ds))[:val_cap]
            val_ds = TensorDataset(*[t[idx] for t in val_ds.tensors])

        print(f"  [capped] train={len(train_ds):,}  val={len(val_ds):,}  "
              f"(proportional to max_samples={max_samples:,})")

    print(f"  num_classes={n_cls}  train={len(train_ds):,}  val={len(val_ds):,}")

    class_w = compute_class_weights(train_ds, n_cls).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)

    # Infer max_len from data shape
    max_len = int(train_ds.tensors[0].shape[1])

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────
    model = DNAClassifier.build(
        num_classes  = n_cls,
        d_model      = d_model,
        n_heads      = n_heads,
        n_layers     = n_layers,
        ffn_dim      = ffn_dim,
        max_len      = max_len,
        encoder_ckpt = encoder_ckpt,
    )

    if freeze_enc:
        model.freeze_encoder()

    params = model.count_parameters()
    print(f"  Params — encoder={params['encoder_total']:,}  "
          f"head={params['head_total']:,}  "
          f"trainable={params['encoder_trainable']+params['head_total']:,}")

    # ── Optimiser ─────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training ──────────────────────────────────────────────────
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
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            ep_loss += loss.item()
            ep_acc  += accuracy(logits, lbls)
            n_steps += 1

            if (step + 1) % log_every == 0:
                print(f"  epoch={epoch} step={step+1}/{len(train_loader)} "
                      f"loss={ep_loss/n_steps:.4f} acc={ep_acc/n_steps:.3f}",
                      flush=True)

        scheduler.step()
        val_m = evaluate(model, val_loader, loss_fn, device)
        tr_m  = {"loss": ep_loss / max(n_steps, 1),
                 "acc":  ep_acc  / max(n_steps, 1)}

        elapsed = time.time() - t0
        print(f"  -- epoch {epoch}/{epochs}  "
              f"train_loss={tr_m['loss']:.4f} train_acc={tr_m['acc']:.3f}  "
              f"val_loss={val_m['loss']:.4f} val_acc={val_m['acc']:.3f}  "
              f"({elapsed:.1f}s)", flush=True)

        history.append({"epoch": epoch,
                        **{f"train_{k}": v for k, v in tr_m.items()},
                        **{f"val_{k}":   v for k, v in val_m.items()}})

        # Save best checkpoint
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
                "freeze_enc":    freeze_enc,
            }, ckpt_path)
            print(f"  Checkpoint saved -> {ckpt_path}", flush=True)

    # Save encoder after Task A for reuse in other tasks
    if task_name == "task_a":
        enc_path = ckpt_dir / "encoder_pretrained.pt"
        torch.save(model.encoder.state_dict(), enc_path)
        print(f"  Pretrained encoder saved -> {enc_path}")

    print(f"  Best val_loss={best_val_loss:.4f}")
    return {"task": task_name, "best_val_loss": best_val_loss,
            "history": history}


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

# Each task trains encoder + head together (no freezing).
# The encoder adapts to the biological length of each task:
#   task_a     : 170 codon tokens (512nt exon/intron windows)
#   task_b     : 111 codon tokens (333nt CDS/non-CDS windows, min 150nt)
#   task_c     :  66 codon tokens (200nt splice site windows)
#   uci_splice :  20 codon tokens (60nt UCI sequences)
# max_len is inferred automatically from the .pt file shape.
TASK_CONFIGS = {
    "task_a": {"freeze_enc": False},   # train encoder + head together
    "task_b": {"freeze_enc": True},    # head only (encoder frozen)
    "task_c": {"freeze_enc": True},    # head only (encoder frozen)
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task",         default="all",
                    choices=list(TASK_CONFIGS.keys()) + ["all"])
    ap.add_argument("--encoded_dir",  default="data/05_encoded")
    ap.add_argument("--ckpt_dir",     default="checkpoints")
    ap.add_argument("--encoder_ckpt", default=None)
    ap.add_argument("--epochs",       type=int,   default=5)
    ap.add_argument("--batch_size",   type=int,   default=256)
    ap.add_argument("--lr",           type=float, default=3e-4)
    ap.add_argument("--d_model",      type=int,   default=128)
    ap.add_argument("--n_heads",      type=int,   default=4)
    ap.add_argument("--n_layers",     type=int,   default=4)
    ap.add_argument("--ffn_dim",      type=int,   default=256)
    ap.add_argument("--max_samples",  type=int,   default=50000,
                    help="Cap training samples per task (default 50000)")
    ap.add_argument("--train_encoder", action="store_true",
                    help="Fine-tune encoder on every task")
    args = ap.parse_args()

    encoded_dir = Path(args.encoded_dir)
    ckpt_dir    = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tasks       = list(TASK_CONFIGS.keys()) if args.task == "all" else [args.task]
    all_results = []

    for task in tasks:
        enc_ckpt = args.encoder_ckpt
        if enc_ckpt is None and task != "task_a":
            auto = ckpt_dir / "encoder_pretrained.pt"
            if auto.exists():
                enc_ckpt = str(auto)
                print(f"  Auto-loading encoder from {auto}")
            else:
                print(f"  WARNING: No encoder checkpoint found for {task}. "
                      f"Run task_a first.")

        # Default: train encoder + head together for all tasks
        # Use --train_encoder=false (default) for full end-to-end training
        freeze = TASK_CONFIGS[task]["freeze_enc"] and not args.train_encoder

        result = train_task(
            task_name    = task,
            encoded_dir  = encoded_dir,
            ckpt_dir     = ckpt_dir,
            encoder_ckpt = enc_ckpt,
            freeze_enc   = freeze,
            epochs       = args.epochs,
            batch_size   = args.batch_size,
            lr           = args.lr,
            d_model      = args.d_model,
            n_heads      = args.n_heads,
            n_layers     = args.n_layers,
            ffn_dim      = args.ffn_dim,
            max_samples  = args.max_samples,
        )
        all_results.append(result)

    print("\n" + "="*60)
    print("Training summary")
    print("="*60)
    for r in all_results:
        print(f"  {r['task']:<16}  best_val_loss={r['best_val_loss']:.4f}")

    summary_path = ckpt_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump([{"task": r["task"],
                    "best_val_loss": r["best_val_loss"]}
                   for r in all_results], fh, indent=2)
    print(f"\nSummary saved -> {summary_path}")


if __name__ == "__main__":
    main()