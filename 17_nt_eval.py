"""
19_nt_linear_probe.py
======================
Evaluates Nucleotide Transformer (NT-500M) on all three tasks
using a linear probe approach.

Model used
----------
  InstaDeepAI/nucleotide-transformer-500m-human-ref
  - 500M parameters, pretrained on 3,202 human reference genomes
  - Uses 6-mer tokenisation (k-mers of length 6)
  - Context window: 1,000 tokens = 6,000 nucleotides
  - Released by InstaDeep/NVIDIA/TUM (Nature Methods 2024)

Why NT-500M-human-ref and not the larger models
------------------------------------------------
  2.5B model : too large for CPU (>10GB RAM)
  500M-1000g : pretrained on 1000 genomes (multi-individual)
  500M-human-ref : pretrained specifically on the human reference
                   genome — most directly relevant to your tasks

Linear probe approach
---------------------
  1. Download pretrained NT weights (once, ~2GB cached locally)
  2. Freeze ALL NT weights — zero training of the backbone
  3. Extract CLS-token embeddings (dim=1024) for each sequence
  4. Train a single linear layer on top for 5 epochs
  5. Evaluate on your chromosome-split test set
  6. Compare MCC/F1 vs your codon transformer

Requirements
------------
  pip install transformers

Usage
-----
  python scripts\\19_nt_linear_probe.py
  python scripts\\19_nt_linear_probe.py --max_train 3000 --epochs 5
  python scripts\\19_nt_linear_probe.py --tasks task_a
"""

import argparse, os, random, sys, time, itertools
from pathlib import Path

# Disable CUDA and flash attention before any torch/transformers import
os.environ["CUDA_VISIBLE_DEVICES"]            = ""
os.environ["USE_FLASH_ATTENTION"]             = "0"
os.environ["TRANSFORMERS_NO_FLASH_ATTENTION"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW

try:
    from transformers import AutoTokenizer, AutoModelForMaskedLM
except ImportError:
    print("[ERROR] transformers not installed. Run: pip install transformers")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

NT_MODEL_NAME = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
NT_EMBED_DIM  = 1280   # hidden size for NT-500M-human-ref

# Sequence lengths per task
TASK_NT_LEN = {
    "task_a": 512,
    "task_b": 333,
    "task_c": 200,
}

TASK_NUM_CLASSES = {
    "task_a": 2,
    "task_b": 2,
    "task_c": 3,
}

TASK_LABELS = {
    "task_a": {0: "intron",    1: "exon"},
    "task_b": {0: "non_cds",   1: "cds"},
    "task_c": {0: "no_splice", 1: "donor", 2: "acceptor"},
}

# FASTA files to read sequences from
TASK_FASTAS = {
    "task_a": {
        "train": [
            ("data/02_splits_chrom/exons_train.fa",   1),
            ("data/02_splits_chrom/introns_train.fa", 0),
        ],
        "test": [
            ("data/02_splits_chrom/exons_test.fa",    1),
            ("data/02_splits_chrom/introns_test.fa",  0),
        ],
    },
    "task_b": {
        "train": [
            ("data/02_splits_chrom/cds_train.fa",     1),
            ("data/02_splits_chrom/non_cds_train.fa", 0),
        ],
        "test": [
            ("data/02_splits_chrom/cds_test.fa",      1),
            ("data/02_splits_chrom/non_cds_test.fa",  0),
        ],
    },
    "task_c": {
        "train": [
            ("data/02_splits_chrom/splice_donors_train.fa",    1),
            ("data/02_splits_chrom/splice_acceptors_train.fa", 2),
            ("data/02_splits_chrom/introns_train.fa",          0),
        ],
        "test": [
            ("data/02_splits_chrom/splice_donors_test.fa",     1),
            ("data/02_splits_chrom/splice_acceptors_test.fa",  2),
            ("data/02_splits_chrom/introns_test.fa",           0),
        ],
    },
}

NEG_RATIO_C = 2

# Your codon transformer results for comparison
BASELINES = {
    "task_a": {"mcc": 0.8719, "f1": 0.9343, "acc": 0.9343,
               "bal_acc": 0.9364, "model": "Codon Transformer (yours)"},
    "task_b": {"mcc": 0.4452, "f1": 0.5566, "acc": 0.5892,
               "bal_acc": 0.6029, "model": "Codon Transformer (yours)"},
    "task_c": {"mcc": 0.4972, "f1": 0.6354, "acc": 0.6535,
               "bal_acc": 0.6536, "model": "Codon Transformer (yours)"},
}


# ═══════════════════════════════════════════════════════════════════
# FASTA READER
# ═══════════════════════════════════════════════════════════════════

def read_fasta(path):
    path = Path(path)
    if not path.exists():
        print(f"  [WARN] FASTA not found: {path}")
        return
    hdr, buf = None, []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if hdr:
                    yield hdr, "".join(buf)
                hdr, buf = line[1:], []
            else:
                buf.append(line)
    if hdr:
        yield hdr, "".join(buf)


def load_sequences(fasta_specs, task, max_seqs=None, seed=42):
    neg_cap   = None
    if task == "task_c":
        n_pos = sum(
            sum(1 for _ in read_fasta(fp))
            for fp, lbl in fasta_specs if lbl != 0
        )
        neg_cap = n_pos * NEG_RATIO_C

    pairs, neg_count = [], 0
    for fpath, label in fasta_specs:
        for _, seq in read_fasta(fpath):
            if len(seq) < 50:
                continue
            if label == 0 and neg_cap is not None:
                if neg_count >= neg_cap:
                    continue
                neg_count += 1
            pairs.append((seq, label))

    random.Random(seed).shuffle(pairs)
    if max_seqs is not None:
        pairs = pairs[:max_seqs]
    return pairs


# ═══════════════════════════════════════════════════════════════════
# NUCLEOTIDE TRANSFORMER EMBEDDER
# ═══════════════════════════════════════════════════════════════════

class NTEmbedder:
    """
    Wraps Nucleotide Transformer 500M.
    All weights frozen — only used for embedding extraction.
    Returns mean-pooled hidden states (dim=1024).
    """

    def __init__(self, device):
        self.device = device
        print(f"[19] Loading Nucleotide Transformer ...")
        print(f"     Model : {NT_MODEL_NAME}")
        print(f"     Size  : 500M parameters (all frozen)")
        print(f"     Note  : First run downloads ~2GB, cached after.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            NT_MODEL_NAME, trust_remote_code=True
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            NT_MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        self.model.eval()
        self.model.to(device)

        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad = False

        n = sum(p.numel() for p in self.model.parameters())
        print(f"     Parameters : {n:,}  (all frozen)")

    @torch.no_grad()
    def embed_batch(self, seqs, max_nt):
        """
        Tokenise a list of DNA strings and return mean-pooled
        hidden state embeddings of shape [B, 1024].
        """
        seqs_trunc = [s[:max_nt].upper() for s in seqs]

        inputs = self.tokenizer(
            seqs_trunc,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=512,    # NT token limit
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
        )

        # Use last hidden layer, mean pool over sequence tokens
        # (exclude CLS and PAD tokens)
        hidden       = outputs.hidden_states[-1]   # [B, T, 1024]
        attention_mask = inputs["attention_mask"]  # [B, T]
        mask_expanded = attention_mask.unsqueeze(-1).float()
        summed = (hidden * mask_expanded).sum(dim=1)
        counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_embed = (summed / counts).cpu()       # [B, 1024]

        return mean_embed


# ═══════════════════════════════════════════════════════════════════
# EMBED ALL SEQUENCES
# ═══════════════════════════════════════════════════════════════════

def embed_all(pairs, embedder, max_nt, batch_size):
    all_embeds, all_labels = [], []
    n = len(pairs)

    for i in range(0, n, batch_size):
        batch  = pairs[i:i+batch_size]
        seqs   = [p[0] for p in batch]
        labels = [p[1] for p in batch]
        embeds = embedder.embed_batch(seqs, max_nt)
        all_embeds.append(embeds)
        all_labels.extend(labels)

        done = min(i+batch_size, n)
        print(f"    [{done:>6}/{n}]  embedded", end="\r", flush=True)

    print()
    return (
        torch.cat(all_embeds, dim=0),
        torch.tensor(all_labels, dtype=torch.long),
    )


# ═══════════════════════════════════════════════════════════════════
# LINEAR PROBE
# ═══════════════════════════════════════════════════════════════════

class LinearProbe(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


def train_probe(train_embeds, train_labels, n_cls, epochs, lr=1e-3,
                batch_size=256):
    counts  = torch.bincount(train_labels,
                             minlength=n_cls).float().clamp(min=1)
    weights = 1.0/counts; weights = weights/weights.sum()*n_cls
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    ds     = TensorDataset(train_embeds, train_labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    probe  = LinearProbe(NT_EMBED_DIM, n_cls)
    opt    = AdamW(probe.parameters(), lr=lr, weight_decay=0.01)

    probe.train()
    for epoch in range(1, epochs+1):
        total = 0.0
        for emb, lbl in loader:
            opt.zero_grad()
            loss = loss_fn(probe(emb), lbl)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"    epoch {epoch}/{epochs}  "
              f"loss={total/len(loader):.4f}")
    return probe


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
    return correct/total if total > 0 else 0.0

def balanced_accuracy(cm, n_cls):
    recalls = []
    for c in range(n_cls):
        tp = cm[c][c]
        fn = sum(cm[c][j] for j in range(n_cls)) - tp
        recalls.append(tp/(tp+fn) if (tp+fn) > 0 else 0.0)
    return sum(recalls)/n_cls

def macro_f1(cm, n_cls):
    f1s = []
    for c in range(n_cls):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(n_cls)) - tp
        fn = sum(cm[c][r] for r in range(n_cls)) - tp
        p  = tp/(tp+fp) if (tp+fp) > 0 else 0.0
        r  = tp/(tp+fn) if (tp+fn) > 0 else 0.0
        f1s.append(2*p*r/(p+r) if (p+r) > 0 else 0.0)
    return sum(f1s)/n_cls

def mcc(cm, n_cls):
    total = sum(cm[i][j] for i in range(n_cls) for j in range(n_cls))
    xy = xx = yy = 0.0
    for k in range(n_cls):
        for l in range(n_cls):
            for m_ in range(n_cls):
                xy += cm[k][k]*cm[m_][l] - cm[l][k]*cm[k][m_]
        s = sum(cm[k][j] for j in range(n_cls))
        t = sum(cm[j][k] for j in range(n_cls))
        xx += s*(total-s); yy += t*(total-t)
    return xy/(xx*yy)**0.5 if xx > 0 and yy > 0 else 0.0

def per_class_metrics(cm, n_cls):
    out = {}
    for c in range(n_cls):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(n_cls)) - tp
        fn = sum(cm[c][r] for r in range(n_cls)) - tp
        p  = tp/(tp+fp) if (tp+fp) > 0 else 0.0
        r  = tp/(tp+fn) if (tp+fn) > 0 else 0.0
        out[c] = {"precision": p, "recall": r,
                  "f1": 2*p*r/(p+r) if (p+r) > 0 else 0.0,
                  "tp": tp, "fp": fp, "fn": fn}
    return out


@torch.no_grad()
def eval_probe(probe, test_embeds, test_labels, n_cls, batch_size=256):
    probe.eval()
    ds     = TensorDataset(test_embeds, test_labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds, labels = [], []
    for emb, lbl in loader:
        preds.extend(probe(emb).argmax(-1).tolist())
        labels.extend(lbl.tolist())
    cm_ = confusion_matrix(preds, labels, n_cls)
    return {
        "n_samples":         len(labels),
        "accuracy":          accuracy(cm_, n_cls),
        "balanced_accuracy": balanced_accuracy(cm_, n_cls),
        "macro_f1":          macro_f1(cm_, n_cls),
        "mcc":               mcc(cm_, n_cls),
        "confusion_matrix":  cm_,
        "per_class":         per_class_metrics(cm_, n_cls),
    }


# ═══════════════════════════════════════════════════════════════════
# WRITERS
# ═══════════════════════════════════════════════════════════════════

def write_tsv(all_metrics, out_dir):
    path = out_dir / "nt_eval_summary.tsv"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("task\tmodel\tparams\ttest_mcc\ttest_macro_f1\t"
                 "test_bal_acc\ttest_accuracy\tn_samples\t"
                 "baseline_mcc\tdelta_mcc\n")
        for task, m in all_metrics.items():
            base  = BASELINES[task]
            delta = m["mcc"] - base["mcc"]
            fh.write(
                f"{task}\tNT-500M (linear probe)\t500M\t"
                f"{m['mcc']:.6f}\t{m['macro_f1']:.6f}\t"
                f"{m['balanced_accuracy']:.6f}\t{m['accuracy']:.6f}\t"
                f"{m['n_samples']}\t{base['mcc']:.6f}\t{delta:+.6f}\n"
            )
        for task, base in BASELINES.items():
            if task in all_metrics:
                fh.write(
                    f"{task}\t{base['model']}\t~1.5M\t"
                    f"{base['mcc']:.6f}\t{base['f1']:.6f}\t"
                    f"{base['bal_acc']:.6f}\t{base['acc']:.6f}\t"
                    f"N/A\t{base['mcc']:.6f}\t+0.000000\n"
                )
    print(f"[19] nt_eval_summary.tsv    -> {path}")


def write_report(all_metrics, out_dir):
    path = out_dir / "nt_eval_report.txt"
    with open(path, "w", encoding="utf-8") as fh:
        def w(s=""): fh.write(s+"\n")

        w("="*70)
        w("NUCLEOTIDE TRANSFORMER (500M) LINEAR PROBE")
        w("vs CODON TRANSFORMER (YOURS)")
        w("-"*70)
        w(f"NT Model : {NT_MODEL_NAME}")
        w(f"NT Size  : 500M parameters (all frozen)")
        w(f"Probe    : linear layer trained 5 epochs")
        w(f"Split    : chromosome-based (test=chr21,22,X,Y)")
        w("="*70)

        w()
        w("MCC COMPARISON")
        w("-"*70)
        w(f"  {'Task':<12} {'NT-500M':>10} {'Yours':>12} "
          f"{'Delta':>10}  Winner")
        w("  "+"-"*56)
        for task, m in all_metrics.items():
            base   = BASELINES[task]["mcc"]
            delta  = m["mcc"] - base
            winner = ("NT-500M" if delta >  0.005 else
                      "Yours"   if delta < -0.005 else "Tie")
            w(f"  {task:<12} {m['mcc']:>10.4f} {base:>12.4f} "
              f"{delta:>+10.4f}  {winner}")

        w()
        w("FULL METRICS")
        w("-"*70)
        w(f"  {'Task':<10} {'Model':<26} {'MCC':>8} {'F1':>8} "
          f"{'BalAcc':>8} {'Acc':>8}")
        w("  "+"-"*68)
        for task, m in all_metrics.items():
            base = BASELINES[task]
            w(f"  {task:<10} {'NT-500M (linear probe)':<26} "
              f"{m['mcc']:>8.4f} {m['macro_f1']:>8.4f} "
              f"{m['balanced_accuracy']:>8.4f} {m['accuracy']:>8.4f}")
            w(f"  {task:<10} {base['model']:<26} "
              f"{base['mcc']:>8.4f} {base['f1']:>8.4f} "
              f"{base['bal_acc']:>8.4f} {base['acc']:>8.4f}")

        w()
        w("MODEL COMPARISON TABLE (paper-ready)")
        w("-"*70)
        w(f"  {'Model':<30} {'Params':>8} {'TaskA MCC':>10} "
          f"{'TaskB MCC':>10} {'TaskC MCC':>10} {'Training':>12}")
        w("  "+"-"*80)
        w(f"  {'NT-500M (linear probe)':<30} {'500M':>8} "
          f"{all_metrics.get('task_a',{}).get('mcc',0):>10.4f} "
          f"{all_metrics.get('task_b',{}).get('mcc',0):>10.4f} "
          f"{all_metrics.get('task_c',{}).get('mcc',0):>10.4f} "
          f"{'Pretrained':>12}")
        w(f"  {'Codon Transformer (ours)':<30} {'~1.5M':>8} "
          f"{BASELINES['task_a']['mcc']:>10.4f} "
          f"{BASELINES['task_b']['mcc']:>10.4f} "
          f"{BASELINES['task_c']['mcc']:>10.4f} "
          f"{'From scratch':>12}")

        w()
        w("PER-CLASS BREAKDOWN — NT-500M")
        w("-"*70)
        for task, m in all_metrics.items():
            labels = TASK_LABELS[task]
            w()
            w(f"  {task.upper()}")
            w(f"  {'Class':<16} {'Precision':>10} {'Recall':>10} "
              f"{'F1':>8} {'TP':>8} {'FP':>8} {'FN':>8}")
            w("  "+"-"*62)
            for c, pc in m["per_class"].items():
                w(f"  {labels.get(c,str(c)):<16} "
                  f"{pc['precision']:>10.4f} {pc['recall']:>10.4f} "
                  f"{pc['f1']:>8.4f} {pc['tp']:>8} "
                  f"{pc['fp']:>8} {pc['fn']:>8}")

        w()
        w("="*70)
        w("NOTE: NT backbone is fully frozen. Only a linear head")
        w("(1,024 x n_classes weights) was trained on 5,000 sequences.")
        w("This is the standard linear probe evaluation protocol from")
        w("the Nucleotide Transformer paper (Nature Methods 2024).")
        w("="*70)

    print(f"[19] nt_eval_report.txt     -> {path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Nucleotide Transformer 500M linear probe evaluation"
    )
    ap.add_argument("--tasks",      nargs="+",
                    default=["task_a","task_b","task_c"],
                    choices=["task_a","task_b","task_c"])
    ap.add_argument("--out_dir",    default="results_nt_probe")
    ap.add_argument("--max_train",  type=int, default=5000,
                    help="Training sequences for linear probe (default 5000)")
    ap.add_argument("--epochs",     type=int, default=5,
                    help="Linear probe training epochs (default 5)")
    ap.add_argument("--batch_size", type=int, default=8,
                    help="Embedding batch size — lower if slow (default 8)")
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cpu")

    print("[19] Nucleotide Transformer Linear Probe Evaluation")
    print(f"     Tasks      : {args.tasks}")
    print(f"     max_train  : {args.max_train:,}")
    print(f"     epochs     : {args.epochs}")
    print(f"     batch_size : {args.batch_size}")
    print(f"     out_dir    : {out_dir}")
    print()

    # Load NT once — shared across all tasks
    embedder = NTEmbedder(device)

    all_metrics = {}

    for task in args.tasks:
        n_cls   = TASK_NUM_CLASSES[task]
        max_nt  = TASK_NT_LEN[task]

        print(f"\n{'='*60}")
        print(f"[19] Task: {task.upper()}   "
              f"n_classes={n_cls}   max_nt={max_nt}")
        print(f"{'='*60}")

        # Load sequences
        print("  Loading train sequences ...")
        train_pairs = load_sequences(
            TASK_FASTAS[task]["train"], task,
            max_seqs=args.max_train, seed=args.seed
        )
        print("  Loading test sequences ...")
        test_pairs = load_sequences(
            TASK_FASTAS[task]["test"], task,
            max_seqs=None, seed=args.seed
        )
        print(f"  train={len(train_pairs):,}  "
              f"test={len(test_pairs):,}")

        # Embed with NT
        print("  Embedding train sequences with NT-500M ...")
        t0 = time.time()
        train_embeds, train_labels = embed_all(
            train_pairs, embedder, max_nt, args.batch_size)

        print("  Embedding test sequences with NT-500M ...")
        test_embeds, test_labels = embed_all(
            test_pairs, embedder, max_nt, args.batch_size)

        embed_min = (time.time()-t0)/60
        print(f"  Embedding done ({embed_min:.1f} min)")

        # Train linear probe
        print(f"  Training linear probe ({args.epochs} epochs) ...")
        probe = train_probe(
            train_embeds, train_labels,
            n_cls=n_cls, epochs=args.epochs
        )

        # Evaluate
        print("  Evaluating on test chromosomes ...")
        m = eval_probe(probe, test_embeds, test_labels, n_cls)
        all_metrics[task] = m

        base  = BASELINES[task]
        delta = m["mcc"] - base["mcc"]
        print(f"\n  NT-500M MCC     : {m['mcc']:.4f}")
        print(f"  Your model MCC  : {base['mcc']:.4f}")
        print(f"  Delta           : {delta:+.4f}  "
              f"({'NT wins' if delta > 0.005 else 'Yours wins' if delta < -0.005 else 'Tie'})")
        print(f"  NT-500M F1      : {m['macro_f1']:.4f}")
        print(f"  NT-500M BalAcc  : {m['balanced_accuracy']:.4f}")

    print("\n[19] Writing outputs ...")
    write_tsv(all_metrics, out_dir)
    write_report(all_metrics, out_dir)

    print(f"\n{'='*60}")
    print(f"[19] FINAL SUMMARY — NT-500M vs Codon Transformer")
    print(f"{'='*60}")
    print(f"  {'Task':<12} {'NT-500M MCC':>13} {'Yours MCC':>12} "
          f"{'Delta':>10}  Winner")
    print(f"  {'-'*56}")
    for task, m in all_metrics.items():
        base   = BASELINES[task]["mcc"]
        delta  = m["mcc"] - base
        winner = ("NT-500M" if delta >  0.005 else
                  "Yours"   if delta < -0.005 else "Tie")
        print(f"  {task:<12} {m['mcc']:>13.4f} {base:>12.4f} "
              f"{delta:>+10.4f}  {winner}")

    print(f"\n[19] Done. Results -> {out_dir}/")


if __name__ == "__main__":
    main()