"""
09_preprocess_bio.py
====================
EXPERIMENT 2 — Re-encode Task B and Task C with biologically
appropriate sequence lengths.

Task A is SKIPPED — Experiment 1 achieved MCC=0.877 which is
excellent and does not need improvement.

UCI is SKIPPED — replaced by cross-species generalisation
validation in a later script.

Biological length rationale
----------------------------
  Task B (CDS/non-CDS)  : 333 nt -> 111 codon tokens
      CDS must be codon-complete (divisible by 3).
      333nt = 111 codons, long enough for codon usage bias to emerge.
      Sequences shorter than 150nt are filtered out (< 50 codons).
      In Experiment 1 this used 512nt causing excessive padding noise.

  Task C (splice sites) : 200 nt -> 66 codon tokens
      Fixed window centred on splice junction.
      GT donor / AG acceptor signals captured as atomic codon tokens.
      In Experiment 1 these 200nt sequences were wrongly padded to 512nt.

Outputs  [data/05_encoded_bio/]
-------
  task_b_{train|val|test}.pt   333nt -> 111 tokens, min_len=150
  task_c_{train|val|test}.pt   200nt ->  66 tokens
  codon_vocab.json
  encoding_stats.txt
"""

import argparse, json, itertools, time
from pathlib import Path
from collections import Counter
import torch

# ── Biologically appropriate nucleotide lengths ───────────────────
SEQ_LEN_NT_TASK_B = 333   # CDS/non-CDS  -> 111 codon tokens (multiple of 3)
SEQ_LEN_NT_TASK_C = 200   # splice sites ->  66 codon tokens
MIN_LEN_TASK_B    = 150   # filter CDS/UTR fragments shorter than 50 codons

CHUNK_SIZE        = 10_000


# ═══════════════════════════════════════════════════════════════════
# CODON VOCABULARY
# ═══════════════════════════════════════════════════════════════════

def _build_codon_vocab():
    bases  = ["A", "C", "G", "T"]
    codons = sorted("".join(c) for c in itertools.product(bases, repeat=3))
    vocab  = {"<PAD>": 0, "<UNK>": 1, "<MASK>": 2}
    for i, codon in enumerate(codons):
        vocab[codon] = i + 3
    return vocab

CODON_VOCAB = _build_codon_vocab()
VOCAB_SIZE  = len(CODON_VOCAB)   # 67

IUPAC_RESOLVE = {
    "R":"A","Y":"C","S":"G","W":"A","K":"G","M":"A",
    "B":"C","D":"A","H":"A","V":"A","N":"A","X":"A",
}

def resolve_iupac(seq):
    return "".join(IUPAC_RESOLVE.get(c, c) for c in seq.upper())

def codon_tokenise(seq, max_nt):
    seq     = resolve_iupac(seq)[:max_nt]
    max_tok = max_nt // 3
    tokens  = []
    for i in range(0, len(seq) - 2, 3):
        tokens.append(CODON_VOCAB.get(seq[i:i+3], CODON_VOCAB["<UNK>"]))
    tokens += [CODON_VOCAB["<PAD>"]] * (max_tok - len(tokens))
    return tokens[:max_tok]


# ═══════════════════════════════════════════════════════════════════
# FASTA READER
# ═══════════════════════════════════════════════════════════════════

def read_fasta(path):
    if not Path(path).exists():
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


# ═══════════════════════════════════════════════════════════════════
# CHUNKED ENCODER
# ═══════════════════════════════════════════════════════════════════

def encode_and_save_chunked(sources, out_path, max_nt, task, split,
                             max_neg=None, chunk_size=CHUNK_SIZE,
                             min_len=0):
    t0         = time.time()
    all_ids    = []
    all_labels = []
    total      = 0
    neg_count  = 0
    chunk_ids  = []
    chunk_lbls = []
    filtered   = 0

    def flush():
        if chunk_ids:
            all_ids.append(torch.tensor(chunk_ids,  dtype=torch.long))
            all_labels.append(torch.tensor(chunk_lbls, dtype=torch.long))
        chunk_ids.clear()
        chunk_lbls.clear()

    for fasta_path, label in sources:
        if not Path(fasta_path).exists():
            print(f"    WARNING: {fasta_path} not found — skipping.")
            continue
        for _, seq in read_fasta(fasta_path):
            if len(seq) < min_len:
                filtered += 1
                continue
            if max_neg is not None and label == 0:
                if neg_count >= max_neg:
                    break
                neg_count += 1

            chunk_ids.append(codon_tokenise(seq, max_nt))
            chunk_lbls.append(label)
            total += 1

            if len(chunk_ids) >= chunk_size:
                flush()
                print(f"    [{task}/{split}] {total:,} encoded ...", flush=True)

    flush()

    if not all_ids:
        print(f"  [{task}/{split}] No sequences found — skipping.")
        return

    final_ids    = torch.cat(all_ids,    dim=0)
    final_labels = torch.cat(all_labels, dim=0)
    torch.save({"input_ids": final_ids, "labels": final_labels}, out_path)

    c        = Counter(final_labels.tolist())
    tok_len  = final_ids.shape[1]
    dist     = "  ".join(
        f"cls{k}={v}({v/total*100:.1f}%)" for k, v in sorted(c.items())
    )
    elapsed  = time.time() - t0
    filt_str = f"  filtered_short={filtered:,}" if filtered else ""
    print(f"  [{task}/{split}] N={total:,}  tok_len={tok_len} (={tok_len*3}nt)"
          f"  {dist}{filt_str}  ({elapsed:.1f}s)")


# ═══════════════════════════════════════════════════════════════════
# TASK ENCODERS
# ═══════════════════════════════════════════════════════════════════

def encode_task_b(split_dir, split, out_dir, chunk_size=CHUNK_SIZE):
    """
    Binary: non_cds=0, cds=1
    333nt -> 111 codon tokens
    Sequences < 150nt filtered (too short for codon usage signal).
    """
    encode_and_save_chunked(
        sources=[
            (split_dir / f"non_cds_{split}.fa", 0),
            (split_dir / f"cds_{split}.fa",     1),
        ],
        out_path   = out_dir / f"task_b_{split}.pt",
        max_nt     = SEQ_LEN_NT_TASK_B,
        task       = "task_b",
        split      = split,
        chunk_size = chunk_size,
        min_len    = MIN_LEN_TASK_B,
    )


def encode_task_c(split_dir, split, out_dir, neg_ratio=2,
                  chunk_size=CHUNK_SIZE):
    """
    3-class: no_splice=0, donor=1, acceptor=2
    200nt -> 66 codon tokens
    Correct biological splice site window (was wrongly 512nt in Exp 1).
    """
    n_donors    = sum(1 for _ in read_fasta(
                      split_dir / f"splice_donors_{split}.fa"))
    n_acceptors = sum(1 for _ in read_fasta(
                      split_dir / f"splice_acceptors_{split}.fa"))
    max_neg     = (n_donors + n_acceptors) * neg_ratio

    encode_and_save_chunked(
        sources=[
            (split_dir / f"splice_donors_{split}.fa",    1),
            (split_dir / f"splice_acceptors_{split}.fa", 2),
            (split_dir / f"introns_{split}.fa",          0),
        ],
        out_path   = out_dir / f"task_c_{split}.pt",
        max_nt     = SEQ_LEN_NT_TASK_C,
        task       = "task_c",
        split      = split,
        max_neg    = max_neg,
        chunk_size = chunk_size,
    )


# ═══════════════════════════════════════════════════════════════════
# VOCAB + STATS
# ═══════════════════════════════════════════════════════════════════

def write_vocab(out_dir):
    payload = {
        "experiment":     "Experiment 2 — biological lengths",
        "strategy":       "codon — non-overlapping 3-mer",
        "vocab_size":     VOCAB_SIZE,
        "special_tokens": {"<PAD>": 0, "<UNK>": 1, "<MASK>": 2},
        "codon_to_idx":   CODON_VOCAB,
        "iupac_resolve":  IUPAC_RESOLVE,
        "task_lengths": {
            "task_b": {
                "nt": SEQ_LEN_NT_TASK_B,
                "tokens": SEQ_LEN_NT_TASK_B // 3,
                "min_len_nt": MIN_LEN_TASK_B,
            },
            "task_c": {
                "nt": SEQ_LEN_NT_TASK_C,
                "tokens": SEQ_LEN_NT_TASK_C // 3,
            },
        },
    }
    with open(out_dir / "codon_vocab.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("  [vocab] codon_vocab.json saved")


def write_stats(out_dir):
    with open(out_dir / "encoding_stats.txt", "w", encoding="utf-8") as fh:
        fh.write("Experiment 2 - Biological Length Encoding\n")
        fh.write("=" * 55 + "\n\n")
        fh.write("Task A : SKIPPED (Exp1 MCC=0.877 sufficient)\n")
        fh.write(f"Task B : {SEQ_LEN_NT_TASK_B}nt -> {SEQ_LEN_NT_TASK_B//3} tokens"
                 f"  (min_len={MIN_LEN_TASK_B}nt)\n")
        fh.write(f"Task C : {SEQ_LEN_NT_TASK_C}nt -> {SEQ_LEN_NT_TASK_C//3} tokens\n")
        fh.write("UCI    : SKIPPED (cross-species generalisation planned)\n\n")
        for p in sorted(out_dir.glob("*.pt")):
            d   = torch.load(p, weights_only=True)
            n, l = d["input_ids"].shape
            c   = Counter(d["labels"].tolist())
            fh.write(f"{p.name}\n")
            fh.write(f"  samples   : {n:,}\n")
            fh.write(f"  token_len : {l}  (= {l*3} nt)\n")
            for cls, cnt in sorted(c.items()):
                fh.write(f"  class {cls}   : {cnt:,} ({cnt/n*100:.1f}%)\n")
            fh.write("\n")
    print("  [stats] encoding_stats.txt written")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Experiment 2 — encode Task B and C with biological lengths"
    )
    ap.add_argument("--chrom_dir",  default="data/02_splits_chrom")
    ap.add_argument("--out_dir",    default="data/05_encoded_bio")
    ap.add_argument("--chunk_size", type=int, default=10000)
    args = ap.parse_args()

    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chrom_dir  = Path(args.chrom_dir)
    chunk_size = args.chunk_size

    print("[09] Experiment 2 - Biological length encoding")
    print(f"     Output dir : {out_dir}")
    print(f"     Task A     : SKIPPED (Exp1 MCC=0.877 sufficient)")
    print(f"     Task B     : {SEQ_LEN_NT_TASK_B}nt -> {SEQ_LEN_NT_TASK_B//3} tokens"
          f"  (min_len={MIN_LEN_TASK_B}nt)")
    print(f"     Task C     : {SEQ_LEN_NT_TASK_C}nt -> {SEQ_LEN_NT_TASK_C//3} tokens")
    print(f"     UCI        : SKIPPED (cross-species generalisation planned)")

    for split in ("train", "val", "test"):
        print(f"\n  -- {split} --")
        encode_task_b(chrom_dir, split, out_dir, chunk_size)
        encode_task_c(chrom_dir, split, out_dir, chunk_size=chunk_size)

    write_vocab(out_dir)
    write_stats(out_dir)
    print(f"\n[09] Done. Files saved to {out_dir}")

if __name__ == "__main__":
    main()