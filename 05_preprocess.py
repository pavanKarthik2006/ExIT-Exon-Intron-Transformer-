"""
05_preprocess.py
================
STEP 4 — Encode all tasks into PyTorch-ready .pt tensor files.

Tokenisation: CODON-BASED (non-overlapping 3-mers)
    512 nt -> 170 codon tokens (GRCh38)
     60 nt ->  20 codon tokens (UCI)
    Vocab: 67 tokens (64 codons + PAD=0, UNK=1, MASK=2)

Memory-safe design
------------------
Sequences are processed in chunks of CHUNK_SIZE (default 10,000).
Each chunk is encoded, converted to tensor, and saved immediately.
Final .pt file is assembled by concatenating all chunk tensors.
Peak RAM per task = CHUNK_SIZE x 170 x 8 bytes = ~14 MB regardless
of dataset size. Safe on any machine.
"""

import argparse, json, random, itertools, time
from pathlib import Path
from collections import Counter
import torch

SEQ_LEN_NT_GRCH = 512
SEQ_LEN_NT_UCI  = 60
CHUNK_SIZE      = 10_000   # sequences per chunk — controls peak RAM (override with --chunk_size)


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

assert VOCAB_SIZE == 67
assert CODON_VOCAB["AAA"] == 3
assert CODON_VOCAB["TTT"] == 66

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
    with open(path) as fh:
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
# CHUNKED ENCODER + SAVER
# ═══════════════════════════════════════════════════════════════════

def encode_and_save_chunked(sources, out_path, max_nt, task, split,
                             max_neg=None, chunk_size=CHUNK_SIZE):
    """
    sources : list of (fasta_path, label_int) tuples
              e.g. [(exons_train.fa, 1), (introns_train.fa, 0)]
    max_neg : cap total negatives (label=0) to this number (for Task C balance)

    Processes CHUNK_SIZE sequences at a time.
    Saves a single .pt file with all input_ids and labels concatenated.
    Peak RAM = CHUNK_SIZE × max_nt//3 × 8 bytes ≈ 14 MB per chunk.
    """
    t0          = time.time()
    tok_len     = max_nt // 3
    all_ids     = []   # list of tensors, one per chunk
    all_labels  = []
    total       = 0
    neg_count   = 0
    chunk_ids   = []
    chunk_lbls  = []

    def flush_chunk():
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
            # Cap negatives for Task C balance
            if max_neg is not None and label == 0:
                if neg_count >= max_neg:
                    break
                neg_count += 1

            chunk_ids.append(codon_tokenise(seq, max_nt))
            chunk_lbls.append(label)
            total += 1

            # Flush chunk to tensor when full
            if len(chunk_ids) >= chunk_size:
                flush_chunk()
                print(f"    [{task}/{split}] {total:,} sequences encoded ...",
                      flush=True)

    flush_chunk()   # flush any remaining

    if not all_ids:
        print(f"  [{task}/{split}] No sequences — skipping.")
        return

    # Concatenate all chunks into one tensor
    final_ids    = torch.cat(all_ids,    dim=0)
    final_labels = torch.cat(all_labels, dim=0)

    torch.save({"input_ids": final_ids, "labels": final_labels}, out_path)

    c     = Counter(final_labels.tolist())
    dist  = "  ".join(
        f"cls{k}={v}({v/total*100:.1f}%)" for k, v in sorted(c.items())
    )
    elapsed = time.time() - t0
    print(f"  [{task}/{split}] N={total:,}  tok_len={tok_len}  "
          f"{dist}  ({elapsed:.1f}s)")


# ═══════════════════════════════════════════════════════════════════
# TASK ENCODERS
# ═══════════════════════════════════════════════════════════════════

def encode_task_a(split_dir, split, out_dir, chunk_size=CHUNK_SIZE):
    """Binary: intron=0, exon=1"""
    encode_and_save_chunked(
        sources=[
            (split_dir / f"introns_{split}.fa", 0),
            (split_dir / f"exons_{split}.fa",   1),
        ],
        out_path=out_dir / f"task_a_{split}.pt",
        max_nt=SEQ_LEN_NT_GRCH,
        task="task_a", split=split,
        chunk_size=chunk_size,
    )


def encode_task_b(split_dir, split, out_dir, chunk_size=CHUNK_SIZE):
    """Binary: non_cds=0, cds=1"""
    encode_and_save_chunked(
        sources=[
            (split_dir / f"non_cds_{split}.fa", 0),
            (split_dir / f"cds_{split}.fa",     1),
        ],
        out_path=out_dir / f"task_b_{split}.pt",
        max_nt=SEQ_LEN_NT_GRCH,
        task="task_b", split=split,
        chunk_size=chunk_size,
    )


def encode_task_c(split_dir, split, out_dir, neg_ratio=2, chunk_size=CHUNK_SIZE):
    """
    3-class: no_splice=0, donor=1, acceptor=2
    Negatives (introns) capped at neg_ratio × n_positives
    """
    # Count positives first to set neg cap
    n_donors    = sum(1 for _ in read_fasta(split_dir / f"splice_donors_{split}.fa"))
    n_acceptors = sum(1 for _ in read_fasta(split_dir / f"splice_acceptors_{split}.fa"))
    n_pos       = n_donors + n_acceptors
    max_neg     = n_pos * neg_ratio

    encode_and_save_chunked(
        sources=[
            (split_dir / f"splice_donors_{split}.fa",    1),
            (split_dir / f"splice_acceptors_{split}.fa", 2),
            (split_dir / f"introns_{split}.fa",          0),
        ],
        out_path=out_dir / f"task_c_{split}.pt",
        max_nt=SEQ_LEN_NT_GRCH,
        task="task_c", split=split,
        max_neg=max_neg,
        chunk_size=chunk_size,
    )


# ═══════════════════════════════════════════════════════════════════
# UCI SPLICE JUNCTION
# ═══════════════════════════════════════════════════════════════════

UCI_LABEL_MAP = {"EI": 0, "IE": 1, "N": 2}

def load_uci_csv(csv_path):
    seqs, labels = [], []
    with open(csv_path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("@") or line.startswith("%"):
                continue
            parts = [p.strip() for p in line.split(",")]
            label_str = seq_str = None
            for p in parts:
                if p.upper() in UCI_LABEL_MAP:
                    label_str = p.upper()
                elif len(p) >= 30:
                    cand = p.replace(" ", "").upper()
                    if all(c in "ACGTNRYSWKMBDHVX" for c in cand):
                        seq_str = cand
            if label_str is None or seq_str is None:
                label_str = parts[-1].upper()
                seq_str   = parts[-2].replace(" ", "").upper()
            if label_str not in UCI_LABEL_MAP:
                continue
            seqs.append(seq_str)
            labels.append(UCI_LABEL_MAP[label_str])
    return seqs, labels


def encode_uci(csv_path, out_dir, seed=42):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"  [uci] {csv_path} not found — skipping.")
        print(f"  [uci] Download: https://archive.ics.uci.edu/dataset/69/")
        return

    seqs, labels = load_uci_csv(csv_path)
    print(f"  [uci] Loaded {len(seqs):,} UCI sequences "
          f"-> {SEQ_LEN_NT_UCI//3} codon tokens each")

    combined = list(zip(seqs, labels))
    random.Random(seed).shuffle(combined)
    seqs, labels = zip(*combined)

    n    = len(seqs)
    n_tr = int(n * 0.80)
    n_va = int(n * 0.10)

    splits = [
        ("train", seqs[:n_tr],            labels[:n_tr]),
        ("val",   seqs[n_tr:n_tr+n_va],   labels[n_tr:n_tr+n_va]),
        ("test",  seqs[n_tr+n_va:],       labels[n_tr+n_va:]),
    ]
    for sp, s_list, l_list in splits:
        ids  = [codon_tokenise(s, SEQ_LEN_NT_UCI) for s in s_list]
        data = {
            "input_ids": torch.tensor(ids,          dtype=torch.long),
            "labels":    torch.tensor(list(l_list), dtype=torch.long),
        }
        torch.save(data, out_dir / f"uci_splice_{sp}.pt")
        c = Counter(list(l_list))
        print(f"  [uci_splice/{sp}] N={len(s_list):,}  "
              + "  ".join(f"cls{k}={v}" for k, v in sorted(c.items())))


# ═══════════════════════════════════════════════════════════════════
# VOCAB + STATS
# ═══════════════════════════════════════════════════════════════════

def write_vocab(out_dir):
    payload = {
        "strategy":          "codon — non-overlapping 3-mer",
        "vocab_size":        VOCAB_SIZE,
        "n_standard_codons": 64,
        "special_tokens":    {"<PAD>": 0, "<UNK>": 1, "<MASK>": 2},
        "codon_to_idx":      CODON_VOCAB,
        "iupac_resolve":     IUPAC_RESOLVE,
        "token_lengths": {
            "grch38_max_nt":  SEQ_LEN_NT_GRCH,
            "grch38_tok_len": SEQ_LEN_NT_GRCH // 3,
            "uci_nt":         SEQ_LEN_NT_UCI,
            "uci_tok_len":    SEQ_LEN_NT_UCI  // 3,
        },
    }
    with open(out_dir / "codon_vocab.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  [vocab] codon_vocab.json saved ({VOCAB_SIZE} tokens)")


def write_stats(out_dir):
    with open(out_dir / "encoding_stats.txt", "w", encoding="utf-8") as fh:
        fh.write("Encoded Dataset Statistics\n" + "="*55 + "\n\n")
        fh.write(f"Tokenisation : codon (non-overlapping 3-mer)\n")
        fh.write(f"Vocab size   : {VOCAB_SIZE} "
                 f"(64 codons + PAD + UNK + MASK)\n")
        fh.write(f"Chunk size   : {CHUNK_SIZE:,} sequences\n\n")
        for p in sorted(out_dir.glob("*.pt")):
            d    = torch.load(p, weights_only=True)
            n, l = d["input_ids"].shape
            c    = Counter(d["labels"].tolist())
            fh.write(f"{p.name}\n")
            fh.write(f"  samples   : {n:,}\n")
            fh.write(f"  token_len : {l}  (= {l*3} nt)\n")
            for cls, cnt in sorted(c.items()):
                fh.write(f"  class {cls}   : {cnt:,} ({cnt/n*100:.1f}%)\n")
            fh.write("\n")
    print(f"  [stats] encoding_stats.txt written")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrom_dir",  default="data/02_splits_chrom")
    ap.add_argument("--uci_csv",    default="data/splice.data")
    ap.add_argument("--out_dir",    default="data/05_encoded")
    ap.add_argument("--chunk_size", type=int, default=10000,
                    help="Sequences per chunk (default 10000, lower = less RAM)")
    args = ap.parse_args()

    # All variables defined here before use
    chunk_size = args.chunk_size
    chrom_dir  = Path(args.chrom_dir)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[05] Codon-based tokenisation  (chunked, memory-safe)")
    print(f"     Vocab size  : {VOCAB_SIZE}")
    print(f"     Chunk size  : {chunk_size:,} sequences  "
          f"(peak RAM ~{chunk_size*170*8/1e6:.0f} MB per chunk)")
    print(f"     GRCh38 toks : {SEQ_LEN_NT_GRCH//3} per sequence")
    print(f"     UCI toks    : {SEQ_LEN_NT_UCI//3}  per sequence")

    for split in ("train", "val", "test"):
        print(f"\n  -- {split} --")
        encode_task_a(chrom_dir, split, out_dir, chunk_size)
        encode_task_b(chrom_dir, split, out_dir, chunk_size)
        encode_task_c(chrom_dir, split, out_dir, chunk_size=chunk_size)

    print("\n[05] UCI splice junction ...")
    encode_uci(args.uci_csv, out_dir)

    write_vocab(out_dir)
    write_stats(out_dir)
    print("\n[05] Done.")

if __name__ == "__main__":
    main()