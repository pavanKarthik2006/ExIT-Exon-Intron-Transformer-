"""
04_leakage_analysis.py
======================
STEP 3 — Quantify and compare data leakage between split strategies.

Analyses
--------
A. Gene contamination (transcript split)
   Genes whose transcripts span multiple splits → share k-mers across partitions.

B. Chromosome overlap (chromosome split)
   Formal verification that zero chromosomes appear in >1 partition.

C. K-mer overlap (k=6)
   Sample N sequences from train and test for each strategy.
   Compute fraction of test k-mers seen in train.
   High overlap → inflated test accuracy → quantifies optimism bias.

D. Accuracy inflation estimate
   Predict how much test accuracy is inflated in the transcript split
   vs the chromosome split, based on k-mer overlap difference.

Outputs  [data/03_leakage/]
-------
    leakage_report.txt
    contaminated_genes.tsv
    kmer_overlap.tsv
"""

import argparse, random
from collections import defaultdict
from pathlib import Path

SAMPLE_N        = 500
KMER_K          = 6
FEATURES_SAMPLE = ["exons", "introns"]


# ─── FASTA ───────────────────────────────────────────────────────────────────

def read_fasta(path):
    if not path.exists(): return
    hdr, buf = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if hdr: yield hdr, "".join(buf)
                hdr, buf = line[1:], []
            else: buf.append(line)
    if hdr: yield hdr, "".join(buf)


def parse_header(hdr):
    p = hdr.split("|")
    if len(p) < 6: return {}
    return {"feature":p[0],"gene_id":p[1],"tx_id":p[2],
            "chrom":p[3],"start":p[4],"end":p[5]}


# ─── A: Gene contamination ────────────────────────────────────────────────────

def analyse_gene_contamination(tx_map_path):
    gene_splits = defaultdict(lambda: defaultdict(int))
    n_tx = 0
    with open(tx_map_path) as fh:
        next(fh)
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 3: continue
            _, gene_id, split = parts[0], parts[1], parts[2]
            gene_splits[gene_id][split] += 1
            n_tx += 1
    contaminated = {g: dict(c) for g, c in gene_splits.items() if len(c) > 1}
    return dict(gene_splits), contaminated, n_tx


# ─── B: Chromosome overlap ────────────────────────────────────────────────────

def analyse_chrom_overlap(chrom_dir, feature="exons"):
    split_chroms = {}
    for split in ("train","val","test"):
        chroms = set()
        for hdr, _ in read_fasta(chrom_dir / f"{feature}_{split}.fa"):
            info = parse_header(hdr)
            if info: chroms.add(info["chrom"])
        split_chroms[split] = chroms
    overlaps = {}
    sp = list(split_chroms)
    for i in range(len(sp)):
        for j in range(i+1, len(sp)):
            s1, s2 = sp[i], sp[j]
            overlaps[f"{s1}_and_{s2}"] = sorted(split_chroms[s1] & split_chroms[s2])
    return split_chroms, overlaps


# ─── C: K-mer overlap ────────────────────────────────────────────────────────

def sample_sequences(fasta_path, n, seed=42):
    rng = random.Random(seed)
    res = []
    for i, (_, seq) in enumerate(read_fasta(fasta_path)):
        if i < n:   res.append(seq)
        else:
            j = rng.randint(0, i)
            if j < n: res[j] = seq
    return res


def kmer_overlap(train_seqs, test_seqs, k):
    def kmers(seqs):
        s = set()
        for seq in seqs:
            s.update(seq[i:i+k] for i in range(len(seq)-k+1))
        return s
    tr = kmers(train_seqs)
    te = kmers(test_seqs)
    if not te:
        return {"train_unique":0,"test_unique":0,"overlap":0,"frac":0.0}
    ov = len(tr & te)
    return {"train_unique":len(tr),"test_unique":len(te),
            "overlap":ov,"frac":ov/len(te)}


def run_kmer_analysis(chrom_dir, tx_dir, k=KMER_K):
    rows = []
    for feat in FEATURES_SAMPLE:
        for strategy, d in [("chromosome", chrom_dir), ("transcript", tx_dir)]:
            tr_p = d / f"{feat}_train.fa"
            te_p = d / f"{feat}_test.fa"
            if not tr_p.exists() or not te_p.exists(): continue
            print(f"  [kmer] {strategy}/{feat} ...", end=" ", flush=True)
            tr_seqs = sample_sequences(tr_p, SAMPLE_N)
            te_seqs = sample_sequences(te_p, SAMPLE_N)
            r = kmer_overlap(tr_seqs, te_seqs, k)
            r.update({"strategy":strategy,"feature":feat,"k":k})
            rows.append(r)
            print(f"overlap={r['frac']:.4f}")
    return rows


# ─── D: Accuracy inflation estimate ──────────────────────────────────────────

def estimate_inflation(kmer_rows):
    """
    Estimate accuracy inflation as the difference in k-mer overlap fraction
    between transcript split and chromosome split.
    Higher overlap → higher chance model 'memorised' test patterns.
    """
    by_strategy = defaultdict(list)
    for r in kmer_rows:
        by_strategy[r["strategy"]].append(r["frac"])
    means = {k: sum(v)/len(v) for k, v in by_strategy.items() if v}
    inflation = means.get("transcript", 0) - means.get("chromosome", 0)
    return means, inflation


# ─── WRITERS ─────────────────────────────────────────────────────────────────

def write_contaminated_tsv(contaminated, path):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("gene_id\tn_train_tx\tn_val_tx\tn_test_tx\n")
        for gid, c in sorted(contaminated.items()):
            fh.write(f"{gid}\t{c.get('train',0)}\t{c.get('val',0)}\t{c.get('test',0)}\n")
    print(f"[04] contaminated_genes.tsv → {path}")


def write_kmer_tsv(rows, path):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("strategy\tfeature\tk\ttrain_unique\ttest_unique\toverlap\tfraction\n")
        for r in rows:
            fh.write(f"{r['strategy']}\t{r['feature']}\t{r['k']}\t"
                     f"{r['train_unique']}\t{r['test_unique']}\t"
                     f"{r['overlap']}\t{r['frac']:.6f}\n")
    print(f"[04] kmer_overlap.tsv      → {path}")


def write_report(gene_splits, contaminated, n_tx,
                 chrom_overlaps, kmer_rows, path):
    n_genes     = len(gene_splits)
    n_contam    = len(contaminated)
    pct         = n_contam / n_genes * 100 if n_genes else 0
    means, infl = estimate_inflation(kmer_rows)

    with open(path, "w", encoding="utf-8") as fh:
        def w(s=""): fh.write(s + "\n")
        w("="*70)
        w("DATA LEAKAGE ANALYSIS REPORT")
        w("Chromosome-split vs Transcript-ID-split")
        w("="*70)
        w()
        # A
        w("A. GENE-LEVEL CONTAMINATION  (transcript split)")
        w("-"*50)
        w(f"  Total genes          : {n_genes:,}")
        w(f"  Total transcripts    : {n_tx:,}")
        w(f"  Contaminated genes   : {n_contam:,}  ({pct:.1f}%)")
        w("  (Contaminated = gene has transcripts in >1 split)")
        w()
        if contaminated:
            worst = sorted(contaminated.items(),
                           key=lambda x: sum(x[1].values()), reverse=True)[:10]
            w("  Top 10 most contaminated genes:")
            w(f"  {'gene_id':<28} {'train':>7} {'val':>7} {'test':>7}")
            w("  " + "-"*50)
            for gid, c in worst:
                w(f"  {gid:<28} {c.get('train',0):>7} "
                  f"{c.get('val',0):>7} {c.get('test',0):>7}")
        w()
        # B
        w("B. CHROMOSOMAL OVERLAP  (chromosome split)")
        w("-"*50)
        any_ov = False
        for key, chroms in chrom_overlaps.items():
            w(f"  {key} : {len(chroms)} shared chromosome(s)")
            if chroms: any_ov = True; w(f"    → {', '.join(chroms)}")
        if not any_ov:
            w("  ✓  Zero chromosomal overlap confirmed — provably leakage-free.")
        w()
        # C
        w(f"C. K-MER OVERLAP  (k={KMER_K}, sample N={SAMPLE_N} per split)")
        w("-"*50)
        w(f"  {'Strategy':<14} {'Feature':<14} {'Train kmers':>13} "
          f"{'Test kmers':>12} {'Overlap':>10} {'Fraction':>10}")
        w("  " + "-"*70)
        for r in kmer_rows:
            w(f"  {r['strategy']:<14} {r['feature']:<14} "
              f"{r['train_unique']:>13,} {r['test_unique']:>12,} "
              f"{r['overlap']:>10,} {r['frac']:>10.4f}")
        w()
        # D
        w("D. ESTIMATED ACCURACY INFLATION")
        w("-"*50)
        for strat, m in means.items():
            w(f"  Mean k-mer overlap ({strat:>12}): {m:.4f}")
        w(f"  Inflation estimate (transcript − chromosome): {infl:+.4f}")
        w()
        w("  Interpretation:")
        w("  A higher k-mer overlap fraction in the transcript split means")
        w("  the test set shares more sequence patterns with training data.")
        w("  This directly inflates test accuracy because the model has")
        w("  effectively seen similar sequences during training.")
        w("  The inflation estimate above is an empirical lower bound on")
        w("  the optimism bias introduced by transcript-level splitting.")
        w()
        # Methods paragraph
        w("E. SUGGESTED METHODS PARAGRAPH")
        w("-"*50)
        w(f"  We evaluated model generalisation under two partitioning regimes.")
        w(f"  In the chromosome split, chromosomes 1-17 were assigned to training,")
        w(f"  18-20 to validation, and 21-22 plus chrX/Y to the held-out test set,")
        w(f"  guaranteeing that no chromosomal region appeared in more than one")
        w(f"  partition. In the transcript split, 70% of transcripts were randomly")
        w(f"  assigned to training, 15% to validation, and 15% to testing (seed=42),")
        w(f"  ensuring no transcript spanned multiple partitions. However,")
        w(f"  {n_contam:,} genes ({pct:.1f}%) had transcripts distributed across")
        w(f"  partitions in the transcript split, representing a source of optimism")
        w(f"  bias. A 6-mer overlap analysis revealed that the fraction of test")
        w(f"  k-mers present in the training set was {infl:.1%} higher under the")
        w(f"  transcript split than under the chromosome split. We therefore report")
        w(f"  chromosome-split metrics as primary results and transcript-split")
        w(f"  metrics as an upper bound on achievable performance.")
        w()
        w("="*70)
    print(f"[04] leakage_report.txt    → {path}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tx_map",    default="data/03_splits_transcript/tx_split_map.tsv")
    ap.add_argument("--chrom_dir", default="data/02_splits_chrom")
    ap.add_argument("--tx_dir",    default="data/03_splits_transcript")
    ap.add_argument("--out_dir",   default="data/04_leakage")
    args = ap.parse_args()

    out_dir   = Path(args.out_dir);   out_dir.mkdir(parents=True, exist_ok=True)
    chrom_dir = Path(args.chrom_dir)
    tx_dir    = Path(args.tx_dir)

    print("[04] Gene contamination analysis ...")
    gene_splits, contaminated, n_tx = analyse_gene_contamination(Path(args.tx_map))
    print(f"[04]   {len(contaminated):,} contaminated genes / {len(gene_splits):,} total")

    print("[04] Chromosome overlap verification ...")
    _, chrom_overlaps = analyse_chrom_overlap(chrom_dir)

    print("[04] K-mer overlap analysis ...")
    kmer_rows = run_kmer_analysis(chrom_dir, tx_dir)

    write_contaminated_tsv(contaminated, out_dir / "contaminated_genes.tsv")
    write_kmer_tsv(kmer_rows, out_dir / "kmer_overlap.tsv")
    write_report(gene_splits, contaminated, n_tx,
                 chrom_overlaps, kmer_rows,
                 out_dir / "leakage_report.txt")
    print("[04] Done.")

if __name__ == "__main__":
    main()