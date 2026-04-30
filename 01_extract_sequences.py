"""
01_extract_sequences.py
=======================
STEP 1 — Extract labelled sequences from GRCh38 FASTA + Ensembl GTF.

Inputs
------
    --fasta      : GRCh38 genome FASTA  (e.g. GRCh38.fa)
    --gtf        : Ensembl genes.gtf
    --out_dir    : output directory     (default: data/01_extracted)
    --max_chroms : (optional) limit to N chromosomes for testing

Outputs  [data/01_extracted/]
-------
    exons.fa            all exon sequences
    introns.fa          inferred intron sequences (inter-exon gaps)
    cds.fa              CDS sequences
    non_cds.fa          UTR / non-coding exonic regions
    splice_donors.fa    200 bp windows centred on 5-prime splice junctions
    splice_acceptors.fa 200 bp windows centred on 3-prime splice junctions
    extraction_stats.txt

FASTA header format (shared by all downstream scripts)
-------------------------------------------------------
>FEATURE|GENE_ID|TRANSCRIPT_ID|CHROM|START|END|STRAND
 e.g. >exon|ENSG00000000003|ENST00000000233|X|99887482|99887565|+

Design notes
------------
- FASTA streamed one chromosome at a time (RAM-safe for full GRCh38).
- Introns inferred as gaps between consecutive sorted exons of the same transcript.
- Sequences < MIN_LEN (50 bp) discarded; > MAX_LEN (5000 bp) centre-cropped.
- Minus-strand features are reverse-complemented to give 5'→3' orientation.
- Splice windows: SPLICE_FLANK nt upstream + downstream of each junction.
"""

import argparse
from collections import defaultdict
from pathlib import Path

MIN_LEN      = 50
MAX_LEN      = 5000
SPLICE_FLANK = 100   # bp each side of junction → 200 bp total window


# ═══════════════════════════════════════════════════════════════════
# GTF PARSING
# ═══════════════════════════════════════════════════════════════════

def parse_gtf(gtf_path: str) -> dict:
    """
    Returns
    -------
    transcripts[chrom][tx_id] = {
        "gene_id": str,
        "strand":  "+" | "-",
        "exons":   [(start, end), ...],   # 0-based half-open, sorted
        "cds":     [(start, end), ...],
    }
    """
    txs = defaultdict(lambda: defaultdict(
        lambda: {"gene_id": "", "strand": "+", "exons": [], "cds": []}
    ))
    print(f"[01] Parsing GTF: {gtf_path}")
    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.strip().split("\t")
            if len(cols) < 9 or cols[2] not in ("exon", "CDS"):
                continue
            chrom  = cols[0].replace("chr", "")
            start  = int(cols[3]) - 1          # 1-based → 0-based
            end    = int(cols[4])
            strand = cols[6]
            attrs  = cols[8]
            tx_id  = _attr(attrs, "transcript_id")
            gene_id= _attr(attrs, "gene_id")
            if not tx_id:
                continue
            txs[chrom][tx_id]["gene_id"] = gene_id
            txs[chrom][tx_id]["strand"]  = strand
            if cols[2] == "exon":
                txs[chrom][tx_id]["exons"].append((start, end))
            else:
                txs[chrom][tx_id]["cds"].append((start, end))

    for chrom in txs:
        for tx in txs[chrom].values():
            tx["exons"].sort()
            tx["cds"].sort()

    n_tx = sum(len(v) for v in txs.values())
    print(f"[01]   {n_tx:,} transcripts on {len(txs)} chromosomes.")
    return dict(txs)


def _attr(attr_str: str, key: str) -> str:
    for field in attr_str.split(";"):
        field = field.strip()
        if field.startswith(key + " ") or field.startswith(key + "\t"):
            return field.split(None, 1)[1].strip().strip('"')
    return ""


# ═══════════════════════════════════════════════════════════════════
# FASTA STREAMING
# ═══════════════════════════════════════════════════════════════════

def stream_fasta(fasta_path: str):
    """Yield (chrom_str, sequence) one chromosome at a time."""
    chrom, buf = None, []
    with open(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if chrom is not None:
                    yield chrom, "".join(buf).upper()
                chrom = line[1:].split()[0].replace("chr", "")
                buf   = []
            else:
                buf.append(line)
    if chrom is not None:
        yield chrom, "".join(buf).upper()


# ═══════════════════════════════════════════════════════════════════
# SEQUENCE UTILITIES
# ═══════════════════════════════════════════════════════════════════

_RC = str.maketrans("ATGCNatgcn", "TACGNtacgn")

def revcomp(seq: str) -> str:
    return seq.translate(_RC)[::-1]

def centre_crop(seq: str, max_len: int) -> str:
    if len(seq) <= max_len:
        return seq
    mid = len(seq) // 2
    return seq[mid - max_len // 2: mid - max_len // 2 + max_len]

def orient(seq: str, strand: str) -> str:
    return revcomp(seq) if strand == "-" else seq

def make_header(feat, gene_id, tx_id, chrom, s, e, strand) -> str:
    return f"{feat}|{gene_id}|{tx_id}|{chrom}|{s}|{e}|{strand}"


# ═══════════════════════════════════════════════════════════════════
# INTERVAL ARITHMETIC
# ═══════════════════════════════════════════════════════════════════

def subtract_intervals(base_list, sub_list):
    """Subtract sub_list intervals from base_list. Returns remaining segments."""
    result = list(base_list)
    for ss, se in sub_list:
        new = []
        for s, e in result:
            if se <= s or ss >= e:
                new.append((s, e))
            else:
                if s < ss: new.append((s, ss))
                if se < e: new.append((se, e))
        result = new
    return result


# ═══════════════════════════════════════════════════════════════════
# EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def extract_from_chrom(chrom: str, seq: str, tx_data: dict) -> dict:
    L = len(seq)
    out = {k: [] for k in
           ("exons","introns","cds","non_cds","splice_donors","splice_acceptors")}

    for tx_id, info in tx_data.items():
        strand  = info["strand"]
        gid     = info["gene_id"]
        exons   = info["exons"]
        cds_ivs = info["cds"]
        if not exons:
            continue

        def _add(bucket, feat, s, e):
            if s < 0 or e > L or e - s < MIN_LEN:
                return
            s2 = orient(centre_crop(seq[s:e], MAX_LEN), strand)
            if len(s2) >= MIN_LEN:
                out[bucket].append((make_header(feat, gid, tx_id, chrom, s, e, strand), s2))

        # Exons
        for i, (s, e) in enumerate(exons):
            _add("exons", "exon", s, e)

        # Introns
        for i in range(len(exons) - 1):
            _add("introns", "intron", exons[i][1], exons[i+1][0])

        # CDS
        for s, e in cds_ivs:
            _add("cds", "cds", s, e)

        # Non-CDS exonic (UTR)
        for s, e in exons:
            for ns, ne in subtract_intervals([(s, e)], cds_ivs):
                _add("non_cds", "non_cds", ns, ne)

        # Splice sites
        for i in range(len(exons) - 1):
            d = exons[i][1]      # donor  = 5' end of intron
            a = exons[i+1][0]   # acceptor = 3' end of intron
            # donor window
            if d - SPLICE_FLANK >= 0 and d + SPLICE_FLANK <= L:
                s2 = orient(seq[d-SPLICE_FLANK:d+SPLICE_FLANK], strand)
                out["splice_donors"].append(
                    (make_header("donor", gid, tx_id, chrom,
                                 d-SPLICE_FLANK, d+SPLICE_FLANK, strand), s2))
            # acceptor window
            if a - SPLICE_FLANK >= 0 and a + SPLICE_FLANK <= L:
                s2 = orient(seq[a-SPLICE_FLANK:a+SPLICE_FLANK], strand)
                out["splice_acceptors"].append(
                    (make_header("acceptor", gid, tx_id, chrom,
                                 a-SPLICE_FLANK, a+SPLICE_FLANK, strand), s2))
    return out


# ═══════════════════════════════════════════════════════════════════
# WRITERS
# ═══════════════════════════════════════════════════════════════════

def write_fasta(records, path: Path, lw=60):
    with open(path, "w") as fh:
        for hdr, seq in records:
            fh.write(f">{hdr}\n")
            for i in range(0, len(seq), lw):
                fh.write(seq[i:i+lw] + "\n")

def write_stats(all_records: dict, path: Path):
    with open(path, "w") as fh:
        fh.write("Extraction Statistics\n" + "="*50 + "\n\n")
        for feat, recs in all_records.items():
            lengths = sorted(len(s) for _, s in recs)
            n = len(lengths)
            if n == 0:
                fh.write(f"[{feat}]  count=0\n\n")
                continue
            fh.write(f"[{feat}]\n")
            fh.write(f"  count  : {n:,}\n")
            fh.write(f"  min    : {lengths[0]}\n")
            fh.write(f"  max    : {lengths[-1]}\n")
            fh.write(f"  mean   : {sum(lengths)/n:.1f}\n")
            fh.write(f"  median : {lengths[n//2]}\n\n")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta",      required=True)
    ap.add_argument("--gtf",        required=True)
    ap.add_argument("--out_dir",    default="data/01_extracted")
    ap.add_argument("--max_chroms", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    txs = parse_gtf(args.gtf)
    all_records = {k: [] for k in
                   ("exons","introns","cds","non_cds",
                    "splice_donors","splice_acceptors")}

    n = 0
    for chrom, seq in stream_fasta(args.fasta):
        if args.max_chroms and n >= args.max_chroms:
            break
        if chrom not in txs:
            continue
        print(f"[01]   chr{chrom} ({len(seq):,} bp) ...", end=" ", flush=True)
        res = extract_from_chrom(chrom, seq, txs[chrom])
        for k in all_records:
            all_records[k].extend(res[k])
        print(" | ".join(f"{k}={len(v)}" for k, v in res.items()))
        n += 1

    file_map = {
        "exons":             "exons.fa",
        "introns":           "introns.fa",
        "cds":               "cds.fa",
        "non_cds":           "non_cds.fa",
        "splice_donors":     "splice_donors.fa",
        "splice_acceptors":  "splice_acceptors.fa",
    }
    for key, fname in file_map.items():
        p = out_dir / fname
        write_fasta(all_records[key], p)
        print(f"[01] {len(all_records[key]):,} records → {p}")

    write_stats(all_records, out_dir / "extraction_stats.txt")
    print("[01] Done.")

if __name__ == "__main__":
    main()
