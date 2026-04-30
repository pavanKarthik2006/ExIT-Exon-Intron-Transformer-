"""
02_build_dataset_chrom.py
=========================
STEP 2A — Train / val / test split by CHROMOSOME (strictest, zero leakage).

Split assignment
----------------
  Train : chr1  – chr17
  Val   : chr18 – chr20
  Test  : chr21, chr22, chrX, chrY

Leakage guarantee
-----------------
Chromosomes are non-overlapping by definition.  No sequence from any
chromosomal region can appear in more than one partition.

Inputs  : data/01_extracted/{feature}.fa
Outputs : data/02_splits_chrom/{feature}_{train|val|test}.fa
          data/02_splits_chrom/split_summary_chrom.txt

Fix vs original
---------------
Streams sequences directly to output files one at a time instead of
loading all records into memory first. Safe for 1.5M+ sequences on
any RAM size.
"""

import argparse
import time
from pathlib import Path

TRAIN_CHROMS = {str(i) for i in range(1, 18)}
VAL_CHROMS   = {str(i) for i in range(18, 21)}
TEST_CHROMS  = {"21", "22", "X", "Y", "x", "y"}

CHROM_TO_SPLIT = {}
for c in TRAIN_CHROMS: CHROM_TO_SPLIT[c] = "train"
for c in VAL_CHROMS:   CHROM_TO_SPLIT[c] = "val"
for c in TEST_CHROMS:  CHROM_TO_SPLIT[c] = "test"

FEATURES = ["exons","introns","cds","non_cds","splice_donors","splice_acceptors"]


def chrom_from_header(hdr):
    """Extract chrom from header: FEATURE|GENE|TX|CHROM|START|END|STRAND"""
    p = hdr.split("|")
    return p[3] if len(p) >= 6 else "unknown"


def split_feature_streaming(in_path, out_dir, feature):
    """
    Stream sequences one at a time from in_path and write directly
    to the correct split file. Never holds more than one sequence
    in memory at a time — safe for very large files.
    """
    counts = {"train": 0, "val": 0, "test": 0, "unassigned": 0}

    # Open all three output files at once
    handles = {
        s: open(out_dir / f"{feature}_{s}.fa", "w")
        for s in ("train", "val", "test")
    }

    hdr, buf = None, []
    with open(in_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                # Process previous record
                if hdr is not None:
                    chrom = chrom_from_header(hdr)
                    split = CHROM_TO_SPLIT.get(chrom)
                    if split:
                        seq = "".join(buf)
                        handles[split].write(f">{hdr}\n")
                        for i in range(0, len(seq), 60):
                            handles[split].write(seq[i:i+60] + "\n")
                        counts[split] += 1
                    else:
                        counts["unassigned"] += 1
                hdr = line[1:]
                buf = []
            else:
                buf.append(line)

        # Don't forget the last record
        if hdr is not None:
            chrom = chrom_from_header(hdr)
            split = CHROM_TO_SPLIT.get(chrom)
            if split:
                seq = "".join(buf)
                handles[split].write(f">{hdr}\n")
                for i in range(0, len(seq), 60):
                    handles[split].write(seq[i:i+60] + "\n")
                counts[split] += 1
            else:
                counts["unassigned"] += 1

    for h in handles.values():
        h.close()

    return counts


def write_summary(summary, path):
    with open(path, "w") as fh:
        fh.write("Chromosome-based split\n" + "="*60 + "\n")
        fh.write("  Train : chr1-17\n")
        fh.write("  Val   : chr18-20\n")
        fh.write("  Test  : chr21, chr22, chrX, chrY\n")
        fh.write("  Leakage: NONE (chromosomes are disjoint)\n\n")
        fh.write(f"{'feature':<24} {'train':>12} {'val':>10} {'test':>10} {'total':>12}\n")
        fh.write("-"*70 + "\n")
        for feat, c in summary.items():
            tot = c['train'] + c['val'] + c['test']
            fh.write(f"{feat:<24} {c['train']:>12,} {c['val']:>10,} "
                     f"{c['test']:>10,} {tot:>12,}\n")
        fh.write("\nProportions (train / val / test)\n" + "-"*40 + "\n")
        for feat, c in summary.items():
            tot = c['train'] + c['val'] + c['test']
            if tot == 0: continue
            fh.write(f"  {feat:<22} {c['train']/tot*100:5.1f}% / "
                     f"{c['val']/tot*100:5.1f}% / {c['test']/tot*100:5.1f}%\n")
    print(f"[02-chrom] Summary written -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",  default="data/01_extracted")
    ap.add_argument("--out_dir", default="data/02_splits_chrom")
    args = ap.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for feat in FEATURES:
        in_path = in_dir / f"{feat}.fa"
        if not in_path.exists():
            print(f"[02-chrom] SKIP {in_path} (not found)")
            continue

        t0 = time.time()
        print(f"[02-chrom] Splitting {feat} ...", flush=True)
        c = split_feature_streaming(in_path, out_dir, feat)
        elapsed = time.time() - t0
        summary[feat] = c
        print(f"[02-chrom] {feat} done in {elapsed:.1f}s")
        print(f"           train={c['train']:,}  val={c['val']:,}  "
              f"test={c['test']:,}  unassigned={c['unassigned']:,}")

    write_summary(summary, out_dir / "split_summary_chrom.txt")

    print("\n[02-chrom] === FINAL SUMMARY ===")
    for feat, c in summary.items():
        tot = c['train'] + c['val'] + c['test']
        print(f"  {feat:<24} total={tot:,}  "
              f"({c['train']/tot*100:.1f}% / "
              f"{c['val']/tot*100:.1f}% / "
              f"{c['test']/tot*100:.1f}%)")
    print("[02-chrom] Done.")

if __name__ == "__main__":
    main()