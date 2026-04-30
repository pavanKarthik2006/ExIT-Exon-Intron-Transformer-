"""
12_chimp_crossspecies.py
========================
CROSS-SPECIES GENERALISATION — Chimpanzee (Pan troglodytes)

Tests whether the human-trained Task A model (exon/intron classifier)
generalises to chimpanzee DNA without any retraining.

Pipeline
--------
1. Parse chimpanzee GTF  -> extract exon/intron coordinates
2. Extract sequences from chimpanzee FASTA
3. Sample 10% of sequences randomly (reproducible, seed=42)
4. Encode with same codon tokenisation as human (512nt -> 170 tokens)
5. Load human Task A checkpoint (checkpoints/task_a_best.pt)
6. Evaluate — full metrics + visualizations vs human baseline

Outputs  [results_crossspecies/]
-------
  chimp_encoded.pt            encoded chimp sequences (10% sample)
  chimp_eval_report.txt       full metrics report (all metrics)
  chimp_eval_summary.tsv      all metrics row for paper Table 3
  chimp_eval_raw.json         all numbers
  chimp_visualizations.html   interactive charts (open in browser)
"""

import argparse, json, itertools, random, time, math
from pathlib import Path
from collections import Counter
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
model_module = import_module("06_model")
DNAClassifier = model_module.DNAClassifier


# ═══════════════════════════════════════════════════════════════════
# CONSTANTS  — must match human pipeline exactly
# ═══════════════════════════════════════════════════════════════════

SEQ_LEN_NT   = 512
MAX_TOK      = SEQ_LEN_NT // 3   # 170 codon tokens
MIN_LEN      = 50
MAX_LEN      = 5000
SAMPLE_SEED  = 42
SAMPLE_FRAC  = 0.10

# Human Exp1 Task A baseline — all metrics from 08_evaluate results
HUMAN_BASELINE = {
    "accuracy":          0.9343,
    "balanced_accuracy": 0.9364,
    "macro_f1":          0.9343,
    "mcc":               0.8719,
    "auc":               0.9810,
    "per_class": {
        "intron": {"precision": 0.9285, "recall": 0.9253, "f1": 0.9269, "specificity": 0.9475},
        "exon":   {"precision": 0.9401, "recall": 0.9475, "f1": 0.9438, "specificity": 0.9253},
    },
}

IUPAC_RESOLVE = {
    "R":"A","Y":"C","S":"G","W":"A","K":"G","M":"A",
    "B":"C","D":"A","H":"A","V":"A","N":"A","X":"A",
}

def _build_codon_vocab():
    bases  = ["A","C","G","T"]
    codons = sorted("".join(c) for c in itertools.product(bases, repeat=3))
    vocab  = {"<PAD>": 0, "<UNK>": 1, "<MASK>": 2}
    for i, codon in enumerate(codons):
        vocab[codon] = i + 3
    return vocab

CODON_VOCAB = _build_codon_vocab()


# ═══════════════════════════════════════════════════════════════════
# SEQUENCE UTILITIES
# ═══════════════════════════════════════════════════════════════════

def reverse_complement(seq):
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(comp)[::-1]

def resolve_iupac(seq):
    return "".join(IUPAC_RESOLVE.get(c, c) for c in seq.upper())

def codon_tokenise(seq):
    seq    = resolve_iupac(seq)[:SEQ_LEN_NT]
    tokens = []
    for i in range(0, len(seq) - 2, 3):
        tokens.append(CODON_VOCAB.get(seq[i:i+3], CODON_VOCAB["<UNK>"]))
    tokens += [CODON_VOCAB["<PAD>"]] * (MAX_TOK - len(tokens))
    return tokens[:MAX_TOK]

def centre_crop(seq, max_len=MAX_LEN):
    if len(seq) <= max_len:
        return seq
    start = (len(seq) - max_len) // 2
    return seq[start:start+max_len]


# ═══════════════════════════════════════════════════════════════════
# GTF PARSER
# ═══════════════════════════════════════════════════════════════════

def _attr(attrs, key):
    for part in attrs.split(";"):
        part = part.strip()
        if part.startswith(key + " "):
            return part.split('"')[1] if '"' in part else part.split()[-1]
    return ""

def parse_chimp_gtf(gtf_path):
    from collections import defaultdict
    txs = defaultdict(lambda: defaultdict(
        lambda: {"gene_id": "", "strand": "+", "exons": []}
    ))
    print(f"[12] Parsing chimpanzee GTF: {gtf_path}")
    n = 0
    with open(gtf_path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.strip().split("\t")
            if len(cols) < 9 or cols[2] != "exon":
                continue
            attrs   = cols[8]
            biotype = _attr(attrs, "transcript_biotype") or _attr(attrs, "gene_biotype")
            if biotype and biotype != "protein_coding":
                continue
            chrom   = cols[0].replace("chr", "")
            start   = int(cols[3]) - 1
            end     = int(cols[4])
            strand  = cols[6]
            tx_id   = _attr(attrs, "transcript_id")
            gene_id = _attr(attrs, "gene_id")
            if not tx_id:
                continue
            txs[chrom][tx_id]["gene_id"] = gene_id
            txs[chrom][tx_id]["strand"]  = strand
            txs[chrom][tx_id]["exons"].append((start, end))
            n += 1

    for chrom in txs:
        for tx in txs[chrom].values():
            tx["exons"].sort()

    total_tx = sum(len(v) for v in txs.values())
    print(f"  Parsed {n:,} exon records across {total_tx:,} transcripts "
          f"on {len(txs)} chromosomes")
    return txs


# ═══════════════════════════════════════════════════════════════════
# FASTA STREAMER
# ═══════════════════════════════════════════════════════════════════

def stream_fasta_chroms(fasta_path):
    chrom, buf = None, []
    print(f"[12] Streaming FASTA: {fasta_path}")
    with open(fasta_path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if chrom:
                    yield chrom, "".join(buf)
                chrom = line[1:].split()[0].replace("chr", "")
                buf   = []
            else:
                buf.append(line.upper())
    if chrom:
        yield chrom, "".join(buf)


# ═══════════════════════════════════════════════════════════════════
# SEQUENCE EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def extract_chimp_sequences(fasta_path, gtf_path):
    txs      = parse_chimp_gtf(gtf_path)
    all_seqs = []
    n_exon = n_intron = n_skip = 0

    for chrom, chrom_seq in stream_fasta_chroms(fasta_path):
        if chrom not in txs:
            continue
        chrom_len = len(chrom_seq)

        for tx_id, tx in txs[chrom].items():
            exons  = tx["exons"]
            strand = tx["strand"]
            if not exons:
                continue

            for start, end in exons:
                if end > chrom_len:
                    continue
                seq = chrom_seq[start:end]
                if strand == "-":
                    seq = reverse_complement(seq)
                if len(seq) < MIN_LEN:
                    n_skip += 1; continue
                all_seqs.append((centre_crop(seq), 1))
                n_exon += 1

            for i in range(len(exons) - 1):
                i_start = exons[i][1]
                i_end   = exons[i+1][0]
                if i_end <= i_start or i_end > chrom_len:
                    continue
                seq = chrom_seq[i_start:i_end]
                if strand == "-":
                    seq = reverse_complement(seq)
                if len(seq) < MIN_LEN:
                    n_skip += 1; continue
                all_seqs.append((centre_crop(seq), 0))
                n_intron += 1

        print(f"  chrom {chrom:>4}: exons={n_exon:,}  introns={n_intron:,}", flush=True)

    print(f"\n[12] Total: exons={n_exon:,}  introns={n_intron:,}  skipped={n_skip:,}")
    return all_seqs


# ═══════════════════════════════════════════════════════════════════
# SUBSAMPLE
# ═══════════════════════════════════════════════════════════════════

def subsample(seqs, frac=SAMPLE_FRAC, seed=SAMPLE_SEED):
    rng     = random.Random(seed)
    exons   = [s for s in seqs if s[1] == 1]
    introns = [s for s in seqs if s[1] == 0]
    n_exon   = max(1, int(len(exons)   * frac))
    n_intron = max(1, int(len(introns) * frac))
    rng.shuffle(exons);   rng.shuffle(introns)
    sample = exons[:n_exon] + introns[:n_intron]
    rng.shuffle(sample)
    print(f"[12] Subsampled {frac*100:.0f}%: exons={n_exon:,}  "
          f"introns={n_intron:,}  total={len(sample):,}")
    return sample


# ═══════════════════════════════════════════════════════════════════
# ENCODE
# ═══════════════════════════════════════════════════════════════════

def encode_sequences(seqs, out_path):
    ids    = [codon_tokenise(s) for s, _ in seqs]
    labels = [l for _, l in seqs]
    data   = {
        "input_ids": torch.tensor(ids,    dtype=torch.long),
        "labels":    torch.tensor(labels, dtype=torch.long),
    }
    torch.save(data, out_path)
    c = Counter(labels)
    print(f"[12] Encoded {len(seqs):,} -> {out_path}")
    print(f"     intron={c[0]:,}  exon={c[1]:,}  token_len={MAX_TOK}")
    return data


# ═══════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════

def confusion_matrix_fn(preds, labels, n=2):
    cm = [[0]*n for _ in range(n)]
    for p, l in zip(preds, labels):
        cm[l][p] += 1
    return cm

def per_class_metrics(cm, n=2):
    total = sum(cm[r][c] for r in range(n) for c in range(n))
    prec, rec, f1, spec = [], [], [], []
    for c in range(n):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(n)) - tp
        fn = sum(cm[c][r] for r in range(n)) - tp
        tn = total - tp - fp - fn
        p  = tp/(tp+fp) if (tp+fp) > 0 else 0.0
        r  = tp/(tp+fn) if (tp+fn) > 0 else 0.0
        f  = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        sp = tn/(tn+fp)  if (tn+fp) > 0 else 0.0
        prec.append(p); rec.append(r); f1.append(f); spec.append(sp)
    return prec, rec, f1, spec

def mcc_fn(cm, n=2):
    total  = sum(cm[r][c] for r in range(n) for c in range(n))
    cov_xy = cov_xx = cov_yy = 0.0
    for k in range(n):
        for l in range(n):
            for m in range(n):
                cov_xy += cm[k][k]*cm[m][l] - cm[l][k]*cm[k][m]
        s = sum(cm[k][j] for j in range(n))
        t = sum(cm[j][k] for j in range(n))
        cov_xx += s*(total-s)
        cov_yy += t*(total-t)
    if cov_xx == 0 or cov_yy == 0:
        return 0.0
    return cov_xy / (cov_xx * cov_yy)**0.5

def roc_auc_binary(probs_pos, labels):
    pos = [p for p,l in zip(probs_pos,labels) if l==1]
    neg = [p for p,l in zip(probs_pos,labels) if l==0]
    if not pos or not neg:
        return float("nan")
    concordant = sum(
        1 if ps>ns else 0.5 if ps==ns else 0
        for ps in pos for ns in neg
    )
    return concordant / (len(pos)*len(neg))

def compute_roc_curve(probs_pos, labels, n_thresholds=100):
    """Compute ROC curve points for plotting."""
    thresholds = [i/n_thresholds for i in range(n_thresholds+1)]
    tpr_list, fpr_list = [], []
    pos_total = sum(1 for l in labels if l == 1)
    neg_total = sum(1 for l in labels if l == 0)
    for t in thresholds:
        tp = sum(1 for p,l in zip(probs_pos,labels) if p >= t and l == 1)
        fp = sum(1 for p,l in zip(probs_pos,labels) if p >= t and l == 0)
        tpr_list.append(tp / pos_total if pos_total > 0 else 0)
        fpr_list.append(fp / neg_total if neg_total > 0 else 0)
    return fpr_list, tpr_list

def compute_confidence_histogram(all_probs, all_preds, all_labels):
    """Confidence distribution for correct vs incorrect predictions."""
    correct_conf   = [max(all_probs[i]) for i in range(len(all_labels))
                      if all_preds[i] == all_labels[i]]
    incorrect_conf = [max(all_probs[i]) for i in range(len(all_labels))
                      if all_preds[i] != all_labels[i]]
    bins = [i/20 for i in range(21)]  # 0.0 to 1.0 in 0.05 steps
    def hist(vals):
        counts = [0] * 20
        for v in vals:
            idx = min(int(v * 20), 19)
            counts[idx] += 1
        return counts
    return bins[:-1], hist(correct_conf), hist(incorrect_conf)


# ═══════════════════════════════════════════════════════════════════
# EVALUATE
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(model, data, batch_size=256):
    dataset = TensorDataset(data["input_ids"], data["labels"])
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=False, num_workers=0)
    all_preds, all_labels, all_probs = [], [], []

    model.eval()
    for ids, lbls in loader:
        logits = model(ids)
        probs  = torch.softmax(logits, dim=-1)
        all_preds.extend(logits.argmax(dim=-1).tolist())
        all_labels.extend(lbls.tolist())
        all_probs.extend(probs.tolist())

    cm              = confusion_matrix_fn(all_preds, all_labels)
    prec, rec, f1, spec = per_class_metrics(cm)
    mcc_val         = mcc_fn(cm)
    total           = len(all_labels)
    correct         = sum(p==l for p,l in zip(all_preds, all_labels))
    probs_pos       = [p[1] for p in all_probs]
    auc             = roc_auc_binary(probs_pos, all_labels)
    fpr, tpr        = compute_roc_curve(probs_pos, all_labels)
    bins, corr_hist, incorr_hist = compute_confidence_histogram(
        all_probs, all_preds, all_labels)

    # Per-class confidence averages
    intron_confs = [all_probs[i][0] for i in range(total) if all_labels[i]==0]
    exon_confs   = [all_probs[i][1] for i in range(total) if all_labels[i]==1]
    avg_intron_conf = sum(intron_confs)/len(intron_confs) if intron_confs else 0
    avg_exon_conf   = sum(exon_confs)/len(exon_confs)     if exon_confs   else 0

    return {
        "n_samples":          total,
        "accuracy":           correct / total,
        "balanced_accuracy":  (rec[0] + rec[1]) / 2,
        "macro_f1":           (f1[0]  + f1[1])  / 2,
        "mcc":                mcc_val,
        "auc":                auc,
        "per_class": {
            "intron": {"precision": prec[0], "recall": rec[0],
                       "f1": f1[0], "specificity": spec[0],
                       "avg_confidence": avg_intron_conf},
            "exon":   {"precision": prec[1], "recall": rec[1],
                       "f1": f1[1], "specificity": spec[1],
                       "avg_confidence": avg_exon_conf},
        },
        "confusion_matrix":   cm,
        "label_dist":         dict(Counter(all_labels)),
        "pred_dist":          dict(Counter(all_preds)),
        "roc_curve":          {"fpr": fpr, "tpr": tpr},
        "confidence_hist":    {"bins": bins,
                               "correct":   corr_hist,
                               "incorrect": incorr_hist},
    }


# ═══════════════════════════════════════════════════════════════════
# TEXT REPORT  — all metrics
# ═══════════════════════════════════════════════════════════════════

def write_report(m, h, out_dir):
    """m = chimp metrics, h = human baseline dict"""
    path = out_dir / "chimp_eval_report.txt"
    with open(path, "w", encoding="utf-8") as fh:
        def w(s=""): fh.write(s + "\n")

        w("=" * 72)
        w("CROSS-SPECIES GENERALISATION REPORT")
        w("Species  : Chimpanzee (Pan troglodytes)")
        w("Task     : Task A — exon vs intron classification")
        w("Model    : Human GRCh38-trained, NO retraining on chimp")
        w("Sample   : 10% stratified (seed=42)")
        w("=" * 72)
        w()
        w(f"  Sequences evaluated : {m['n_samples']:,}")
        w(f"  Introns             : {m['label_dist'].get(0,0):,}")
        w(f"  Exons               : {m['label_dist'].get(1,0):,}")
        w()

        # Full metric comparison table
        w("  METRIC COMPARISON — Human (GRCh38) vs Chimpanzee")
        w("  " + "-" * 62)
        w(f"  {'Metric':<26} {'Human':>10} {'Chimp':>10} "
          f"{'Delta':>10} {'Drop%':>8}")
        w("  " + "-" * 62)

        metrics_list = [
            ("Accuracy",          "accuracy"),
            ("Balanced Accuracy", "balanced_accuracy"),
            ("Macro F1",          "macro_f1"),
            ("MCC",               "mcc"),
            ("AUC (ROC)",         "auc"),
        ]
        for label, key in metrics_list:
            hv    = h[key]
            cv    = m[key]
            delta = cv - hv
            drop  = (delta / hv * 100) if hv > 0 else 0
            arrow = "+" if delta > 0.005 else ("-" if delta < -0.005 else "~")
            w(f"  {label:<26} {hv:>10.4f} {cv:>10.4f} "
              f"{delta:>+10.4f} {drop:>7.1f}% {arrow}")

        w()
        w("  PER-CLASS METRICS — Intron")
        w("  " + "-" * 62)
        w(f"  {'Metric':<26} {'Human':>10} {'Chimp':>10} {'Delta':>10}")
        w("  " + "-" * 62)
        for metric in ["precision", "recall", "f1", "specificity"]:
            hv    = h["per_class"]["intron"][metric]
            cv    = m["per_class"]["intron"][metric]
            delta = cv - hv
            w(f"  {metric.capitalize():<26} {hv:>10.4f} {cv:>10.4f} {delta:>+10.4f}")

        w()
        w("  PER-CLASS METRICS — Exon")
        w("  " + "-" * 62)
        w(f"  {'Metric':<26} {'Human':>10} {'Chimp':>10} {'Delta':>10}")
        w("  " + "-" * 62)
        for metric in ["precision", "recall", "f1", "specificity"]:
            hv    = h["per_class"]["exon"][metric]
            cv    = m["per_class"]["exon"][metric]
            delta = cv - hv
            w(f"  {metric.capitalize():<26} {hv:>10.4f} {cv:>10.4f} {delta:>+10.4f}")

        w()
        w("  CONFUSION MATRIX — Chimpanzee (rows=actual, cols=predicted):")
        w("              intron    exon")
        cm = m["confusion_matrix"]
        w(f"    intron    {cm[0][0]:>7,}  {cm[0][1]:>7,}")
        w(f"    exon      {cm[1][0]:>7,}  {cm[1][1]:>7,}")
        w()
        w("  MODEL CONFIDENCE:")
        w(f"    Avg confidence on intron sequences: "
          f"{m['per_class']['intron']['avg_confidence']:.4f}")
        w(f"    Avg confidence on exon sequences  : "
          f"{m['per_class']['exon']['avg_confidence']:.4f}")

        # Interpretation
        w()
        w("  INTERPRETATION:")
        mcc_chimp = m["mcc"]
        delta_mcc = mcc_chimp - h["mcc"]
        if mcc_chimp >= 0.80:
            w("  STRONG generalisation — model transfers excellently to chimpanzee.")
            w("  Suggests learned features are deeply conserved across primates.")
        elif mcc_chimp >= 0.65:
            w("  GOOD generalisation — small performance drop vs human.")
            w("  Core splice signals conserved; minor species-specific variation.")
        elif mcc_chimp >= 0.40:
            w("  MODERATE generalisation — noticeable species gap.")
            w("  Some learned features may be human-specific.")
        else:
            w("  POOR generalisation — model struggles on chimpanzee DNA.")
        w(f"  MCC retention: {mcc_chimp/h['mcc']*100:.1f}% of human performance")
        w(f"  MCC delta    : {delta_mcc:+.4f}")
        w("=" * 72)

    print(f"[12] Report  -> {path}")

    # TSV — all metrics for paper
    tsv_path = out_dir / "chimp_eval_summary.tsv"
    with open(tsv_path, "w", encoding="utf-8") as fh:
        headers = ("species\ttask\tn_samples\t"
                   "accuracy\tbal_acc\tmacro_f1\tmcc\tauc\t"
                   "intron_prec\tintron_rec\tintron_f1\tintron_spec\t"
                   "exon_prec\texon_rec\texon_f1\texon_spec\t"
                   "human_mcc\tdelta_mcc\tmcc_retention_pct\n")
        fh.write(headers)
        pc = m["per_class"]
        fh.write(
            f"chimpanzee\ttask_a\t{m['n_samples']}\t"
            f"{m['accuracy']:.6f}\t{m['balanced_accuracy']:.6f}\t"
            f"{m['macro_f1']:.6f}\t{m['mcc']:.6f}\t{m['auc']:.6f}\t"
            f"{pc['intron']['precision']:.6f}\t{pc['intron']['recall']:.6f}\t"
            f"{pc['intron']['f1']:.6f}\t{pc['intron']['specificity']:.6f}\t"
            f"{pc['exon']['precision']:.6f}\t{pc['exon']['recall']:.6f}\t"
            f"{pc['exon']['f1']:.6f}\t{pc['exon']['specificity']:.6f}\t"
            f"{h['mcc']:.6f}\t{m['mcc']-h['mcc']:.6f}\t"
            f"{m['mcc']/h['mcc']*100:.2f}\n"
        )
    print(f"[12] TSV     -> {tsv_path}")


# ═══════════════════════════════════════════════════════════════════
# HTML VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════

def write_visualizations(m, h, out_dir):
    """
    Generates a standalone HTML file with 6 charts comparing human vs chimp:
      1. Metric bar chart (all 5 metrics side by side)
      2. Per-class F1 grouped bars
      3. Per-class Precision/Recall/Specificity radar-style bars
      4. ROC curve (chimp)
      5. Confidence histogram (correct vs incorrect predictions)
      6. Confusion matrix heatmap
    """
    pc_c = m["per_class"]
    pc_h = h["per_class"]

    # Serialize data for JS
    bar_metrics   = ["Accuracy", "Bal Accuracy", "Macro F1", "MCC", "AUC"]
    human_vals    = [h["accuracy"], h["balanced_accuracy"],
                     h["macro_f1"], h["mcc"], h["auc"]]
    chimp_vals    = [m["accuracy"], m["balanced_accuracy"],
                     m["macro_f1"], m["mcc"], m["auc"]]

    classes       = ["Intron", "Exon"]
    human_f1      = [pc_h["intron"]["f1"], pc_h["exon"]["f1"]]
    chimp_f1      = [pc_c["intron"]["f1"], pc_c["exon"]["f1"]]

    prec_labels   = ["Intron Prec", "Intron Rec", "Intron Spec",
                     "Exon Prec",   "Exon Rec",   "Exon Spec"]
    human_pcm     = [pc_h["intron"]["precision"], pc_h["intron"]["recall"],
                     pc_h["intron"]["specificity"],
                     pc_h["exon"]["precision"],   pc_h["exon"]["recall"],
                     pc_h["exon"]["specificity"]]
    chimp_pcm     = [pc_c["intron"]["precision"], pc_c["intron"]["recall"],
                     pc_c["intron"]["specificity"],
                     pc_c["exon"]["precision"],   pc_c["exon"]["recall"],
                     pc_c["exon"]["specificity"]]

    roc           = m["roc_curve"]
    conf          = m["confidence_hist"]
    cm            = m["confusion_matrix"]
    cm_max        = max(cm[r][c] for r in range(2) for c in range(2))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Cross-Species Generalisation — Human vs Chimpanzee</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
  :root {{
    --bg:      #0d1117;
    --surface: #161b22;
    --border:  #30363d;
    --human:   #58a6ff;
    --chimp:   #3fb950;
    --accent:  #f78166;
    --text:    #e6edf3;
    --muted:   #8b949e;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
    padding: 2rem;
  }}
  h1 {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--human);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1rem;
    margin-bottom: 0.5rem;
  }}
  .subtitle {{
    color: var(--muted);
    font-size: 0.85rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 2.5rem;
  }}
  .grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    max-width: 1200px;
  }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
  }}
  .card.wide {{ grid-column: 1 / -1; }}
  .card h2 {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1.2rem;
  }}
  .legend {{
    display: flex;
    gap: 1.5rem;
    margin-bottom: 1rem;
    font-size: 0.8rem;
    font-family: 'IBM Plex Mono', monospace;
  }}
  .legend-dot {{
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-right: 5px;
  }}
  canvas {{ width: 100%; }}
  .stat-row {{
    display: flex;
    justify-content: space-between;
    padding: 0.4rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.85rem;
  }}
  .stat-row:last-child {{ border-bottom: none; }}
  .stat-label {{ color: var(--muted); font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; }}
  .stat-human {{ color: var(--human); font-family: 'IBM Plex Mono', monospace; }}
  .stat-chimp {{ color: var(--chimp); font-family: 'IBM Plex Mono', monospace; }}
  .stat-delta {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; }}
  .neg {{ color: var(--accent); }}
  .pos {{ color: var(--chimp); }}
  .badge {{
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 1rem;
  }}
</style>
</head>
<body>

<h1>Cross-Species Generalisation Report</h1>
<div class="subtitle">
  Task A (exon/intron) &nbsp;|&nbsp;
  Human model (GRCh38) tested on Chimpanzee (Pan troglodytes) &nbsp;|&nbsp;
  10% stratified sample &nbsp;|&nbsp;
  n={m['n_samples']:,} sequences
</div>

<div class="grid">

  <!-- Chart 1: All metrics bar -->
  <div class="card">
    <h2>All Metrics — Human vs Chimpanzee</h2>
    <div class="legend">
      <span><span class="legend-dot" style="background:var(--human)"></span>Human (GRCh38)</span>
      <span><span class="legend-dot" style="background:var(--chimp)"></span>Chimpanzee</span>
    </div>
    <canvas id="c1" height="220"></canvas>
  </div>

  <!-- Chart 2: Per-class F1 -->
  <div class="card">
    <h2>Per-Class F1 Score</h2>
    <div class="legend">
      <span><span class="legend-dot" style="background:var(--human)"></span>Human</span>
      <span><span class="legend-dot" style="background:var(--chimp)"></span>Chimpanzee</span>
    </div>
    <canvas id="c2" height="220"></canvas>
  </div>

  <!-- Chart 3: Precision/Recall/Specificity grouped -->
  <div class="card wide">
    <h2>Per-Class Precision / Recall / Specificity</h2>
    <div class="legend">
      <span><span class="legend-dot" style="background:var(--human)"></span>Human</span>
      <span><span class="legend-dot" style="background:var(--chimp)"></span>Chimpanzee</span>
    </div>
    <canvas id="c3" height="180"></canvas>
  </div>

  <!-- Chart 4: ROC curve -->
  <div class="card">
    <h2>ROC Curve — Chimpanzee (Exon class)</h2>
    <canvas id="c4" height="220"></canvas>
  </div>

  <!-- Chart 5: Confidence histogram -->
  <div class="card">
    <h2>Prediction Confidence — Correct vs Incorrect</h2>
    <div class="legend">
      <span><span class="legend-dot" style="background:var(--chimp)"></span>Correct</span>
      <span><span class="legend-dot" style="background:var(--accent)"></span>Incorrect</span>
    </div>
    <canvas id="c5" height="220"></canvas>
  </div>

  <!-- Chart 6: Confusion matrix heatmap -->
  <div class="card">
    <h2>Confusion Matrix — Chimpanzee</h2>
    <canvas id="c6" height="220"></canvas>
  </div>

  <!-- Summary table -->
  <div class="card wide">
    <h2>Metric Summary Table</h2>
    <div style="display:grid;grid-template-columns:2fr 1fr 1fr 1fr;gap:0.2rem;">
      <div class="stat-row"><span class="stat-label">Metric</span><span class="stat-human">Human</span><span class="stat-chimp">Chimp</span><span class="stat-label">Delta</span></div>
      {"".join(
        f'<div class="stat-row"><span class="stat-label">{lbl}</span>'
        f'<span class="stat-human">{hv:.4f}</span>'
        f'<span class="stat-chimp">{cv:.4f}</span>'
        f'<span class="stat-delta {"neg" if cv-hv < -0.005 else "pos"}">{cv-hv:+.4f}</span></div>'
        for lbl, hv, cv in zip(
            ["Accuracy","Balanced Acc","Macro F1","MCC","AUC",
             "Intron F1","Exon F1","Intron Precision","Exon Precision",
             "Intron Recall","Exon Recall","Intron Specificity","Exon Specificity"],
            [h["accuracy"],h["balanced_accuracy"],h["macro_f1"],h["mcc"],h["auc"],
             pc_h["intron"]["f1"],pc_h["exon"]["f1"],
             pc_h["intron"]["precision"],pc_h["exon"]["precision"],
             pc_h["intron"]["recall"],pc_h["exon"]["recall"],
             pc_h["intron"]["specificity"],pc_h["exon"]["specificity"]],
            [m["accuracy"],m["balanced_accuracy"],m["macro_f1"],m["mcc"],m["auc"],
             pc_c["intron"]["f1"],pc_c["exon"]["f1"],
             pc_c["intron"]["precision"],pc_c["exon"]["precision"],
             pc_c["intron"]["recall"],pc_c["exon"]["recall"],
             pc_c["intron"]["specificity"],pc_c["exon"]["specificity"]]
        )
      )}
    </div>
    <div>
      <span class="badge" style="background:{"#1a3a1a" if m["mcc"]>=0.65 else "#3a1a1a"};
            color:{"var(--chimp)" if m["mcc"]>=0.65 else "var(--accent)"}">
        MCC retention: {m["mcc"]/h["mcc"]*100:.1f}% of human performance
      </span>
    </div>
  </div>

</div>

<script>
// ── Utility functions ───────────────────────────────────────────────
function hexToRgb(hex) {{
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return `${{r}},${{g}},${{b}}`;
}}

const HUMAN_COLOR  = '#58a6ff';
const CHIMP_COLOR  = '#3fb950';
const ACCENT_COLOR = '#f78166';
const TEXT_COLOR   = '#8b949e';
const GRID_COLOR   = '#21262d';

function setupCanvas(id) {{
  const canvas = document.getElementById(id);
  const dpr    = window.devicePixelRatio || 1;
  const rect   = canvas.getBoundingClientRect();
  canvas.width  = canvas.offsetParent.clientWidth * dpr;
  canvas.height = parseInt(canvas.getAttribute('height')) * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  canvas._w = canvas.offsetParent.clientWidth;
  canvas._h = parseInt(canvas.getAttribute('height'));
  return ctx;
}}

function drawGrid(ctx, w, h, pad, maxY, steps=5) {{
  ctx.strokeStyle = GRID_COLOR;
  ctx.lineWidth   = 1;
  ctx.fillStyle   = TEXT_COLOR;
  ctx.font        = '10px IBM Plex Mono';
  ctx.textAlign   = 'right';
  for (let i=0; i<=steps; i++) {{
    const y  = pad.top + (h - pad.top - pad.bottom) * (1 - i/steps);
    const val = (maxY * i / steps).toFixed(2);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();
    ctx.fillText(val, pad.left - 4, y + 3);
  }}
}}

// ── Chart 1: All metrics grouped bars ──────────────────────────────
(function() {{
  const ctx  = setupCanvas('c1');
  const W    = ctx.canvas._w, H = ctx.canvas._h;
  const pad  = {{top:20, right:20, bottom:50, left:45}};
  const labels = {json.dumps(bar_metrics)};
  const hVals  = {json.dumps([round(v,4) for v in human_vals])};
  const cVals  = {json.dumps([round(v,4) for v in chimp_vals])};
  const n      = labels.length;
  const gw     = (W - pad.left - pad.right) / n;
  const bw     = gw * 0.32;

  drawGrid(ctx, W, H, pad, 1.0);

  labels.forEach((lbl, i) => {{
    const x0 = pad.left + i * gw + gw * 0.1;
    [hVals[i], cVals[i]].forEach((v, j) => {{
      const x   = x0 + j * (bw + 3);
      const bh  = (H - pad.top - pad.bottom) * v;
      const y   = H - pad.bottom - bh;
      ctx.fillStyle = j === 0 ? HUMAN_COLOR : CHIMP_COLOR;
      ctx.beginPath();
      ctx.roundRect(x, y, bw, bh, [3,3,0,0]);
      ctx.fill();
      ctx.fillStyle = j === 0 ? HUMAN_COLOR : CHIMP_COLOR;
      ctx.font      = '9px IBM Plex Mono';
      ctx.textAlign = 'center';
      ctx.fillText(v.toFixed(3), x + bw/2, y - 3);
    }});
    ctx.fillStyle = TEXT_COLOR;
    ctx.font      = '9px IBM Plex Mono';
    ctx.textAlign = 'center';
    ctx.fillText(lbl, x0 + bw + 1.5, H - pad.bottom + 14);
    ctx.save();
    ctx.translate(x0 + bw + 1.5, H - pad.bottom + 14);
    ctx.restore();
  }});
}})();

// ── Chart 2: Per-class F1 ───────────────────────────────────────────
(function() {{
  const ctx    = setupCanvas('c2');
  const W = ctx.canvas._w, H = ctx.canvas._h;
  const pad    = {{top:20, right:20, bottom:50, left:45}};
  const labels = {json.dumps(classes)};
  const hVals  = {json.dumps([round(v,4) for v in human_f1])};
  const cVals  = {json.dumps([round(v,4) for v in chimp_f1])};
  const n      = labels.length;
  const gw     = (W - pad.left - pad.right) / n;
  const bw     = gw * 0.35;

  drawGrid(ctx, W, H, pad, 1.0);

  labels.forEach((lbl, i) => {{
    const x0 = pad.left + i * gw + gw * 0.1;
    [hVals[i], cVals[i]].forEach((v, j) => {{
      const x  = x0 + j * (bw + 5);
      const bh = (H - pad.top - pad.bottom) * v;
      const y  = H - pad.bottom - bh;
      ctx.fillStyle = j === 0 ? HUMAN_COLOR : CHIMP_COLOR;
      ctx.beginPath();
      ctx.roundRect(x, y, bw, bh, [3,3,0,0]);
      ctx.fill();
      ctx.fillStyle = j === 0 ? HUMAN_COLOR : CHIMP_COLOR;
      ctx.font = '10px IBM Plex Mono';
      ctx.textAlign = 'center';
      ctx.fillText(v.toFixed(3), x + bw/2, y - 4);
    }});
    ctx.fillStyle = TEXT_COLOR;
    ctx.font = '11px IBM Plex Mono';
    ctx.textAlign = 'center';
    ctx.fillText(lbl, x0 + bw + 2.5, H - pad.bottom + 16);
  }});
}})();

// ── Chart 3: Prec/Rec/Spec grouped ────────────────────────────────
(function() {{
  const ctx    = setupCanvas('c3');
  const W = ctx.canvas._w, H = ctx.canvas._h;
  const pad    = {{top:20, right:20, bottom:50, left:45}};
  const labels = {json.dumps(prec_labels)};
  const hVals  = {json.dumps([round(v,4) for v in human_pcm])};
  const cVals  = {json.dumps([round(v,4) for v in chimp_pcm])};
  const n      = labels.length;
  const gw     = (W - pad.left - pad.right) / n;
  const bw     = gw * 0.35;

  drawGrid(ctx, W, H, pad, 1.0);

  // Separator line between intron and exon groups
  const sepX = pad.left + 3 * gw;
  ctx.strokeStyle = '#30363d';
  ctx.setLineDash([4,4]);
  ctx.beginPath();
  ctx.moveTo(sepX, pad.top);
  ctx.lineTo(sepX, H - pad.bottom);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = TEXT_COLOR;
  ctx.font = '9px IBM Plex Mono';
  ctx.textAlign = 'center';
  ctx.fillText('INTRON', pad.left + 1.5*gw, pad.top - 4);
  ctx.fillText('EXON', pad.left + 4.5*gw, pad.top - 4);

  labels.forEach((lbl, i) => {{
    const x0 = pad.left + i * gw + gw * 0.1;
    [hVals[i], cVals[i]].forEach((v, j) => {{
      const x  = x0 + j*(bw+3);
      const bh = (H - pad.top - pad.bottom) * v;
      const y  = H - pad.bottom - bh;
      ctx.fillStyle = j === 0 ? HUMAN_COLOR : CHIMP_COLOR;
      ctx.beginPath();
      ctx.roundRect(x, y, bw, bh, [2,2,0,0]);
      ctx.fill();
    }});
    ctx.fillStyle = TEXT_COLOR;
    ctx.font = '8px IBM Plex Mono';
    ctx.textAlign = 'center';
    const shortLabel = lbl.split(' ')[1] || lbl;
    ctx.fillText(shortLabel, x0 + bw + 1.5, H - pad.bottom + 13);
  }});
}})();

// ── Chart 4: ROC curve ─────────────────────────────────────────────
(function() {{
  const ctx  = setupCanvas('c4');
  const W = ctx.canvas._w, H = ctx.canvas._h;
  const pad  = {{top:20, right:20, bottom:40, left:45}};
  const fpr  = {json.dumps([round(v,4) for v in roc["fpr"]])};
  const tpr  = {json.dumps([round(v,4) for v in roc["tpr"]])};
  const auc  = {round(m["auc"], 4)};

  drawGrid(ctx, W, H, pad, 1.0);

  // Diagonal baseline
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 1;
  ctx.setLineDash([4,4]);
  ctx.beginPath();
  ctx.moveTo(pad.left, H - pad.bottom);
  ctx.lineTo(W - pad.right, pad.top);
  ctx.stroke();
  ctx.setLineDash([]);

  // ROC curve
  ctx.strokeStyle = CHIMP_COLOR;
  ctx.lineWidth   = 2;
  ctx.beginPath();
  fpr.forEach((x, i) => {{
    const px = pad.left + x * (W - pad.left - pad.right);
    const py = H - pad.bottom - tpr[i] * (H - pad.top - pad.bottom);
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  }});
  ctx.stroke();

  // AUC label
  ctx.fillStyle = CHIMP_COLOR;
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'left';
  ctx.fillText(`AUC = ${{auc.toFixed(4)}}`, pad.left + 10, pad.top + 20);

  // Axis labels
  ctx.fillStyle = TEXT_COLOR;
  ctx.font = '9px IBM Plex Mono';
  ctx.textAlign = 'center';
  ctx.fillText('False Positive Rate', W/2, H - 4);
}})();

// ── Chart 5: Confidence histogram ─────────────────────────────────
(function() {{
  const ctx      = setupCanvas('c5');
  const W = ctx.canvas._w, H = ctx.canvas._h;
  const pad      = {{top:20, right:20, bottom:40, left:45}};
  const bins     = {json.dumps([round(v,2) for v in conf["bins"]])};
  const correct  = {json.dumps(conf["correct"])};
  const incorrec = {json.dumps(conf["incorrect"])};
  const maxVal   = Math.max(...correct, ...incorrec, 1);
  const n        = bins.length;
  const bw       = (W - pad.left - pad.right) / n;

  // Grid
  ctx.strokeStyle = GRID_COLOR;
  for (let i=0; i<=4; i++) {{
    const y = pad.top + (H - pad.top - pad.bottom) * (1 - i/4);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W-pad.right, y); ctx.stroke();
  }}

  bins.forEach((b, i) => {{
    const x  = pad.left + i * bw;
    [[correct[i], CHIMP_COLOR, 0], [incorrec[i], ACCENT_COLOR, bw/2]].forEach(([v, col, off]) => {{
      if (v === 0) return;
      const bh = (H - pad.top - pad.bottom) * (v / maxVal);
      const y  = H - pad.bottom - bh;
      ctx.fillStyle = col + '99';
      ctx.fillRect(x + off, y, bw/2 - 1, bh);
    }});
  }});

  ctx.fillStyle = TEXT_COLOR;
  ctx.font = '9px IBM Plex Mono';
  ctx.textAlign = 'center';
  ctx.fillText('0.0', pad.left, H - pad.bottom + 12);
  ctx.fillText('0.5', pad.left + (W-pad.left-pad.right)/2, H - pad.bottom + 12);
  ctx.fillText('1.0', W - pad.right, H - pad.bottom + 12);
  ctx.fillText('Prediction Confidence', W/2, H - 2);
}})();

// ── Chart 6: Confusion matrix heatmap ─────────────────────────────
(function() {{
  const ctx  = setupCanvas('c6');
  const W = ctx.canvas._w, H = ctx.canvas._h;
  const cm   = {json.dumps(cm)};
  const labs = ['Intron', 'Exon'];
  const pad  = {{top:50, right:20, bottom:50, left:70}};
  const cw   = (W - pad.left - pad.right)  / 2;
  const ch   = (H - pad.top  - pad.bottom) / 2;
  const maxV = {cm_max};

  cm.forEach((row, r) => {{
    row.forEach((val, c) => {{
      const intensity = val / maxV;
      const isCorrect = r === c;
      const alpha     = 0.15 + intensity * 0.75;
      const baseColor = isCorrect
        ? `rgba(${{hexToRgb(CHIMP_COLOR)}},${{alpha}})`
        : `rgba(${{hexToRgb(ACCENT_COLOR)}},${{alpha * 0.6}})`
      const x = pad.left + c * cw;
      const y = pad.top  + r * ch;
      ctx.fillStyle = baseColor;
      ctx.fillRect(x, y, cw - 2, ch - 2);
      ctx.fillStyle = intensity > 0.4 ? '#0d1117' : '#e6edf3';
      ctx.font = 'bold 13px IBM Plex Mono';
      ctx.textAlign = 'center';
      ctx.fillText(val.toLocaleString(), x + cw/2, y + ch/2 + 5);
    }});
  }});

  // Labels
  ctx.fillStyle = TEXT_COLOR;
  ctx.font = '10px IBM Plex Mono';
  ctx.textAlign = 'center';
  labs.forEach((l, i) => {{
    ctx.fillText(l, pad.left + i*cw + cw/2, pad.top - 8);
    ctx.textAlign = 'right';
    ctx.fillText(l, pad.left - 6, pad.top + i*ch + ch/2 + 4);
    ctx.textAlign = 'center';
  }});

  ctx.fillStyle = TEXT_COLOR;
  ctx.font = '9px IBM Plex Mono';
  ctx.fillText('Predicted', W/2, 15);
  ctx.save();
  ctx.translate(12, H/2);
  ctx.rotate(-Math.PI/2);
  ctx.fillText('Actual', 0, 0);
  ctx.restore();
}})();
</script>
</body>
</html>"""

    vis_path = out_dir / "chimp_visualizations.html"
    with open(vis_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"[12] Charts  -> {vis_path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Cross-species test: human Task A model on chimpanzee DNA"
    )
    ap.add_argument("--chimp_fasta", required=True)
    ap.add_argument("--chimp_gtf",   required=True)
    ap.add_argument("--ckpt",        default="checkpoints/task_a_best.pt")
    ap.add_argument("--out_dir",     default="results_crossspecies")
    ap.add_argument("--sample_frac", type=float, default=0.10)
    ap.add_argument("--batch_size",  type=int,   default=256)
    ap.add_argument("--seed",        type=int,   default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[12] Cross-species generalisation — Chimpanzee")
    print(f"     FASTA  : {args.chimp_fasta}")
    print(f"     GTF    : {args.chimp_gtf}")
    print(f"     Ckpt   : {args.ckpt}")
    print(f"     Sample : {args.sample_frac*100:.0f}%  seed={args.seed}")

    # 1. Extract
    t0   = time.time()
    seqs = extract_chimp_sequences(args.chimp_fasta, args.chimp_gtf)
    print(f"     Extraction: {time.time()-t0:.1f}s")

    # 2. Subsample
    sample = subsample(seqs, frac=args.sample_frac, seed=args.seed)

    # 3. Encode
    data = encode_sequences(sample, out_dir / "chimp_encoded.pt")

    # 4. Load model
    print(f"\n[12] Loading checkpoint: {args.ckpt}")
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Not found: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg   = ckpt["config"]
    model = DNAClassifier.build(
        num_classes=ckpt["num_classes"], d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        ffn_dim=cfg["ffn_dim"], max_len=cfg["max_len"],
    )
    model.encoder.load_state_dict(ckpt["encoder_state"])
    model.classifier.load_state_dict(ckpt["head_state"])
    model.eval()
    print(f"     epoch={ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('best_val_loss',0):.4f}")

    # 5. Evaluate
    print("\n[12] Evaluating ...")
    t0      = time.time()
    metrics = evaluate_model(model, data, args.batch_size)
    print(f"     Done in {time.time()-t0:.1f}s")

    h = HUMAN_BASELINE
    print(f"\n     {'Metric':<22} {'Human':>8} {'Chimp':>8} {'Delta':>8}")
    print(f"     {'-'*50}")
    for label, key in [("Accuracy","accuracy"),("Bal Accuracy","balanced_accuracy"),
                        ("Macro F1","macro_f1"),("MCC","mcc"),("AUC","auc")]:
        delta = metrics[key] - h[key]
        print(f"     {label:<22} {h[key]:>8.4f} {metrics[key]:>8.4f} {delta:>+8.4f}")

    # 6. Outputs
    write_report(metrics, h, out_dir)
    write_visualizations(metrics, h, out_dir)

    with open(out_dir / "chimp_eval_raw.json", "w", encoding="utf-8") as fh:
        # Remove roc_curve from json (too large) — keep everything else
        save_m = {k: v for k, v in metrics.items()
                  if k not in ("roc_curve", "confidence_hist")}
        json.dump(save_m, fh, default=lambda o: round(o,6) if isinstance(o,float) else o,
                  indent=2)
    print(f"[12] JSON    -> {out_dir}/chimp_eval_raw.json")
    print("\n[12] Done. Open chimp_visualizations.html in your browser.")


if __name__ == "__main__":
    main()