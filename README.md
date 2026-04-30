# 🧬 ExIT: Exon–Intron Transformer

> A lightweight Transformer-based framework for genomic sequence analysis using codon-level representations.

---

## 🌟 Overview

**ExIT (Exon–Intron Transformer)** is designed to model genomic sequences efficiently using **codon-level tokenization** and a **shared Transformer encoder**.

Unlike large genomic foundation models that require GPUs, ExIT is built for:

- ⚡ CPU-based training  
- 🧠 biologically meaningful representation  
- 🔀 multi-task genomic prediction  

---

## 🚀 Key Highlights

- 🧬 Codon-based (3-mer) tokenization  
- 🧠 Transformer encoder with self-attention  
- 🔀 Multi-task learning (3 genomic tasks)  
- ⚡ Runs on consumer hardware (~8GB RAM)  
- 🌍 Cross-species generalization (Human → Chimpanzee)  

---

## 🏗️ Model Architecture
      DNA Sequence
      ↓
      Codon Tokenization (3-mer)
      ↓
      Embedding + Positional Encoding
      ↓
      Transformer Encoder (Self-Attention + FFN)
      ↓
      [CLS] Token Representation
      ↓
      Task-Specific Heads

---

<details>
<summary><b>🔬 Detailed Architecture (Click to Expand)</b></summary>

### Input Representation
- DNA is split into **non-overlapping codons**
- Vocabulary:
  - 64 codons  
  - Special tokens: `<PAD>`, `<UNK>`, `<MASK>`

### Positional Encoding
- Sinusoidal encoding added to embeddings
- Preserves sequence order

### Transformer Encoder
Each layer includes:
- Multi-head self-attention  
- Feed-forward network (GELU)  
- Residual connections + layer normalization  

### [CLS] Token
- Represents entire sequence
- Used for downstream classification  

</details>

---

## 📊 Tasks

| Task | Description | Output |
|------|------------|--------|
| **Task A** | Exon vs Intron | Binary |
| **Task B** | CDS Identification | Binary |
| **Task C** | Splice Site Detection | 3-class |

---

## 📂 Dataset

### Source
- Ensembl (Release 109 / 111)

### Genomes Used
- Human (GRCh38)  
- Chimpanzee (Pan_tro_3.0)

---

<details>
<summary><b>🧪 Data Processing Pipeline (Click to Expand)</b></summary>

- Extracted exon/CDS regions from GTF  
- Inferred introns from exon gaps  
- Generated splice site windows  
- Reverse-complemented negative strand sequences  
- Resolved ambiguous nucleotides  
- Filtered sequence length (50–5000 bp)  
- Converted sequences to codon tokens  

</details>

---

## 📈 Evaluation Metrics

- **Matthews Correlation Coefficient (MCC)** (Primary)  
- Accuracy  
- Precision / Recall  
- F1 Score  

---
📊 Outputs

The model generates:

📌 Predictions for each sequence
📊 Evaluation metrics (MCC, accuracy, etc.)
📉 Visualizations:
Confusion matrix
Prediction distribution

🔬 Interpretability & Insights

ExIT is designed to capture biologically meaningful patterns such as:

Coding vs non-coding regions
Splice junction signals
Codon-level sequence structure

Future extensions aim to include:

Attention visualization
Gene expression prediction
Codon usage bias analysis

🤝 Contributing

We welcome contributions!

Open issues for bugs/features
Submit pull requests
Suggest new datasets or tasks
📜 License

MIT License

## 🧪 Experimental Highlights

- ✔ Strong exon–intron classification performance  
- ✔ Robust cross-species generalization  
- ✔ Efficient CPU-based training  
- ✔ No reliance on large pretrained genomic models  

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/ExIT.git
cd ExIT
pip install -r requirements.txt
```
## Workflow
# Step 1: Preprocess
python preprocess.py --fasta genome.fa --gtf genes.gtf

# Step 2: Train
python train.py --task A

# Step 3: Evaluate
python evaluate.py --checkpoint best_model.pt
