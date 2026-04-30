ExIT: Exon-Intron Transformer
ExIT is a high-efficiency genomic modeling framework designed to bring advanced sequence analysis to consumer-grade hardware. By leveraging codon-level tokenization and a multi-task Transformer architecture, it provides a robust alternative to high-parameter models that typically require expensive GPU clusters.

🏗️ Detailed Architecture
The model utilizes a hybrid Transformer structure, featuring a shared encoder that serves as a universal feature extractor for various genomic classification tasks.

1. Input & Tokenization
Codon-Level Processing: Unlike traditional k-mer approaches, ExIT tokenizes DNA sequences into biologically relevant 3-base triplets (codons).

Embeddings: Raw tokens are mapped to a high-dimensional vector space where semantic genomic relationships are captured.

Positional Encoding: Since Transformers lack inherent sequence order awareness, positional encodings are injected to maintain the relative and absolute location of codons within the gene structure.

2. The Shared Encoder
Multi-Head Attention: This allows the model to simultaneously focus on different regulatory motifs—such as splice sites and start codons—across the sequence.

FFN (GELU): A Feed-Forward Network using Gaussian Error Linear Unit activation handles the non-linear transformations of the attended features.

[CLS] Pooling: A dedicated classification token is pooled to represent the aggregate sequence context, which is then branched into independent task-specific heads.

3. Multi-Task Prediction Heads
The architecture is designed to perform three distinct genomic tasks in a single forward pass:

Task A (Exon/Intron): Classifies regions as coding exons or non-coding introns.

Task B (CDS Identification): Specifically identifies the Coding DNA Sequence regions essential for protein synthesis.

Task C (Splice Site Prediction): A 3-class classification head that predicts donor sites, acceptor sites, or non-splice regions.

🛠️ Detailed Development Process
1. Data Procurement & Engineering
Human & Primate Datasets: Genomic sequences were sourced from Ensembl Release 109 and 111, covering both the human reference genome (GRCh38) and chimpanzee (Pan_tro_3.0) for cross-species validation.

Ground Truth Generation: Annotation files (GTF) were parsed to generate precise labels for Task A, B, and C, ensuring the model was trained on high-fidelity, curated biological data.

BioBERT Integration: A gene entity recognition pipeline was implemented by fine-tuning BioBERT on the JNLPBA dataset to assist in entity normalization and data cleaning before training the core Transformer.

2. Optimization for Accessibility
Hardware Constraints: The development was centered on a "CPU-first" philosophy, ensuring the model performs optimally on devices with as little as 8GB of RAM.

Tokenization Refinement: By utilizing codon-level modeling instead of k-mer tokenization, the model maintains high accuracy while significantly reducing the computational footprint.

Portable Utility: The final model is designed for deployment on portable sequencing devices, enabling real-time genomic analysis in field environments without needing high-performance computing (HPC) access.

3. Experimental Validation
Performance Metrics: The model's success was validated using Matthews Correlation Coefficient (MCC), which provides a more realistic assessment of performance on unbalanced genomic data compared to standard accuracy.
