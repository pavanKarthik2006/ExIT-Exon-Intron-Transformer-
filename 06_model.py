"""
06_model.py
===========
STEP 5 — Transformer encoder with adaptive classifier heads.

Architecture
------------

                    ┌────────────────────────────────────┐
                    │  Input: token IDs  (B, L)           │
                    │  Embedding  +  Positional Encoding  │
                    └──────────────┬─────────────────────┘
                                   │
                    ┌──────────────▼─────────────────────┐
                    │   SHARED DNA ENCODER  (frozen)      │
                    │   N × TransformerEncoderLayer        │
                    │   output: (B, L, d_model)            │
                    │                                     │
                    │   [CLS] pooling → (B, d_model)       │
                    └──────────────┬─────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
   ┌──────────▼──────┐  ┌──────────▼──────┐  ┌─────────▼──────────┐
   │  Task A Head     │  │  Task B Head     │  │  Task C Head        │
   │  (n_classes=2)   │  │  (n_classes=2)   │  │  (n_classes=3)      │
   │  exon / intron   │  │  CDS / non-CDS   │  │  donor/accpt/none   │
   └──────────────────┘  └──────────────────┘  └─────────────────────┘

Key design choices
------------------
1. SHARED ENCODER (frozen after pre-training):
   The same weights process sequences for all tasks, forcing the model to
   learn general DNA representations.

2. ADAPTIVE CLASSIFIER HEAD:
   Each head is constructed with num_classes as a parameter.
   The exact same head class is used for all tasks:
     - Task A  (2 classes)  exon / intron
     - Task B  (2 classes)  CDS / non-CDS
     - Task C  (3 classes)  donor / acceptor / no-splice
   The encoder weights do NOT change between tasks; only the head changes.

3. [CLS] POOLING:
   The embedding at position 0 is used as the sequence representation,
   following BERT convention.  Sequence-level classification tasks use this.
   (Per-nucleotide tasks would use the full (B,L,d) output instead.)

4. CPU-OPTIMISED SIZE:
   Default d_model=128, n_layers=4, n_heads=4 → ~1.5M parameters.
   Can be scaled up once a GPU is available.

Public API
----------
    from 06_model import DNAEncoder, AdaptiveClassifier, DNAClassifier

    # Build encoder once
    encoder = DNAEncoder(vocab_size=67, d_model=128, ...)

    # Task A (2 classes)
    head_a  = AdaptiveClassifier(d_model=128, num_classes=2)
    model_a = DNAClassifier(encoder, head_a)

    # Task C (3 classes) — same encoder, different head
    head_c  = AdaptiveClassifier(d_model=128, num_classes=3)
    model_c = DNAClassifier(encoder, head_c)
"""

import math
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════
# POSITIONAL ENCODING
# ═══════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ═══════════════════════════════════════════════════════════════════
# SINGLE ENCODER LAYER
# ═══════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """Standard pre-norm transformer encoder block."""

    def __init__(self, d_model: int, n_heads: int,
                 ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.attn    = nn.MultiheadAttention(d_model, n_heads,
                                             dropout=dropout, batch_first=True)
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.drop    = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        # Pre-norm attention
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                         key_padding_mask=pad_mask)
        x = x + self.drop(h)
        # Pre-norm FFN
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ═══════════════════════════════════════════════════════════════════
# SHARED DNA ENCODER
# ═══════════════════════════════════════════════════════════════════

class DNAEncoder(nn.Module):
    """
    Shared transformer encoder backbone.

    Input  : (B, L) token IDs
    Output : (B, d_model) — [CLS] token representation

    Parameters
    ----------
    vocab_size : int   67 (64 standard codons + PAD=0 + UNK=1 + MASK=2)
                         Codon vocabulary built in 05_preprocess.py / 09_preprocess_bio.py
    d_model    : int   embedding / hidden dimension
    n_heads    : int   number of attention heads
    n_layers   : int   number of encoder layers
    ffn_dim    : int   feed-forward intermediate size
    max_len    : int   maximum sequence length
    dropout    : float dropout probability
    pad_idx    : int   index of <PAD> token (used to build attention mask)
    """

    def __init__(
        self,
        vocab_size: int = 67,   # 64 codons + PAD + UNK + MASK
        d_model:    int = 128,
        n_heads:    int = 4,
        n_layers:   int = 4,
        ffn_dim:    int = 256,
        max_len:    int = 170,  # 512 nt // 3 = 170 codon tokens (GRCh38 default)
        dropout:    float = 0.1,
        pad_idx:    int = 0,
    ):
        super().__init__()
        self.pad_idx   = pad_idx
        self.d_model   = d_model

        # Learnable [CLS] token embedding prepended to every sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc   = PositionalEncoding(d_model, max_len + 1, dropout)
        self.layers    = nn.ModuleList([
            EncoderLayer(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.cls_token,        std=0.02)
        for layer in self.layers:
            for p in layer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        input_ids : (B, L)

        Returns
        -------
        cls_repr  : (B, d_model)   — [CLS] representation for classification
        all_repr  : (B, L+1, d_model) — full sequence output (for per-nucleotide tasks)
        """
        B, L = input_ids.shape

        # Pad mask: True where padded  → ignored in attention
        pad_mask = (input_ids == self.pad_idx)   # (B, L)
        # Prepend False for CLS token
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=input_ids.device)
        pad_mask = torch.cat([cls_mask, pad_mask], dim=1)   # (B, L+1)

        # Embed + prepend CLS
        x   = self.embedding(input_ids)                      # (B, L, d)
        cls = self.cls_token.expand(B, -1, -1)               # (B, 1, d)
        x   = torch.cat([cls, x], dim=1)                     # (B, L+1, d)
        x   = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x, pad_mask=pad_mask)

        x = self.norm(x)
        return x[:, 0, :], x   # (B, d_model), (B, L+1, d_model)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════
# ADAPTIVE CLASSIFIER HEAD
# ═══════════════════════════════════════════════════════════════════

class AdaptiveClassifier(nn.Module):
    """
    Sequence-level classification head.

    num_classes is set at construction time, making this head reusable
    for any task with any number of classes — the encoder stays identical.

    Input  : (B, d_model)   — from DNAEncoder [CLS] output
    Output : (B, num_classes) — raw logits

    Architecture: Linear → GELU → LayerNorm → Dropout → Linear
    """

    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, cls_repr: torch.Tensor) -> torch.Tensor:
        return self.net(cls_repr)   # (B, num_classes)


# ═══════════════════════════════════════════════════════════════════
# COMBINED MODEL  (encoder + one head)
# ═══════════════════════════════════════════════════════════════════

class DNAClassifier(nn.Module):
    """
    Wraps one DNAEncoder + one AdaptiveClassifier.

    The encoder is shared across tasks; the classifier head is swapped
    per task.  freeze_encoder() / unfreeze_encoder() control whether
    encoder weights are updated during training.
    """

    def __init__(self, encoder: DNAEncoder, classifier: AdaptiveClassifier):
        super().__init__()
        self.encoder    = encoder
        self.classifier = classifier

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns logits: (B, num_classes)"""
        cls_repr, _ = self.encoder(input_ids)
        return self.classifier(cls_repr)

    def freeze_encoder(self):
        """Freeze all encoder parameters (for external validation / head-only fine-tuning)."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        print("[model] Encoder FROZEN — only classifier head will be updated.")

    def unfreeze_encoder(self):
        """Unfreeze encoder for full fine-tuning."""
        for p in self.encoder.parameters():
            p.requires_grad = True
        print("[model] Encoder UNFROZEN — full model will be updated.")

    def count_parameters(self) -> dict:
        enc_total  = sum(p.numel() for p in self.encoder.parameters())
        enc_train  = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        head_total = sum(p.numel() for p in self.classifier.parameters())
        return {
            "encoder_total":     enc_total,
            "encoder_trainable": enc_train,
            "head_total":        head_total,
            "model_total":       enc_total + head_total,
        }

    @classmethod
    def build(
        cls,
        num_classes: int,
        vocab_size:  int   = 67,   # matches CODON_VOCAB in 05_preprocess.py
        d_model:     int   = 128,
        n_heads:     int   = 4,
        n_layers:    int   = 4,
        ffn_dim:     int   = 256,
        max_len:     int   = 170,  # 512 nt // 3 codon tokens
        dropout:     float = 0.1,
        encoder_ckpt: str  = None,
    ) -> "DNAClassifier":
        """
        Factory method.  Builds encoder + head.  If encoder_ckpt is provided,
        loads pre-trained encoder weights (used when attaching a new head for
        a different task or for external validation).
        """
        encoder    = DNAEncoder(vocab_size, d_model, n_heads, n_layers,
                                ffn_dim, max_len, dropout)
        classifier = AdaptiveClassifier(d_model, num_classes, dropout)
        model      = cls(encoder, classifier)

        if encoder_ckpt:
            ckpt = torch.load(encoder_ckpt, map_location="cpu")
            # Support both full checkpoint dict and bare state dict
            state = ckpt.get("encoder_state", ckpt)
            model.encoder.load_state_dict(state, strict=False)
            print(f"[model] Encoder weights loaded from {encoder_ckpt}")

        return model


# ═══════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    for n_cls, name in [(2, "Task A/B (2-class)"),
                        (3, "Task C (3-class)")]:
        # Task A/B: 170 codon tokens (512nt). Task C: 66 tokens (200nt).
        model = DNAClassifier.build(num_classes=n_cls)
        dummy = torch.randint(0, 67, (4, 170))  # 67-token codon vocab, 170 tokens
        out   = model(dummy)
        params = model.count_parameters()
        print(f"\n{name}")
        print(f"  Output shape     : {out.shape}")
        print(f"  Encoder params   : {params['encoder_total']:,}")
        print(f"  Head params      : {params['head_total']:,}")
        print(f"  Total params     : {params['model_total']:,}")

    # Test freeze / unfreeze
    model = DNAClassifier.build(num_classes=3)
    model.freeze_encoder()
    p = model.count_parameters()
    print(f"\nAfter freeze — trainable encoder params: {p['encoder_trainable']:,}")
    model.unfreeze_encoder()
    p = model.count_parameters()
    print(f"After unfreeze — trainable encoder params: {p['encoder_trainable']:,}")