"""
Earnings Call Anomaly Detector
BERT-style transformer trained on financial language
with deception pattern classification head
"""
import torch
import torch.nn as nn
import math


class FinancialTokenizer:
    """
    Simple word-level tokenizer trained on financial vocabulary
    Handles financial terms, numbers, and deception markers
    """
    def __init__(self, vocab_size=8000, max_len=256):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.word2idx = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
        self.idx2word = {0: "[PAD]", 1: "[UNK]", 2: "[CLS]", 3: "[SEP]"}
        self.is_trained = False

    def train(self, texts: list[str]):
        """Build vocabulary from financial texts"""
        from collections import Counter
        import re

        word_counts = Counter()
        for text in texts:
            text = text.lower()
            text = re.sub(r'[^a-z0-9\s]', ' ', text)
            words = text.split()
            word_counts.update(words)

        # Keep top vocab_size - 4 words (reserve 4 for special tokens)
        top_words = [w for w, _ in word_counts.most_common(self.vocab_size - 4)]
        for word in top_words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        self.is_trained = True
        print(f"Vocabulary built: {len(self.word2idx)} tokens")

    def encode(self, text: str) -> list[int]:
        import re
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        words = text.split()[:self.max_len - 2]

        ids = [self.word2idx["[CLS]"]]
        for word in words:
            ids.append(self.word2idx.get(word, self.word2idx["[UNK]"]))
        ids.append(self.word2idx["[SEP]"])

        # Pad to max_len
        ids = ids + [self.word2idx["[PAD]"]] * (self.max_len - len(ids))
        return ids[:self.max_len]

    def save(self, path: str):
        import json
        with open(path, "w") as f:
            json.dump({"word2idx": self.word2idx}, f)
        print(f"Tokenizer saved to {path}")

    def load(self, path: str):
        import json
        with open(path) as f:
            data = json.load(f)
        self.word2idx = data["word2idx"]
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.is_trained = True
        print(f"Tokenizer loaded: {len(self.word2idx)} tokens")


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        Q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out), attn


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, attn_weights


class EarningsAnomalyDetector(nn.Module):
    """
    BERT-style encoder for detecting deceptive language in earnings calls
    Returns:
      - fraud probability
      - attention weights (for interpretability)
      - deception category scores
    """
    def __init__(
        self,
        vocab_size: int = 8000,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        max_len: int = 256,
        dropout: float = 0.1,
        num_deception_categories: int = 5,
    ):
        super().__init__()
        self.d_model = d_model

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Classification heads
        self.fraud_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self.deception_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_deception_categories),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor):
        B, T = input_ids.shape
        device = input_ids.device

        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        # Padding mask
        mask = (input_ids != 0).unsqueeze(1).unsqueeze(2)

        all_attn = []
        for block in self.blocks:
            x, attn = block(x, mask)
            all_attn.append(attn)

        x = self.norm(x)

        # Use [CLS] token for classification
        cls = x[:, 0, :]

        fraud_logit = self.fraud_head(cls).squeeze(-1)
        deception_scores = self.deception_head(cls)

        return {
            "fraud_logit": fraud_logit,
            "fraud_prob": torch.sigmoid(fraud_logit),
            "deception_scores": torch.sigmoid(deception_scores),
            "attention_weights": all_attn[-1],
            "cls_embedding": cls,
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = EarningsAnomalyDetector()
    print(f"Model parameters: {model.count_parameters():,}")

    x = torch.randint(0, 8000, (4, 256))
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Fraud prob: {out['fraud_prob']}")
    print(f"Deception scores shape: {out['deception_scores'].shape}")
    print("Forward pass successful!")