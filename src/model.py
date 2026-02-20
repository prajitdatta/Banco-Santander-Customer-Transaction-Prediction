"""
Custom Tabular Neural Network with Attention-Weighted Feature Embeddings.

Architecture:
    - Three parallel embedding pathways (raw, categorical, not-unique)
    - Learned attention weights per feature via softmax
    - Weighted aggregation → MLP → binary prediction

Inspired by the 1st place solution approach to the Santander competition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class TabularAttentionModel(nn.Module):
    """Neural network for tabular data with per-feature attention weighting.

    Instead of concatenating all features into a flat vector, this model:
    1. Processes each feature through three parallel embedding pathways
    2. Learns an importance weight for each feature via a small attention network
    3. Computes a weighted average across all features
    4. Feeds the result through a final MLP for classification

    Args:
        emb_szs: Embedding sizes for categorical features [(vocab, dim)]
        n_cont: Number of continuous features (raw + not_unique = 400)
        out_sz: Output size (2 for binary classification)
        layers: Hidden layer sizes for final MLP
        ps: Dropout rates per layer
        emb_drop: Dropout on embeddings
        cont_emb: (hidden, output) dims for raw feature embedding network
        cont_emb_notu: (hidden, output) dims for not-unique embedding network
    """

    def __init__(
        self,
        emb_szs: List[Tuple[int, int]],
        n_cont: int,
        out_sz: int,
        layers: List[int],
        ps: float = 0.0,
        emb_drop: float = 0.0,
        cont_emb: Tuple[int, int] = (50, 10),
        cont_emb_notu: Tuple[int, int] = (50, 10),
        use_bn: bool = True,
    ):
        super().__init__()

        self.cont_emb_dim = cont_emb[1]
        self.cont_emb_notu_dim = cont_emb_notu[1]

        # ── Raw Feature Embedding Network ────────────────────────────────
        # Each of 200 raw values → (value + 2D intercept) → hidden → output
        self.cont_emb_l1 = nn.Linear(1 + 2, cont_emb[0])
        self.cont_emb_l2 = nn.Linear(cont_emb[0], cont_emb[1])

        # ── Not-Unique Feature Embedding Network ─────────────────────────
        # Each of 200 not_unique counts → same architecture, separate weights
        self.notu_emb_l1 = nn.Linear(1 + 2, cont_emb_notu[0])
        self.notu_emb_l2 = nn.Linear(cont_emb_notu[0], cont_emb_notu[1])

        # ── Categorical Embedding (has_one flags) ────────────────────────
        self.embeds = nn.Embedding(emb_szs[0][0], emb_szs[0][1])
        n_emb = emb_szs[0][1]

        # ── Intercept Embeddings (shared bias per feature position) ──────
        self.embeds_feat = nn.Embedding(201, 2)
        self.embeds_feat_w = nn.Embedding(201, 2)

        self.emb_drop = nn.Dropout(emb_drop)

        # ── Attention Weight Network ─────────────────────────────────────
        # Predicts a scalar importance weight per feature
        inp_w = n_emb + 2 + cont_emb[1] + cont_emb_notu[1]
        self.weight_l1 = nn.Linear(inp_w, 5)
        self.weight_l2 = nn.Linear(5, 1)

        # ── Final MLP ────────────────────────────────────────────────────
        inp_final = n_emb + cont_emb[1] + cont_emb_notu[1]
        mlp_layers = []
        sizes = [inp_final] + layers + [out_sz]
        for i in range(len(sizes) - 1):
            if i > 0 and use_bn:
                mlp_layers.append(nn.BatchNorm1d(sizes[i]))
            if i > 0:
                mlp_layers.append(nn.Dropout(ps))
            mlp_layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                mlp_layers.append(nn.ReLU(inplace=True))

        self.mlp = nn.Sequential(*mlp_layers)

        # ── Batch Normalization ──────────────────────────────────────────
        mom = 0.1
        self.bn_cat = nn.BatchNorm1d(200, momentum=mom)
        self.bn_raw = nn.BatchNorm1d(200, momentum=mom)
        self.bn_notu = nn.BatchNorm1d(200, momentum=mom)
        self.bn_feat = nn.BatchNorm1d(200, momentum=mom)
        self.bn_feat_w = nn.BatchNorm1d(200, momentum=mom)
        self.bn_w = nn.BatchNorm1d(inp_w, momentum=mom)
        self.bn_final = nn.BatchNorm1d(inp_final, momentum=mom)

        self.n_emb = n_emb

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        b_size = x_cont.size(0)

        # ── Categorical embeddings (has_one flags) ───────────────────────
        x = torch.stack([self.embeds(x_cat[:, i]) for i in range(200)], dim=1)

        # ── Intercept embeddings (shared learned bias) ───────────────────
        x_feat = self.embeds_feat(x_cat[:, 200])
        x_feat = torch.stack([x_feat] * 200, 1)
        x_feat = self.bn_feat(x_feat)

        x_feat_w = self.embeds_feat_w(x_cat[:, 200])
        x_feat_w = torch.stack([x_feat_w] * 200, 1)

        # ── Raw feature embeddings ───────────────────────────────────────
        x_raw = x_cont[:, :200].contiguous().view(-1, 1)
        x_raw = torch.cat([x_raw, x_feat.view(-1, 2)], 1)
        x_raw = F.relu(self.cont_emb_l1(x_raw))
        x_raw = self.cont_emb_l2(x_raw)
        x_raw = x_raw.view(b_size, 200, self.cont_emb_dim)

        # ── Not-unique feature embeddings ────────────────────────────────
        x_notu = x_cont[:, 200:].contiguous().view(-1, 1)
        x_notu = torch.cat([x_notu, x_feat.view(-1, 2)], 1)
        x_notu = F.relu(self.notu_emb_l1(x_notu))
        x_notu = self.notu_emb_l2(x_notu)
        x_notu = x_notu.view(b_size, 200, self.cont_emb_notu_dim)

        # ── Normalize and dropout ────────────────────────────────────────
        x = self.bn_cat(x)
        x_raw = self.bn_raw(x_raw)
        x_notu = self.bn_notu(x_notu)
        x_feat_w = self.bn_feat_w(x_feat_w)

        x = self.emb_drop(x)
        x_raw = self.emb_drop(x_raw)
        x_notu = self.emb_drop(x_notu)

        # ── Compute attention weights per feature ────────────────────────
        x_w = torch.cat([
            x.view(-1, self.n_emb),
            x_feat_w.view(-1, 2),
            x_raw.view(-1, self.cont_emb_dim),
            x_notu.view(-1, self.cont_emb_notu_dim),
        ], 1)
        x_w = self.bn_w(x_w)

        w = F.relu(self.weight_l1(x_w))
        w = self.weight_l2(w).view(b_size, -1)
        w = F.softmax(w, dim=-1).unsqueeze(-1)

        # ── Weighted aggregation across features ─────────────────────────
        x = (w * x).sum(dim=1)
        x_raw = (w * x_raw).sum(dim=1)
        x_notu = (w * x_notu).sum(dim=1)

        # ── Final prediction ─────────────────────────────────────────────
        out = torch.cat([x, x_raw, x_notu], 1)
        out = self.bn_final(out)
        out = self.mlp(out)

        return out
