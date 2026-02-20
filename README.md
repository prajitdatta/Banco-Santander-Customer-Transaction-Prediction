<div align="center">

# 🏦 Santander Customer Transaction Prediction

### A Custom Neural Network with Attention-Weighted Feature Embeddings and Shuffle Augmentation

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/santander-customer-transaction-prediction)
[![PyTorch](https://img.shields.io/badge/PyTorch-Neural%20Net-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![fast.ai](https://img.shields.io/badge/fast.ai-v1-00B4D8?style=for-the-badge)](https://docs.fast.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

---

## The Problem

Santander Bank asked: **given 200 anonymized numeric features about a customer, predict whether they will make a specific transaction.** No feature names, no domain context, no metadata — just 200 columns of numbers and a binary target.

This sounds simple until you look at the data. The target is heavily imbalanced (~10% positive), the features are completely anonymous, and the signal is diffuse — no single feature is strongly predictive. The competition's key twist: the test set contained **synthetic (fake) rows** mixed with real data, and the winning solutions figured out how to detect them.

**Metric:** ROC-AUC

---

## The Data

| Property | Value |
|----------|-------|
| Training samples | 200,000 |
| Test samples | 200,000 |
| Features | 200 anonymized numeric variables (`var_0` to `var_199`) |
| Target | Binary — `1` = made transaction, `0` = didn't |
| Class balance | ~10% positive, ~90% negative |
| Engineered features | `has_one` (binary: is the value unique?), `not_unique` (count of duplicates) |

The critical discovery in this competition was that **whether a feature value is unique in the dataset** carries strong predictive signal. This led to two engineered features per variable: a binary `has_one` flag and a `not_unique` count — effectively tripling the feature space from 200 to 600.

---

## My Approach — Custom Neural Network with Learned Feature Weights

This solution goes well beyond a standard tabular neural network. I built a **custom architecture with three parallel embedding pathways, attention-weighted feature aggregation, and a novel shuffle augmentation strategy.**

### Architecture Overview

```
Input: 200 raw features + 200 has_one flags + 200 not_unique counts
        │                     │                      │
        ▼                     ▼                      ▼
┌──────────────────┐ ┌────────────────┐  ┌─────────────────────┐
│ Continuous        │ │ Categorical     │  │ Continuous           │
│ Embedding         │ │ Embedding       │  │ Embedding            │
│ (raw features)    │ │ (has_one flags) │  │ (not_unique counts)  │
│                   │ │                 │  │                      │
│ Linear(3→50)      │ │ Embedding(6,12) │  │ Linear(3→50)         │
│ ReLU              │ │                 │  │ ReLU                 │
│ Linear(50→10)     │ │                 │  │ Linear(50→10)        │
└────────┬─────────┘ └───────┬─────────┘  └──────────┬──────────┘
         │                   │                       │
         └───────────┬───────┘───────────────────────┘
                     ▼
        ┌────────────────────────┐
        │  Attention Weight NN    │
        │  (learns per-feature    │
        │   importance weights)   │
        │                         │
        │  Linear(all→5) → ReLU   │
        │  Linear(5→1) → Softmax  │
        └────────────┬────────────┘
                     │ w₁, w₂, ..., w₂₀₀
                     ▼
        ┌────────────────────────┐
        │  Weighted Average       │
        │  across 200 features    │
        └────────────┬────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │  Final MLP              │
        │  BatchNorm → Linear(32) │
        │  → ReLU → Linear(2)    │
        └────────────┬────────────┘
                     │
                     ▼
              Binary prediction
```

### The Three Embedding Pathways

Instead of feeding all 600 features into a flat MLP, I created **three specialized sub-networks** that each process one type of information differently:

**1. Raw Feature Embeddings** — each of the 200 numeric values is individually passed through a small neural network (`Linear(3→50) → ReLU → Linear(50→10)`). The input is 3-dimensional: the raw value plus a 2-dimensional learned intercept embedding. This is what the original authors called "continuous embeddings" — treating each continuous variable as its own mini-prediction problem.

```python
x_cont_raw = x_cont[:,:200].contiguous().view(-1, 1)
x_cont_raw = torch.cat([x_cont_raw, x_feat_emb.view(-1, self.n_emb_feat)], 1)
x_cont_raw = F.relu(self.cont_emb_l(x_cont_raw))
x_cont_raw = self.cont_emb_l2(x_cont_raw)
```

**2. Categorical Embeddings** — the `has_one` binary flags are processed through a standard embedding layer (`Embedding(6, 12)`). This captures the uniqueness signal in a learnable representation.

**3. Not-Unique Embeddings** — the duplicate count features get their own embedding pathway (`Linear(3→50) → ReLU → Linear(50→10)`), separate from the raw values. This separation lets the model learn different representations for "what is the value" vs. "how common is this value."

### Attention-Weighted Feature Aggregation

The key architectural innovation: instead of concatenating all 200 features into a single flat vector, I let **a small neural network learn how much weight to give each feature**:

```python
# Predict attention weight per feature
w = F.relu(self.weight(x_w))
w = self.weight2(w).view(b_size, -1)
w = torch.nn.functional.softmax(w, dim=-1).unsqueeze(-1)

# Weighted average across all 200 features
x = (w * x).sum(dim=1)
x_cont_raw = (w * x_cont_raw).sum(dim=1)
x_cont_notu = (w * x_cont_notu).sum(dim=1)
```

The weight network takes in all three embeddings for each feature, produces a scalar weight, then softmax normalizes across all 200 features. The final representation is a **weighted sum** — features that matter more for a given sample get higher weights. This is essentially a **self-attention mechanism over features** rather than tokens.

### Shuffle Augmentation

The most creative part: a custom data augmentation strategy that **shuffles feature values within each class** during training:

```python
class AugShuffCallback(LearnerCallback):
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        if not train: return
        for f in range(200):
            shuffle_pos = torch.randperm(pos_cat.size(0))
            pos_cat[:,f] = pos_cat[shuffle_pos, f]
            pos_cont[:,f] = pos_cont[shuffle_pos, f]
```

For each feature independently, positive samples have their values randomly swapped with other positive samples, and negative samples swap with other negatives. This creates **new synthetic training examples** that preserve the marginal distribution of each feature within each class, while breaking inter-feature correlations. It's a form of augmentation specifically designed for the independence assumption in tabular data.

### Training Strategy

**10-Fold Stratified Cross-Validation** — each fold trains a separate model, and the final submission is a **rank-averaged ensemble** of all 10 folds:

```python
sub_preds[target] = sub_preds.rank().mean(axis=1)
```

Rank averaging is more robust than probability averaging when models have different calibration — it only cares about the relative ordering.

**1-Cycle Learning Rate Policy** — fast.ai's `fit_one_cycle` with `max_lr=1e-2` over 15 epochs, which ramps up the learning rate, then anneals it down.

**2× Oversampling** — `LongerRandomSampler` repeats each sample twice per epoch, effectively doubling the training data seen per epoch without duplicating it in memory:

```python
class LongerRandomSampler(Sampler):
    def __iter__(self):
        return iter(torch.randperm(n).tolist() * self.mult)
```

**Model Checkpointing** — saved the best model per fold based on validation AUROC.

---

## Results

| Component | Detail |
|-----------|--------|
| Architecture | Custom NN with 3 embedding pathways + attention |
| Augmentation | Intra-class feature shuffle |
| Validation | 10-fold Stratified CV |
| Ensemble | Rank-averaged predictions across folds |
| Metric | ROC-AUC |

---

## What I Learned

**Feature-level attention beats flat concatenation on high-dimensional tabular data.** With 200 anonymous features, most are noise. Letting the model learn per-feature importance weights (via softmax attention) was more effective than relying on dropout alone to ignore irrelevant features. The attention mechanism acts as a learned, input-dependent feature selector.

**"Continuous embeddings" make neural networks competitive with gradient boosting on tabular data.** The standard approach — normalize and concatenate all features into one vector — doesn't work well because it treats every feature identically. Processing each feature through its own mini-network lets the model learn feature-specific nonlinear transformations, similar to what tree splits achieve naturally.

**Shuffle augmentation is the tabular equivalent of image augmentation.** On images, you can flip, rotate, crop. On tabular data with independent features, you can shuffle values within a class — this preserves the marginal distributions while creating novel combinations. The key insight is that if features are approximately independent given the label, then shuffling within a class produces valid synthetic examples.

**Rank averaging is more robust than probability averaging for ensembles.** When combining 10 models with potentially different probability calibrations, rank averaging only cares about the ordering. A model that predicts probabilities in [0.1, 0.3] and one that predicts [0.4, 0.9] can still be meaningfully combined through ranks.

**The "uniqueness" of feature values is a powerful meta-feature.** The `has_one` and `not_unique` features — which encode whether a value appears once or multiple times in the dataset — were among the most important signals. In this competition, non-unique values often indicated synthetic (fake) test rows, and the model learned to exploit this.

---

## Repository Structure

```
santander-transaction-prediction/
│
├── src/
│   ├── model.py                       # Custom NN architecture
│   ├── callbacks.py                   # AUROC, shuffle augmentation, sampler
│   └── train.py                       # Full training pipeline
│
├── README.md
└── LICENSE
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Framework | PyTorch + fast.ai v1 |
| Architecture | Custom tabular NN with attention-weighted feature embeddings |
| Augmentation | Intra-class feature shuffling (custom callback) |
| Validation | 10-fold Stratified CV with rank-averaged ensemble |
| Training | 1-cycle LR policy, 2× oversampling, model checkpointing |
| Preprocessing | StandardScaler, engineered `has_one` + `not_unique` features |

---

## References

- [Santander Customer Transaction Prediction — Kaggle](https://www.kaggle.com/competitions/santander-customer-transaction-prediction)
- [1st Place Solution Discussion](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/88939)
- [fast.ai Documentation](https://docs.fast.ai)
- [1-Cycle Learning Rate Policy (Smith, 2018)](https://arxiv.org/abs/1708.07120)

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

**[Prajit Datta](https://www.prajitdatta.com)** · [Blog](https://www.prajitdatta.com/thoughtsandresearch) · [Kaggle](https://www.kaggle.com/prajitdatta) · [Medium](https://medium.com/@prajitdatta)

</div>
