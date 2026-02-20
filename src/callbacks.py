"""
Custom Callbacks for Santander Transaction Prediction.

Includes:
    - AUROC: Epoch-level ROC-AUC metric computation
    - AugShuffCallback: Intra-class feature shuffle augmentation
    - LongerRandomSampler: Oversampling via repeated random permutations
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Sampler
from sklearn.metrics import roc_auc_score


def auroc_score(input: torch.Tensor, target: torch.Tensor) -> float:
    """Compute ROC-AUC from model output and targets."""
    input = input.cpu().numpy()[:, 1]
    target = target.cpu().numpy()
    return roc_auc_score(target, input)


class LongerRandomSampler(Sampler):
    """Random sampler that repeats the dataset `mult` times per epoch.

    Each epoch sees every sample `mult` times in a different random order.
    This effectively increases training data per epoch without duplicating
    it in memory — useful for small datasets with weak signal.

    Args:
        data_source: Dataset to sample from
        mult: Number of times to repeat each sample per epoch
    """

    def __init__(self, data_source, mult: int = 3):
        self.data_source = data_source
        self.mult = mult

    def __iter__(self):
        n = len(self.data_source)
        return iter(torch.randperm(n).tolist() * self.mult)

    def __len__(self):
        return len(self.data_source) * self.mult


def shuffle_augment(x_cat, x_cont, target, n_features: int = 200):
    """Intra-class feature shuffle augmentation.

    For each feature independently:
    - Positive samples have values shuffled among other positives
    - Negative samples have values shuffled among other negatives

    This creates synthetic training examples that preserve the marginal
    distribution within each class while breaking inter-feature correlations.
    Effective when features are approximately independent given the label.

    Args:
        x_cat: Categorical input tensor (batch × features)
        x_cont: Continuous input tensor (batch × features)
        target: Binary target tensor
        n_features: Number of original features (200)

    Returns:
        Augmented (x_cat, x_cont, target) tuple
    """
    device = x_cat.device
    m_pos = target == 1
    m_neg = target == 0

    pos_cat, pos_cont = x_cat[m_pos], x_cont[m_pos]
    neg_cat, neg_cont = x_cat[m_neg], x_cont[m_neg]

    for f in range(n_features):
        # Shuffle positive samples
        shuffle_pos = torch.randperm(pos_cat.size(0)).to(device)
        pos_cat[:, f] = pos_cat[shuffle_pos, f]
        pos_cont[:, f] = pos_cont[shuffle_pos, f]
        pos_cont[:, f + n_features] = pos_cont[shuffle_pos, f + n_features]

        # Shuffle negative samples
        shuffle_neg = torch.randperm(neg_cat.size(0)).to(device)
        neg_cat[:, f] = neg_cat[shuffle_neg, f]
        neg_cont[:, f] = neg_cont[shuffle_neg, f]
        neg_cont[:, f + n_features] = neg_cont[shuffle_neg, f + n_features]

    new_cat = torch.cat([pos_cat, neg_cat])
    new_cont = torch.cat([pos_cont, neg_cont])
    new_target = torch.cat([target[m_pos], target[m_neg]])

    return new_cat, new_cont, new_target
