"""GPU residual metric adaptation of frozen, aligned visual descriptors."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MetricAdapter(nn.Module):
    def __init__(self, dimensions, width=512):
        super().__init__()
        generator = torch.Generator().manual_seed(20261106)
        self.register_buffer('projection', torch.randn(dimensions, width, generator=generator) / math.sqrt(width))
        self.residual = nn.Sequential(nn.Linear(dimensions, 256, bias=False), nn.GELU(),
                                      nn.Dropout(.1), nn.Linear(256, width, bias=False))
        nn.init.zeros_(self.residual[-1].weight)

    def forward(self, x):
        return F.normalize(x @ self.projection + self.residual(x), dim=-1)


def supervised_contrastive(z, labels, temperature=.1):
    eye = torch.eye(len(labels), device=z.device, dtype=torch.bool)
    positives = labels[:, None].eq(labels[None, :]) & ~eye
    logp = F.log_softmax((z @ z.T / temperature).masked_fill(eye, -1e4), dim=1)
    valid = positives.any(1)
    if not valid.any():
        return z.sum() * 0
    return -((logp * positives).sum(1) / positives.sum(1).clamp_min(1))[valid].mean()


def normalized_mean(features):
    result = features.mean(1)
    return result / np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1e-8)


def identity_retrieval(z, labels):
    # Each query excludes itself; singleton identities do not contribute AP.
    from sklearn.metrics import average_precision_score
    scores = z @ z.T
    values = []
    for i in range(len(labels)):
        valid = np.arange(len(labels)) != i
        truth = labels[valid] == labels[i]
        if truth.any():
            values.append(average_precision_score(truth, scores[i, valid]))
    return float(np.mean(values)) if values else 0.
