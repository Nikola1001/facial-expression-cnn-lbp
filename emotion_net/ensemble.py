"""Validation-selected ensemble of CNN+LBP models with horizontal-flip TTA."""

import torch
from torch import nn


class FlipEnsemble(nn.Module):
    def __init__(self, models, weights):
        super().__init__()
        if len(models) < 1 or len(models) != len(weights):
            raise ValueError("Expected models and matching weights")
        if any(weight < 0 for weight in weights) or abs(sum(weights) - 1) > 1e-6:
            raise ValueError("Ensemble weights must be nonnegative and sum to one")
        self.models = nn.ModuleList(models)
        self.weights = tuple(float(weight) for weight in weights)

    def forward(self, image, lbp):
        flipped_lbp = lbp.reshape(-1, 4, 4, 10).flip(2).flatten(1)
        total = None
        for weight, model in zip(self.weights, self.models):
            probabilities = (model(image, lbp).softmax(1).float()
                             + model(image.flip(3), flipped_lbp).softmax(1).float()) * 0.5
            weighted = probabilities * weight
            total = weighted if total is None else total + weighted
        return total.clamp_min(1e-8).log()


class LBPFusion(nn.Module):
    """Blend a CNN ensemble with an independently trained LBP expert."""

    def __init__(self, cnn, lbp_expert, lbp_weight: float):
        super().__init__()
        if not 0 <= lbp_weight <= 1:
            raise ValueError("LBP weight must be between zero and one")
        self.cnn = cnn
        self.lbp_expert = lbp_expert
        self.lbp_weight = float(lbp_weight)

    def forward(self, image, lbp):
        cnn_probability = self.cnn(image, lbp).softmax(1).float()
        lbp_probability = self.lbp_expert(image, lbp).softmax(1).float()
        return ((1 - self.lbp_weight) * cnn_probability
                + self.lbp_weight * lbp_probability).clamp_min(1e-8).log()
