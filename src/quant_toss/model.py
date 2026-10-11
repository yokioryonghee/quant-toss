from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    features: int = 9
    lookback: int = 64
    patch: int = 8
    dim: int = 64
    horizons: tuple = (1, 5, 20)
    experts: int = 3
    architecture: str = 'transformer'


class QuantileForecaster(nn.Module):
    """Feature gating + historical patch transformer + mixture quantile experts.

    The mixture is learned, not an identified economic-regime classifier.
    All tokens are historical, so bidirectional attention within the input
    window does not expose future prices. Quantiles cannot cross.
    """
    def __init__(self, config):
        super().__init__()
        if config.lookback % config.patch or config.dim % 4 or config.experts < 1:
            raise ValueError('lookback divisible by patch, dim divisible by 4, experts>=1 required')
        self.config = config
        if config.architecture == 'transformer':
            self.feature_gate = nn.Sequential(nn.Linear(config.features, 32), nn.GELU(),
                                              nn.Linear(32, config.features))
            self.patch_embed = nn.Linear(config.patch * config.features, config.dim)
            self.position = nn.Parameter(torch.randn(1, config.lookback // config.patch, config.dim) * .01)
            layer = nn.TransformerEncoderLayer(config.dim, 4, config.dim * 2, .1,
                                                activation='gelu', batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(config.dim)
        elif config.architecture == 'linear':
            self.linear = nn.Linear(config.lookback * config.features, config.dim)
        else:
            raise ValueError('architecture must be transformer or linear')
        self.regime_gate = nn.Linear(config.dim, config.experts)
        self.expert_heads = nn.ModuleList([nn.Linear(config.dim, len(config.horizons) * 3)
                                          for _ in range(config.experts)])

    def forward(self, x):
        config = self.config
        feature_weights = None
        if config.architecture == 'transformer':
            feature_weights = self.feature_gate(x.mean(1)).softmax(-1)
            selected = x * feature_weights[:, None] * config.features
            tokens = self.patch_embed(selected.reshape(len(x), config.lookback // config.patch, -1))
            encoded = self.norm(self.encoder(tokens + self.position).mean(1))
        else:
            encoded = self.linear(x.flatten(1))
        regime = self.regime_gate(encoded).softmax(1)
        quantiles = []
        for head in self.expert_heads:
            raw = head(encoded).reshape(len(x), len(config.horizons), 3)
            center = raw[:, :, 0]
            quantiles.append(torch.stack([center - F.softplus(raw[:, :, 1]), center,
                                          center + F.softplus(raw[:, :, 2])], -1))
        prediction = (torch.stack(quantiles, 1) * regime[:, :, None, None]).sum(1)
        return {'quantiles': prediction, 'regime_weights': regime, 'feature_weights': feature_weights}


def pinball(quantiles, targets):
    levels = quantiles.new_tensor([.1, .5, .9])
    errors = targets[..., None] - quantiles
    return torch.maximum(levels * errors, (levels - 1) * errors).mean()
