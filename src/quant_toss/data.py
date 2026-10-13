import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

FEATURES = ['log_return', 'overnight_gap', 'intraday_return', 'range', 'volume_change',
            'momentum_5', 'momentum_20', 'volatility_20', 'close_position']


@dataclass
class Bars:
    dates: list
    prices: np.ndarray  # open, high, low, close, volume; one symbol/currency only

    def __len__(self):
        return len(self.dates)


def read_bars(path):
    with open(path, newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {'date', 'open', 'high', 'low', 'close', 'volume'} <= set(rows[0]):
        raise ValueError('CSV columns: date,open,high,low,close,volume (one symbol/currency)')
    for key in ('symbol', 'currency'):
        if key in rows[0] and len({row[key] for row in rows}) != 1:
            raise ValueError('Mixed symbols/currencies in daily CSV')
    dates = [date.fromisoformat(row['date']).isoformat() for row in rows]
    if dates != sorted(set(dates)):
        raise ValueError('Daily bars must have unique ascending local trading dates')
    values = np.array([[float(row[k]) for k in ('open', 'high', 'low', 'close', 'volume')]
                       for row in rows], dtype=np.float64)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError('Nonfinite/invalid OHLCV')
    op, hi, lo, cl, _ = values.T
    if (hi < np.maximum(op, cl)).any() or (lo > np.minimum(op, cl)).any() or (lo > hi).any():
        raise ValueError('Inconsistent OHLC range')
    return Bars(dates, values)


def features(bars):
    """Each row depends only on that row and earlier completed sessions."""
    op, hi, lo, cl, volume = bars.prices.T
    previous = np.r_[cl[0], cl[:-1]]
    returns = np.log(cl / previous)
    columns = [returns, np.log(op / previous), np.log(cl / op), (hi - lo) / previous,
               np.r_[0., np.diff(np.log1p(volume))]]
    for lag in (5, 20):
        shifted = np.r_[np.repeat(cl[0], min(lag, len(cl))), cl[:-lag]]
        columns.append(np.log(cl / shifted))
    volatility = np.array([returns[max(0, i - 19):i + 1].std() for i in range(len(cl))])
    columns += [volatility, (cl - lo) / np.maximum(hi - lo, 1e-8)]
    return np.stack(columns, axis=1).astype(np.float32)


def split_indices(n, lookback, max_horizon, embargo=5, boundaries=None):
    if lookback < 8 or max_horizon < 1 or embargo < 0:
        raise ValueError('Invalid lookback/horizon/embargo')
    if any(not isinstance(v, (int, np.integer)) for v in (n,lookback,max_horizon,embargo)):
        raise ValueError('Integer session parameters required')
    bounds = boundaries or [int(n * .6), int(n * .75), int(n * .85), n]
    if len(bounds) != 4 or any(not isinstance(v,(int,np.integer)) for v in bounds) or not 0 < bounds[0] < bounds[1] < bounds[2] < bounds[3] <= n:
        raise ValueError('Invalid chronological boundaries')
    splits, start = {}, 20 + lookback - 1
    for name, stop in zip(('train', 'val', 'calibration', 'test'), bounds):
        # Signal at i close, last label at i+max_horizon; strictly before stop.
        indices = np.arange(start, stop - max_horizon, dtype=np.int64)
        if len(indices) < 8:
            raise ValueError(f'{name}: fewer than 8 usable signals after purge; provide more history')
        splits[name] = indices
        start = stop + embargo
    return splits


def fit_scaler(x, train_indices):
    history = x[20:int(train_indices.max()) + 1]
    return history.mean(0), np.maximum(history.std(0), 1e-6)


class Windows(Dataset):
    def __init__(self, bars, x, indices, mean, scale, lookback, horizons):
        self.bars, self.indices = bars, indices
        self.x = (x - mean) / scale
        self.lookback, self.horizons = lookback, horizons

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, row):
        i = int(self.indices[row])
        window = self.x[i - self.lookback + 1:i + 1]
        # Fixed percentage-point units, not a data-fitted target scaler.
        target = [(self.bars.prices[i + h, 3] / self.bars.prices[i + 1, 0] - 1) * 100
                  for h in self.horizons]
        return torch.from_numpy(window.copy()), torch.tensor(target, dtype=torch.float32)

