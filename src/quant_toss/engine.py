import csv
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .backtest import Policy, simulate
from .data import FEATURES, Windows, features, fit_scaler, read_bars, split_indices
from .model import ModelConfig, QuantileForecaster, pinball


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def verify_identity(path, symbol, currency):
    rows=[]
    if Path(path).is_file():
        with open(path, newline='') as handle:
            rows = list(csv.DictReader(handle))
    for key, expected in [('symbol',symbol),('currency',currency)]:
        if rows and key in rows[0] and {row[key] for row in rows} != {expected}:
            raise ValueError(f'CSV {key} differs from model/research identity')
    metadata_path = Path(path).with_suffix('.meta.json')
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get('symbol') != symbol or metadata.get('currency') != currency:
            raise ValueError('CSV metadata symbol/currency differs from model or research settings')


@torch.inference_mode()
def infer(model, loader):
    model.eval()
    predictions, targets = [], []
    for x, y in loader:
        predictions.append(model(x)['quantiles'].cpu())
        targets.append(y)
    return torch.cat(predictions), torch.cat(targets)


def interval_margin(prediction, targets, alpha=.2):
    """Residual widening on a chronological calibration block.

    This is empirically calibrated, NOT a distribution-free time-series
    coverage guarantee: financial observations are dependent/nonstationary.
    """
    if not 0 < alpha < 1:
        raise ValueError('alpha must be in (0,1)')
    scores = torch.maximum(prediction[:, :, 0] - targets, targets - prediction[:, :, 2])
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    if k > len(scores):
        raise ValueError('Insufficient independent calibration observations for requested level')
    return scores.sort(0).values[k - 1].clamp(min=0)


def apply_margin(prediction, margin):
    calibrated = prediction.clone()
    calibrated[:, :, 0] -= margin
    calibrated[:, :, 2] += margin
    return calibrated


def forecast_metrics(prediction, targets, horizons):
    result = {}
    for j, horizon in enumerate(horizons):
        q, y = prediction[:, j], targets[:, j]
        result[str(horizon)] = {'mae_percentage_points': float((q[:, 1] - y).abs().mean()),
                                'direction_accuracy': float(((q[:, 1] > 0) == (y > 0)).float().mean()),
                                'interval_coverage': float(((q[:, 0] <= y) & (y <= q[:, 2])).float().mean()),
                                'interval_width_percentage_points': float((q[:, 2] - q[:, 0]).mean()),
                                'pinball': float(pinball(q[:, None], y[:, None]))}
    return result


def run(csv_path, output, symbol, currency, *, epochs=15, batch=32, lookback=64,
        horizons=(1, 5, 20), architecture='transformer', seed=42, policy=None,
        boundaries=None, embargo=5, synthetic=False):
    if epochs < 1 or batch < 2 or 1 not in horizons or min(horizons) < 1 or len(set(horizons)) != len(horizons):
        raise ValueError('Need epochs>=1, batch>=2, unique positive horizons including 1')
    if currency not in ('KRW', 'USD'):
        raise ValueError('currency must be KRW or USD; do not mix balances')
    policy = policy or Policy()
    policy.validate()
    verify_identity(csv_path, symbol, currency)
    bars = read_bars(csv_path)
    splits = split_indices(len(bars), lookback, max(horizons), embargo, boundaries)
    x = features(bars)
    mean, scale = fit_scaler(x, splits['train'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=False)
    config = ModelConfig(lookback=lookback, horizons=tuple(horizons),
                         architecture=architecture, experts=3 if architecture == 'transformer' else 1)
    model = QuantileForecaster(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    def batches(split, shuffle=False):
        return DataLoader(Windows(bars, x, splits[split], mean, scale, lookback, horizons),
                          batch_size=batch, shuffle=shuffle, generator=torch.Generator().manual_seed(seed))
    training, validation = batches('train', True), batches('val')
    best, best_state, history = float('inf'), None, []
    for epoch in range(epochs):
        model.train()
        loss_sum, examples = 0., 0
        for window, targets in training:
            loss = pinball(model(window)['quantiles'], targets)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(window)
            examples += len(window)
        prediction, targets = infer(model, validation)
        val_loss = float(pinball(prediction, targets))
        history.append({'epoch': epoch + 1, 'train_pinball': loss_sum / examples, 'val_pinball': val_loss})
        if val_loss < best:
            best, best_state = val_loss, {key: value.detach().clone() for key, value in model.state_dict().items()}
        print(f'{architecture} epoch={epoch+1} train={loss_sum/examples:.4f} val={val_loss:.4f}', flush=True)
    model.load_state_dict(best_state)
    calibration_prediction, calibration_targets = infer(model, batches('calibration'))
    margin = interval_margin(calibration_prediction, calibration_targets)
    checkpoint = {'model_config': asdict(config), 'state_dict': best_state,
                  'mean': torch.tensor(mean), 'scale': torch.tensor(scale), 'interval_margin': margin,
                  'symbol': symbol, 'currency': currency, 'features': FEATURES,
                  'data_sha256': hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
                  'synthetic': synthetic, 'trained_through_signal_date': bars.dates[int(splits['train'].max())],
                  'calibrated_through_label_date': bars.dates[int(splits['calibration'].max()) + max(horizons)]}
    torch.save(checkpoint, folder / 'model.pt')
    raw, targets = infer(model, batches('test'))
    calibrated = apply_margin(raw, margin)
    # Historical empirical-return quantiles are fit on train targets only.
    train_dataset = batches('train').dataset
    train_targets = torch.stack([train_dataset[i][1] for i in range(len(train_dataset))])
    empirical = train_targets.quantile(torch.tensor([.1, .5, .9]), dim=0).T
    empirical_test = empirical[None].repeat(len(targets), 1, 1)
    empirical_calibration = empirical[None].repeat(len(calibration_targets), 1, 1)
    empirical_test = apply_margin(empirical_test, interval_margin(empirical_calibration, calibration_targets))
    policy_horizon = list(horizons).index(1)
    backtest = simulate(bars, splits['test'], calibrated[:, policy_horizon].numpy() / 100, policy)
    empirical_bt = simulate(bars, splits['test'], empirical_test[:, policy_horizon].numpy() / 100, policy)
    # Same open/close execution convention for an always-exposed comparator.
    always_exposed = np.full((len(targets), 3), 1e3)
    always_bt = simulate(bars, splits['test'], always_exposed, policy)
    report = {'symbol': symbol, 'currency': currency, 'synthetic': synthetic,
              'data_sha256': checkpoint['data_sha256'], 'architecture': architecture, 'seed': seed,
              'horizons_sessions': list(horizons), 'embargo_sessions': embargo,
              'split_ranges': {name: {'count': len(index), 'first_signal': bars.dates[int(index.min())],
                                     'last_signal': bars.dates[int(index.max())],
                                     'last_label': bars.dates[int(index.max()) + max(horizons)]}
                               for name, index in splits.items()},
              'forecast_raw': forecast_metrics(raw, targets, horizons),
              'forecast_calibrated': forecast_metrics(calibrated, targets, horizons),
              'empirical_train_quantiles': forecast_metrics(empirical_test, targets, horizons),
              'strategy': {key: value for key, value in backtest.items() if key != 'ledger'},
              'empirical_strategy': {key: value for key, value in empirical_bt.items() if key != 'ledger'},
              'always_exposed_intraday': {key: value for key, value in always_bt.items() if key != 'ledger'},
              'cash_total_return': 0., 'note': 'No live orders. No overnight holding. Not an outperformance claim.'}
    write_json(folder / 'report.json', report)
    write_json(folder / 'ledger.json', backtest['ledger'])
    write_json(folder / 'history.json', history)
    write_json(folder / 'config.json', {'model': asdict(config), 'epochs': epochs, 'batch': batch,
                                      'policy': asdict(policy), 'torch': str(torch.__version__),
                                      'device': 'cpu', 'synthetic': synthetic})
    print(f"Saved {folder}; test net return={report['strategy']['total_return']:.6f}; traded sessions={report['strategy']['traded_sessions']}", flush=True)
    return report


@torch.inference_mode()
def predict(csv_path, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    verify_identity(csv_path, checkpoint['symbol'], checkpoint['currency'])
    model = QuantileForecaster(ModelConfig(**checkpoint['model_config'])).eval()
    model.load_state_dict(checkpoint['state_dict'])
    bars = read_bars(csv_path)
    if len(bars) < model.config.lookback + 20:
        raise ValueError('Not enough completed daily bars')
    if bars.dates[-1] <= checkpoint['calibrated_through_label_date']:
        raise ValueError('Forward prediction requires a signal date later than calibration labels')
    window = torch.tensor(features(bars)[-model.config.lookback:])
    window = (window - checkpoint['mean']) / checkpoint['scale']
    result = model(window[None])
    quantiles = apply_margin(result['quantiles'], checkpoint['interval_margin'])[0] / 100
    return {'symbol': checkpoint['symbol'], 'currency': checkpoint['currency'], 'synthetic': checkpoint['synthetic'],
            'signal_date': bars.dates[-1], 'execution': 'next trading-session open; signal only',
            'return_quantiles_from_next_open': {str(h): {'q10': float(q[0]), 'median': float(q[1]), 'q90': float(q[2])}
                                               for h, q in zip(model.config.horizons, quantiles)},
            'mixture_weights': result['regime_weights'][0].tolist(),
            'feature_weights': dict(zip(FEATURES, result['feature_weights'][0].tolist())) if result['feature_weights'] is not None else None,
            'live_order_submitted': False}

