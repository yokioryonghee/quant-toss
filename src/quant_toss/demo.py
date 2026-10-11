"""Offline stochastic price process, never presented as historical returns."""
import argparse
import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from .engine import predict, run, write_json


def create_csv(path, seed=123, sessions=1100):
    rng = np.random.default_rng(seed)
    price, current, rows, latent = 100., date(2021, 1, 1), [], 0.
    while len(rows) < sessions:
        current += timedelta(days=1)
        if current.weekday() >= 5:
            continue
        latent = .85 * latent + rng.normal(0, .0005)
        op = price * np.exp(rng.normal(0, .003))
        cl = op * np.exp(latent + rng.normal(0, .008))
        high = max(op, cl) * np.exp(abs(rng.normal(0, .004)))
        low = min(op, cl) * np.exp(-abs(rng.normal(0, .004)))
        rows.append([current.isoformat(), op, high, low, cl, int(rng.lognormal(12, .25))])
        price = cl
    with Path(path).open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['date', 'open', 'high', 'low', 'close', 'volume'])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=False)
    csv_path = folder / 'synthetic.csv'
    create_csv(csv_path)
    for architecture in ('linear', 'transformer'):
        run(csv_path, folder / architecture, 'SYNTHETIC', 'USD', epochs=2,
            lookback=32, architecture=architecture, synthetic=True)
    forecast = predict(csv_path, folder / 'transformer' / 'model.pt')
    write_json(folder / 'forecast.json', forecast)
    print('PASS: synthetic chronological training, linear comparator, calibration, cost-aware paper backtest, checkpoint reload, forecast. No account connection or live order.')


if __name__ == '__main__':
    main()
