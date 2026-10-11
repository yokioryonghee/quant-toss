"""Offline stochastic price process, never presented as historical returns."""
import argparse
import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np

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
    parser.add_argument('--open', action='store_true', help='Open the HTML results after the demo finishes')
    args = parser.parse_args()
    folder = Path(args.output)
    if folder.exists():
        parser.error('Output folder already exists. Choose a new name, or view its report with python -m quant_toss.report --input ' + str(folder))
    print('Starting offline demo: 1,100 synthetic sessions, CPU training. No live orders.', flush=True)
    from .engine import predict, run, write_json
    from .report import render
    folder.mkdir(parents=True, exist_ok=False)
    csv_path = folder / 'synthetic.csv'
    create_csv(csv_path)
    for architecture in ('linear', 'transformer'):
        run(csv_path, folder / architecture, 'SYNTHETIC', 'USD', epochs=2,
            lookback=32, architecture=architecture, synthetic=True)
    forecast = predict(csv_path, folder / 'transformer' / 'model.pt')
    write_json(folder / 'forecast.json', forecast)
    report_path = render(folder)
    print('PASS: synthetic chronological training, linear comparator, calibration, cost-aware paper backtest, checkpoint reload, forecast. No account connection or live order.')
    print(f'Results ready: {report_path}\nOpen report.html in your browser to view charts and forecasts.', flush=True)
    if args.open:
        import webbrowser
        if not webbrowser.open(report_path.as_uri()):
            print('Open report.html directly in your browser.')


if __name__ == '__main__':
    main()
