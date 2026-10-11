import argparse
import json
from pathlib import Path

from .backtest import Policy
from .data import read_bars
from .engine import predict, run, write_json
from .toss import TossMarketData


def main():
    parser = argparse.ArgumentParser(description='Quant Toss: forecast research and paper execution, no live order submission')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('research', 'walk-forward'):
        command = commands.add_parser(name)
        command.add_argument('--csv', required=True)
        command.add_argument('--output', required=True)
        command.add_argument('--symbol', required=True)
        command.add_argument('--currency', required=True, choices=['USD', 'KRW'])
        command.add_argument('--epochs', type=int, default=15)
        command.add_argument('--batch', type=int, default=32)
        command.add_argument('--lookback', type=int, default=64)
        command.add_argument('--architecture', choices=['transformer', 'linear'], default='transformer')
        command.add_argument('--seed', type=int, default=42)
        command.add_argument('--cost-bps', type=float, default=10.)
        command.add_argument('--sell-tax-bps', type=float, default=0.)
        command.add_argument('--max-exposure', type=float, default=.25)
        command.add_argument('--max-drawdown', type=float, default=.1)
        command.add_argument('--synthetic', action='store_true')
        if name == 'walk-forward':
            command.add_argument('--folds', type=int, default=3)
    prediction = commands.add_parser('predict')
    prediction.add_argument('--csv', required=True)
    prediction.add_argument('--checkpoint', required=True)
    fetch = commands.add_parser('fetch-toss', help='Official read-only daily candles; local bearer token required')
    fetch.add_argument('--symbol', required=True)
    fetch.add_argument('--output', required=True)
    fetch.add_argument('--pages', type=int, default=10)
    args = parser.parse_args()
    if args.command == 'fetch-toss':
        print(json.dumps(TossMarketData().save_daily(args.symbol, args.output, args.pages), indent=2))
    elif args.command == 'predict':
        print(json.dumps(predict(args.csv, args.checkpoint), indent=2))
    else:
        policy = Policy(cost_bps=args.cost_bps, sell_tax_bps=args.sell_tax_bps,
                        max_exposure=args.max_exposure, max_drawdown=args.max_drawdown)
        options = dict(epochs=args.epochs, batch=args.batch, lookback=args.lookback,
                       architecture=args.architecture, seed=args.seed, policy=policy, synthetic=args.synthetic)
        if args.command == 'research':
            run(args.csv, args.output, args.symbol, args.currency, **options)
        else:
            if not 2 <= args.folds <= 10:
                raise ValueError('folds must be between 2 and 10')
            bars = read_bars(args.csv)
            start = int(len(bars) * .65)
            block = (len(bars) - start) // args.folds
            folder = Path(args.output)
            folder.mkdir(parents=True, exist_ok=False)
            reports = []
            for fold in range(args.folds):
                test_start = start + fold * block
                end = len(bars) if fold == args.folds - 1 else test_start + block
                boundaries = [int(test_start * .75), int(test_start * .9), test_start, end]
                reports.append(run(args.csv, folder / f'fold_{fold}', args.symbol, args.currency,
                                   boundaries=boundaries, **options))
                write_json(folder / 'folds.json', reports)
            print('Expanding-window folds saved separately; no stitched PnL claim across purged gaps.')


if __name__ == '__main__':
    main()
