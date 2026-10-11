from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Policy:
    cost_bps: float = 10.  # hypothetical per-side commission + slippage; not a current broker fee
    sell_tax_bps: float = 0.
    max_exposure: float = .25
    max_drawdown: float = .1
    min_edge_bps: float = 5.

    def validate(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Policy must be finite')
        if min(self.cost_bps, self.sell_tax_bps, self.min_edge_bps) < 0 or not 0 <= self.max_exposure <= 1 or not 0 < self.max_drawdown < 1:
            raise ValueError('Invalid cost/exposure/drawdown policy')
        if self.cost_bps + self.sell_tax_bps >= 10000:
            raise ValueError('Per-side transaction costs must be below 100%')


def target_exposure(quantiles, policy):
    policy.validate()
    q10, median, q90 = quantiles
    round_trip = (2 * policy.cost_bps + policy.sell_tax_bps + policy.min_edge_bps) / 10000
    # Explicit conservative long/cash rule, no shorting or leverage.
    return policy.max_exposure if q10 > round_trip and median > round_trip else 0.


def simulate(bars, indices, quantiles, policy):
    """Close(i) signal -> next open buy -> same-session close liquidation.

    Predictions are simple return fractions, not percentage-point units.
    Daily mark-to-market; fractional notional research accounting, no actual
    order/lot model. The next open is not known when the signal is generated.
    """
    policy.validate()
    if len(indices) != len(quantiles) or len(indices) == 0 or not np.isfinite(quantiles).all():
        raise ValueError('Invalid backtest predictions')
    if (np.diff(indices) != 1).any():
        raise ValueError('Daily simulation requires consecutive trading-session signals')
    equity, peak, halted, ledger = 1., 1., False, []
    for i, prediction in zip(indices, quantiles):
        i = int(i)
        weight = 0. if halted else target_exposure(prediction, policy)
        realized = bars.prices[i + 1, 3] / bars.prices[i + 1, 0] - 1
        buy_rate = policy.cost_bps / 10000
        sell_rate = (policy.cost_bps + policy.sell_tax_bps) / 10000
        # Buy with a fraction of equity inclusive of buy costs; liquidate
        # gross exit notional with sell costs, preserving cash exactly.
        net = weight * ((1 + realized) * (1 - sell_rate) / (1 + buy_rate) - 1)
        equity *= 1 + net
        peak = max(peak, equity)
        drawdown = 1 - equity / peak
        if drawdown >= policy.max_drawdown:
            halted = True  # takes effect next session, cannot undo today's loss
        ledger.append({'signal_date': bars.dates[i], 'execution_date': bars.dates[i + 1],
                       'exposure': weight, 'gross_asset_return': float(realized),
                       'net_portfolio_return': float(net), 'equity': float(equity),
                       'drawdown': float(drawdown), 'halted': halted})
    returns = np.array([row['net_portfolio_return'] for row in ledger])
    std = returns.std(ddof=1) if len(returns) > 1 else 0.
    return {'total_return': float(equity - 1), 'max_drawdown': float(max(row['drawdown'] for row in ledger)),
            'annualized_sharpe_zero_rf': float(returns.mean() / std * np.sqrt(252)) if std > 1e-12 else None,
            'session_count': len(ledger), 'traded_sessions': sum(row['exposure'] > 0 for row in ledger),
            'halted': halted, 'policy': asdict(policy),
            'scope': 'single asset; close signal, next-open entry, next-close exit; fractional notional',
            'ledger': ledger}
