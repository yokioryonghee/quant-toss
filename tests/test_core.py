import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from quant_toss.backtest import Policy, simulate
from quant_toss.data import Bars, features, fit_scaler, split_indices
from quant_toss.engine import interval_margin, verify_identity
from quant_toss.model import ModelConfig, QuantileForecaster, pinball
from quant_toss.toss import TossMarketData


def bars(n=1100):
    close = 100 * np.exp(np.arange(n) * .001)
    prices = np.stack([close * .999, close * 1.01, close * .99, close, np.full(n, 1000.)], 1)
    return Bars([f'session_{i}' for i in range(n)], prices)


class Tests(unittest.TestCase):
    def test_features_are_causal(self):
        original = bars()
        old = features(original)
        original.prices[500:] *= 100
        new = features(original)
        np.testing.assert_array_equal(old[:500], new[:500])

    def test_price_unit_invariance(self):
        original = bars()
        before = features(original)
        original.prices[:, :4] *= 1000
        np.testing.assert_allclose(before, features(original), atol=1e-7)

    def test_currency_identity_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bars.csv'
            path.with_suffix('.meta.json').write_text(json.dumps({'symbol': '005930', 'currency': 'KRW'}))
            verify_identity(path, '005930', 'KRW')
            with self.assertRaises(ValueError):
                verify_identity(path, 'AAPL', 'USD')

    def test_purge_and_train_only_scaler(self):
        indices = split_indices(1100, 64, 20)
        names = ['train', 'val', 'calibration', 'test']
        for left, right in zip(names, names[1:]):
            self.assertLess(indices[left].max() + 20, indices[right].min())
        x = features(bars())
        mean, scale = fit_scaler(x, indices['train'])
        x[indices['train'].max() + 1:] = 1e6
        new_mean, new_scale = fit_scaler(x, indices['train'])
        np.testing.assert_array_equal(mean, new_mean)
        np.testing.assert_array_equal(scale, new_scale)

    def test_ordered_quantiles_and_gradients(self):
        torch.set_num_threads(2)
        for architecture in ('linear', 'transformer'):
            model = QuantileForecaster(ModelConfig(lookback=32, dim=32, architecture=architecture))
            output = model(torch.randn(4, 32, 9))
            q = output['quantiles']
            self.assertEqual(tuple(q.shape), (4, 3, 3))
            self.assertTrue((q[:, :, 0] <= q[:, :, 1]).all())
            self.assertTrue((q[:, :, 1] <= q[:, :, 2]).all())
            loss = pinball(q, torch.randn(4, 3))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertIsNotNone(model.expert_heads[0].weight.grad)

    def test_zero_exposure_and_cost_accounting(self):
        sample = bars(4)
        sample.prices[1, 0], sample.prices[1, 3] = 100., 110.
        policy = Policy(cost_bps=10, max_exposure=.25)
        result = simulate(sample, np.array([0]), np.array([[.01, .02, .03]]), policy)
        expected = .25 * (1.1 * .999 / 1.001 - 1)
        self.assertAlmostEqual(result['total_return'], expected)
        no_trade = simulate(sample, np.array([0]), np.array([[-.02, 0., .02]]), policy)
        self.assertEqual(no_trade['total_return'], 0.)

    def test_drawdown_halt_applies_next_session(self):
        sample = bars(4)
        sample.prices[1, 0], sample.prices[1, 3] = 100., 10.
        result = simulate(sample, np.array([0, 1]), np.ones((2, 3)), Policy())
        self.assertTrue(result['halted'])
        self.assertEqual(result['ledger'][1]['exposure'], 0.)

    def test_calibration_widens_only(self):
        prediction = torch.zeros(20, 3, 3)
        margin = interval_margin(prediction, torch.ones(20, 3))
        self.assertTrue(torch.equal(margin, torch.ones(3)))

    def test_toss_get_only_and_local_dates(self):
        calls = []
        candle = {'timestamp': '2024-01-02T00:00:00+09:00', 'openPrice': '100',
                  'highPrice': '110', 'lowPrice': '95', 'closePrice': '105', 'volume': '1000', 'currency': 'KRW'}
        response = {'result': {'candles': [candle], 'nextBefore': None}}
        def transport(request, timeout):
            calls.append(request)
            return io.BytesIO(json.dumps(response).encode())
        client = TossMarketData(token='FAKE_TOKEN_FOR_UNIT_TEST_ONLY', transport=transport)
        rows, currency = client.daily_bars('005930')
        self.assertEqual(rows[0]['date'], '2024-01-02')
        self.assertEqual(currency, 'KRW')
        self.assertEqual(calls[0].method, 'GET')
        self.assertIn('/api/v1/candles?', calls[0].full_url)
        with self.assertRaises(ValueError):
            client.get('/api/v1/orders', {})

    def test_toss_pagination_repeated_cursor(self):
        candle = {'timestamp': '2024-01-02T00:00:00-05:00', 'openPrice': '100',
                  'highPrice': '110', 'lowPrice': '95', 'closePrice': '105', 'volume': '1000', 'currency': 'USD'}
        def transport(request, timeout):
            return io.BytesIO(json.dumps({'result': {'candles': [candle], 'nextBefore': candle['timestamp']}}).encode())
        with patch('quant_toss.toss.time.sleep'):
            with self.assertRaisesRegex(ValueError, 'Repeated'):
                TossMarketData(token='FAKE', transport=transport).daily_bars('AAPL', pages=3)


if __name__ == '__main__':
    unittest.main()
