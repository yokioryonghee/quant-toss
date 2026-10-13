import tempfile
import unittest
from pathlib import Path
import numpy as np
from quant_toss.engine import verify_identity
from quant_toss.backtest import Policy,simulate,target_exposure
from quant_toss.data import Bars

class Guards(unittest.TestCase):
    def test_csv_identity_columns_are_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bars.csv'
            path.write_text('date,symbol,currency\n2024-01-01,AAPL,USD\n')
            verify_identity(path,'AAPL','USD')
            with self.assertRaisesRegex(ValueError,'symbol'): verify_identity(path,'005930','USD')
            with self.assertRaisesRegex(ValueError,'currency'): verify_identity(path,'AAPL','KRW')
    def test_invalid_indices_and_quantile_order_fail(self):
        bars=Bars(['a','b'],np.full((2,5),100.))
        with self.assertRaises(ValueError): simulate(bars,np.array([-1]),np.ones((1,3)),Policy())
        with self.assertRaises(ValueError): simulate(bars,np.array([1]),np.ones((1,3)),Policy())
        with self.assertRaises(ValueError): target_exposure([.03,.01,.02],Policy())
