import csv
import numpy as np
import pytest
from quant_toss.engine import verify_identity
from quant_toss.backtest import Policy,simulate,target_exposure
from quant_toss.data import Bars

def test_csv_identity_columns_are_checked(tmp_path):
    path=tmp_path/'bars.csv'
    path.write_text('date,symbol,currency\n2024-01-01,AAPL,USD\n')
    verify_identity(path,'AAPL','USD')
    with pytest.raises(ValueError,match='symbol'): verify_identity(path,'005930','USD')
    with pytest.raises(ValueError,match='currency'): verify_identity(path,'AAPL','KRW')

def test_invalid_indices_and_quantile_order_fail():
    bars=Bars(['a','b'],np.full((2,5),100.))
    with pytest.raises(ValueError): simulate(bars,np.array([-1]),np.ones((1,3)),Policy())
    with pytest.raises(ValueError): simulate(bars,np.array([1]),np.ones((1,3)),Policy())
    with pytest.raises(ValueError): target_exposure([.03,.01,.02],Policy())
