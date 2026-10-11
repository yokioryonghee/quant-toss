# Validation status — 2026-10-11

Verified locally on CPU with PyTorch 2.8.0:

- 10 unit tests: causal features, price-unit invariance, symbol/currency metadata guard, purged labels and train-only scaling, noncrossing quantiles and gradients, exact buy/sell accounting, next-session drawdown halt, residual interval widening, read-only mock Toss daily bars/local dates, repeated pagination cursor detection.
- Offline 1,100-session synthetic data run completed. Linear and transformer each trained for 2 epochs. Validation selection, calibration, test forecasting, cost-aware paper accounting, checkpoint reload and forward prediction completed.
- Both synthetic policies stayed in cash: zero traded sessions and zero paper return. This is disclosed, not described as trading success.
- Two expanding-window folds completed with fresh transformer training, calibration and separate test reports (one epoch per fold). This verifies the runner, not robust strategy performance.
- A standalone Python wheel built locally without downloading build dependencies.
- Official Toss OpenAPI schema checked at version 1.2.24. Authenticated API calls have not been tested; the user's API account setup is not enabled yet.

No real market benchmark, actual profit, live order execution, or evidence of model superiority has been established. Synthetic price processes do not represent either the Korean or US exchange calendar; they only skip weekends.

The stored synthetic demo reports are reproducible through `python -m quant_toss.demo --output artifacts/new_run`. Artifacts and credentials are excluded from source publication.

GitHub CI passed for the initial published implementation: [run #1](https://github.com/yokioryonghee/quant-toss/actions/runs/38127483491), commit `6d6ccbb`. A fresh Ubuntu / Python 3.11 install completed all 10 unit tests and the offline training/backtest demo.

The standalone HTML report was generated locally both from a previously completed demo and from a fresh end-to-end run. It displays the stored input prices, validation loss, calibrated forecast ranges, comparison metrics and paper ledger. The browser-open option is optional; CI renders the file without opening a browser.
