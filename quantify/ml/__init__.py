"""Machine-learning / Deep-learning factor mining module.

This module is **independent** of the JoinQuant-compatible strategy pipeline.
It does NOT target JoinQuant portability — instead it fully leverages ML/DL
to discover factors and predict cross-sectional stock returns.

Three phases (implemented incrementally):

1. **ML factor synthesis** — use existing factor library as features, train
   XGBoost / LightGBM / sklearn models to predict forward returns, output
   daily stock scores.  Vectorized backtest for fast iteration.

2. **GP factor discovery** — genetic programming (gplearn) evolves Qlib
   factor expressions, fitness = IC/ICIR.  Output expressions feed back
   into the existing ``factor_library``.

3. **DL end-to-end** — PyTorch LSTM / Transformer takes raw OHLCV +
   fundamentals time series, directly predicts cross-sectional returns.

All heavy dependencies (sklearn, xgboost, lightgbm, torch, gplearn) are
lazily imported so the base package stays lightweight.
"""
