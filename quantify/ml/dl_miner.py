"""Phase 3: Deep Learning end-to-end stock selection.

Uses PyTorch LSTM / Transformer to take raw OHLCV time series and directly
predict cross-sectional forward returns.  No manual factor engineering —
the model learns features from raw price/volume data.

Usage::

    from quantify.ml.dl_miner import DLMiner, DLConfig

    miner = DLMiner(DLConfig(universe="000300.SH", model_type="lstm"))
    result = miner.run()
    print(result.summary())

Requires ``pip install -e ".[dl]"`` (PyTorch).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantify.utils.logger import log

from .backtest import VectorBacktestResult, compute_ic, vectorized_backtest
from .data import load_forward_returns, load_raw_ohlcv


@dataclass
class DLConfig:
    """Configuration for DL end-to-end stock selection."""

    universe: str | list[str] | None = None
    start_date: str | None = None
    end_date: str | None = None
    forward_period: int = 5
    # Sequence
    lookback: int = 20  # past N days of OHLCV as input sequence
    # Model
    model_type: str = "lstm"  # "lstm" or "transformer"
    hidden_dim: int = 64
    num_layers: int = 2
    n_heads: int = 4  # transformer only
    dropout: float = 0.1
    # Training
    batch_size: int = 256
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-5
    early_stop_patience: int = 10
    # Data
    fields: tuple[str, ...] = ("open", "high", "low", "close", "volume", "amount")
    test_ratio: float = 0.3
    # Backtest
    top_n: int = 20
    rebalance_days: int = 5
    # Device
    device: str = "auto"  # "auto", "cpu", "cuda"
    random_state: int = 42


@dataclass
class DLResult:
    """Result of DL end-to-end stock selection."""

    model: object
    train_scores: pd.DataFrame
    test_scores: pd.DataFrame
    train_ic: dict
    test_ic: dict
    train_backtest: VectorBacktestResult
    test_backtest: VectorBacktestResult
    train_history: list[float]
    val_history: list[float]
    config: DLConfig

    def summary(self) -> str:
        lines = [
            "=== DL End-to-End Stock Selection Results ===",
            f"Model: {self.config.model_type} (hidden={self.config.hidden_dim}, "
            f"layers={self.config.num_layers}, lookback={self.config.lookback})",
            f"Fields: {self.config.fields}",
            f"Train dates: {len(self.train_scores)}, Test dates: {len(self.test_scores)}",
            "",
            "--- IC Metrics ---",
            f"Train IC={self.train_ic.get('ic_mean', 0):.4f} IR={self.train_ic.get('icir', 0):.4f}",
            f"Test  IC={self.test_ic.get('ic_mean', 0):.4f} IR={self.test_ic.get('icir', 0):.4f}",
            "",
            "--- Train Backtest ---",
            f"Total return: {self.train_backtest.total_return:.2f}%",
            f"Sharpe: {self.train_backtest.sharpe:.2f}",
            f"Max DD: {self.train_backtest.max_drawdown:.2f}%",
            "",
            "--- Test Backtest ---",
            f"Total return: {self.test_backtest.total_return:.2f}%",
            f"Sharpe: {self.test_backtest.sharpe:.2f}",
            f"Max DD: {self.test_backtest.max_drawdown:.2f}%",
            "",
            f"Training: {len(self.train_history)} epochs, best val_loss={min(self.val_history):.6f}"
            if self.val_history
            else "",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PyTorch Models
# ---------------------------------------------------------------------------


def _build_model(config: DLConfig, n_features: int):
    """Build the PyTorch model."""
    import torch
    import torch.nn as nn

    class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, dropout):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, x):
            # x: (batch, seq_len, input_dim)
            out, _ = self.lstm(x)
            # Use last hidden state
            out = out[:, -1, :]
            return self.head(out).squeeze(-1)

    class TransformerModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, n_heads, dropout):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.pos_encoding = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, x):
            # x: (batch, seq_len, input_dim)
            seq_len = x.size(1)
            x = self.input_proj(x) + self.pos_encoding[:, :seq_len, :]
            x = self.encoder(x)
            # Mean pooling over sequence
            x = x.mean(dim=1)
            return self.head(x).squeeze(-1)

    if config.model_type == "lstm":
        return LSTMModel(n_features, config.hidden_dim, config.num_layers, config.dropout)
    if config.model_type == "transformer":
        return TransformerModel(
            n_features, config.hidden_dim, config.num_layers, config.n_heads, config.dropout
        )
    raise ValueError(f"Unknown model_type: {config.model_type}")


# ---------------------------------------------------------------------------
# DL Miner
# ---------------------------------------------------------------------------


class DLMiner:
    """Deep learning end-to-end stock selection."""

    def __init__(self, config: DLConfig | None = None) -> None:
        self.config = config or DLConfig()

    def _resolve_device(self) -> str:
        import torch

        cfg = self.config
        if cfg.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return cfg.device

    def _build_sequences(self) -> dict:
        """Build (date × stock) sequences of past OHLCV for DL input.

        Returns dict with keys: X_train, y_train, X_test, y_test,
        train_dates, test_dates, assets, close_panel, fwd_returns.
        """
        cfg = self.config
        from quantify.factor.evaluator import evaluation_window_default

        if not cfg.start_date or not cfg.end_date:
            ds, de = evaluation_window_default()
            cfg.start_date = cfg.start_date or ds
            cfg.end_date = cfg.end_date or de

        log.info(f"DL 加载原始数据: fields={cfg.fields}, lookback={cfg.lookback}")

        # Load raw OHLCV panels
        panels = load_raw_ohlcv(cfg.universe, cfg.start_date, cfg.end_date, cfg.fields)
        if not panels:
            raise RuntimeError("无法加载 OHLCV 数据")

        # Load forward returns
        fwd = load_forward_returns(cfg.universe, cfg.start_date, cfg.end_date, cfg.forward_period)

        # Align dates and assets
        common_dates = sorted(set.intersection(*[set(p.index) for p in panels.values()]) & set(fwd.index))
        common_assets = sorted(
            set.intersection(*[set(p.columns) for p in panels.values()]) & set(fwd.columns)
        )

        if len(common_dates) < cfg.lookback + cfg.forward_period + 50:
            raise RuntimeError(f"数据天数不足: {len(common_dates)}")

        # Build per-field arrays: (n_dates, n_assets) → normalize per stock
        field_arrays = {}
        for fld in cfg.fields:
            panel = panels[fld].loc[common_dates, common_assets]
            # Per-stock normalization: z-score using expanding window
            arr = panel.to_numpy(dtype=np.float32)
            # Simple normalization: rank-based to handle different scales
            # Normalize per column (per stock) using rolling z-score
            df = pd.DataFrame(arr, index=common_dates, columns=common_assets)
            # Global per-stock standardization
            mean = df.expanding(min_periods=cfg.lookback).mean()
            std = df.expanding(min_periods=cfg.lookback).std()
            normalized = (df - mean) / std.replace(0, 1)
            field_arrays[fld] = normalized.to_numpy(dtype=np.float32)

        # Chronological split
        split_idx = int(len(common_dates) * (1 - cfg.test_ratio))
        # Ensure we have enough lookback before the first training date
        start_idx = cfg.lookback  # first date we can make a prediction
        train_end = split_idx
        test_start = split_idx

        # Build sequences: for each date t (>= lookback), input = past lookback days
        # output = forward return at date t
        def _build_set(date_indices: range) -> tuple:
            X_list = []
            y_list = []
            score_dates = []
            for t in date_indices:
                if t < cfg.lookback:
                    continue
                if t + cfg.forward_period >= len(common_dates):
                    continue  # no forward return available

                # Input: (n_assets, lookback, n_fields)
                seq = np.stack(
                    [field_arrays[field][t - cfg.lookback : t, :] for field in cfg.fields],
                    axis=-1,
                )  # (lookback, n_assets, n_fields)

                # Target: forward returns at date t
                y = fwd.iloc[t].loc[common_assets].to_numpy(dtype=np.float32)

                # Valid stocks: both input and target are finite
                # seq shape: (lookback, n_assets, n_fields) → check all time×field per asset
                valid_input = np.all(np.isfinite(seq), axis=(0, 2))  # (n_assets,)
                valid = valid_input & np.isfinite(y)

                if valid.sum() < 10:
                    continue

                # Only keep valid stocks, transpose to (n_valid, lookback, n_fields)
                seq_valid = seq[:, valid, :].transpose(1, 0, 2)
                y_valid = y[valid]

                X_list.append(seq_valid)
                y_list.append(y_valid)
                score_dates.append(common_dates[t])

            if not X_list:
                return np.array([]), np.array([]), []

            # Stack: (total_valid_samples, lookback, n_fields)
            X_flat = np.concatenate(X_list, axis=0)
            y_flat = np.concatenate(y_list, axis=0)

            return X_flat, y_flat, score_dates

        # Training set
        X_train, y_train, train_dates = _build_set(range(start_idx, train_end))
        # Test set
        X_test, y_test, test_dates = _build_set(range(test_start, len(common_dates) - cfg.forward_period))

        log.info(
            f"DL 数据: train={len(X_train)} samples, test={len(X_test)} samples, "
            f"features={len(cfg.fields)}, lookback={cfg.lookback}"
        )

        # Also prepare close price panel for backtest
        close_panel = panels["close"].loc[common_dates, common_assets]

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "train_dates": train_dates,
            "test_dates": test_dates,
            "assets": common_assets,
            "close_panel": close_panel,
            "fwd_returns": fwd.loc[common_dates, common_assets],
            "panels": panels,
            "field_arrays": field_arrays,
            "common_dates": common_dates,
        }

    def _train_model(self, X_train, y_train, X_val, y_val, device: str):
        """Train the DL model."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        cfg = self.config
        n_features = X_train.shape[-1]

        model = _build_model(cfg, n_features).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        criterion = nn.MSELoss()

        # DataLoaders
        train_ds = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).float(),
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).float(),
        )
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

        train_history = []
        val_history = []
        best_val = float("inf")
        best_state = None
        patience_counter = 0

        log.info(f"训练 {cfg.model_type} 模型: {cfg.epochs} epochs, device={device}")

        for epoch in range(cfg.epochs):
            model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            train_history.append(train_loss)
            val_history.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                log.info(
                    f"  Epoch {epoch + 1}/{cfg.epochs}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
                )

            if patience_counter >= cfg.early_stop_patience:
                log.info(f"  Early stopping at epoch {epoch + 1} (patience={cfg.early_stop_patience})")
                break

        if best_state:
            model.load_state_dict(best_state)

        return model, train_history, val_history

    def _predict_scores(self, model, data: dict, device: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Predict scores for all dates and rebuild (date × asset) panels."""
        import torch

        cfg = self.config
        model.eval()

        common_dates = data["common_dates"]
        common_assets = data["assets"]
        field_arrays = data["field_arrays"]

        def _predict_for_dates(date_indices: range, dates: list[str]) -> pd.DataFrame:
            panel = pd.DataFrame(np.nan, index=common_dates, columns=common_assets)
            with torch.no_grad():
                for t in date_indices:
                    if t < cfg.lookback or t >= len(common_dates):
                        continue
                    dt = common_dates[t]
                    if dt not in dates:
                        continue

                    seq = np.stack(
                        [field_arrays[field][t - cfg.lookback : t, :] for field in cfg.fields],
                        axis=-1,
                    )  # (lookback, n_assets, n_fields)

                    valid_input = np.all(np.isfinite(seq), axis=(0, 2))  # (n_assets,)

                    if valid_input.sum() < 10:
                        continue

                    seq_valid = seq[:, valid_input, :].transpose(1, 0, 2)  # (n_valid, lookback, n_fields)
                    x = torch.from_numpy(seq_valid).float().to(device)
                    pred = model(x).cpu().numpy()

                    valid_assets = np.array(common_assets)[valid_input]
                    panel.loc[dt, valid_assets] = pred

            return panel.loc[dates]

        train_scores = _predict_for_dates(range(cfg.lookback, len(common_dates)), data["train_dates"])
        test_scores = _predict_for_dates(range(cfg.lookback, len(common_dates)), data["test_dates"])

        return train_scores, test_scores

    def run(self) -> DLResult:
        """Run the full DL pipeline.

        1. Load raw OHLCV + forward returns
        2. Build time-series sequences (lookback window)
        3. Train LSTM/Transformer
        4. Predict scores
        5. Compute IC + vectorized backtest
        """
        cfg = self.config
        device = self._resolve_device()
        log.info(f"DL 端到端选股: model={cfg.model_type}, device={device}, lookback={cfg.lookback}")

        # 1. Build sequences
        data = self._build_sequences()
        if len(data["X_train"]) < 500:
            raise RuntimeError(f"训练样本不足: {len(data['X_train'])}")

        # Split training into train/val (80/20)
        n_train = len(data["X_train"])
        val_size = int(n_train * 0.2)
        X_val = data["X_train"][-val_size:]
        y_val = data["y_train"][-val_size:]
        X_tr = data["X_train"][:-val_size]
        y_tr = data["y_train"][:-val_size]

        # 2. Train
        model, train_history, val_history = self._train_model(X_tr, y_tr, X_val, y_val, device)

        # 3. Predict scores
        train_scores, test_scores = self._predict_scores(model, data, device)

        # 4. IC
        fwd = data["fwd_returns"]
        train_ic = compute_ic(train_scores, fwd)
        test_ic = compute_ic(test_scores, fwd)
        log.info(f"Train IC={train_ic['ic_mean']:.4f} IR={train_ic['icir']:.4f}")
        log.info(f"Test  IC={test_ic['ic_mean']:.4f} IR={test_ic['icir']:.4f}")

        # 5. Backtest
        close_panel = data["close_panel"]
        train_bt = vectorized_backtest(
            train_scores,
            close_panel,
            top_n=cfg.top_n,
            rebalance_days=cfg.rebalance_days,
        )
        test_bt = vectorized_backtest(
            test_scores,
            close_panel,
            top_n=cfg.top_n,
            rebalance_days=cfg.rebalance_days,
        )
        log.info(f"Train: return={train_bt.total_return:.2f}% sharpe={train_bt.sharpe:.2f}")
        log.info(f"Test:  return={test_bt.total_return:.2f}% sharpe={test_bt.sharpe:.2f}")

        return DLResult(
            model=model,
            train_scores=train_scores,
            test_scores=test_scores,
            train_ic=train_ic,
            test_ic=test_ic,
            train_backtest=train_bt,
            test_backtest=test_bt,
            train_history=train_history,
            val_history=val_history,
            config=cfg,
        )
