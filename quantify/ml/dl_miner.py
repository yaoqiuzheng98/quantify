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
    # Factor-augmented input: load Qlib atomic factors as cross-sectional features
    use_factors: bool = True
    factor_dim: int = 32  # MLP hidden dim for factor branch
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


def _build_model(config: DLConfig, n_features: int, n_factors: int = 0):
    """Build the PyTorch model.

    If n_factors > 0, builds a hybrid model: LSTM/Transformer for raw OHLCV
    time series + MLP for cross-sectional factor features, concatenated.
    """
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
        def __init__(self, input_dim, hidden_dim, num_layers, n_heads, dropout, max_seq_len=512):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            # Dynamic position encoding sized to max sequence length
            self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
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

    class HybridModel(nn.Module):
        """LSTM/Transformer for time series + MLP for cross-sectional factors."""

        def __init__(self, ts_model, n_factors, factor_dim, hidden_dim, dropout):
            super().__init__()
            self.ts_model = ts_model  # LSTM or Transformer (without its head)
            # Replace ts_model.head with identity — we'll concat before head
            self.ts_feat_dim = hidden_dim
            self.factor_mlp = nn.Sequential(
                nn.Linear(n_factors, factor_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(factor_dim, factor_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_dim + factor_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x_ts, x_factor):
            # x_ts: (batch, seq_len, input_dim), x_factor: (batch, n_factors)
            # Extract ts features (before head)
            if isinstance(self.ts_model, LSTMModel):
                out, _ = self.ts_model.lstm(x_ts)
                ts_feat = out[:, -1, :]  # (batch, hidden_dim)
            else:  # TransformerModel
                seq_len = x_ts.size(1)
                proj = self.ts_model.input_proj(x_ts) + self.ts_model.pos_encoding[:, :seq_len, :]
                enc = self.ts_model.encoder(proj)
                ts_feat = enc.mean(dim=1)  # (batch, hidden_dim)
            fac_feat = self.factor_mlp(x_factor)  # (batch, factor_dim)
            combined = torch.cat([ts_feat, fac_feat], dim=1)
            return self.head(combined).squeeze(-1)

    if config.model_type == "lstm":
        ts_model = LSTMModel(n_features, config.hidden_dim, config.num_layers, config.dropout)
        if n_factors > 0:
            return HybridModel(ts_model, n_factors, config.factor_dim, config.hidden_dim, config.dropout)
        return ts_model
    if config.model_type == "transformer":
        ts_model = TransformerModel(
            n_features,
            config.hidden_dim,
            config.num_layers,
            config.n_heads,
            config.dropout,
            max_seq_len=config.lookback,
        )
        if n_factors > 0:
            return HybridModel(ts_model, n_factors, config.factor_dim, config.hidden_dim, config.dropout)
        return ts_model
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

        If config.use_factors, also loads Qlib atomic factors as cross-sectional
        features. Returns dict with keys: X_train, y_train, X_test, y_test,
        train_dates, test_dates, assets, close_panel, fwd_returns, and optionally
        factor_train/factor_test.
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
            arr = panel.to_numpy(dtype=np.float32)
            df = pd.DataFrame(arr, index=common_dates, columns=common_assets)
            mean = df.expanding(min_periods=cfg.lookback).mean()
            std = df.expanding(min_periods=cfg.lookback).std()
            normalized = (df - mean) / std.replace(0, 1)
            field_arrays[fld] = normalized.to_numpy(dtype=np.float32)

        # Load atomic factors as cross-sectional features (if enabled)
        factor_arrays = None  # (n_dates, n_assets, n_factors)
        if cfg.use_factors:
            from .gp_miner import ATOMIC_FACTORS
            from .data import load_factor_panels

            log.info(f"DL 加载 {len(ATOMIC_FACTORS)} 个原子因子 (横截面特征)")
            factor_exprs = [expr for _, expr in ATOMIC_FACTORS]
            factor_panels = load_factor_panels(factor_exprs, cfg.universe, cfg.start_date, cfg.end_date)
            # Build (n_dates, n_assets, n_factors) array
            n_dates = len(common_dates)
            n_assets = len(common_assets)
            n_factors = len(ATOMIC_FACTORS)
            factor_arrays = np.full((n_dates, n_assets, n_factors), np.nan, dtype=np.float32)
            for i, (name, _) in enumerate(ATOMIC_FACTORS):
                panel = factor_panels[factor_exprs[i]].loc[common_dates, common_assets]
                factor_arrays[:, :, i] = panel.to_numpy(dtype=np.float32)
            # Cross-sectional z-score per (date, factor) to normalize
            with np.errstate(invalid="ignore"):
                cs_mean = np.nanmean(factor_arrays, axis=1, keepdims=True)
                cs_std = np.nanstd(factor_arrays, axis=1, keepdims=True)
                factor_arrays = (factor_arrays - cs_mean) / np.where(cs_std > 1e-8, cs_std, 1.0)
            factor_arrays = np.nan_to_num(factor_arrays, nan=0.0, posinf=0.0, neginf=0.0)
            log.info(f"DL 因子特征: {n_factors} 个因子, shape={factor_arrays.shape}")

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
            F_list = []  # factor features
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

                # Also check factor validity if enabled
                if factor_arrays is not None:
                    fac_t = factor_arrays[t]  # (n_assets, n_factors)
                    valid = valid & np.all(np.isfinite(fac_t), axis=1)

                if valid.sum() < 10:
                    continue

                # Only keep valid stocks, transpose to (n_valid, lookback, n_fields)
                seq_valid = seq[:, valid, :].transpose(1, 0, 2)
                y_valid = y[valid]

                X_list.append(seq_valid)
                y_list.append(y_valid)
                score_dates.append(common_dates[t])
                if factor_arrays is not None:
                    F_list.append(fac_t[valid])

            if not X_list:
                if factor_arrays is not None:
                    return np.array([]), np.array([]), np.array([]), []
                return np.array([]), np.array([]), []

            # Stack: (total_valid_samples, lookback, n_fields)
            X_flat = np.concatenate(X_list, axis=0)
            y_flat = np.concatenate(y_list, axis=0)

            if factor_arrays is not None:
                F_flat = np.concatenate(F_list, axis=0)
                return X_flat, F_flat, y_flat, score_dates
            return X_flat, y_flat, score_dates

        # Training set
        if factor_arrays is not None:
            X_train, F_train, y_train, train_dates = _build_set(range(start_idx, train_end))
            X_test, F_test, y_test, test_dates = _build_set(
                range(test_start, len(common_dates) - cfg.forward_period)
            )
        else:
            X_train, y_train, train_dates = _build_set(range(start_idx, train_end))
            X_test, y_test, test_dates = _build_set(range(test_start, len(common_dates) - cfg.forward_period))

        log.info(
            f"DL 数据: train={len(X_train)} samples, test={len(X_test)} samples, "
            f"features={len(cfg.fields)}, lookback={cfg.lookback}"
            + (f", factors={factor_arrays.shape[2]}" if factor_arrays is not None else "")
        )

        # Also prepare close price panel for backtest
        close_panel = panels["close"].loc[common_dates, common_assets]

        result = {
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
            "use_factors": factor_arrays is not None,
            "n_factors": factor_arrays.shape[2] if factor_arrays is not None else 0,
        }
        if factor_arrays is not None:
            result["F_train"] = F_train
            result["F_test"] = F_test
            result["factor_arrays"] = factor_arrays
        return result

    def _train_model(self, X_train, y_train, X_val, y_val, device: str, F_train=None, F_val=None):
        """Train the DL model. If F_train is provided, uses hybrid (ts + factor) model."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        cfg = self.config
        n_features = X_train.shape[-1]
        n_factors = F_train.shape[-1] if F_train is not None else 0

        # Set random seeds for reproducibility
        torch.manual_seed(cfg.random_state)
        np.random.seed(cfg.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.random_state)

        model = _build_model(cfg, n_features, n_factors).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        criterion = nn.MSELoss()
        # Learning rate scheduler: reduce LR by half on plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=False
        )

        # DataLoaders — shuffle=False for time series (preserve temporal order)
        if n_factors > 0:
            train_ds = TensorDataset(
                torch.from_numpy(X_train).float(),
                torch.from_numpy(F_train).float(),
                torch.from_numpy(y_train).float(),
            )
            val_ds = TensorDataset(
                torch.from_numpy(X_val).float(),
                torch.from_numpy(F_val).float(),
                torch.from_numpy(y_val).float(),
            )
        else:
            train_ds = TensorDataset(
                torch.from_numpy(X_train).float(),
                torch.from_numpy(y_train).float(),
            )
            val_ds = TensorDataset(
                torch.from_numpy(X_val).float(),
                torch.from_numpy(y_val).float(),
            )
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

        train_history = []
        val_history = []
        best_val = float("inf")
        best_state = None
        patience_counter = 0

        model_desc = f"{cfg.model_type}" + ("+factor" if n_factors > 0 else "")
        log.info(f"训练 {model_desc} 模型: {cfg.epochs} epochs, device={device}")

        for epoch in range(cfg.epochs):
            model.train()
            train_losses = []
            for batch in train_loader:
                if n_factors > 0:
                    xb, fb, yb = batch
                    xb, fb, yb = xb.to(device), fb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    pred = model(xb, fb)
                else:
                    xb, yb = batch
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    if n_factors > 0:
                        xb, fb, yb = batch
                        xb, fb, yb = xb.to(device), fb.to(device), yb.to(device)
                        pred = model(xb, fb)
                    else:
                        xb, yb = batch
                        xb, yb = xb.to(device), yb.to(device)
                        pred = model(xb)
                    loss = criterion(pred, yb)
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            train_history.append(train_loss)
            val_history.append(val_loss)

            # Step the LR scheduler based on validation loss
            scheduler.step(val_loss)

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
        use_factors = data.get("use_factors", False)
        factor_arrays = data.get("factor_arrays")

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

                    if use_factors and factor_arrays is not None:
                        fac_t = factor_arrays[t]  # (n_assets, n_factors)
                        valid_input = valid_input & np.all(np.isfinite(fac_t), axis=1)

                    if valid_input.sum() < 10:
                        continue

                    seq_valid = seq[:, valid_input, :].transpose(1, 0, 2)  # (n_valid, lookback, n_fields)
                    x = torch.from_numpy(seq_valid).float().to(device)

                    if use_factors and factor_arrays is not None:
                        f_valid = fac_t[valid_input]
                        f = torch.from_numpy(f_valid).float().to(device)
                        pred = model(x, f).cpu().numpy()
                    else:
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

        # Factor features for hybrid model
        F_tr = F_val = None
        if data.get("use_factors"):
            F_val = data["F_train"][-val_size:]
            F_tr = data["F_train"][:-val_size]

        # 2. Train
        model, train_history, val_history = self._train_model(
            X_tr, y_tr, X_val, y_val, device, F_train=F_tr, F_val=F_val
        )

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
