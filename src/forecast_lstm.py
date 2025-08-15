import os, math, argparse, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ------------------- Config defaults -------------------
DEFAULT_LOOKBACK = 252         # ~1 trading year
DEFAULT_HORIZON  = 20          # ~1 trading month
DEFAULT_PERIOD   = "5y"        # download window
VAL_TAIL_DAYS    = 120         # time-based validation window within each fold
MIN_TRAIN_DAYS   = 800         # skip if less history
EPOCHS           = 50
BATCH_SIZE       = 64
MC_SAMPLES       = 100         # for uncertainty bands via MC Dropout

# ------------------- Helpers -------------------
def to_windows_cum(series_1d, lookback, horizon):
    X, y = [], []
    for i in range(len(series_1d) - lookback - horizon + 1):
        X.append(series_1d[i:i+lookback])
        y.append(series_1d[i+lookback:i+lookback+horizon].sum())
    X = np.array(X)[..., None]  # (N, lookback, 1)
    y = np.array(y).reshape(-1, 1)  # (N, 1)
    return X, y

def build_model(lookback, horizon, dropout=0.2):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    m = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(dropout),
        LSTM(32, return_sequences=False),
        Dropout(dropout),
        Dense(1)  # cumulative H-day return
    ])
    m.compile(optimizer='adam', loss='mse')
    return m

def mape(y_true, y_pred):
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = t != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((t[mask] - p[mask]) / t[mask])) * 100

def rmse(y_true, y_pred):
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((t - p)**2)))

def directional_accuracy(returns_true, returns_pred):
    # returns arrays aligned (N, H). Score 1-step & all-steps average
    sign_true = np.sign(returns_true)
    sign_pred = np.sign(returns_pred)
    hit = (sign_true == sign_pred).mean()
    return float(hit)

def business_days_from(start_date, periods):
    return pd.bdate_range(start=start_date, periods=periods)

# ------------------- Core -------------------
def walk_forward_backtest(adj_close: pd.Series, lookback, horizon):
    # 1) log-returns
    logp = np.log(adj_close.astype(float).replace(0, np.nan)).dropna()
    r = logp.diff().dropna().values.reshape(-1, 1)  # (T-1, 1)

    if len(r) < max(MIN_TRAIN_DAYS, lookback + horizon + 50):
        return None

    cut_points = range(MIN_TRAIN_DAYS, len(r) - horizon, horizon)

    all_fore_price, all_true_price = [], []
    all_fore_ret,   all_true_ret   = [], []

    for cut in cut_points:
        train_r = r[:cut]
        scaler = StandardScaler().fit(train_r)
        r_scaled = scaler.transform(r)

        X_all, y_all = to_windows_cum(r_scaled[:cut].flatten(), lookback, horizon)
        if len(X_all) == 0:
            continue

        val_start = max(0, len(X_all) - VAL_TAIL_DAYS)
        X_tr, y_tr = X_all[:val_start], y_all[:val_start]
        X_val, y_val = X_all[val_start:], y_all[val_start:]

        model = build_model(lookback, horizon)
        cb = [EarlyStopping(patience=5, restore_best_weights=True),
              ReduceLROnPlateau(patience=3, factor=0.5)]
        decay = 0.999  # tune 0.997–0.9995
        n = len(X_tr)
        weights = np.power(decay, np.arange(n)[::-1])
        model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
          sample_weight=weights,
          epochs=EPOCHS, batch_size=BATCH_SIZE,
          callbacks=cb, verbose=0)

        last_window = r_scaled[cut-lookback:cut].reshape(1, lookback, 1)
        pred_r_scaled = model.predict(last_window, verbose=0).flatten()
        pred_r = scaler.inverse_transform(pred_r_scaled.reshape(-1,1)).flatten()

        # actuals
        true_r = r[cut:cut+horizon].flatten()

        # prices
        P0 = float(adj_close.iloc[cut])  # base price
        pred_price = P0 * np.exp(np.cumsum(pred_r))
        true_price = P0 * np.exp(np.cumsum(true_r))

        all_fore_price.append(pred_price)
        all_true_price.append(true_price)
        all_fore_ret.append(pred_r)
        all_true_ret.append(true_r)

    if not all_fore_price:
        return None

    # stack
    f_price = np.vstack(all_fore_price)
    t_price = np.vstack(all_true_price)
    f_ret   = np.vstack(all_fore_ret)
    t_ret   = np.vstack(all_true_ret)

    # Align dates for reporting (optional)
    # We won’t return dates here; only metrics.
    metrics = {
        "rmse_price": rmse(t_price, f_price),
        "mape_price": mape(t_price, f_price),
        "rmse_return": rmse(t_ret, f_ret),
        "directional_accuracy": directional_accuracy(t_ret, f_ret)
    }
    return metrics

def final_forecast_with_intervals(adj_close: pd.Series, lookback, horizon):
    # Train on full history and forecast next horizon with MC Dropout
    logp = np.log(adj_close.astype(float).replace(0, np.nan)).dropna()
    r = logp.diff().dropna().values.reshape(-1,1)
    if len(r) < lookback + horizon + 10:
        return None

    scaler = StandardScaler().fit(r)
    r_scaled = scaler.transform(r)

    X, y = to_windows_cum(r_scaled.flatten(), lookback, horizon)
    if len(X) == 0:
        return None

    val_start = max(0, len(X) - VAL_TAIL_DAYS)
    X_tr, y_tr = X[:val_start], y[:val_start]
    X_val, y_val = X[val_start:], y[val_start:]

    model = build_model(lookback, horizon, dropout=0.2)
    cb = [EarlyStopping(patience=5, restore_best_weights=True),
          ReduceLROnPlateau(patience=3, factor=0.5)]
    model.fit(X_tr, y_tr,
              validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH_SIZE,
              verbose=0, callbacks=cb)

    last_win = r_scaled[-lookback:].reshape(1, lookback, 1)

    # MC Dropout samples (keep dropout active by calling with training=True)
    preds = []
    for _ in range(MC_SAMPLES):
        pred_scaled = model(last_win, training=True).numpy().flatten()
        pred = scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()
        preds.append(pred)
    preds = np.stack(preds, axis=0)  # (S, H)

    # convert returns to price paths
    P0 = float(adj_close.iloc[-1])
    paths = P0 * np.exp(np.cumsum(preds, axis=1))  # (S, H)

    p50 = np.median(paths, axis=0)
    p10 = np.percentile(paths, 10, axis=0)
    p90 = np.percentile(paths, 90, axis=0)

    dates = business_days_from(adj_close.index[-1] + pd.Timedelta(days=1), horizon)
    out = pd.DataFrame({"Date": dates, "P10": p10, "P50": p50, "P90": p90})
    return out

def process_ticker(ticker, args, out_base):
    print(f"→ {ticker}")
    try:
        df = yf.download(ticker, period=args.period, interval="1d", auto_adjust=True, progress=False)
        if df.empty or "Close" not in df:
            print(f"   Skipped — no data.")
            return
        adj_close = df["Close"].dropna()
        os.makedirs(f"{out_base}/predictions", exist_ok=True)
        os.makedirs(f"{out_base}/reports", exist_ok=True)

        # backtest
        metrics = walk_forward_backtest(adj_close, args.lookback, args.horizon)
        if metrics is None:
            print("   Skipped — not enough history for backtest.")
            return

        # final forecast with intervals
        f_df = final_forecast_with_intervals(adj_close, args.lookback, args.horizon)
        if f_df is None:
            print("   Skipped — not enough history for final forecast.")
            return

        # save
        safe = ticker.replace(".", "_")
        f_df.to_csv(f"{out_base}/predictions/{safe}_forecast.csv", index=False)
        with open(f"{out_base}/reports/{safe}_eval.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"   Saved predictions + metrics. "
              f"DA={metrics['directional_accuracy']:.2f} | MAPE={metrics['mape_price']:.2f}%")

    except Exception as e:
        print(f"   Error: {e}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", type=str, required=True,
                   help="Comma-separated list, e.g. EMAAR.AE,ADNOCDRILL.AE")
    p.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--period", type=str, default=DEFAULT_PERIOD)
    p.add_argument("--outdir", type=str, default="stock_prediction")
    return p.parse_args()

def main():
    args = parse_args()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    out_base = args.outdir
    for t in tickers:
        process_ticker(t, args, out_base)

if __name__ == "__main__":
    # Make TF quieter & deterministic-ish
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.get_logger().setLevel("ERROR")
    main()
