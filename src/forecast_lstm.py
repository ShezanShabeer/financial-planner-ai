# forecast_lstm.py  (ensemble LSTM with meta-calibration, short-history safe)
import os, json, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_percentage_error

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Layer, Activation, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# =========================
# Global knobs (can be overridden by CLI where available)
# =========================
SEED = 42
tf.keras.utils.set_random_seed(SEED)

EPOCHS = 70
BATCH = 128
DROPOUT_P = 0.25
DECAY_W = 0.998

DEFAULT_LOOKBACKS = [36, 48, 54]      # ensemble members
DEFAULT_SEEDS = [11, 22, 33]          # ensemble members
DEFAULT_HVEC = 12
DEFAULT_HORIZONS = [12, 24, 60]
DEFAULT_VAL_TAIL_WINDOWS = 30         # smaller so short histories work
MC_SAMPLES = 30

LOSS_W_VEC = 0.45
LOSS_W_CUM = 0.25
LOSS_W_CLS = 0.30

DEFAULT_RMAX = 0.12
USE_FOCAL_DEFAULT = 1
FOCAL_GAMMA_DEFAULT = 2.0

# =========================
# Data utils
# =========================
def _coerce_close_series(df_close):
    if isinstance(df_close, pd.DataFrame):
        if df_close.shape[1] == 1:
            s = df_close.squeeze("columns")
        else:
            col = df_close.notna().sum().sort_values(ascending=False).index[0]
            s = df_close[col]
    else:
        s = df_close
    s.index = pd.to_datetime(s.index)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s

def monthly_close(ticker, period="max", debug=0):
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty or "Close" not in df.columns:
        if debug: print(f"   [data] No data for {ticker} (period={period})")
        return None
    close = _coerce_close_series(df["Close"])
    if close.empty:
        if debug: print(f"   [data] empty Close for {ticker}")
        return None
    m = close.resample("M").last().dropna().astype(float)
    m = m[~m.index.duplicated(keep="last")].sort_index()
    if debug:
        print(f"   [data] {ticker}: {len(m)} monthly bars ({m.index.min().date()} → {m.index.max().date()})")
    return m

def compute_features_from_close(close_m: pd.Series):
    if not isinstance(close_m, pd.Series): close_m = pd.Series(close_m)
    close_m = close_m.dropna()
    if len(close_m) < 18: return None

    logp = np.log(close_m.astype(float))
    r = logp.diff().rename("r")

    mo = close_m.index.month
    sin_m = pd.Series(np.sin(2*np.pi*mo/12), index=close_m.index, name="sin_m")
    cos_m = pd.Series(np.cos(2*np.pi*mo/12), index=close_m.index, name="cos_m")

    mom12 = r.rolling(12, min_periods=12).mean().rename("mom12")
    vol12 = r.rolling(12, min_periods=12).std().rename("vol12")

    feats = pd.concat([r, mom12, vol12, sin_m, cos_m], axis=1).dropna()
    return feats if not feats.empty else None

def add_exog(feats_df: pd.DataFrame, exog_list, period="max", debug=0):
    if not exog_list: return feats_df
    out = feats_df.copy()
    for t in exog_list:
        try:
            em = monthly_close(t, period=period, debug=0)
            if em is None or len(em) < 24:
                if debug: print(f"   [exog] skip {t}: insufficient")
                continue
            rex = np.log(em).diff().rename(f"exog_{t}_r")
            out = out.join(rex, how="left")
        except Exception as e:
            if debug: print(f"   [exog] {t} error: {e}")
    return out.dropna()

def to_windows(feat_df: pd.DataFrame, lookback, h_vec, cum_horizons, r_col="r"):
    if feat_df is None or feat_df.empty or (r_col not in feat_df.columns): return None
    r = feat_df[r_col].values
    F = feat_df.values
    idx = feat_df.index
    max_h = max(cum_horizons + [h_vec])
    T = len(feat_df)
    last_start = T - lookback - max_h + 1
    if last_start <= 0: return None

    X, yv, yc, yk, ends = [], [], [], [], []
    for i in range(last_start):
        past = F[i:i+lookback]
        fut_r = r[i+lookback:i+lookback+h_vec]
        if len(fut_r) != h_vec: continue
        cums, bins = [], []
        for H in cum_horizons:
            future_H = r[i+lookback:i+lookback+H]
            if len(future_H) != H:
                cums.append(0.0); bins.append(0.0)
            else:
                c = future_H.sum()
                cums.append(c)
                bins.append(1.0 if c > 0 else 0.0)
        X.append(past); yv.append(fut_r); yc.append(cums); yk.append(bins)
        ends.append(idx[i+lookback-1])

    return (np.array(X, np.float32),
            np.array(yv, np.float32),
            np.array(yc, np.float32),
            np.array(yk, np.float32),
            np.array(ends))

# =========================
# Model & losses
# =========================
class TemporalAttention(Layer):
    def build(self, input_shape):
        h = int(input_shape[-1])
        self.W = self.add_weight(shape=(h, h), name="att_W", initializer="glorot_uniform")
        self.b = self.add_weight(shape=(h,),   name="att_b", initializer="zeros")
        self.u = self.add_weight(shape=(h,),   name="att_u", initializer="glorot_uniform")
        super().build(input_shape)
    def call(self, x):
        uit = tf.tensordot(x, self.W, axes=1) + self.b
        uit = tf.tanh(uit)
        ait = tf.tensordot(uit, self.u, axes=1)
        ait = tf.nn.softmax(ait)
        ait = tf.expand_dims(ait, -1)
        return tf.reduce_sum(x * ait, axis=1)

def make_cls_loss_from_pos_weights(pos_w, use_focal=False, gamma=1.5, eps=1e-7):
    pos_w = tf.constant(np.asarray(pos_w, dtype=np.float32))
    neg_w = tf.ones_like(pos_w, dtype=tf.float32)
    def _loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
        p_t = y_true * y_pred + (1. - y_true) * (1. - y_pred)
        bce = -(y_true * tf.math.log(y_pred) + (1. - y_true) * tf.math.log(1. - y_pred))
        w = y_true * pos_w + (1. - y_true) * neg_w
        loss = w * bce
        if use_focal:
            loss *= tf.pow(1. - p_t, gamma)
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return _loss

def build_lstm(lookback, feat_dim, h_vec, k_cum, rmax=DEFAULT_RMAX, dropout=DROPOUT_P, units=64):
    inp = Input(shape=(lookback, feat_dim))
    x = LSTM(units, return_sequences=True)(inp)
    x = Dropout(dropout)(x)
    x = LSTM(units, return_sequences=True)(x)
    x = Dropout(dropout)(x)
    ctx = TemporalAttention()(x)

    # y_vec: monthly returns bounded by tanh * rmax
    vec_h = Dense(128, activation="relu")(ctx)
    y_vec_lin = Dense(h_vec)(vec_h)
    y_vec = Activation("tanh")(y_vec_lin)
    y_vec = Lambda(lambda t: t * rmax, name="y_vec")(y_vec)

    # y_cum: cumulative returns per horizon
    cum_h = Dense(64, activation="relu")(ctx)
    y_cum = Dense(k_cum, name="y_cum")(cum_h)

    # y_cls: prob up per horizon
    cls_h = Dense(64, activation="relu")(ctx)
    y_cls = Dense(k_cum, activation="sigmoid", name="y_cls")(cls_h)

    model = Model(inp, [y_vec, y_cum, y_cls])
    return model

# =========================
# Helpers
# =========================
def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
def mape_price(true_prices, pred_prices):
    return float(mean_absolute_percentage_error(true_prices, pred_prices) * 100.0)

def pos_weights_from_labels(y_cls_true, clip=10.0, eps=1e-6):
    pos_rate = np.clip(y_cls_true.mean(axis=0), eps, 1.0 - eps)
    pos_w = (1.0 - pos_rate) / pos_rate
    return np.clip(pos_w, 1.0, clip)

def predict_mc_mean(model, X, mc_samples=1):
    if mc_samples <= 1:
        return model.predict(X, verbose=0)
    ys_vec, ys_cum, ys_cls = [], [], []
    for _ in range(mc_samples):
        y_vec, y_cum, y_cls = model(X, training=True)
        ys_vec.append(y_vec.numpy()); ys_cum.append(y_cum.numpy()); ys_cls.append(y_cls.numpy())
    return np.mean(ys_vec, 0), np.mean(ys_cum, 0), np.mean(ys_cls, 0)

# =========================
# Train one model config for a fold
# =========================
def fit_one_config(X_tr, yv_tr, yc_tr, yk_tr, X_val, yv_val, yc_val, yk_val,
                   lookback, feat_dim, h_vec, k_cum, rmax, use_focal, focal_gamma, seed):
    tf.keras.utils.set_random_seed(seed)
    pos_w = pos_weights_from_labels(yk_tr, clip=10.0)
    cls_loss = make_cls_loss_from_pos_weights(pos_w, use_focal=bool(use_focal), gamma=focal_gamma)

    model = build_lstm(lookback, feat_dim, h_vec, k_cum, rmax=rmax, dropout=DROPOUT_P, units=64)
    model.compile(optimizer="adam",
                  loss=[tf.keras.losses.Huber(), tf.keras.losses.Huber(), cls_loss],
                  loss_weights=[LOSS_W_VEC, LOSS_W_CUM, LOSS_W_CLS])

    cb = [EarlyStopping(patience=6, restore_best_weights=True),
          ReduceLROnPlateau(patience=3, factor=0.5)]

    n = len(X_tr)
    sw = np.power(DECAY_W, np.arange(n)[::-1]).astype(np.float32)
    model.fit(X_tr, [yv_tr, yc_tr, yk_tr],
              validation_data=(X_val, [yv_val, yc_val, yk_val]),
              epochs=EPOCHS, batch_size=BATCH,
              sample_weight=[sw, sw, sw],
              verbose=0, callbacks=cb)
    return model

# =========================
# Alignment helpers for meta-calibration
# =========================
def align_by_common_ends(val_stack, ref_labels):
    """
    Ensure all models' validation predictions align on the same end-date set.
    val_stack: list of dicts with keys ['y_vec','y_cum','y_cls','ends_val']
    ref_labels: (yv_val, yc_val, yk_val, ends_ref)
    Returns:
      val_stack_aligned, ref_labels_aligned, common_ends
    """
    if len(val_stack) == 0:
        return [], None, None
    # build intersection of all ends_val
    common = None
    for d in val_stack:
        e = d["ends_val"].astype('datetime64[M]')
        common = e if common is None else np.intersect1d(common, e)
    if common is None or len(common) == 0:
        return [], None, None

    # filter each model's arrays to common ends
    aligned = []
    for d in val_stack:
        e = d["ends_val"].astype('datetime64[M]')
        mask = np.isin(e, common)
        if mask.sum() == 0: 
            continue
        aligned.append({
            "y_vec": d["y_vec"][mask],
            "y_cum": d["y_cum"][mask],
            "y_cls": d["y_cls"][mask],
            "ends_val": d["ends_val"][mask],
        })

    # filter reference labels to common ends
    yv_ref, yc_ref, yk_ref, ends_ref = ref_labels
    e_ref = ends_ref.astype('datetime64[M]')
    mref = np.isin(e_ref, common)
    if mref.sum() == 0:
        return [], None, None
    ref_aligned = (yv_ref[mref], yc_ref[mref], yk_ref[mref], ends_ref[mref])
    return aligned, ref_aligned, common

def make_meta_features(stack_list, cum_horizons):
    """
    For each horizon, aggregate across models with [mean, median, min, max, std]
    over y_cum, vecsum, y_cls. Returns list of (Nv, 15) arrays, one per horizon.
    """
    feats_by_h = []
    for k, H in enumerate(cum_horizons):
        per_m_ycum = [d["y_cum"][:, k] for d in stack_list]             # (Nv,) lists
        per_m_vecs = [d["y_vec"][:, :H].sum(axis=1) for d in stack_list]
        per_m_ycls = [d["y_cls"][:, k] for d in stack_list]

        A = np.stack(per_m_ycum, axis=1)  # (Nv, M)
        B = np.stack(per_m_vecs, axis=1)  # (Nv, M)
        C = np.stack(per_m_ycls, axis=1)  # (Nv, M)

        def agg(M):
            return np.column_stack([
                M.mean(axis=1), np.median(M, axis=1),
                M.min(axis=1), M.max(axis=1), M.std(axis=1) + 1e-9
            ])
        Xk = np.column_stack([agg(A), agg(B), agg(C)])  # (Nv, 15)
        feats_by_h.append(Xk)
    return feats_by_h

# =========================
# Walk-forward with ensemble
# =========================
def walk_forward_ensemble(close_m, feats, lookbacks, seeds, h_vec, cum_horizons,
                          rmax, use_focal, focal_gamma, min_fold_windows=20, val_tail_windows=DEFAULT_VAL_TAIL_WINDOWS, debug=1):
    feats_full = feats
    if feats_full is None or len(feats_full) < 24:
        return None
    if debug:
        print(f"   [feat] rows={len(feats_full)}  cols={list(feats_full.columns)}")

    max_lb = max(lookbacks)
    max_h = max(cum_horizons + [h_vec])

    win_ref = to_windows(feats_full, max_lb, h_vec, cum_horizons)
    if win_ref is None:
        return None
    _, _, _, _, ends_ref = win_ref
    ends_sorted = np.sort(np.unique(ends_ref))
    cuts = []
    for i in range(min_fold_windows, len(ends_sorted) - 1):
        if (i - min_fold_windows) % h_vec == 0:
            cuts.append(ends_sorted[i])
    if debug: print(f"   [wf] #cuts={len(cuts)}")
    if len(cuts) == 0:
        return None

    folds = 0
    all_price_true, all_price_pred = [], []
    all_cumhead, all_vecsum, all_cls, all_meta = [], [], [], []

    for cut in cuts:
        feats_cut = feats_full.loc[:cut]
        if len(feats_cut) < max_lb + max_h: 
            continue
        sc = StandardScaler().fit(feats_cut.values)

        per_model = []
        val_stack = []
        ref_labels = None

        for lb in lookbacks:
            win = to_windows(feats_full, lb, h_vec, cum_horizons)
            if win is None: 
                continue
            X_all, yv_all, yc_all, yk_all, ends_all = win

            mask_train = ends_all <= cut
            X_all, yv_all, yc_all, yk_all, ends_all = X_all[mask_train], yv_all[mask_train], yc_all[mask_train], yk_all[mask_train], ends_all[mask_train]
            if len(X_all) < (min_fold_windows + 8): 
                continue

            def scale_seq(Xraw, sc):
                N,L,F = Xraw.shape
                if N == 0: 
                    return Xraw
                flat = Xraw.reshape(N*L, F)
                flat = sc.transform(flat)
                return flat.reshape(N, L, F)

            X_all_s = scale_seq(X_all, sc)

            ord_idx = np.argsort(ends_all.astype('datetime64[M]'))
            X_all_s, yv_all, yc_all, yk_all, ends_all = X_all_s[ord_idx], yv_all[ord_idx], yc_all[ord_idx], yk_all[ord_idx], ends_all[ord_idx]

            vstart = max(1, len(X_all_s) - val_tail_windows)
            X_tr, X_val = X_all_s[:vstart], X_all_s[vstart:]
            yv_tr, yv_val = yv_all[:vstart], yv_all[vstart:]
            yc_tr, yc_val = yc_all[:vstart], yc_all[vstart:]
            yk_tr, yk_val = yk_all[:vstart], yk_all[vstart:]
            ends_val = ends_all[vstart:]

            if len(X_val) == 0: 
                continue

            if ref_labels is None:
                ref_labels = (yv_val.copy(), yc_val.copy(), yk_val.copy(), ends_val.copy())

            feat_dim = X_tr.shape[-1]
            for sd in seeds:
                mdl = fit_one_config(X_tr, yv_tr, yc_tr, yk_tr, X_val, yv_val, yc_val, yk_val,
                                     lb, feat_dim, h_vec, len(cum_horizons),
                                     rmax, use_focal, focal_gamma, sd)
                y_vec_v, y_cum_v, y_cls_v = predict_mc_mean(mdl, X_val, mc_samples=10)
                val_stack.append({
                    "lb": lb, "seed": sd,
                    "y_vec": y_vec_v, "y_cum": y_cum_v, "y_cls": y_cls_v,
                    "ends_val": ends_val.copy()
                })
                per_model.append({"lb": lb, "seed": sd, "model": mdl, "scaler": sc, "feat_dim": feat_dim})

        # align all model validation preds to common ends
        val_stack, ref_labels_aligned, common = align_by_common_ends(val_stack, ref_labels) if ref_labels else ([], None, None)
        if len(val_stack) == 0 or ref_labels_aligned is None:
            continue

        yv_val_ref, yc_val_ref, yk_val_ref, ends_ref_val = ref_labels_aligned
        Hs = cum_horizons

        # === Meta calibrators from aligned validation ===
        X_meta_by_h = make_meta_features(val_stack, Hs)
        y_true_bin = (yc_val_ref > 0).astype(int)

        calibrators = []
        for k in range(len(Hs)):
            Xk = X_meta_by_h[k]
            yk = y_true_bin[:, k]
            if yk.min() == yk.max():
                lr = LogisticRegression()
                lr.coef_ = np.zeros((1, Xk.shape[1])); lr.intercept_ = np.zeros(1)
                lr.classes_ = np.array([0,1])
            else:
                lr = LogisticRegression(max_iter=300, solver="lbfgs")
                lr.fit(Xk, yk)
            calibrators.append(lr)

        # === Test prediction at this cut (one point) ===
        feats_up_to_cut = feats_full.loc[:cut]
        if len(feats_up_to_cut) < max_lb: 
            continue
        per_model_pred = []
        for m in per_model:
            lb = m["lb"]; sc2 = m["scaler"]; mdl = m["model"]
            X_last = feats_up_to_cut.iloc[-lb:].values.reshape(1, lb, -1)
            flat = X_last.reshape(lb, -1)
            flat = sc2.transform(flat)
            X_last = flat.reshape(1, lb, -1)
            y_vec_p, y_cum_p, y_cls_p = predict_mc_mean(mdl, X_last, mc_samples=MC_SAMPLES)
            per_model_pred.append({"lb": lb, "seed": m["seed"], "y_vec": y_vec_p, "y_cum": y_cum_p, "y_cls": y_cls_p})

        # Ensemble path (mean y_vec across models)
        y_vec_stack = np.stack([d["y_vec"].reshape(-1) for d in per_model_pred], axis=0)
        y_vec_ens = y_vec_stack.mean(axis=0)

        # Meta features for the single test point
        meta_test = []
        for k, H in enumerate(Hs):
            A = np.array([d["y_cum"].reshape(-1)[k] for d in per_model_pred])
            B = np.array([d["y_vec"].reshape(-1)[:H].sum() for d in per_model_pred])
            C = np.array([d["y_cls"].reshape(-1)[k] for d in per_model_pred])
            def agg(v): return np.array([v.mean(), np.median(v), v.min(), v.max(), v.std()+1e-9])
            meta_test.append(np.concatenate([agg(A), agg(B), agg(C)]))
        meta_probs = np.array([calibrators[k].predict_proba(meta_test[k].reshape(1, -1))[:,1][0]
                               for k in range(len(Hs))], dtype=float)

        # Ground truth next 12m price path for metrics
        r_all = feats_full["r"].values
        cut_pos = feats_full.index.get_loc(cut)
        r_future_true = r_all[cut_pos+1:cut_pos+1+h_vec]

        # P0: nearest close at/just before cut
        if cut in close_m.index:
            idx_pos = close_m.index.get_indexer_for([cut])[0]
        else:
            idx_pos = close_m.index.get_indexer([cut], method="pad")[0]
        P0 = float(close_m.iloc[idx_pos])

        pred_path = P0 * np.exp(np.cumsum(y_vec_ens))
        true_path = P0 * np.exp(np.cumsum(r_future_true))

        cum_true = np.array([r_all[cut_pos+1:cut_pos+1+H].sum() for H in Hs])
        y_cum_mean = np.mean([d["y_cum"].reshape(-1) for d in per_model_pred], axis=0)
        y_vecsum_mean = np.array([y_vec_ens[:H].sum() for H in Hs])
        y_cls_mean = np.mean([d["y_cls"].reshape(-1) for d in per_model_pred], axis=0)

        s_cum  = (np.sign(cum_true) == np.sign(y_cum_mean)).astype(float)
        s_vec  = (np.sign(cum_true) == np.sign(y_vecsum_mean)).astype(float)
        s_cls  = ((y_cls_mean >= 0.5).astype(int) == (cum_true > 0).astype(int)).astype(float)
        s_meta = ((meta_probs >= 0.5).astype(int) == (cum_true > 0).astype(int)).astype(float)

        all_price_true.append(true_path)
        all_price_pred.append(pred_path)
        all_cumhead.append(s_cum)
        all_vecsum.append(s_vec)
        all_cls.append(s_cls)
        all_meta.append(s_meta)
        folds += 1

    if folds == 0:
        return None

    price_true = np.vstack(all_price_true)
    price_pred = np.vstack(all_price_pred)

    da_cum = np.mean(np.vstack(all_cumhead), axis=0)
    da_vec = np.mean(np.vstack(all_vecsum), axis=0)
    da_cls = np.mean(np.vstack(all_cls), axis=0)
    da_meta= np.mean(np.vstack(all_meta), axis=0)
    da_best= np.maximum.reduce([da_cum, da_vec, da_cls, da_meta])

    metrics = {
        "rmse_price_1y": rmse(price_true, price_pred),
        "mape_price_1y": mape_price(price_true, price_pred),
        "directional_accuracy_from_cumhead": [float(x) for x in da_cum],
        "directional_accuracy_from_vecsum":  [float(x) for x in da_vec],
        "directional_accuracy_from_cls":     [float(x) for x in da_cls],
        "directional_accuracy_from_meta":    [float(x) for x in da_meta],
        "directional_accuracy_best":         [float(x) for x in da_best],
        "cum_horizons_used": [int(x) for x in cum_horizons],
        "h_vec_used": int(h_vec),
        "lookbacks_used": lookbacks,
        "seeds_used": seeds,
        "num_folds": int(folds),
        "mode": "ensemble_walk_forward"
    }
    return metrics

# =========================
# Final forecast with ensemble
# =========================
def final_forecast_ensemble(close_m, feats, lookbacks, seeds, h_vec, cum_horizons,
                            rmax, use_focal, focal_gamma, val_tail_windows=DEFAULT_VAL_TAIL_WINDOWS, debug=1):
    if feats is None or len(feats) < max(lookbacks) + max(cum_horizons + [h_vec]):
        return None
    sc = StandardScaler().fit(feats.values)

    # base windows for max lookback to define reference validation labels/ends
    win_ref = to_windows(feats, max(lookbacks), h_vec, cum_horizons)
    if win_ref is None: return None
    X_all_ref, yv_all_ref, yc_all_ref, yk_all_ref, ends_ref = win_ref
    ord_idx = np.argsort(ends_ref.astype('datetime64[M]'))
    X_all_ref, yv_all_ref, yc_all_ref, yk_all_ref, ends_ref = X_all_ref[ord_idx], yv_all_ref[ord_idx], yc_all_ref[ord_idx], yk_all_ref[ord_idx], ends_ref[ord_idx]

    # time split ensuring at least 1 val sample, and not exceeding n-1
    n_all = len(X_all_ref)
    split = int(max(1, min(n_all - 1, max(40, n_all * 0.85))))

    def scale_seq(X_raw):
        N,L,F = X_raw.shape
        if N == 0:
            return X_raw
        flat = X_raw.reshape(N*L, F)
        flat = sc.transform(flat)
        return flat.reshape(N, L, F)

    X_tr_ref, X_val_ref = X_all_ref[:split], X_all_ref[split:]
    yv_tr_ref, yv_val_ref = yv_all_ref[:split], yv_all_ref[split:]
    yc_tr_ref, yc_val_ref = yc_all_ref[:split], yc_all_ref[split:]
    yk_tr_ref, yk_val_ref = yk_all_ref[:split], yk_all_ref[split:]
    ends_val_ref = ends_ref[split:]

    X_tr_ref, X_val_ref = scale_seq(X_tr_ref), scale_seq(X_val_ref)

    members = []
    val_stack = []

    # Train members and collect validation predictions WITH their ends for alignment
    for lb in lookbacks:
        win_lb = to_windows(feats, lb, h_vec, cum_horizons)
        if win_lb is None: 
            continue
        X_all_lb, yv_all_lb, yc_all_lb, yk_all_lb, ends_lb = win_lb
        ord_idx = np.argsort(ends_lb.astype('datetime64[M]'))
        X_all_lb, yv_all_lb, yc_all_lb, yk_all_lb, ends_lb = X_all_lb[ord_idx], yv_all_lb[ord_idx], yc_all_lb[ord_idx], yk_all_lb[ord_idx], ends_lb[ord_idx]

        # create split index in "relative time" consistent with ref split proportion
        # we use the same proportion and minimums
        n_lb = len(X_all_lb)
        split_lb = int(max(1, min(n_lb - 1, max(40, n_lb * 0.85))))
        X_tr_lb, X_val_lb = X_all_lb[:split_lb], X_all_lb[split_lb:]
        yv_tr_lb, yv_val_lb = yv_all_lb[:split_lb], yv_all_lb[split_lb:]
        yc_tr_lb, yc_val_lb = yc_all_lb[:split_lb], yc_all_lb[split_lb:]
        yk_tr_lb, yk_val_lb = yk_all_lb[:split_lb], yk_all_lb[split_lb:]
        ends_val_lb = ends_lb[split_lb:]

        X_tr_lb, X_val_lb = scale_seq(X_tr_lb), scale_seq(X_val_lb)
        if len(X_val_lb) == 0:
            continue

        feat_dim = X_tr_lb.shape[-1]
        for sd in seeds:
            mdl = fit_one_config(X_tr_lb, yv_tr_lb, yc_tr_lb, yk_tr_lb,
                                 X_val_lb, yv_val_lb, yc_val_lb, yk_val_lb,
                                 lb, feat_dim, h_vec, len(cum_horizons),
                                 rmax, use_focal, focal_gamma, sd)
            y_vec_v, y_cum_v, y_cls_v = predict_mc_mean(mdl, X_val_lb, mc_samples=10)
            val_stack.append({"lb": lb, "seed": sd, "y_vec": y_vec_v, "y_cum": y_cum_v, "y_cls": y_cls_v, "ends_val": ends_val_lb})
            members.append({"lb": lb, "seed": sd, "model": mdl})

    if len(members) == 0 or len(val_stack) == 0:
        return None

    # align all members to the intersection of validation end dates
    val_stack_aligned, ref_labels_aligned, common_ends = align_by_common_ends(
        val_stack, (yv_val_ref, yc_val_ref, yk_val_ref, ends_val_ref)
    )
    if len(val_stack_aligned) == 0 or ref_labels_aligned is None:
        return None

    yv_val_ref, yc_val_ref, yk_val_ref, ends_val_ref = ref_labels_aligned
    Hs = cum_horizons

    # Meta calibrators
    X_meta_by_h = make_meta_features(val_stack_aligned, Hs)
    y_true_bin = (yc_val_ref > 0).astype(int)
    calibrators = []
    for k in range(len(Hs)):
        Xk = X_meta_by_h[k]
        yk = y_true_bin[:, k]
        if yk.min() == yk.max():
            lr = LogisticRegression()
            lr.coef_ = np.zeros((1, Xk.shape[1])); lr.intercept_ = np.zeros(1)
            lr.classes_ = np.array([0,1])
        else:
            lr = LogisticRegression(max_iter=300, solver="lbfgs")
            lr.fit(Xk, yk)
        calibrators.append(lr)

    # Final ensemble forecast at last date
    last_idx = feats.index[-1]
    P0 = float(close_m.iloc[-1])
    dates = pd.date_range(last_idx + pd.offsets.MonthEnd(), periods=h_vec, freq="M")

    per_pred = []
    for m in members:
        lb = m["lb"]; mdl = m["model"]
        X_last = feats.iloc[-lb:].values.reshape(1, lb, -1)
        flat = X_last.reshape(lb, -1)
        flat = sc.transform(flat)
        X_last = flat.reshape(1, lb, -1)
        y_vec_p, y_cum_p, y_cls_p = predict_mc_mean(mdl, X_last, mc_samples=MC_SAMPLES)
        per_pred.append({"lb": lb, "seed": m["seed"], "y_vec": y_vec_p, "y_cum": y_cum_p, "y_cls": y_cls_p})

    # Price bands from ensemble of y_vec
    paths = []
    for d in per_pred:
        rvec = d["y_vec"].reshape(-1)
        paths.append(P0 * np.exp(np.cumsum(rvec)))
    paths = np.stack(paths, axis=0)  # (M, H)
    band_df = pd.DataFrame({
        "Date": dates,
        "P10": np.percentile(paths, 10, axis=0),
        "P50": np.percentile(paths, 50, axis=0),
        "P90": np.percentile(paths, 90, axis=0),
    })

    # Meta probability today
    meta_test = []
    for k, H in enumerate(Hs):
        A = np.array([d["y_cum"].reshape(-1)[k] for d in per_pred])
        B = np.array([d["y_vec"].reshape(-1)[:H].sum() for d in per_pred])
        C = np.array([d["y_cls"].reshape(-1)[k] for d in per_pred])
        def agg(v): return np.array([v.mean(), np.median(v), v.min(), v.max(), v.std()+1e-9])
        meta_test.append(np.concatenate([agg(A), agg(B), agg(C)]))
    meta_prob = np.array([calibrators[k].predict_proba(meta_test[k].reshape(1,-1))[:,1][0]
                          for k in range(len(Hs))], dtype=float)

    # Raw prob_up (mean of y_cls)
    raw_prob = np.mean([d["y_cls"].reshape(-1) for d in per_pred], axis=0)

    return band_df, meta_prob, raw_prob

# =========================
# Runner
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True, help="e.g., EMAAR.AE")
    ap.add_argument("--outdir", default="stock_prediction_ens")
    ap.add_argument("--period", default="15y")
    ap.add_argument("--lookbacks", type=str, default="36,48,54")
    ap.add_argument("--seeds", type=str, default="11,22,33")
    ap.add_argument("--hvec", type=int, default=DEFAULT_HVEC)
    ap.add_argument("--horizons", type=str, default="12,24,60")
    ap.add_argument("--rmax", type=float, default=DEFAULT_RMAX)
    ap.add_argument("--use_focal", type=int, default=USE_FOCAL_DEFAULT)
    ap.add_argument("--focal_gamma", type=float, default=FOCAL_GAMMA_DEFAULT)
    ap.add_argument("--exog", type=str, default="BZ=F,^GSPC,^VIX,DX-Y.NYB,EEM")
    ap.add_argument("--min_fold_windows", type=int, default=20)
    ap.add_argument("--val_tail_windows", type=int, default=DEFAULT_VAL_TAIL_WINDOWS)
    ap.add_argument("--forecast_only", type=int, default=0, help="Skip backtest for speed (1=yes)")
    ap.add_argument("--debug", type=int, default=1)
    args = ap.parse_args()

    lookbacks = [int(x) for x in args.lookbacks.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    cum_horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    exog_list = [x.strip() for x in args.exog.split(",") if x.strip()]

    os.makedirs(f"{args.outdir}/predictions", exist_ok=True)
    os.makedirs(f"{args.outdir}/reports", exist_ok=True)

    print(f"→ {args.ticker}")
    cm = monthly_close(args.ticker, period=args.period, debug=args.debug)
    if cm is None:
        print("   [skip] no data"); return

    feats = compute_features_from_close(cm)
    if feats is None:
        print("   [skip] features empty"); return
    feats = add_exog(feats, exog_list, period=args.period, debug=args.debug)
    if args.debug:
        print(f"   [feat] rows={len(feats)}  cols={list(feats.columns)}")
    if len(feats) < max(lookbacks) + max(cum_horizons + [args.hvec]) + 12:
        print("   [skip] not enough usable history"); return

    # BACKTEST (out-of-sample)
    if not args.forecast_only:
        metrics = walk_forward_ensemble(cm, feats, lookbacks, seeds, args.hvec, cum_horizons,
                                    args.rmax, args.use_focal, args.focal_gamma,
                                    min_fold_windows=args.min_fold_windows,
                                    val_tail_windows=args.val_tail_windows,
                                    debug=args.debug)
    else:
        print("   [info] forecast_only=1 → skipping backtest.", flush=True)
    if metrics is None:
        print("   [warn] walk-forward not feasible with current settings.")
    else:
        safe = args.ticker.replace(".", "_")
        with open(f"{args.outdir}/reports/{safe}_ens_eval.json", "w") as f:
            json.dump({"metrics": metrics}, f, indent=2)
        print(f"   [saved] reports/{safe}_ens_eval.json")

    # FINAL FORECAST (today → future)
    res = final_forecast_ensemble(cm, feats, lookbacks, seeds, args.hvec, cum_horizons,
                                  args.rmax, args.use_focal, args.focal_gamma,
                                  val_tail_windows=args.val_tail_windows,
                                  debug=args.debug)
    if res is None:
        print("   [warn] final forecast not produced.")
    else:
        band_df, meta_prob, raw_prob = res
        safe = args.ticker.replace(".", "_")
        band_df.to_csv(f"{args.outdir}/predictions/{safe}_ens_bands_{args.hvec}m.csv", index=False)
        out = {
            "prob_up_by_horizon_meta": {f"{h}m": float(meta_prob[i]) for i,h in enumerate(cum_horizons)},
            "prob_up_by_horizon_raw":  {f"{h}m": float(raw_prob[i])  for i,h in enumerate(cum_horizons)}
        }
        with open(f"{args.outdir}/reports/{safe}_ens_forecast.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"   [saved]\n     predictions/{safe}_ens_bands_{args.hvec}m.csv\n     reports/{safe}_ens_forecast.json")

if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.get_logger().setLevel("ERROR")
    main()