# src/forecast_tabular.py
import os, json, argparse, warnings, time
warnings.filterwarnings("ignore")

from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_percentage_error, brier_score_loss, roc_curve
from sklearn.isotonic import IsotonicRegression

# Try LightGBM; fall back to sklearn if not available
HAS_LGBM = True
try:
    from lightgbm import LGBMRegressor
except Exception:
    HAS_LGBM = False
    from sklearn.ensemble import HistGradientBoostingRegressor

# Try SciPy sparse (preferred); fall back to dense hstack if unavailable
HAS_SP = True
try:
    import scipy.sparse as sp
except Exception:
    HAS_SP = False

RANDOM_SEED = 1337
np.random.seed(RANDOM_SEED)

# =========================
# Utilities / metrics
# =========================
def rmse(a, b) -> float:
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def smape(a, b) -> float:
    a = np.asarray(a); b = np.asarray(b)
    den = (np.abs(a) + np.abs(b)) / 2.0
    z = np.where(den == 0, 0.0, np.abs(a - b) / den)
    return float(np.mean(z) * 100.0)

def rmse_price(true_paths: np.ndarray, pred_paths: np.ndarray) -> float:
    return rmse(true_paths, pred_paths)

def mape_price(true_paths: np.ndarray, pred_paths: np.ndarray) -> float:
    return float(mean_absolute_percentage_error(true_paths, pred_paths) * 100.0)

def smape_price(true_paths: np.ndarray, pred_paths: np.ndarray) -> float:
    return smape(true_paths, pred_paths)

def robust_sigma(x: np.ndarray) -> float:
    """Robust std estimate via IQR (~1.349 * sigma for normal)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.1
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    return float(max(1e-6, iqr / 1.349))

def prob_shrink(p: np.ndarray, p_base: float, shrink: float) -> np.ndarray:
    """Blend probability toward base rate: p' = (1-s)*p + s*p_base."""
    s = np.clip(float(shrink), 0.0, 1.0)
    return (1.0 - s) * p + s * float(p_base)

def make_gbt_reg(random_state: int):
    """Small-sample-friendly GBT settings."""
    if HAS_LGBM:
        return LGBMRegressor(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=7,
            max_depth=3,
            min_data_in_leaf=6,
            min_gain_to_split=0.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="regression",
            force_col_wise=True,
            verbosity=-1,
            random_state=random_state,
        )
    else:
        return HistGradientBoostingRegressor(
            max_depth=3,
            max_iter=600,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=random_state,
        )

def fit_with_fallback(model, X, y, label: str = "", debug: int = 1):
    model.fit(X, y)
    try:
        yhat = model.predict(X)
    except Exception:
        yhat = np.array([np.nan])
    bad = (not np.isfinite(yhat).all()) or (np.std(yhat) < 1e-6)
    if bad:
        if debug:
            print(f"   [fit] {label}: degenerate tree model → Ridge fallback")
        try:
            lin = Ridge(alpha=1.0, random_state=RANDOM_SEED)
        except TypeError:
            lin = Ridge(alpha=1.0)
        lin.fit(X, y)
        return lin
    return model

# =========================
# Panel intercepts (fixed effects)
# =========================
class PanelInterceptEncoder:
    """
    Adds per-ticker fixed effects (one-hot with drop='first') to base X.
    Fit on TRAIN ONLY inside each panel fold.
    """
    def __init__(self, add_for_classifier=True, add_for_regressor=False, fe_scale: float = 0.5, dtype=np.float32):
        self.add_for_classifier = add_for_classifier
        self.add_for_regressor  = add_for_regressor
        self.fe_scale = float(fe_scale)
        self.dtype = dtype
        self.ohe_ = None

    def fit(self, tickers_train: List[str]):
        self.ohe_ = OneHotEncoder(
            handle_unknown="ignore",
            drop="first",
            sparse_output=True if "sparse_output" in OneHotEncoder.__init__.__code__.co_varnames else True,
            dtype=self.dtype
        )
        self.ohe_.fit(np.asarray(tickers_train).reshape(-1,1))
        return self

    def _hstack(self, X_base, Z):
        if HAS_SP:
            Xb = X_base if sp.issparse(X_base) else sp.csr_matrix(X_base)
            Zc = Z if sp.issparse(Z) else sp.csr_matrix(Z)
            if self.fe_scale != 1.0:
                Zc = Zc.multiply(self.fe_scale)
            return sp.hstack([Xb, Zc], format="csr")
        Xd = np.asarray(X_base)
        Zd = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
        if self.fe_scale != 1.0:
            Zd = Zd * self.fe_scale
        return np.hstack([Xd, Zd])

    def transform(self, tickers, X_base, for_role="classifier"):
        use = (for_role=="classifier" and self.add_for_classifier) or \
              (for_role=="regressor"  and self.add_for_regressor)
        if not use:
            return X_base
        Z = self.ohe_.transform(np.asarray(tickers).reshape(-1,1))
        return self._hstack(X_base, Z)

# =========================
# Data: yfinance → monthly
# =========================
def _coerce_close_series(close_obj):
    if isinstance(close_obj, pd.Series):
        return close_obj
    if isinstance(close_obj, pd.DataFrame):
        if close_obj.shape[1] == 1:
            try:
                return close_obj.squeeze("columns")
            except Exception:
                return close_obj.iloc[:, 0]
        col = close_obj.notna().sum().sort_values(ascending=False).index[0]
        return close_obj[col]
    return pd.Series(close_obj)

def monthly_close(ticker: str, period="max", debug=1) -> Optional[pd.Series]:
    df = None
    for k in range(2):
        try:
            df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
            break
        except Exception as e:
            if debug: print(f"   [warn] yfinance error ({e}), retry {k+1}/2...")
            time.sleep(2)
    if df is None or df.empty:
        if debug: print(f"   [data] No data for {ticker} (period={period})")
        return None
    if "Close" not in df.columns:
        if debug: print(f"   [data] no 'Close' column for {ticker}")
        return None

    s = _coerce_close_series(df["Close"]).copy()
    s.index = pd.to_datetime(s.index)
    try:
        s = pd.to_numeric(s, errors="coerce")
    except TypeError:
        s = pd.to_numeric(s.iloc[:, 0], errors="coerce")
    s = s.dropna()

    m = s.resample("M").last().dropna().astype(float)
    m = m[~m.index.duplicated(keep="last")].sort_index()
    m.index = m.index.to_period("M")
    if debug and len(m) > 0:
        print(f"   [data] {ticker}: {len(m)} monthly bars ({m.index.min().to_timestamp().date()} → {m.index.max().to_timestamp().date()})")
    return m if len(m) else None

# =========================
# Features
# =========================
def compute_features_from_close(close_m: pd.Series) -> Optional[pd.DataFrame]:
    if not isinstance(close_m, pd.Series):
        close_m = pd.Series(close_m)
    close_m = close_m.dropna()
    if len(close_m) < 18:
        return None

    if not isinstance(close_m.index, pd.PeriodIndex):
        close_m.index = close_m.index.to_period("M")

    logp = np.log(close_m.astype(float))
    r = logp.diff().rename("r")

    mo = close_m.index.month
    sin_m = pd.Series(np.sin(2*np.pi*mo/12), index=close_m.index, name="sin_m")
    cos_m = pd.Series(np.cos(2*np.pi*mo/12), index=close_m.index, name="cos_m")

    mom3  = r.rolling(3,  min_periods=3).mean().rename("mom3")
    mom6  = r.rolling(6,  min_periods=6).mean().rename("mom6")
    mom12 = r.rolling(12, min_periods=12).mean().rename("mom12")
    vol12 = r.rolling(12, min_periods=12).std().rename("vol12")
    mom12_cum = r.rolling(12, min_periods=12).sum().rename("mom12_cum")

    z6  = (r - r.rolling(6,  min_periods=6).mean()) / (r.rolling(6,  min_periods=6).std() + 1e-9)
    z12 = (r - r.rolling(12, min_periods=12).mean()) / (r.rolling(12, min_periods=12).std() + 1e-9)
    z6.name = "z6"; z12.name = "z12"

    cum = (1.0 + r.fillna(0)).cumprod()
    peak = cum.cummax()
    dd = (cum / peak - 1.0).rename("dd12")
    up_vol12 = r.clip(lower=0).rolling(12, min_periods=12).std().rename("up_vol12")
    dn_vol12 = (-r.clip(upper=0)).rolling(12, min_periods=12).std().rename("dn_vol12")

    feats = pd.concat(
        [r, mom3, mom6, mom12, mom12_cum, vol12, z6, z12, dd, up_vol12, dn_vol12, sin_m, cos_m],
        axis=1
    ).dropna()
    return feats if not feats.empty else None

def add_exog(feats_df: pd.DataFrame, exog_list: List[str], period="max", debug=1) -> pd.DataFrame:
    if not exog_list:
        return feats_df
    out = feats_df.copy()
    for t in exog_list:
        t = t.strip()
        if not t:
            continue
        try:
            em = monthly_close(t, period=period, debug=0)
            if em is None or len(em) < 24:
                if debug: print(f"   [exog] skip {t}: insufficient")
                continue
            rex = np.log(em).diff().rename(f"exog_{t}_r")
            mom3  = rex.rolling(3,  min_periods=3).mean().rename(f"exog_{t}_mom3")
            mom6  = rex.rolling(6,  min_periods=6).mean().rename(f"exog_{t}_mom6")
            mom12 = rex.rolling(12, min_periods=12).mean().rename(f"exog_{t}_mom12")
            out = out.join(pd.concat([rex, mom3, mom6, mom12], axis=1), how="left")
        except Exception as e:
            if debug: print(f"   [exog] {t} error: {e}")
    return out.dropna()

# =========================
# Samples (tabular windows)
# =========================
def _window_agg(past: pd.DataFrame) -> np.ndarray:
    cols = past.columns
    feats = []
    idx = np.arange(len(past), dtype=float)
    if len(idx) > 1:
        idx = (idx - idx.mean()) / (idx.std() + 1e-9)
    for c in cols:
        v = past[c].values
        last = v[-1]
        mean = v.mean()
        std  = v.std()
        mn   = v.min()
        mx   = v.max()
        slope = np.polyfit(idx, v, 1)[0] if len(v) > 1 else 0.0
        feats.extend([last, mean, std, mn, mx, slope])
        if c == "r":
            for k in [1,2,3]:
                feats.append(v[-k] if len(v) >= k else 0.0)
    return np.asarray(feats, dtype=float)

def make_tabular_samples(feats_df: pd.DataFrame, lookback: int, horizons: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_h = max(horizons)
    T = len(feats_df)
    if T < lookback + max_h + 1:
        return np.empty((0,0)), np.empty((0,0)), np.empty((0,0)), np.array([], dtype="datetime64[M]")

    X_rows, Y_cum_rows, Y_up_rows, ends = [], [], [], []
    r = feats_df["r"].values
    idx_periods = feats_df.index
    for end_pos in range(lookback - 1, T - max_h):
        past = feats_df.iloc[end_pos - lookback + 1 : end_pos + 1]
        x = _window_agg(past)
        ycum = []
        yup  = []
        for H in horizons:
            fut = r[end_pos + 1 : end_pos + 1 + H]
            if len(fut) != H:
                ycum.append(0.0); yup.append(0.0)
            else:
                csum = float(fut.sum())
                ycum.append(csum)
                yup.append(1.0 if csum > 0 else 0.0)
        X_rows.append(x)
        Y_cum_rows.append(ycum)
        Y_up_rows.append(yup)
        ends.append(idx_periods[end_pos])   # Period('M')
    X = np.vstack(X_rows)
    Y_c = np.vstack(Y_cum_rows)
    Y_u = np.vstack(Y_up_rows)
    ends = np.array(ends, dtype="object")   # store Periods
    return X, Y_c, Y_u, ends

# =========================
# Time weights & embargo
# =========================
def exponential_time_weights(ends: np.ndarray, cutoff_month: pd.Period, half_life_months: int) -> np.ndarray:
    ends_p = pd.PeriodIndex(list(ends), freq="M")
    cut_p = pd.Period(str(cutoff_month), freq="M")
    cut_ord = int(cut_p.ordinal)
    end_ord = ends_p.asi8
    age = (cut_ord - end_ord).astype(float)
    age = np.clip(age, 0, None)
    hl = max(1, int(half_life_months))
    lam = np.log(2) / hl
    w = np.exp(-lam * age)
    w = w / (w.sum() + 1e-9)
    return w

# =========================
# Calibration & thresholds
# =========================
def calibrate_probs(y_val: np.ndarray, p_val: np.ndarray, mode: str) -> Tuple[np.ndarray, str]:
    mode = (mode or "none").lower()
    if mode == "none":
        return p_val, "raw"
    best_tag = "raw"
    best = p_val
    best_brier = brier_score_loss(y_val, p_val) if np.unique(y_val).size == 2 else np.nan

    def _sigmoid(p):
        lr = LogisticRegression(max_iter=1000)
        X = p.reshape(-1, 1)
        try:
            lr.fit(X, y_val)
            return lr.predict_proba(X)[:,1]
        except Exception:
            return p

    def _isotonic(p):
        try:
            iso = IsotonicRegression(out_of_bounds="clip")
            return iso.fit_transform(p, y_val)
        except Exception:
            return p

    cands = [("sigmoid", _sigmoid(p_val)), ("isotonic", _isotonic(p_val))]
    for tag, pv in cands:
        if np.unique(y_val).size < 2 or not np.isfinite(pv).all():
            continue
        bs = brier_score_loss(y_val, pv)
        if np.isnan(best_brier) or bs < best_brier:
            best_brier = bs; best = pv; best_tag = tag
    if mode == "auto":
        return best, best_tag
    if mode == "sigmoid":
        return cands[0][1], "sigmoid"
    if mode == "isotonic":
        return cands[1][1], "isotonic"
    return p_val, "raw"

def _grid_balanced_threshold(y: np.ndarray, p: np.ndarray, floor=0.05, cap=0.95) -> Tuple[float, float]:
    if np.unique(y).size < 2:
        return 0.5, np.nan
    grid = np.linspace(floor, cap, 19)
    best_t, best_s = 0.5, -1.0
    for t in grid:
        yhat = (p >= t).astype(int)
        tp = ((yhat==1) & (y==1)).sum()
        tn = ((yhat==0) & (y==0)).sum()
        fp = ((yhat==1) & (y==0)).sum()
        fn = ((yhat==0) & (y==1)).sum()
        tpr = tp / (tp+fn+1e-9); tnr = tn / (tn+fp+1e-9)
        s = 0.5*(tpr+tnr)
        if s > best_s:
            best_s, best_t = s, t
    return float(best_t), float(best_s)

def tune_threshold_balanced(y_true: np.ndarray, p_up: np.ndarray,
                            sample_weight=None, floor=0.05, cap=0.95) -> Tuple[float, float]:
    y = np.asarray(y_true); p = np.asarray(p_up)
    if np.unique(y).size < 2:
        return 0.5, np.nan
    fpr, tpr, thr = roc_curve(y, p, sample_weight=sample_weight)
    ba = 0.5*(tpr + (1.0 - fpr))
    mask = np.isfinite(thr)
    if mask.any():
        idx = int(np.nanargmax(ba[mask]))
        t = float(thr[mask][idx])
        s = float(ba[mask][idx])
        if not np.isfinite(t):
            t, s = _grid_balanced_threshold(y, p, floor=floor, cap=cap)
    else:
        t, s = _grid_balanced_threshold(y, p, floor=floor, cap=cap)
    t = float(np.clip(t, floor, cap))
    return t, s

def tune_threshold_grid(y_val: np.ndarray, p_val: np.ndarray, metric: str = "balanced") -> Tuple[float, float]:
    if np.unique(y_val).size < 2:
        return 0.5, np.nan
    grid = np.linspace(0.05, 0.95, 19)
    best_t, best_s = 0.5, -1
    for t in grid:
        yhat = (p_val >= t).astype(int)
        tp = ((yhat==1) & (y_val==1)).sum()
        tn = ((yhat==0) & (y_val==0)).sum()
        fp = ((yhat==1) & (y_val==0)).sum()
        fn = ((yhat==0) & (y_val==1)).sum()
        if metric == "precision":
            s = tp / (tp+fp+1e-9)
        elif metric == "recall":
            s = tp / (tp+fn+1e-9)
        elif metric == "accuracy":
            s = (tp+tn) / (tp+tn+fp+fn+1e-9)
        else: # balanced accuracy
            tpr = tp / (tp+fn+1e-9); tnr = tn / (tn+fp+1e-9)
            s = 0.5*(tpr+tnr)
        if s > best_s:
            best_s, best_t = s, t
    return float(best_t), float(best_s)

# =========================
# Group scaling helpers
# =========================
def fit_group_scalers(X: np.ndarray, tickers: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    d = {}
    uniq = np.unique(tickers)
    for tk in uniq:
        m = (tickers == tk)
        Xg = X[m]
        mu = Xg.mean(axis=0)
        sd = Xg.std(axis=0)
        sd[sd < 1e-9] = 1.0
        d[str(tk)] = (mu, sd)
    return d

def transform_by_group(X: np.ndarray, tickers: np.ndarray,
                       d: Dict[str, Tuple[np.ndarray, np.ndarray]],
                       mu_global: np.ndarray, sd_global: np.ndarray) -> np.ndarray:
    Xz = np.empty_like(X, dtype=float)
    for i, tk in enumerate(tickers):
        mu, sd = d.get(str(tk), (mu_global, sd_global))
        Xz[i] = (X[i] - mu) / sd
    return Xz

# =========================
# Walk-forward backtest (single ticker)
# =========================
def walk_forward_backtest(close_m: pd.Series, feats_df: pd.DataFrame, lookback: int, horizons: List[int],
                          min_fold_windows: int = 10, step: Optional[int] = None, val_tail: int = 12,
                          embargo: int = 0, half_life_months: int = 24,
                          prob_floor: float = 0.1, prob_cap: float = 0.9,
                          min_shrink: float = 0.25, calib_mode: str = "auto",
                          tune_thresh: Optional[str] = None, debug=1) -> Optional[Dict[str, Any]]:

    X, Yc, Yu, ends = make_tabular_samples(feats_df, lookback, horizons)
    if X.size == 0:
        if debug: print("   [wf] not enough samples after feature building")
        return None

    unique_ends = pd.PeriodIndex(sorted(set(list(ends))), freq="M")
    step = step or min(horizons)
    cuts: List[pd.Period] = []
    for i in range(min_fold_windows, len(unique_ends)-1):
        if (i - min_fold_windows) % step == 0:
            cuts.append(unique_ends[i])
    if debug: print(f"   [wf] #cuts={len(cuts)}")
    if len(cuts) == 0:
        return None

    H_main = 12 if 12 in horizons else horizons[0]
    all_true_paths, all_pred_paths = [], []
    acc_cls_default = [[] for _ in horizons]
    acc_cls_tuned   = [[] for _ in horizons]
    resid_pool_per_h = [[] for _ in horizons]

    P = close_m.copy()
    P.index = P.index.asfreq("M")

    for cut in cuts:
        mask_cut = pd.PeriodIndex(list(ends), freq="M") <= (cut - embargo)
        if mask_cut.sum() < min_fold_windows:
            if debug: print("   [wf] skip cut (train too small after embargo)")
            continue

        X_tr = X[mask_cut]
        Yc_tr, Yu_tr = Yc[mask_cut], Yu[mask_cut]
        ends_tr = ends[mask_cut]

        w_tr = exponential_time_weights(ends_tr, cutoff_month=cut, half_life_months=half_life_months)

        sc = StandardScaler().fit(X_tr)
        X_tr = sc.transform(X_tr)

        vt = min(val_tail, len(X_tr))
        X_val = X_tr[-vt:]
        Yc_val = Yc_tr[-vt:]
        Yu_val = Yu_tr[-vt:]

        regs_pos, regs_neg, clfs = [], [], []
        for k, H in enumerate(horizons):
            clf = LogisticRegression(max_iter=1200, class_weight="balanced", random_state=RANDOM_SEED + k)
            clf.fit(X_tr, Yu_tr[:, k], sample_weight=w_tr)
            clfs.append(clf)

            pos_mask = Yc_tr[:, k] > 0
            neg_mask = ~pos_mask
            n_pos, n_neg = int(pos_mask.sum()), int(neg_mask.sum())
            if n_pos < 5 or n_neg < 5:
                reg_all = fit_with_fallback(make_gbt_reg(RANDOM_SEED + k), X_tr, Yc_tr[:, k],
                                            label=f"H{H}_all", debug=debug)
                regs_pos.append(reg_all); regs_neg.append(reg_all)
            else:
                reg_pos = fit_with_fallback(make_gbt_reg(RANDOM_SEED + 100 + k), X_tr[pos_mask], Yc_tr[pos_mask, k],
                                            label=f"H{H}_pos", debug=debug)
                reg_neg = fit_with_fallback(make_gbt_reg(RANDOM_SEED + 200 + k), X_tr[neg_mask], Yc_tr[neg_mask, k],
                                            label=f"H{H}_neg", debug=debug)
                regs_pos.append(reg_pos); regs_neg.append(reg_neg)

            p_val = clfs[k].predict_proba(X_val)[:,1]
            p_val = np.clip(p_val, prob_floor, prob_cap)
            base_rate = float(np.mean(Yu_tr[-vt:, k])) if vt > 1 else float(np.mean(Yu_tr[:, k]))
            p_val = prob_shrink(p_val, base_rate, min_shrink)
            if calib_mode != "none":
                p_val, _ = calibrate_probs(Yu_val[:,k], p_val, "auto" if calib_mode=="auto" else calib_mode)
                p_val = np.clip(p_val, prob_floor, prob_cap)

            mu_pos = regs_pos[k].predict(X_val)
            mu_neg = regs_neg[k].predict(X_val)
            mu_pos = np.maximum(mu_pos, 0.0)
            mu_neg = np.minimum(mu_neg, 0.0)
            mu_exp = p_val * mu_pos + (1.0 - p_val)*mu_neg
            resid = (Yc_val[:, k] - mu_exp)
            resid_pool_per_h[k].extend(list(resid))

        sel = pd.PeriodIndex(list(ends), freq="M") == cut
        if sel.sum() == 0:
            continue
        x_test = sc.transform(X[sel])[-1].reshape(1, -1)

        try:
            P0 = float(P.loc[cut])
        except KeyError:
            P0 = float(P.iloc[P.index.get_indexer([cut], method="pad")][0])

        k_main = horizons.index(H_main)
        end_pos = np.where(pd.PeriodIndex(list(feats_df.index), freq="M") == cut)[0]
        if len(end_pos)==0:
            continue
        pos = end_pos[0]
        r_all = feats_df["r"].values
        true_r_future = r_all[pos+1: pos+1+H_main]

        p = float(clfs[k_main].predict_proba(x_test)[0,1])
        p = np.clip(p, prob_floor, prob_cap)
        base_rate = float(np.mean(Yu_tr[-vt:, k_main])) if vt > 1 else float(np.mean(Yu_tr[:, k_main]))
        p = float(prob_shrink(np.array([p]), base_rate, min_shrink)[0])

        y_true_up = int(true_r_future.sum() > 0)
        acc_cls_default[k_main].append(1.0 if (p>=0.5)==bool(y_true_up) else 0.0)

        if tune_thresh:
            pv = clfs[k_main].predict_proba(X_val)[:,1]
            pv = np.clip(pv, prob_floor, prob_cap)
            pv = prob_shrink(pv, base_rate, min_shrink)
            pv, _ = calibrate_probs(Yu_val[:,k_main], pv, calib_mode if calib_mode!="none" else "auto")
            pv = np.clip(pv, prob_floor, prob_cap)
            if tune_thresh == "balanced":
                best_t, _ = tune_threshold_balanced(Yu_val[:,k_main], pv, floor=prob_floor, cap=prob_cap)
            else:
                best_t, _ = tune_threshold_grid(Yu_val[:,k_main], pv, metric=tune_thresh)
            acc_cls_tuned[k_main].append(1.0 if (p>=best_t)==bool(y_true_up) else 0.0)

        mu_pos = float(regs_pos[k_main].predict(x_test)[0])
        mu_neg = float(regs_neg[k_main].predict(x_test)[0])
        mu_pos = max(mu_pos, 0.0)
        mu_neg = min(mu_neg, 0.0)

        mu_exp = p * mu_pos + (1.0 - p) * mu_neg
        pred_cum_monthly = np.linspace(mu_exp / H_main, mu_exp, num=H_main)
        pred_path = P0 * np.exp(pred_cum_monthly)

        true_path = P0 * np.exp(np.cumsum(true_r_future))
        all_true_paths.append(true_path)
        all_pred_paths.append(pred_path)

    if len(all_true_paths) == 0:
        return None

    price_true = np.vstack(all_true_paths)
    price_pred = np.vstack(all_true_paths if len(all_pred_paths)==0 else all_pred_paths)

    metrics = {
        "rmse_price_mainH": rmse_price(price_true, price_pred),
        "mape_price_mainH": mape_price(price_true, price_pred),
        "rmse_price_mainH_norm": rmse(price_true/ (price_true.mean(axis=1, keepdims=True)+1e-9),
                                      price_pred/(price_true.mean(axis=1, keepdims=True)+1e-9)),
        "smape_price_mainH_norm": smape_price(price_true, price_pred),
        "rmse_cum_return_mainH": float(rmse(np.log(price_true[:,-1]/price_true[:,0]),
                                            np.log(price_pred[:,-1]/price_true[:,0]))),
        "smape_cum_return_mainH": float(smape(np.log(price_true[:,-1]/price_true[:,0]),
                                              np.log(price_pred[:,-1]/price_true[:,0]))),
        "directional_accuracy_default": [float(np.mean(acc)) if len(acc) else None for acc in acc_cls_default],
        "directional_accuracy_tuned": [float(np.mean(acc)) if len(acc) else None for acc in acc_cls_tuned],
        "horizons_used": [int(h) for h in horizons],
        "lookback_used": int(lookback),
        "num_folds": int(len(all_true_paths)),
        "regressor": "LightGBM" if HAS_LGBM else "HGBR",
        "classifier": "LogReg (single) + sign-aware expected return"
    }
    resid_std = [robust_sigma(np.array(resid_pool_per_h[i], dtype=float)) for i,_ in enumerate(horizons)]
    metrics["residual_std_per_h"] = {f"{h}m": float(resid_std[i]) for i, h in enumerate(horizons)}
    return metrics

# =========================
# Bands helpers
# =========================
_Z_MAP = {0.10: 1.2815515655446004, 0.15: 1.0364333894937898, 0.20: 0.8416212335729143, 0.25: 0.6744897501960817}
def z_from_q(q: float) -> float:
    q = float(q)
    if q in _Z_MAP: return _Z_MAP[q]
    # fallback to normal inverse via approximation if needed
    from math import sqrt, log, pi
    # Beasley-Springer/Moro-ish rough inverse for central quantiles
    # map q to tail prob
    p = min(max(q, 1e-6), 0.499999)
    t = sqrt(-2.0*log(2.0*p))
    # crude polynomial
    z = t - (2.515517 + 0.802853*t + 0.010328*t*t)/(1 + 1.432788*t + 0.189269*t*t + 0.001308*t**3)
    return float(z)

# =========================
# Final forecast (single)
# =========================
def final_forecast(close_m: pd.Series, feats_df: pd.DataFrame, lookback: int, horizons: List[int],
                   metrics_for_bands: Optional[Dict[str, Any]], prob_floor: float, prob_cap: float,
                   min_shrink: float, calib_mode: str, debug=1,
                   band_q: float = 0.20, band_scale: float = 0.80, sqrt_time_bands: bool = True):
    X, Yc, Yu, ends = make_tabular_samples(feats_df, lookback, horizons)
    if X.size == 0:
        return None

    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)

    regs_pos, regs_neg, clfs = [], [], []
    for k, H in enumerate(horizons):
        clf = LogisticRegression(max_iter=1200, class_weight="balanced", random_state=RANDOM_SEED + 500 + k)
        clf.fit(Xs, Yu[:, k]); clfs.append(clf)

        pos_mask = Yc[:, k] > 0
        neg_mask = ~pos_mask
        n_pos, n_neg = int(pos_mask.sum()), int(neg_mask.sum())
        if n_pos < 5 or n_neg < 5:
            reg_all = fit_with_fallback(make_gbt_reg(RANDOM_SEED + 500 + k), Xs, Yc[:, k],
                                        label=f"H{H}_all_final", debug=debug)
            regs_pos.append(reg_all); regs_neg.append(reg_all)
        else:
            reg_pos = fit_with_fallback(make_gbt_reg(RANDOM_SEED + 600 + k), Xs[pos_mask], Yc[pos_mask, k],
                                        label=f"H{H}_pos_final", debug=debug)
            reg_neg = fit_with_fallback(make_gbt_reg(RANDOM_SEED + 700 + k), Xs[neg_mask], Yc[neg_mask, k],
                                        label=f"H{H}_neg_final", debug=debug)
            regs_pos.append(reg_pos); regs_neg.append(reg_neg)

    last_end_idx = len(feats_df) - 1
    past = feats_df.iloc[last_end_idx - lookback + 1 : last_end_idx + 1]
    x_last = _window_agg(past).reshape(1, -1)
    x_last = sc.transform(x_last)

    pred_cum, prob_up, cond_up, cond_dn = [], [], [], []
    for k in range(len(horizons)):
        p = float(clfs[k].predict_proba(x_last)[0, 1])
        p = np.clip(p, prob_floor, prob_cap)
        base_rate = float(np.mean(Yu[:,k])) if len(Yu)>0 else 0.5
        p = float(prob_shrink(np.array([p]), base_rate, min_shrink)[0])

        mu_pos = float(regs_pos[k].predict(x_last)[0])
        mu_neg = float(regs_neg[k].predict(x_last)[0])
        mu_pos = max(mu_pos, 0.0)
        mu_neg = min(mu_neg, 0.0)

        mu_exp = p * mu_pos + (1.0 - p) * mu_neg

        pred_cum.append(mu_exp); prob_up.append(p)
        cond_up.append(mu_pos); cond_dn.append(mu_neg)

    pred_cum = np.array(pred_cum, dtype=float)
    prob_up = np.array(prob_up, dtype=float)

    try:
        k12 = horizons.index(12)
        H_for_bands = 12
    except ValueError:
        k12 = 0
        H_for_bands = horizons[0]

    # Global panel-level residual std as fallback
    resid_std = 0.10
    if metrics_for_bands and "residual_std_per_h" in metrics_for_bands:
        resid_std = float(metrics_for_bands["residual_std_per_h"].get(f"{H_for_bands}m", resid_std))

    # NOTE: for single-ticker mode we don't have per-ticker pool; keep global
    sigma_eff = float(resid_std) * float(band_scale)
    zq = z_from_q(band_q)

    mu = float(pred_cum[k12])
    P = close_m.copy(); P.index = P.index.asfreq("M")
    P0 = float(P.iloc[-1])
    dates = pd.period_range(feats_df.index[-1] + 1, periods=H_for_bands, freq="M")

    def cum_to_price(mu_cum, m_idx):
        frac = (m_idx+1) / H_for_bands
        if sqrt_time_bands:
            sig_m = sigma_eff * np.sqrt(frac)
        else:
            sig_m = sigma_eff * frac
        # p10/p90 around the *median* cumulative mean path
        mid = P0 * np.exp(mu_cum * frac)
        lo  = P0 * np.exp(mu_cum * frac - zq * sig_m)
        hi  = P0 * np.exp(mu_cum * frac + zq * sig_m)
        return lo, mid, hi

    p10_list, p50_list, p90_list = [], [], []
    for m in range(H_for_bands):
        lo, mid, hi = cum_to_price(mu, m)
        p10_list.append(lo); p50_list.append(mid); p90_list.append(hi)

    band_df = pd.DataFrame({
        "Date": dates.astype(str),
        "P10": p10_list,
        "P50": p50_list,
        "P90": p90_list,
    })

    out_json = {
        "prob_up_by_horizon": {f"{h}m": float(prob_up[i]) for i, h in enumerate(horizons)},
        "exp_cum_return_by_horizon": {f"{h}m": float(pred_cum[i]) for i, h in enumerate(horizons)},
        "cond_cum_return_if_up": {f"{h}m": float(cond_up[i]) for i, h in enumerate(horizons)},
        "cond_cum_return_if_down": {f"{h}m": float(cond_dn[i]) for i, h in enumerate(horizons)},
    }
    return band_df, out_json

# =========================
# Panel walk-forward (FE + threshold policies + balanced val + group scaling)
# =========================
def walk_forward_backtest_panel(
    feats_map: Dict[str, pd.DataFrame], price_map: Dict[str, pd.Series],
    lookback: int, horizons: List[int], step: int, embargo: int,
    half_life_months: int, prob_floor: float, prob_cap: float,
    min_shrink: float, calib_mode: str, tune_thresh: Optional[str],
    panel_intercepts: str = "clf", cls_class_weight: str = "none",
    fe_scale: float = 0.5, threshold_policy: str = "per_ticker_gated",
    min_gain: float = 0.002, vt_per_ticker: int = 3, panel_scaler: str = "global",
    debug=1
) -> Optional[Dict[str, Any]]:

    rows = []
    for tk, fdf in feats_map.items():
        X, Yc, Yu, ends = make_tabular_samples(fdf, lookback, horizons)
        if X.size == 0:
            continue
        rows.append((tk, X, Yc, Yu, ends))
    if not rows:
        print("   [panel] no usable tickers after preprocessing.")
        return None

    all_ends = []
    for _, _, _, _, ends in rows:
        all_ends.extend(list(ends))
    unique_months = pd.PeriodIndex(sorted(set(all_ends)), freq="M")

    min_fold_windows = 10
    cuts: List[pd.Period] = []
    for i in range(min_fold_windows, len(unique_months)-1):
        if (i - min_fold_windows) % step == 0:
            cuts.append(unique_months[i])
    if debug: print(f"   [wf] #cuts={len(cuts)}")
    if len(cuts)==0:
        return None

    H_main = 12 if 12 in horizons else horizons[0]
    price_true_all, price_pred_all = [], []
    acc_default, acc_tuned = [], []
    brier_collect = []
    tuned_thresholds_snapshots = []
    invalid_thr_count_global = 0

    # collect residuals by symbol for per-ticker σ
    resid_by_symbol: Dict[str, List[float]] = {tk: [] for tk,_X,_Yc,_Yu,_e in rows}

    for cut in cuts:
        X_tr_list, Yc_tr_list, Yu_tr_list, w_tr_list, last_row_list = [], [], [], [], []
        tickers_tr_list, ends_tr_list = [], []

        for tk, X, Yc, Yu, ends in rows:
            ends_p = pd.PeriodIndex(list(ends), freq="M")
            m_train = ends_p <= (cut - embargo)
            if m_train.sum() < min_fold_windows:
                continue

            X_tr = X[m_train]
            Yc_tr = Yc[m_train]
            Yu_tr = Yu[m_train]
            ends_tr = np.asarray(ends[m_train], dtype=object)

            w_tr = exponential_time_weights(ends_tr, cutoff_month=cut, half_life_months=half_life_months)

            m_cut = (ends_p == cut)
            x_last = X[m_cut][-1] if m_cut.sum()>0 else None

            X_tr_list.append(X_tr); Yc_tr_list.append(Yc_tr); Yu_tr_list.append(Yu_tr); w_tr_list.append(w_tr)
            tickers_tr_list.append(np.array([tk]*len(X_tr), dtype=object))
            ends_tr_list.append(ends_tr)
            last_row_list.append((tk, x_last))

        if not X_tr_list:
            continue

        # Stack training
        X_tr = np.vstack(X_tr_list)
        Yc_tr = np.vstack(Yc_tr_list)
        Yu_tr = np.vstack(Yu_tr_list)
        w_tr  = np.concatenate(w_tr_list)
        tickers_tr = np.concatenate(tickers_tr_list)
        ends_tr_all = np.concatenate(ends_tr_list)

        # ---------- Scaling
        if (panel_scaler or "global").lower() == "by_ticker":
            # Per-ticker z-scoring
            mu_g = X_tr.mean(axis=0); sd_g = X_tr.std(axis=0); sd_g[sd_g<1e-9]=1.0
            d_scalers = fit_group_scalers(X_tr, tickers_tr)
            X_tr_base = transform_by_group(X_tr, tickers_tr, d_scalers, mu_g, sd_g)
            # transform function for new rows:
            def _transform_base(x, tk):
                return ((x - d_scalers.get(str(tk),(mu_g,sd_g))[0]) /
                        d_scalers.get(str(tk),(mu_g,sd_g))[1])
        else:
            sc = StandardScaler().fit(X_tr)
            X_tr_base = sc.transform(X_tr)
            def _transform_base(x, tk):  # noqa: ARG001
                return sc.transform(x)

        # ---------- Panel FE encoder
        add_clf = panel_intercepts in ("clf","both")
        add_reg = panel_intercepts in ("reg","both")
        enc = PanelInterceptEncoder(add_for_classifier=add_clf, add_for_regressor=add_reg, fe_scale=fe_scale).fit(tickers_tr)

        Xc_tr = enc.transform(tickers_tr, X_tr_base, for_role="classifier")
        Xr_tr = enc.transform(tickers_tr, X_tr_base, for_role="regressor")

        # ---------- Balanced validation tail: last vt_per_ticker rows per ticker
        idx_val = []
        unique_tk = np.unique(tickers_tr)
        for tk in unique_tk:
            m_tk = (tickers_tr == tk)
            idxs = np.where(m_tk)[0]
            take = min(vt_per_ticker, len(idxs))
            if take > 0:
                idx_val.extend(list(idxs[-take:]))
        idx_val = np.array(sorted(idx_val))
        if idx_val.size == 0:
            continue

        Xc_val = Xc_tr[idx_val]
        Xr_val = Xr_tr[idx_val]
        Yc_val = Yc_tr[idx_val]
        Yu_val = Yu_tr[idx_val]
        tickers_val = tickers_tr[idx_val]

        # ---------- Classifier/regressors
        cw = None if (cls_class_weight or "none").lower() == "none" else "balanced"
        regs_pos, regs_neg, clfs = [], [], []
        for k, H in enumerate(horizons):
            clf = LogisticRegression(max_iter=1200, C=0.5, class_weight=cw, random_state=RANDOM_SEED + 900 + k)
            clf.fit(Xc_tr, Yu_tr[:, k], sample_weight=w_tr)
            clfs.append(clf)

            pos_mask = Yc_tr[:, k] > 0
            neg_mask = ~pos_mask
            if pos_mask.sum() < 10 or neg_mask.sum() < 10:
                reg_all = fit_with_fallback(make_gbt_reg(RANDOM_SEED + 900 + k), Xr_tr, Yc_tr[:, k],
                                            label=f"[panel]H{H}_all", debug=debug)
                regs_pos.append(reg_all); regs_neg.append(reg_all)
            else:
                reg_pos = fit_with_fallback(make_gbt_reg(RANDOM_SEED + 910 + k), Xr_tr[pos_mask], Yc_tr[pos_mask, k],
                                            label=f"[panel]H{H}_pos", debug=debug)
                reg_neg = fit_with_fallback(make_gbt_reg(RANDOM_SEED + 920 + k), Xr_tr[neg_mask], Yc_tr[neg_mask, k],
                                            label=f"[panel]H{H}_neg", debug=debug)
                regs_pos.append(reg_pos); regs_neg.append(reg_neg)

        # ---- Prob calibration & Brier on balanced val (use main horizon index 0)
        pv = clfs[0].predict_proba(Xc_val)[:,1]
        pv = np.clip(pv, prob_floor, prob_cap)
        base_rate = float(np.mean(Yu_val[:,0])) if len(Yu_val)>0 else 0.5
        pv = prob_shrink(pv, base_rate, min_shrink)
        pv_cal, _ = calibrate_probs(Yu_val[:,0], pv, "auto" if calib_mode!="none" else "none")
        pv_cal = np.clip(pv_cal, prob_floor, prob_cap)
        try:
            brier_collect.append(float(brier_score_loss(Yu_val[:,0], pv_cal)))
        except Exception:
            pass

        # --- Baseline BA @ 0.5 on balanced val
        y0_all = Yu_val[:,0].astype(int)
        yhat0 = (pv_cal >= 0.5).astype(int)
        tp = ((yhat0==1) & (y0_all==1)).sum()
        tn = ((yhat0==0) & (y0_all==0)).sum()
        fp = ((yhat0==1) & (y0_all==0)).sum()
        fn = ((yhat0==0) & (y0_all==1)).sum()
        ba0_all = 0.5*((tp/(tp+fn+1e-9)) + (tn/(tn+fp+1e-9)))

        # --- Tune thresholds
        thr_global = 0.5
        thr_by_ticker: Dict[str, float] = {}
        invalid_thr_count = 0

        def _tune(y_true, p_cal):
            if tune_thresh == "balanced" or tune_thresh is None:
                return tune_threshold_balanced(y_true, p_cal, floor=prob_floor, cap=prob_cap)
            else:
                return tune_threshold_grid(y_true, p_cal, metric=tune_thresh)

        if tune_thresh:
            policy = (threshold_policy or "per_ticker_gated").lower()
            if policy in ("panel","panel_gated"):
                t, s = _tune(y0_all, pv_cal)
                if not np.isfinite(t): invalid_thr_count += 1; t = 0.5
                if policy.endswith("gated") and s < ba0_all + float(min_gain):
                    t = 0.5
                thr_global = float(np.clip(t, prob_floor, prob_cap))
                tuned_thresholds_snapshots.append(thr_global)
            elif policy in ("per_ticker","per_ticker_gated"):
                uniq = np.unique(tickers_val)
                for tk in uniq:
                    m = (tickers_val == tk)
                    y_t = y0_all[m]; p_t = pv_cal[m]
                    if len(y_t) < 4 or np.unique(y_t).size < 2:
                        t, s = _tune(y0_all, pv_cal)
                    else:
                        t, s = _tune(y_t, p_t)
                    if not np.isfinite(t): invalid_thr_count += 1; t = 0.5
                    if policy.endswith("gated"):
                        yhat0_t = (p_t >= 0.5).astype(int)
                        tp = ((yhat0_t==1) & (y_t==1)).sum()
                        tn = ((yhat0_t==0) & (y_t==0)).sum()
                        fp = ((yhat0_t==1) & (y_t==0)).sum()
                        fn = ((yhat0_t==0) & (y_t==1)).sum()
                        ba0_t = 0.5*((tp/(tp+fn+1e-9)) + (tn/(tn+fp+1e-9)))
                        if s < ba0_t + float(min_gain):
                            t = 0.5
                    thr_by_ticker[str(tk)] = float(np.clip(t, prob_floor, prob_cap))
                tuned_thresholds_snapshots.append(np.nanmean(list(thr_by_ticker.values())) if thr_by_ticker else 0.5)
            else:
                thr_global = 0.5

        invalid_thr_count_global += invalid_thr_count

        # ---- Collect residuals by ticker for bands (main horizon only)
        mu_pos_val = regs_pos[0].predict(Xr_val)
        mu_neg_val = regs_neg[0].predict(Xr_val)
        mu_pos_val = np.maximum(mu_pos_val, 0.0)
        mu_neg_val = np.minimum(mu_neg_val, 0.0)
        mu_exp_val = pv_cal * mu_pos_val + (1.0 - pv_cal) * mu_neg_val
        resid_val = (Yc_val[:,0] - mu_exp_val)
        for tk_val, r_ in zip(tickers_val, resid_val):
            try:
                resid_by_symbol[str(tk_val)].append(float(r_))
            except Exception:
                pass

        # ---- Score each ticker at the cut
        for tk, x_last in last_row_list:
            if x_last is None:
                continue
            fdf = feats_map[tk]
            P = price_map[tk]
            P.index = P.index.asfreq("M")

            x_last_base = _transform_base(x_last.reshape(1, -1), tk)
            x_last_c = enc.transform([tk], x_last_base, for_role="classifier")
            x_last_r = enc.transform([tk], x_last_base, for_role="regressor")

            p = float(clfs[0].predict_proba(x_last_c)[0,1])
            p = np.clip(p, prob_floor, prob_cap)
            p = float(prob_shrink(np.array([p]), base_rate, min_shrink)[0])
            p_cal = p

            try:
                P0 = float(P.loc[cut])
            except KeyError:
                P0 = float(P.iloc[P.index.get_indexer([cut], method="pad")][0])

            fidx = np.where(pd.PeriodIndex(list(fdf.index), freq="M") == cut)[0]
            if len(fidx)==0:
                continue
            pos = fidx[0]
            r_all = fdf["r"].values
            true_r_future = r_all[pos+1: pos+1+H_main]
            if len(true_r_future) < H_main:
                continue
            y_true_up = int(true_r_future.sum() > 0)

            acc_default.append(1.0 if (p_cal>=0.5)==bool(y_true_up) else 0.0)

            if tune_thresh:
                if threshold_policy in ("per_ticker","per_ticker_gated") and (tk in thr_by_ticker):
                    thr_use = thr_by_ticker[tk]
                else:
                    thr_use = thr_global
                acc_tuned.append(1.0 if (p_cal>=thr_use)==bool(y_true_up) else 0.0)

            mu_pos = float(regs_pos[0].predict(x_last_r)[0])
            mu_neg = float(regs_neg[0].predict(x_last_r)[0])
            mu_pos = max(mu_pos, 0.0)
            mu_neg = min(mu_neg, 0.0)

            mu_exp = p_cal * mu_pos + (1.0 - p_cal) * mu_neg
            pred_cum_monthly = np.linspace(mu_exp / H_main, mu_exp, num=H_main)
            pred_path = P0 * np.exp(pred_cum_monthly)

            true_path = P0 * np.exp(np.cumsum(true_r_future))
            price_true_all.append(true_path)
            price_pred_all.append(pred_path)

    if not price_true_all:
        return None

    price_true = np.vstack(price_true_all)
    price_pred = np.vstack(price_pred_all)

    metrics = {
        "rmse_price_mainH": rmse_price(price_true, price_pred),
        "mape_price_mainH": mape_price(price_true, price_pred),
        "rmse_price_mainH_norm": rmse(price_true/(price_true.mean(axis=1, keepdims=True)+1e-9),
                                      price_pred/(price_true.mean(axis=1, keepdims=True)+1e-9)),
        "smape_price_mainH_norm": smape_price(price_true, price_pred),
        "rmse_cum_return_mainH": float(rmse(np.log(price_true[:,-1]/price_true[:,0]),
                                            np.log(price_pred[:,-1]/price_true[:,0]))),
        "smape_cum_return_mainH": float(smape(np.log(price_true[:,-1]/price_true[:,0]),
                                              np.log(price_pred[:,-1]/price_true[:,0]))),
        "directional_accuracy_default": [float(np.mean(acc_default))] if acc_default else [None],
        "directional_accuracy_tuned": [float(np.mean(acc_tuned))] if acc_tuned else [None],
        "horizons_used": [int(h) for h in horizons],
        "lookback_used": int(lookback),
        "num_folds": int(len(price_true_all)),
        "regressor": "LightGBM" if HAS_LGBM else "HGBR",
        "classifier": f"LogReg (panel+FE) + sign-aware expected return",
        "threshold_policy_used": threshold_policy,
        "panel_scaler": panel_scaler,
        "vt_per_ticker": int(vt_per_ticker),
    }
    if brier_collect:
        metrics["brier_score_per_h"] = {f"{h}m": float(np.mean(brier_collect)) for h in horizons}
    resid_sigma = robust_sigma(np.log(price_true[:,-1]/price_true[:,0]) - np.log(price_pred[:,-1]/price_true[:,0]))
    metrics["residual_std_per_h"] = {f"{h}m": float(resid_sigma) for h in horizons}

    # per-symbol residual stats (main horizon)
    per_sym_std = {}
    per_sym_n = {}
    for tk, rr in resid_by_symbol.items():
        rr = np.array(rr, dtype=float)
        rr = rr[np.isfinite(rr)]
        if rr.size >= 3:
            per_sym_std[tk] = float(robust_sigma(rr))
            per_sym_n[tk] = int(rr.size)
    metrics["residual_std_per_symbol"] = {f"{H_main}m": per_sym_std}
    metrics["residual_n_per_symbol"]   = {f"{H_main}m": per_sym_n}

    if tuned_thresholds_snapshots:
        arr = np.array(tuned_thresholds_snapshots, dtype=float)
        finite = arr[np.isfinite(arr)]
        metrics["panel_thresholds_used"] = {
            "mean": float(np.nanmean(finite)) if finite.size else None,
            "median": float(np.nanmedian(finite)) if finite.size else None,
            "invalid_count": int(np.sum(~np.isfinite(arr)))
        }
    return metrics

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--tickers", default=None, help="Comma-separated for panel mode")
    ap.add_argument("--outdir", default="src/stock_predictions_tabular")
    ap.add_argument("--period", default="max")
    ap.add_argument("--lookback", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--horizons", type=str, default=None)
    ap.add_argument("--exog", type=str,
                    default="BZ=F,DX-Y.NYB,^TNX,^VIX,EEM,GLD",
                    help="Comma-separated exog tickers (Yahoo). Example: BZ=F (Brent), DX-Y.NYB (DXY), ^TNX (US10Y), ^VIX, EEM (MSCI EM ETF), GLD.")
    ap.add_argument("--min_fold_windows", type=int, default=10)
    ap.add_argument("--val_tail_windows", type=int, default=12)  # (single-ticker only)
    ap.add_argument("--embargo", type=int, default=6)
    ap.add_argument("--half_life", type=int, default=24)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--prob_floor", type=float, default=0.1)
    ap.add_argument("--prob_cap", type=float, default=0.9)
    ap.add_argument("--min_shrink", type=float, default=0.30)
    ap.add_argument("--calib", type=str, default="auto", help="auto|none|sigmoid|isotonic")
    ap.add_argument("--tune_threshold", type=str, default="balanced", help="balanced|precision|recall|accuracy|none")
    ap.add_argument("--threshold_policy", type=str, default="per_ticker_gated",
                    choices=["none","panel","panel_gated","per_ticker","per_ticker_gated"],
                    help="How to apply tuned thresholds during evaluation.")
    ap.add_argument("--min_gain", type=float, default=0.002,
                    help="Minimum BA improvement over 0.5 to accept tuned threshold (for *_gated).")
    ap.add_argument("--panel_intercepts", type=str, default="clf", choices=["none","clf","reg","both"],
                    help="Add per-ticker fixed effects to classifier/regressor in panel backtest.")
    ap.add_argument("--cls_class_weight", type=str, default="none", choices=["none","balanced"],
                    help="Classifier class_weight for panel mode (use 'none' if you rely on threshold tuning).")
    ap.add_argument("--fe_scale", type=float, default=0.5, help="Multiplier for FE columns to shrink their effect.")
    ap.add_argument("--vt_per_ticker", type=int, default=3, help="Validation rows per ticker for tuning.")
    ap.add_argument("--panel_scaler", type=str, default="global", choices=["global","by_ticker"],
                    help="Feature scaling mode in panel training.")

    # ---- NEW (added, nothing removed)
    ap.add_argument("--band_q", type=float, default=0.20, help="Lower quantile for bands (e.g., 0.10=10th; symmetric)")
    ap.add_argument("--band_scale", type=float, default=0.80, help="Multiply residual std for band width")
    ap.add_argument("--no_sqrt_time_bands", action="store_true",
                    help="If set, band width grows linearly with time instead of sqrt(time)")

    ap.add_argument("--debug", type=int, default=1)
    args = ap.parse_args()

    if args.horizons:
        horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    elif args.horizon:
        horizons = [args.horizon]
    else:
        horizons = [12]
    step = args.step or min(horizons)

    exog_list = []
    if isinstance(args.exog, str) and args.exog.strip():
        exog_list = [x.strip() for x in args.exog.split(",") if x.strip()]

    os.makedirs(os.path.join(args.outdir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "reports"), exist_ok=True)

    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
        print(f"→ Panel tickers: {tickers}")

        feats_map: Dict[str, pd.DataFrame] = {}
        price_map: Dict[str, pd.Series] = {}
        for tk in tickers:
            cm = monthly_close(tk, period=args.period, debug=args.debug)
            if cm is None:
                continue
            f = compute_features_from_close(cm)
            if f is None:
                if args.debug: print(f"   [feat] {tk}: empty")
                continue
            f = add_exog(f, exog_list, period=args.period, debug=args.debug)
            if f is None or f.empty:
                if args.debug: print(f"   [feat] {tk}: empty")
                continue
            feats_map[tk] = f
            price_map[tk] = cm

        if not feats_map:
            print("   [panel] no usable tickers after preprocessing.")
            return

        metrics = walk_forward_backtest_panel(
            feats_map, price_map, lookback=args.lookback, horizons=horizons,
            step=step, embargo=args.embargo, half_life_months=args.half_life,
            prob_floor=args.prob_floor, prob_cap=args.prob_cap, min_shrink=args.min_shrink,
            calib_mode=args.calib, tune_thresh=None if args.tune_threshold=="none" else args.tune_threshold,
            panel_intercepts=args.panel_intercepts, cls_class_weight=args.cls_class_weight,
            fe_scale=args.fe_scale, threshold_policy=args.threshold_policy, min_gain=args.min_gain,
            vt_per_ticker=args.vt_per_ticker, panel_scaler=args.panel_scaler,
            debug=args.debug
        )
        if metrics is None:
            print("   [warn] panel walk-forward not feasible with current settings.")
            return
        panel_eval_path = os.path.join(args.outdir, "reports", "panel_tab_eval.json")
        with open(panel_eval_path, "w") as f:
            json.dump({"metrics": metrics}, f, indent=2)
        print(f"   [saved] reports/panel_tab_eval.json")

        # Per-ticker final forecast with calibrated bands
        consolidated = {}
        for tk, f in feats_map.items():
            cm = price_map[tk]
            res = final_forecast(
                cm, f, lookback=args.lookback, horizons=horizons,
                metrics_for_bands=metrics,
                prob_floor=args.prob_floor, prob_cap=args.prob_cap,
                min_shrink=args.min_shrink, calib_mode=args.calib, debug=args.debug,
                band_q=args.band_q, band_scale=args.band_scale,
                sqrt_time_bands=(not args.no_sqrt_time_bands)
            )
            if res is None:
                continue
            band_df, out_json = res

            safe = tk.replace(".", "_")
            band_name = f"{safe}_tab_bands_{horizons[0]}m.csv" if len(horizons)==1 else f"{safe}_tab_bands_12m.csv"
            band_path = os.path.join(args.outdir, "predictions", band_name)
            band_df.to_csv(band_path, index=False)

            fc_path = os.path.join(args.outdir, "reports", f"{safe}_tab_panel_forecast.json")
            with open(fc_path, "w") as f:
                json.dump(out_json, f, indent=2)

            consolidated[tk] = out_json

        # Singular filename (matches your app’s expectation)
        with open(os.path.join(args.outdir, "reports", "panel_tab_forecast.json"), "w") as f:
            json.dump(consolidated, f, indent=2)
        print("   [saved] reports/panel_tab_forecast.json")
        return

    # ---------- Single ticker mode ----------
    if not args.ticker:
        print("   [error] provide --ticker or --tickers")
        return

    print(f"→ {args.ticker}")
    safe = args.ticker.replace(".", "_")
    cm = monthly_close(args.ticker, period=args.period, debug=args.debug)
    if cm is None:
        print("   [skip] no data")
        return

    feats = compute_features_from_close(cm)
    if feats is None:
        print("   [skip] features empty")
        return
    feats = add_exog(feats, exog_list, period=args.period, debug=args.debug)
    if args.debug:
        print(f"   [feat] rows={len(feats)}  cols={list(feats.columns)}")
    if len(feats) < args.lookback + max(horizons) + 4:
        print("   [skip] not enough usable history")
        return

    metrics = walk_forward_backtest(
        cm, feats, lookback=args.lookback, horizons=horizons,
        min_fold_windows=args.min_fold_windows, step=step,
        val_tail=args.val_tail_windows, embargo=args.embargo, half_life_months=args.half_life,
        prob_floor=args.prob_floor, prob_cap=args.prob_cap, min_shrink=args.min_shrink,
        calib_mode=args.calib, tune_thresh=None if args.tune_threshold=="none" else args.tune_threshold, debug=args.debug
    )
    if metrics is None:
        print("   [warn] walk-forward not feasible with current settings.")
    else:
        with open(os.path.join(args.outdir, "reports", f"{safe}_tab_eval.json"), "w") as f:
            json.dump({"metrics": metrics}, f, indent=2)
        print(f"   [saved] reports/{safe}_tab_eval.json")

    res = final_forecast(
        cm, feats, lookback=args.lookback, horizons=horizons,
        metrics_for_bands=metrics, prob_floor=args.prob_floor, prob_cap=args.prob_cap,
        min_shrink=args.min_shrink, calib_mode=args.calib, debug=args.debug,
        band_q=args.band_q, band_scale=args.band_scale,
        sqrt_time_bands=(not args.no_sqrt_time_bands)
    )
    if res is None:
        print("   [warn] final forecast not produced.")
        return
    band_df, out_json = res

    band_name = f"{safe}_tab_bands_{horizons[0]}m.csv" if len(horizons) == 1 else f"{safe}_tab_bands_12m.csv"
    band_path = os.path.join(args.outdir, "predictions", band_name)
    band_df.to_csv(band_path, index=False)

    fc_path = os.path.join(args.outdir, "reports", f"{safe}_tab_forecast.json")
    with open(fc_path, "w") as f:
        json.dump(out_json, f, indent=2)

    print("   [saved]")
    print(f"     predictions/{os.path.basename(band_path)}")
    print(f"     reports/{os.path.basename(fc_path)}")

if __name__ == "__main__":
    main()
