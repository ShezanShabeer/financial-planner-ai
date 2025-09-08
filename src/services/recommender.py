# src/services/recommender.py
from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable

import pandas as pd


# -------------------------
# Path helpers
# -------------------------

# Hard override (Windows absolute path)
_FORCE_DATA_ROOT = Path(
    r"C:\Users\shezan.shabeer\AI\financial-planner-ai\stock_prediction_ens"
).expanduser().resolve()


def _src_dir() -> Path:
    """
    Resolve the absolute path to the 'src' directory:
    .../<project>/src/services/recommender.py  -> returns .../<project>/src
    """
    return Path(__file__).resolve().parents[1]


def _default_roots(model: str = "ens") -> list[Path]:
    """
    Default locations we will search for data if no override/env is set.
    Supports both 'stock_predictions_ens' and 'stock_prediction_ens'.
    """
    s = _src_dir()
    return [
        s / f"stock_predictions_{model}",
        s / f"stock_prediction_{model}",
    ]


def _data_root(model: str = "ens") -> list[Path]:
    """
    Build the ordered list of roots to search, with the hard override first,
    then FORECAST_DATA_ROOT (if set; ';' separated), then sensible fallbacks.
    """
    roots: list[Path] = []

    # 1) Hard override (your requested path)
    roots.append(_FORCE_DATA_ROOT)

    # 2) Optional env var override(s)
    env = os.environ.get("FORECAST_DATA_ROOT")
    if env:
        for part in env.split(";"):
            part = part.strip()
            if part:
                roots.append(Path(part).expanduser().resolve())

    # 3) Fallbacks under src/
    roots.extend(_default_roots(model))

    # De-dup while preserving order
    seen: set[str] = set()
    uniq: list[Path] = []
    for r in roots:
        key = str(r)
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    return uniq


def _safe_ticker(t: str) -> str:
    """
    File-system safe ticker: replace any non [A-Za-z0-9_-] with underscore.
    Also convert dots to underscores (EMAAR.AE -> EMAAR_AE).
    """
    return re.sub(r"[^A-Za-z0-9_-]+", "_", t.strip())


def _candidates_reports(ticker: str, model: str = "ens") -> list[Path]:
    """
    Candidate report JSON file paths across all roots.
    """
    t_safe = _safe_ticker(ticker)
    t_raw = ticker.strip()

    names = [
        f"{t_safe}_{model}_forecast.json",
        f"{t_raw}_{model}_forecast.json",
        f"{t_safe}_forecast.json",
        f"{t_raw}_forecast.json",
        f"{t_safe}.json",
        f"{t_raw}.json",
    ]

    out: list[Path] = []
    for root in _data_root(model):
        for name in names:
            out.append(root / "reports" / name)
    return out


def _candidates_eval(ticker: str, model: str = "ens") -> list[Path]:
    t_safe = _safe_ticker(ticker)
    t_raw = ticker.strip()
    names = [
        f"{t_safe}_{model}_eval.json",
        f"{t_raw}_{model}_eval.json",
        f"{t_safe}_eval.json",
        f"{t_raw}_eval.json",
        f"{t_safe}.eval.json",
        f"{t_raw}.eval.json",
    ]
    out: list[Path] = []
    for root in _data_root(model):
        for name in names:
            out.append(root / "reports" / name)
    return out


def _candidates_bands_12m(ticker: str, model: str = "ens") -> list[Path]:
    t_safe = _safe_ticker(ticker)
    t_raw = ticker.strip()
    names = [
        f"{t_safe}_{model}_bands_12m.csv",
        f"{t_raw}_{model}_bands_12m.csv",
        f"{t_safe}_bands_12m.csv",
        f"{t_raw}_bands_12m.csv",
    ]
    out: list[Path] = []
    for root in _data_root(model):
        # tolerate both 'predictions' and 'prediction'
        for sub in ("predictions", "prediction"):
            for name in names:
                out.append(root / sub / name)
    return out


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        try:
            if p.exists():
                return p
        except OSError:
            continue
    return None


# -------------------------
# Public loaders
# -------------------------

def load_forecast(ticker: str, model: str = "ens") -> Dict[str, Any]:
    """
    Loads forecast JSON for a ticker.

    Search order:
      HARD OVERRIDE (this machine)/reports/{ticker_variant}_{model}_forecast.json
      FORECAST_DATA_ROOT[;...]/reports/...
      <src>/stock_predictions_{model}/reports/...
      <src>/stock_prediction_{model}/reports/...

    Accepts several filename variants (ticker, safe_ticker; with/without suffixes).
    """
    candidates = _candidates_reports(ticker, model=model)
    found = _first_existing(candidates)
    if not found:
        tried = "\n - ".join(str(c) for c in candidates)
        raise FileNotFoundError(
            f"Missing forecast JSON for '{ticker}'. Tried:\n - {tried}\n"
            f"Module: {__file__}\nCWD: {Path.cwd()}"
        )
    with found.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_backtest(ticker: str, model: str = "ens") -> Optional[Dict[str, Any]]:
    """
    Loads backtest/eval JSON (optional).
    """
    candidates = _candidates_eval(ticker, model=model)
    found = _first_existing(candidates)
    if not found:
        return None
    with found.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_bands_12m(ticker: str, model: str = "ens") -> Optional[pd.DataFrame]:
    """
    Loads 12m bands CSV (optional) into a DataFrame with parsed dates.
    """
    candidates = _candidates_bands_12m(ticker, model=model)
    found = _first_existing(candidates)
    if not found:
        return None
    return pd.read_csv(found, parse_dates=["Date"])


# -------------------------
# Scoring / Recommendation
# -------------------------

def _risk_weight(level: str) -> float:
    return {"low": 0.7, "medium": 1.0, "high": 1.2}.get(level, 1.0)


def score_one(ticker: str, risk_level: str = "medium", model: str = "ens") -> Optional[Dict[str, Any]]:
    # 1) Probabilities (priors about future)
    try:
        fc = load_forecast(ticker, model=model)
    except Exception:
        return None

    # prefer calibrated meta prob if present
    p_meta = (fc.get("prob_up_by_horizon_meta") or {}).get("12m")
    if p_meta is None:
        p_meta = (fc.get("prob_up_by_horizon_raw") or {}).get("12m", 0.5)

    # 2) Backtest directional accuracy (evidence from past OOS performance)
    bt = load_backtest(ticker, model=model)
    da12 = None
    if bt and "metrics" in bt:
        m = bt["metrics"]
        if isinstance(m.get("directional_accuracy_best"), list) and m["directional_accuracy_best"]:
            try:
                da12 = float(m["directional_accuracy_best"][0])
            except Exception:
                da12 = None
        elif isinstance(m.get("directional_accuracy"), dict):
            try:
                da12 = float(m["directional_accuracy"].get("12m", 0.5))
            except Exception:
                da12 = None

    # 3) Simple score: prob + (da-0.5)*0.6; then risk adjust
    score = float(p_meta) + (((da12 or 0.5) - 0.5) * 0.6)
    score *= _risk_weight(risk_level)

    reason = f"Prob(up,12m): {float(p_meta):.2f}"
    if da12 is not None:
        reason += f"; OOS DA(12m): {da12:.2f}"
    reason += f"; risk adj: {_risk_weight(risk_level):.2f}"

    return {
        "ticker": ticker,
        "score": score,
        "reason": reason,
        "prob_up_12m": float(p_meta),
        "backtest_da_12m": da12,
    }


def recommend(universe: List[str], risk_level: str = "medium", topk: int = 5, model: str = "ens") -> Dict[str, Any]:
    rows: list[Dict[str, Any]] = []
    missing: list[str] = []
    for t in universe:
        r = score_one(t, risk_level=risk_level, model=model)
        if r:
            rows.append(r)
        else:
            missing.append(t)
    rows.sort(key=lambda x: x["score"], reverse=True)
    k = min(topk, len(rows))
    return {"picks": rows[:k], "missing": missing, "universe": universe}
