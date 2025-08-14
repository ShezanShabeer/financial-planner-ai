# src/planner_recommender.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable
import math

# ========= Core compounding helpers =========
def real_return(r_annual: float, inflation: float) -> float:
    """Convert nominal return & inflation to approximate real return."""
    return (1 + (r_annual or 0)) / (1 + (inflation or 0)) - 1

def pmt_for_target(fv: float, years: float, r_annual: float) -> float:
    """
    Required monthly contribution to reach FV in 'years' at annual rate r_annual (monthly compounding).
    Falls back to straight division if r_annual<=0 or years<=0.
    """
    months = max(1, int(round((years or 0) * 12)))
    if (r_annual or 0) <= 0:
        return (fv or 0) / months
    rm = r_annual / 12.0
    denom = (1 + rm) ** months - 1
    if denom <= 0:
        return (fv or 0) / months
    return fv * rm / denom

def fv_from_pmt(pmt: float, years: float, r_annual: float) -> float:
    """Future value achieved by investing PMT per month for N years at annual rate r_annual."""
    months = max(1, int(round((years or 0) * 12)))
    if (r_annual or 0) <= 0:
        return pmt * months
    rm = r_annual / 12.0
    return pmt * ((1 + rm) ** months - 1) / rm

def solve_required_return_for_budget(fv_target: float, years: float, monthly_budget: float,
                                     r_min: float = -0.10, r_max: float = 0.30, tol: float = 1e-6,
                                     max_iter: int = 200) -> Dict[str, Any]:
    """
    Solve for the annual return required to reach FV target with a fixed monthly budget.
    Uses bisection on r in [r_min, r_max]. Returns the solution and diagnostics.
    """
    # Handle trivial cases
    months = max(1, int(round((years or 0) * 12)))
    if months <= 0 or monthly_budget <= 0:
        return {"solved": False, "reason": "Non-positive horizon or budget.", "required_return_annual": None}

    # If simple accumulation already exceeds target at 0% return:
    fv0 = monthly_budget * months
    if fv0 >= fv_target:
        return {"solved": True, "required_return_annual": 0.0, "fv_at_solution": fv0, "iterations": 0}

    # Bisection
    lo, hi = r_min, r_max
    fv_lo = fv_from_pmt(monthly_budget, years, lo)
    fv_hi = fv_from_pmt(monthly_budget, years, hi)
    # If even hi can't reach target, report infeasible
    if fv_hi < fv_target:
        return {"solved": False, "reason": "Budget too low even at optimistic return.", "required_return_annual": None,
                "fv_at_hi": fv_hi, "hi": hi}
    # If lo already reaches, return low rate
    if fv_lo >= fv_target:
        return {"solved": True, "required_return_annual": lo, "fv_at_solution": fv_lo, "iterations": 0}

    it = 0
    while it < max_iter:
        mid = (lo + hi) / 2.0
        fv_mid = fv_from_pmt(monthly_budget, years, mid)
        if abs(fv_mid - fv_target) <= max(1.0, fv_target * 1e-6):
            return {"solved": True, "required_return_annual": mid, "fv_at_solution": fv_mid, "iterations": it + 1}
        if fv_mid < fv_target:
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) < tol:
            return {"solved": True, "required_return_annual": (lo + hi) / 2.0,
                    "fv_at_solution": fv_mid, "iterations": it + 1}
        it += 1
    return {"solved": True, "required_return_annual": (lo + hi) / 2.0,
            "fv_at_solution": fv_from_pmt(monthly_budget, years, (lo + hi) / 2.0), "iterations": it}

# ========= Assumptions & risk constraints =========
@dataclass
class PlanAssumptions:
    expected_return_annual: float = 0.07   # Nominal expectation
    inflation_annual: float = 0.03         # UAE default ~2.5–3%
    feasible_ratio: float = 0.80           # <=80% of free cash is "feasible"
    tight_ratio: float = 1.00              # 80–100% is "tight"

def affordability_status(required_sum: float, income: float, expenses: float,
                         feasible_ratio: float, tight_ratio: float) -> Dict[str, Any]:
    free_cash = (income or 0) - (expenses or 0)
    ratio = (required_sum / free_cash) if free_cash > 0 else float("inf")
    status = "feasible" if ratio <= feasible_ratio else ("tight" if ratio <= tight_ratio else "unaffordable")
    return {
        "free_cash_monthly_aed": round(free_cash, 2),
        "required_monthly_sum_aed": round(required_sum, 2),
        "affordability_ratio": (round(ratio, 2) if ratio != float("inf") else 1e9),
        "status": status
    }

def max_equity_from_age(age: Optional[int]) -> float:
    """Simple rule: equity_max ~ 110 - age (%) clamped to [0.3, 0.9]."""
    if age is None:
        return 0.7
    return min(0.9, max(0.3, (110 - age) / 100.0))

def risk_bucket_from_frs(frs: float) -> str:
    if frs >= 80: return "Growth"
    if frs >= 60: return "Balanced"
    return "Conservative"

MODEL_PORTFOLIOS = {
    "Conservative": {"equity": 0.35, "bonds": 0.45, "cash": 0.20},
    "Balanced":     {"equity": 0.60, "bonds": 0.30, "cash": 0.10},
    "Growth":       {"equity": 0.80, "bonds": 0.15, "cash": 0.05},
}

def clamp_weights_to_equity_cap(weights: Dict[str, float], equity_cap: float) -> Dict[str, float]:
    w = dict(weights)
    if w.get("equity", 0) > equity_cap:
        excess = w["equity"] - equity_cap
        w["equity"] = equity_cap
        # Redistribute excess to bonds/cash proportionally
        bc = w.get("bonds", 0) + w.get("cash", 0)
        if bc <= 0:
            w["cash"] = round(1 - w["equity"], 4)
            return w
        w["bonds"] += (w["bonds"] / bc) * excess
        w["cash"]  += (w["cash"]  / bc) * excess
    # renormalize
    s = sum(w.values()) or 1.0
    for k in w:
        w[k] = round(w[k] / s, 4)
    return w

# ========= Planning per goal =========
def plan_goal_assumed_return(amount_aed: float, years: float, assumptions: PlanAssumptions) -> Dict[str, Any]:
    r_real = real_return(assumptions.expected_return_annual, assumptions.inflation_annual)
    pmt = pmt_for_target(amount_aed, years, r_real)
    return {
        "mode": "assumed_return",
        "required_monthly_aed": round(pmt, 2),
        "assumptions": {
            "expected_return_annual": assumptions.expected_return_annual,
            "inflation_annual": assumptions.inflation_annual,
            "real_return_annual": round(r_real, 5)
        }
    }

def plan_goal_required_return(amount_aed: float, years: float, monthly_budget_aed: float, assumptions: PlanAssumptions) -> Dict[str, Any]:
    sol = solve_required_return_for_budget(amount_aed, years, monthly_budget_aed)
    out = {"mode": "required_return", **sol, "budget_monthly_aed": monthly_budget_aed}
    return out

# ========= LSTM stock recommendation (adapter) =========
"""
We assume you have an LSTM module that can produce an expected CAGR for (symbol, years).
Define an adapter function signature. You can inject your real function here.
"""
# Type signature for a predictor: (symbol, years) -> expected CAGR (e.g., 0.08 for 8%/yr)
PredictorFn = Callable[[str, int], float]

def default_dummy_predictor(symbol: str, years: int) -> float:
    # Placeholder: you will replace this with your model's prediction.
    # Here we just return a small random-like mapping to keep deterministic behavior per symbol.
    base = sum(ord(c) for c in symbol) % 7
    return 0.05 + (base/100.0)  # 5% to 11%

def recommend_stocks(symbols: List[str], horizons: List[int],
                     predictor: PredictorFn = default_dummy_predictor,
                     top_k: int = 5, per_symbol_max_weight: float = 0.15) -> Dict[str, Any]:
    """
    For each horizon, rank symbols by predicted CAGR, choose top_k, and assign equal weights
    capped by per_symbol_max_weight. Returns per-horizon lists of {symbol, exp_cagr}.
    """
    recos: Dict[str, Any] = {}
    for h in horizons:
        scored = [{"symbol": s, "exp_cagr": round(predictor(s, h), 4)} for s in symbols]
        scored.sort(key=lambda x: x["exp_cagr"], reverse=True)
        top = scored[:top_k]
        # equal-weight and cap per-symbol if you later transform into a portfolio
        weight = min(1.0 / max(1, len(top)), per_symbol_max_weight)
        for t in top:
            t["suggested_weight"] = round(weight, 3)
        recos[str(h) + "y"] = top
    return recos

# ========= Orchestration for whole plan =========
def build_plan(profile: Dict[str, Any], goals: List[Dict[str, Any]],
               frs_score: Optional[float] = None,
               mode_per_goal: Optional[str] = "assumed_return",
               monthly_budget_map: Optional[Dict[int, float]] = None,
               lstm_symbols: Optional[List[str]] = None,
               lstm_predictor: PredictorFn = default_dummy_predictor) -> Dict[str, Any]:
    """
    profile: includes age, income_monthly_aed, expenses_monthly_aed, risk_tolerance (optional),
             expected_return_annual, inflation_annual.
    goals:   list of parsed goals (your NLP output items). Each should have amount & timeframe.
    frs_score: Financial Readiness Score to gate risk bucket.
    mode_per_goal: "assumed_return" (default) or "required_return".
    monthly_budget_map: optional dict mapping goal index -> monthly budget (only used if mode_per_goal="required_return").
    lstm_symbols: list of UAE stock tickers to score with LSTM.
    """
    assump = PlanAssumptions(
        expected_return_annual=profile.get("expected_return_annual", 0.07) or 0.07,
        inflation_annual=profile.get("inflation_annual", 0.03) or 0.03,
    )

    per_goal = []
    total_required = 0.0

    for idx, g in enumerate(goals):
        amt = g.get("amount")
        years = (g.get("timeframe_years") or 0) + (g.get("timeframe_months") or 0)/12.0
        if amt is None or years <= 0:
            per_goal.append({
                "original_text": g.get("original_text"),
                "goal_type": g.get("goal_type"),
                "required_monthly_aed": None,
                "warning": "Missing amount or timeframe."
            })
            continue

        if mode_per_goal == "required_return":
            budget = None
            if monthly_budget_map:
                budget = monthly_budget_map.get(idx)
            if budget is None:
                per_goal.append({
                    "original_text": g.get("original_text"),
                    "goal_type": g.get("goal_type"),
                    "required_monthly_aed": None,
                    "warning": "No budget provided for required_return mode."
                })
                continue
            res = plan_goal_required_return(float(amt), float(years), float(budget), assump)
            per_goal.append({
                "original_text": g.get("original_text"),
                "goal_type": g.get("goal_type"),
                **res
            })
            # In required_return mode we do not sum monthly requirements (user provided budget).
            total_required += float(budget)
        else:
            res = plan_goal_assumed_return(float(amt), float(years), assump)
            per_goal.append({
                "original_text": g.get("original_text"),
                "goal_type": g.get("goal_type"),
                **res
            })
            total_required += res["required_monthly_aed"]

    afford = affordability_status(
        total_required,
        profile.get("income_monthly_aed", 0),
        profile.get("expenses_monthly_aed", 0),
        assump.feasible_ratio,
        assump.tight_ratio
    )

    # Risk bucket and equity cap
    bucket = (profile.get("risk_tolerance") or "").title()
    if bucket not in MODEL_PORTFOLIOS:
        bucket = risk_bucket_from_frs(frs_score or 60.0)
    weights = MODEL_PORTFOLIOS[bucket]
    eq_cap = max_equity_from_age(profile.get("age"))
    weights_capped = clamp_weights_to_equity_cap(weights, eq_cap)

    # LSTM stock recos (AI-powered section)
    stock_rec = None
    if lstm_symbols:
        stock_rec = recommend_stocks(lstm_symbols, horizons=[2,3,5], predictor=lstm_predictor, top_k=5)

    return {
        "per_goal": per_goal,
        "summary": afford,
        "portfolio": {
            "bucket": bucket,
            "base": weights,
            "equity_cap_from_age": eq_cap,
            "final_weights": weights_capped
        },
        "stock_recommendations": stock_rec,
        "assumptions": {
            "expected_return_annual": assump.expected_return_annual,
            "inflation_annual": assump.inflation_annual,
        }
    }