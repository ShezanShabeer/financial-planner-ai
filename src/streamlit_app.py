# streamlit_app.py
import os, json, requests
import numpy as np
import pandas as pd
import streamlit as st

# Optional: only for latest price → expected return vs P50
try:
    import yfinance as yf
except Exception:
    yf = None

# -----------------------------
# Config
# -----------------------------
API = os.environ.get("PLANNER_API_BASE", "http://localhost:8000")
TAB_FE_DIR = os.environ.get("TAB_FE_DIR", "src/stock_predictions_tabular")

st.set_page_config(page_title="Financial Planner (MVP)", page_icon="📈", layout="wide")

# Scoped CSS (no global dark background)
st.markdown("""
<style>
.badge {display:inline-block;padding:4px 10px;border-radius:14px;background:#eef2ff;
        color:#1f2a44;font-size:12px;font-weight:600;border:1px solid #dbe3ff;}
.card  {background:#ffffff;border:1px solid #ebeef3;border-radius:12px;padding:16px;margin-bottom:12px;}
.kpi   {background:#fff;border:1px solid #edf0f5;border-radius:12px;padding:10px 12px;margin-bottom:8px;}
.kpi .kpi-label{font-size:12px;color:#667085;font-weight:600;margin-bottom:2px;}
.kpi .kpi-value{font-size:22px;font-weight:700;line-height:1.1;word-break:break-word;}
.kpi .kpi-sub{font-size:11px;color:#8a94a6;margin-top:2px;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Helpers
# -----------------------------
def format_aed(x) -> str:
    try:
        return f"AED {float(x):,.2f}"
    except Exception:
        return "AED —"

def pct(x, digits=1) -> str:
    try:
        return f"{100*float(x):.{digits}f}%"
    except Exception:
        return "—"

def last_close(ticker: str):
    if yf is None:
        return None
    try:
        df = yf.download(ticker, period="5d", interval="1d", auto_adjust=True, progress=False)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        return float(df["Close"].iloc[-1])
    except Exception:
        return None

def post_json(url: str, payload: dict, timeout=45) -> dict:
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def get_json(url: str, timeout=30) -> dict:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

# Small HTML KPI card (wraps nicely in narrow columns)
def kpi(label: str, value: str, sub: str | None = None):
    sub_html = f"<div class='kpi-sub'>{sub}</div>" if sub else ""
    st.markdown(f"<div class='kpi'><div class='kpi-label'>{label}</div>"
                f"<div class='kpi-value'>{value}</div>{sub_html}</div>", unsafe_allow_html=True)

# -----------------------------
# Tabular-FE local file I/O
# -----------------------------
def _safe_name(ticker: str) -> str:
    return ticker.replace(".", "_")

def _unsafety_fix(s: str) -> str:
    # Convert file-safe to ticker (AIRARABIA_AE -> AIRARABIA.AE)
    return s.replace("_", ".")

def tabfe_paths(ticker: str) -> dict:
    s = _safe_name(ticker)
    preds = os.path.join(TAB_FE_DIR, "predictions")
    reps  = os.path.join(TAB_FE_DIR, "reports")
    return {
        "bands_tab":   os.path.join(preds, f"{s}_tab_bands_12m.csv"),
        "bands_panel": os.path.join(preds, f"{s}_panel_bands_12m.csv"),
        "ticker_fc":   os.path.join(reps,  f"{s}_tab_panel_forecast.json"),
        "panel_fc_all":os.path.join(reps,  "panel_tab_forecasts.json"),
        "panel_eval":  os.path.join(reps,  "panel_tab_eval.json"),
    }

def _read_json(fp: str):
    if not fp or not os.path.exists(fp): return None
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _read_csv(fp: str):
    if not fp or not os.path.exists(fp): return None
    try:
        df = pd.read_csv(fp)
        return df if not df.empty else None
    except Exception:
        return None

def list_tabfe_tickers() -> list:
    """Discover all tickers with Tabular-FE results."""
    tickers = set()
    preds = os.path.join(TAB_FE_DIR, "predictions")
    reps  = os.path.join(TAB_FE_DIR, "reports")
    try:
        if os.path.isdir(preds):
            for f in os.listdir(preds):
                if f.endswith("_tab_bands_12m.csv") or f.endswith("_panel_bands_12m.csv"):
                    base = f.replace("_tab_bands_12m.csv", "").replace("_panel_bands_12m.csv","")
                    tickers.add(_unsafety_fix(base))
    except Exception:
        pass
    try:
        jf = os.path.join(reps, "panel_tab_forecasts.json")
        obj = _read_json(jf) or {}
        for k in obj.keys():
            tickers.add(k if "." in k else _unsafety_fix(k))
    except Exception:
        pass
    return sorted(tickers)

def load_tabular_fe_local(ticker: str) -> dict:
    """Load Tabular-FE bands + per-ticker forecast + panel metrics from local files."""
    p = tabfe_paths(ticker)

    # Bands: prefer tab_bands, fallback panel_bands
    bands_df = _read_csv(p["bands_tab"])
    if bands_df is None or bands_df.empty:
        bands_df = _read_csv(p["bands_panel"])

    bands = []
    if bands_df is not None and not bands_df.empty:
        df = bands_df.copy()
        # Normalize column names
        lower_map = {c.lower(): c for c in df.columns}
        if "date" in lower_map and lower_map["date"] != "Date":
            df.rename(columns={lower_map["date"]: "Date"}, inplace=True)
        for c in ["P10","P50","P90"]:
            if c not in df.columns and c.lower() in lower_map:
                df.rename(columns={lower_map[c.lower()]: c}, inplace=True)
        keep = [c for c in ["Date","P10","P50","P90"] if c in df.columns]
        df = df[keep].dropna()
        for _, r in df.iterrows():
            d = {
                "Date": str(pd.to_datetime(r["Date"]).date()) if "Date" in r else None,
                "P10":  float(r["P10"]) if "P10" in r and pd.notna(r["P10"]) else None,
                "P50":  float(r["P50"]) if "P50" in r and pd.notna(r["P50"]) else None,
                "P90":  float(r["P90"]) if "P90" in r and pd.notna(r["P90"]) else None,
            }
            if all(v is not None for v in (d["P10"], d["P50"], d["P90"])):
                bands.append(d)

    # Forecast JSON (per-ticker) or consolidated
    fc = _read_json(p["ticker_fc"])
    if not fc:
        all_fc = _read_json(p["panel_fc_all"]) or {}
        fc = all_fc.get(ticker) or all_fc.get(_safe_name(ticker)) or {}

    # Panel metrics
    panel_eval = _read_json(p["panel_eval"]) or {}
    metrics = panel_eval.get("metrics", panel_eval)

    return {"bands": bands, "forecast": fc, "backtest": {"metrics": metrics}}

def tabfe_recommend(universe, risk_level="medium"):
    """Simple in-app recommender for Tabular-FE (no API)."""
    picks, missing = [], []
    for t in universe:
        payload = load_tabular_fe_local(t)
        fc = payload.get("forecast") or {}
        if not fc:
            missing.append(t); continue

        probs = (fc.get("prob_up_by_horizon") or
                 fc.get("prob_up_by_horizon_meta") or
                 fc.get("prob_up_by_horizon_raw") or {})
        exps  = fc.get("exp_cum_return_by_horizon") or {}
        p12   = probs.get("12m")
        mu12  = exps.get("12m")
        if p12 is None or mu12 is None:
            missing.append(t); continue

        risk_w = {"low":0.8,"medium":1.0,"high":1.2}.get(str(risk_level).lower(),1.0)
        score = (float(p12) - 0.5) * float(mu12) * risk_w
        picks.append({"ticker": t, "score": float(score), "reason": "Tabular-FE: 12m μ × (Prob↑−0.5)"})
    picks = sorted(picks, key=lambda x: x["score"], reverse=True)
    return {"picks": picks, "missing": missing}

# -----------------------------
# LSTM API helpers
# -----------------------------
def lstm_recommend(universe, risk_level="medium"):
    try:
        return post_json(f"{API}/recommend/stocks", {"risk_level": risk_level, "candidate_universe": universe}, timeout=30)
    except Exception:
        return {"picks": [], "missing": universe}

def load_lstm_forecast(ticker: str):
    try:
        return get_json(f"{API}/forecast/{ticker}", timeout=30)
    except Exception:
        return {}

def list_lstm_tickers_fallback(universe):
    """Try to retrieve tickers from API (several likely endpoints), else fallback to universe."""
    for ep in ["/forecast/tickers", "/data/lstm_tickers", "/tickers"]:
        try:
            arr = get_json(f"{API}{ep}", timeout=15)
            if isinstance(arr, dict):
                arr = arr.get("tickers") or arr.get("symbols") or []
            if isinstance(arr, list) and arr:
                return sorted(arr)
        except Exception:
            pass
    try:
        rec = lstm_recommend(universe, "medium")
        cands = set([p["ticker"] for p in rec.get("picks", [])])
        cands.update(rec.get("missing", []))
        if cands:
            return sorted(cands)
    except Exception:
        pass
    return sorted(universe)

# -----------------------------
# PAGE — Planner & Inputs
# -----------------------------
st.title("📈 Financial Planner — UAE (MVP)")

with st.form("goals"):
    st.write("Describe your goals. Backend NLP structures them; the planner computes the monthly needs.")
    txt = st.text_area(
        "Goal text",
        height=120,
        placeholder="Example: I’m saving AED 200,000 for a house in 5 years. I earn 15k, spend 9k. Medium risk."
    )

    colA, colB, colC, colD = st.columns([1,1,1,1])
    with colA:
        age = st.number_input("Age", min_value=18, max_value=90, value=32, step=1)
    with colB:
        inc = st.number_input("Monthly income (AED)", min_value=0, value=15000, step=500)
    with colC:
        exp = st.number_input("Monthly expenses (AED)", min_value=0, value=9000, step=500)
    with colD:
        risk_tol = st.selectbox("Risk tolerance", ["Conservative","Balanced","Growth"], index=1)

    colE, colF, colG = st.columns([1,1,1])
    with colE:
        mode = st.selectbox("Planning mode", ["assumed_return","required_return"], index=0)
    with colF:
        exp_ret = st.number_input("Expected return (annual, %)", min_value=-50.0, max_value=50.0, value=7.0, step=0.5)
    with colG:
        infl = st.number_input("Inflation (annual, %)", min_value=-5.0, max_value=20.0, value=3.0, step=0.5)

    st.markdown("---")
    st.caption("Recommendation settings")

    risk_ui = st.selectbox("Scoring risk weight", ["low","medium","high"], index=1)

    model_choice = st.radio("Signal source", ["Panel (Tabular FE)", "LSTM (API)"], index=0, horizontal=True)
    compare_toggle = st.checkbox("Compare Tabular-FE vs LSTM for a selected ticker")

    default_universe = ["EMAAR.AE","ALDAR.AE","ADNOCDRILL.AE","EAND.AE","DIB.AE","DEWA.DU","FAB.AE","EMIRATESNBD.AE","AIRARABIA.AE","ARMX.AE","CBD.AE","SUKOON.AE","WATANIA.AE","ALRAMZ.AE","DEYAAR.AE","DFM.AE"]
    tabfe_pool = list_tabfe_tickers()
    lstm_pool  = list_lstm_tickers_fallback(default_universe)

    universe = st.multiselect(
        "Candidate stocks",
        sorted(set(default_universe) | set(tabfe_pool) | set(lstm_pool)),
        default=[x for x in default_universe if x in (set(tabfe_pool) | set(lstm_pool))] or default_universe[:5]
    )

    pool_for_select = tabfe_pool if model_choice == "Panel (Tabular FE)" else lstm_pool
    select_candidates = sorted(set(pool_for_select) & set(universe)) or pool_for_select or universe
    selected_ticker = st.selectbox("Select ticker to view details", select_candidates, index=0)

    submitted = st.form_submit_button("Plan")

if not submitted:
    st.stop()

# -----------------------------
# NLP
# -----------------------------
try:
    nlp = post_json(f"{API}/nlp/parse", {"text": txt}, timeout=30)
except Exception as e:
    st.error(f"NLP error: {e}")
    st.stop()

goals = nlp.get("goals") or []
st.subheader("Structured goals")
budget_map = {}
if not goals:
    st.warning("No goals extracted.")
else:
    for i, g in enumerate(goals, start=1):
        with st.expander(f"Goal {i}: {g.get('goal_type','(unknown)')}"):
            col1, col2, col3 = st.columns(3)

            amt = g.get("amount")
            if amt is None:
                notes = g.get("notes") or {}
                try: amt = float((notes.get("amount_text") or "").replace(",",""))
                except Exception: amt = None
            with col1:
                st.caption("Amount"); st.markdown(f"**{format_aed(amt) if amt else '—'}**")

            yrs = g.get("timeframe_years") or 0
            mos = g.get("timeframe_months") or 0
            if yrs or mos:
                parts = []
                if yrs: parts.append(f"{yrs} yr{'s' if yrs!=1 else ''}")
                if mos: parts.append(f"{mos} mo")
                tf_txt = " ".join(parts)
            elif g.get("timeframe_months_total"):
                m = g["timeframe_months_total"]; tf_txt = f"{m//12} yrs {m%12} mo" if m>=12 else f"{m} mo"
            else:
                tf_txt = "—"
            with col2:
                st.caption("Timeframe"); st.markdown(f"**{tf_txt}**")

            with col3:
                st.caption("Classifier confidence"); st.markdown(f"**{pct(g.get('classifier_confidence',0.0),0)}**")

            if mode == "required_return":
                st.markdown("---")
                budget = st.number_input(
                    f"Your monthly budget for Goal {i} (AED)",
                    min_value=0.0, value=0.0, step=100.0, key=f"budget_{i}"
                )
                budget_map[i-1] = float(budget)

# -----------------------------
# PLAN
# -----------------------------
st.subheader("Plan")

plan_body = {
    "profile": {
        "age": age,
        "income_monthly_aed": float(inc),
        "expenses_monthly_aed": float(exp),
        "expected_return_annual": float(exp_ret)/100.0,
        "inflation_annual": float(infl)/100.0,
        "risk_tolerance": risk_tol,
    },
    "goals_text": txt,
    "frs_score": 65,
    "mode_per_goal": mode,
    "monthly_budget_map": (budget_map if mode=="required_return" else None),
    "lstm_symbols": universe
}

try:
    plan_resp = post_json(f"{API}/plan/from-text-advanced", plan_body, timeout=45)
except Exception as e:
    st.error(f"Planner error: {e}")
    st.stop()

plan = plan_resp.get("plan") or {}
summary = plan.get("summary") or {}
per_goal = plan.get("per_goal") or []
portfolio = plan.get("portfolio") or {}
weights = portfolio.get("final_weights") or {}

# Affordability cards
st.markdown("### Affordability Summary")
c1, c2, c3, c4 = st.columns([1,1,1,0.8])
with c1:
    st.caption("Free cash / mo"); st.markdown(f"**{format_aed(summary.get('free_cash_monthly_aed'))}**")
with c2:
    st.caption("Required / mo"); st.markdown(f"**{format_aed(summary.get('required_monthly_sum_aed'))}**")
with c3:
    st.caption("Affordability ratio"); st.markdown(f"**{summary.get('affordability_ratio','—')}**")
with c4:
    st.caption("&nbsp;")
    status = (summary.get("status") or "—").title()
    color = {"Feasible":"green","Tight":"orange","Unaffordable":"red"}.get(status,"gray")
    st.markdown(f"<span class='badge' style='background:{color};color:white;border:none;'>{status}</span>", unsafe_allow_html=True)

# Per-goal outputs
if per_goal:
    st.markdown("### Per-goal outputs")
    for g in per_goal:
        with st.container(border=True):
            st.caption(g.get("goal_type","(goal)"))
            if g.get("mode") == "required_return":
                colr_top = st.columns(2)
                with colr_top[0]:
                    kpi("Budget / mo", format_aed(g.get("budget_monthly_aed")))
                with colr_top[1]:
                    rr = g.get("required_return_annual")
                    kpi("Required annual return", pct(rr,1) if rr is not None else "—")
                st.caption("Solver status")
                st.markdown(("✅ feasible" if g.get("solved") else "⚠️ infeasible"))
            else:
                kpi("Required / mo", format_aed(g.get("required_monthly_aed")))

# Portfolio cards
st.markdown("### Suggested Strategic Portfolio")
pcols = st.columns(4)
for i,(k,v) in enumerate(weights.items()):
    with pcols[i % 4]:
        kpi(k.capitalize(), pct(v,1))

# -----------------------------
# Recommended Stocks & Ticker view
# -----------------------------
st.subheader("Recommended Stocks")

if model_choice == "Panel (Tabular FE)":
    st.caption("Using **Panel (Tabular FE)** — Read from local files in `src/stock_predictions_tabular`.")
    rec = tabfe_recommend(universe, risk_ui)
else:
    st.caption(f"Using **LSTM (API)** — base = {API}")
    rec = lstm_recommend(universe, risk_ui)

if not rec.get("picks"):
    st.info("No picks yet — generate forecasts or broaden your universe.")
else:
    for p in rec["picks"][:10]:
        st.write(f"- **{p['ticker']}** — Score **{p['score']:.3f}**")

st.markdown("---")

# ---- Single ticker details (based on current model choice)
def show_ticker_panel_tabfe(ticker: str):
    payload = load_tabular_fe_local(ticker)
    bands = payload.get("bands") or []
    fc    = payload.get("forecast") or {}
    panel_metrics = payload.get("backtest",{}).get("metrics",{})

    probs = (fc.get("prob_up_by_horizon") or
             fc.get("prob_up_by_horizon_meta") or
             fc.get("prob_up_by_horizon_raw") or {})
    p12 = probs.get("12m")
    da_vec = panel_metrics.get("directional_accuracy_default") or panel_metrics.get("directional_accuracy_tuned") or []
    da12 = da_vec[0] if isinstance(da_vec, list) and len(da_vec) else None

    exp12, lo12, hi12 = None, None, None
    price_now = last_close(ticker)
    if bands and price_now:
        last = bands[-1]
        try:
            p50 = float(last["P50"]); p10 = float(last["P10"]); p90 = float(last["P90"])
            exp12 = (p50 / price_now) - 1.0
            lo12  = (p10 / price_now) - 1.0
            hi12  = (p90 / price_now) - 1.0
        except Exception:
            pass

    st.markdown(f"#### **{ticker}** — Tabular-FE")

    # KPIs arranged 2×2 to avoid truncation in compare view
    r1c1, r1c2 = st.columns(2)
    with r1c1: kpi("Exp. 12-mo return", pct(exp12,1) if exp12 is not None else "N/A")
    with r1c2: kpi("12-mo band", f"{pct(lo12,1)} to {pct(hi12,1)}" if lo12 is not None and hi12 is not None else "N/A", "P10–P90")

    r2c1, r2c2 = st.columns(2)
    with r2c1: kpi("Prob ↑ (12m)", pct(p12,0) if p12 is not None else "—")
    with r2c2: kpi("OOS DA (panel, 12m)", pct(da12,0) if da12 is not None else "—")

    if bands:
        with st.expander("Show 12-month bands chart"):
            df = pd.DataFrame(bands).rename(columns={"Date":"date"}).set_index("date")[["P10","P50","P90"]]
            st.line_chart(df)

def show_ticker_lstm(ticker: str):
    fres = load_lstm_forecast(ticker)
    bands = fres.get("bands") or []
    fc = (fres.get("forecast") or {})
    bt = (fres.get("backtest") or {}).get("metrics", {})

    meta = fc.get("prob_up_by_horizon_meta") or {}
    raw  = fc.get("prob_up_by_horizon_raw") or {}
    p12  = meta.get("12m", raw.get("12m"))
    da_list = bt.get("directional_accuracy_best") or []
    da12 = da_list[0] if len(da_list) >= 1 else None

    exp12, lo12, hi12 = None, None, None
    price_now = last_close(ticker)
    if bands and price_now:
        last = bands[-1]
        try:
            p50 = float(last["P50"]); p10 = float(last["P10"]); p90 = float(last["P90"])
            exp12 = (p50 / price_now) - 1.0
            lo12  = (p10 / price_now) - 1.0
            hi12  = (p90 / price_now) - 1.0
        except Exception:
            pass

    st.markdown(f"#### **{ticker}** — LSTM")

    r1c1, r1c2 = st.columns(2)
    with r1c1: kpi("Exp. 12-mo return", pct(exp12,1) if exp12 is not None else "N/A")
    with r1c2: kpi("12-mo band", f"{pct(lo12,1)} to {pct(hi12,1)}" if lo12 is not None and hi12 is not None else "N/A", "P10–P90")

    r2c1, r2c2 = st.columns(2)
    with r2c1: kpi("Prob ↑ (12m)", pct(p12,0) if p12 is not None else "—")
    with r2c2: kpi("DA OOS (12m)", pct(da12,0) if da12 is not None else "—")

    if bands:
        with st.expander("Show 12-month bands chart"):
            df = pd.DataFrame(bands).rename(columns={"Date":"date"}).set_index("date")[["P10","P50","P90"]]
            st.line_chart(df)

# show one ticker for current model
if model_choice == "Panel (Tabular FE)":
    show_ticker_panel_tabfe(selected_ticker)
else:
    show_ticker_lstm(selected_ticker)

# ---- Compare models (side-by-side)
if compare_toggle:
    st.markdown("---")
    st.subheader("Compare models (same ticker)")
    tabfe_pool = list_tabfe_tickers()            # refresh, in case files changed
    lstm_pool  = list_lstm_tickers_fallback(universe)
    both_pool = sorted(set(tabfe_pool) & set(lstm_pool))
    if not both_pool:
        st.info("No tickers found in **both** Tabular-FE and LSTM results.")
    else:
        cmp_ticker = st.selectbox("Select ticker to compare", [selected_ticker] + [t for t in both_pool if t != selected_ticker])
        colL, colR = st.columns(2)
        with colL:
            show_ticker_panel_tabfe(cmp_ticker)
        with colR:
            show_ticker_lstm(cmp_ticker)

        # overlay P50 comparison, if both bands exist
        tf_payload = load_tabular_fe_local(cmp_ticker)
        lstm_payload = load_lstm_forecast(cmp_ticker)
        tf_bands = tf_payload.get("bands") or []
        ls_bands = lstm_payload.get("bands") or []
        if tf_bands and ls_bands:
            st.caption("Overlay: P50 (Tabular-FE vs LSTM)")
            dfa = pd.DataFrame(tf_bands).rename(columns={"Date":"date"})[["date","P50"]].set_index("date")
            dfa.rename(columns={"P50":"TabFE_P50"}, inplace=True)
            dfb = pd.DataFrame(ls_bands).rename(columns={"Date":"date"})[["date","P50"]].set_index("date")
            dfb.rename(columns={"P50":"LSTM_P50"}, inplace=True)
            join = dfa.join(dfb, how="inner")
            if not join.empty:
                st.line_chart(join)

st.caption("Not financial advice. Educational demo.")
