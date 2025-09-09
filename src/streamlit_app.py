# src/streamlit_app.py
import os
import requests
import streamlit as st
import pandas as pd

API = os.environ.get("PLANNER_API_BASE", "http://localhost:8000")

# ---------------------------
# Pretty-format helpers
# ---------------------------
def fmt_aed(x):
    try:
        return f"AED {float(x):,.2f}"
    except Exception:
        return "—"

def fmt_pct(x, digits=1):
    try:
        return f"{float(x)*100:.{digits}f}%"
    except Exception:
        return "—"

def fmt_years_months(y, m):
    y = int(y or 0)
    m = int(m or 0)
    parts = []
    if y: parts.append(f"{y} yr{'s' if y!=1 else ''}")
    if m: parts.append(f"{m} mo{'s' if m!=1 else ''}")
    return " ".join(parts) if parts else "—"

def badge(text, color="#4caf50"):
    return f"""
    <span style="padding:4px 10px;border-radius:999px;background:{color};color:white;font-size:12px;">
      {text}
    </span>
    """

def status_color(status):
    s = (status or "").lower()
    if s == "feasible": return "#2e7d32"
    if s == "tight":    return "#ef6c00"
    return "#c62828"

# ---------------------------
# Custom metric card (no truncation)
# ---------------------------
METRIC_CSS = """
<style>
.bv {border:1px solid #eee;border-radius:8px;background:#fafafa;padding:10px 12px;}
.bv .lbl {font-size:0.82rem;color:#666;margin-bottom:3px;}
.bv .val {font-size:1.6rem;font-weight:700;line-height:1.25;
          word-break:break-word;white-space:normal;overflow-wrap:anywhere;}
.bv .sub {font-size:0.75rem;color:#999;margin-top:2px;}
</style>
"""
st.markdown(METRIC_CSS, unsafe_allow_html=True)

def bigvalue(label, value, sub=None):
    st.markdown(
        f"""
        <div class="bv">
          <div class="lbl">{label}</div>
          <div class="val">{value}</div>
          {f'<div class="sub">{sub}</div>' if sub else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------
# Renderers
# ---------------------------
def render_goals(nlp_payload: dict):
    st.subheader("Structured Goals")
    if not nlp_payload or not nlp_payload.get("goals"):
        st.info("No goals parsed.")
        return

    # Summary table
    rows = []
    for g in nlp_payload["goals"]:
        rows.append({
            "Goal type": g.get("goal_type", "—"),
            "Amount": f"{g.get('currency','AED') or 'AED'} {g.get('amount'):,}" if g.get("amount") else "—",
            "Timeframe": fmt_years_months(g.get("timeframe_years"), g.get("timeframe_months")),
            "Location": g.get("location") or "—",
            "Confidence": f"{(g.get('classifier_confidence') or 0)*100:.0f}%"
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    # Per-goal cards
    for i, g in enumerate(nlp_payload["goals"]):
        with st.expander(f"Goal {i+1}: {g.get('goal_type','(unknown)')}", expanded=False):
            # widen the amount column so big numbers fit comfortably
            c1, c2, c3 = st.columns([2.2, 1.1, 1.1])
            with c1:
                amt = fmt_aed(g.get("amount")) if g.get("amount") is not None else "—"
                bigvalue("Amount", amt)
            with c2:
                bigvalue("Timeframe", fmt_years_months(g.get("timeframe_years"), g.get("timeframe_months")))
            with c3:
                bigvalue("Confidence", f"{(g.get('classifier_confidence') or 0)*100:.0f}%")
            if g.get("notes"):
                st.caption("Notes")
                st.write({k:v for k,v in g["notes"].items() if v})

def render_plan(plan_payload: dict):
    if not plan_payload:
        st.info("No plan available.")
        return

    plan = plan_payload.get("plan") or {}

    # Affordability summary
    st.subheader("Affordability Summary")
    summary = plan.get("summary", {})

    # Wider first two columns to avoid truncation
    c1, c2, c3, c4 = st.columns([1.4, 1.4, 1.0, 0.9])
    with c1:
        bigvalue("Free cash / mo", fmt_aed(summary.get("free_cash_monthly_aed")))
    with c2:
        bigvalue("Required / mo", fmt_aed(summary.get("required_monthly_sum_aed")))
    with c3:
        ratio = summary.get("affordability_ratio")
        try:
            txt = f"{float(ratio):.2f}"
        except Exception:
            txt = "—"
        bigvalue("Affordability ratio", txt)
    with c4:
        s = summary.get("status", "unknown").title()
        st.markdown(f"<div style='margin-top:30px'>{badge(s, status_color(s))}</div>", unsafe_allow_html=True)

    # Portfolio block
    st.subheader("Suggested Strategic Portfolio")
    pf = plan.get("portfolio", {})
    bucket = pf.get("bucket", "—")
    cap = pf.get("equity_cap_from_age", None)
    st.write(f"Risk bucket: **{bucket}**  |  Equity cap from age: **{fmt_pct(cap,1) if cap is not None else '—'}**")

    final_w = pf.get("final_weights") or {}
    if final_w:
        dfw = pd.DataFrame([final_w]).T.reset_index()
        dfw.columns = ["Asset", "Weight"]
        dfw = dfw.sort_values("Weight", ascending=False)
        c1, c2 = st.columns([1,1])
        with c1:
            st.dataframe(dfw, use_container_width=True)
        with c2:
            st.bar_chart(dfw.set_index("Asset"))

    # Per-goal requirements
    st.subheader("Per-Goal Requirements")
    for i, g in enumerate(plan.get("per_goal", [])):
        title = g.get("goal_type", f"Goal {i+1}")
        with st.container(border=True):
            st.markdown(f"**{title}**")
            mode = g.get("mode")
            if mode == "required_return":
                r = g.get("required_return_annual")
                bigvalue("Required annual return", fmt_pct(r, 2))
                st.caption(f"Budget provided: {fmt_aed(g.get('budget_monthly_aed'))}")
            else:
                bigvalue("Monthly needed", fmt_aed(g.get("required_monthly_aed")))
                ass = g.get("assumptions", {})
                st.caption(
                    f"Assumptions: Expected {fmt_pct(ass.get('expected_return_annual'),1)}, "
                    f"Inflation {fmt_pct(ass.get('inflation_annual'),1)}, "
                    f"Real {fmt_pct(ass.get('real_return_annual'),2)}"
                )

    # LSTM stock ideas inside the plan (if any)
    st.subheader("AI Stock Ideas (from plan)")
    si = plan.get("stock_recommendations")
    if not si:
        st.caption("No symbols were provided or model returned nothing.")
    else:
        tabs = st.tabs(list(si.keys()))
        for tab, k in zip(tabs, si.keys()):
            with tab:
                df = pd.DataFrame(si[k])
                if not df.empty:
                    df["Exp CAGR"] = df["exp_cagr"].apply(lambda v: fmt_pct(v,2))
                    df["Weight"] = df["suggested_weight"].apply(lambda v: fmt_pct(v,1))
                    st.dataframe(df[["symbol","Exp CAGR","Weight"]], use_container_width=True)
                else:
                    st.caption("No picks for this horizon.")

# ---------------------------
# App layout
# ---------------------------
st.set_page_config(page_title="Financial Planner (MVP)", page_icon="📈", layout="centered")
st.title("📈 Financial Planner — UAE (MVP)")
st.caption(f"API base: {API}")

# Health (optional)
with st.expander("API health", expanded=False):
    try:
        h = requests.get(f"{API}/health", timeout=5).json()
        if h.get("status") == "ok":
            st.success("OK")
        else:
            st.warning(h)
    except Exception as e:
        st.warning(f"Health check failed: {e}")

# Input form
with st.form("goals"):
    st.write("Describe your goals. Your FastAPI NLP+planner will parse and plan.")
    txt = st.text_area(
        "Goal text",
        height=120,
        placeholder="Example: I’m saving for a 200k AED down payment in 5 years, medium risk, prefer dividends."
    )

    col1, col2 = st.columns(2)
    with col1:
        risk = st.selectbox("Risk level (for recommender weighting)", ["low", "medium", "high"], index=1)
    with col2:
        rt_opt = st.selectbox("Risk tolerance (planner)", ["Conservative", "Balanced", "Growth"], index=1)

    st.markdown("**Profile for the planner**")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=32, step=1)
    with c2:
        income = st.number_input("Monthly income (AED)", min_value=0.0, value=15000.0, step=500.0, format="%.2f")
    with c3:
        expenses = st.number_input("Monthly expenses (AED)", min_value=0.0, value=8000.0, step=500.0, format="%.2f")

    universe = st.multiselect(
        "Candidate stocks (also sent to planner's AI ideas)",
        ["EMAAR.AE","ALDAR.AE","ADNOCDRILL.AE","EAND.AE","DIB.AE","DEWA.DU","FAB.AE","EMIRATESNBD.AE","AIRARABIA.AE"],
        default=["EMAAR.AE","ALDAR.AE","ADNOCDRILL.AE","EAND.AE","DIB.AE"]
    )

    submitted = st.form_submit_button("Plan")

if submitted:
    # 1) NLP parse
    nlp_payload = None
    try:
        nlp_res = requests.post(f"{API}/nlp/parse", json={"text": txt}, timeout=30)
        nlp_res.raise_for_status()
        nlp_payload = nlp_res.json()
    except Exception as e:
        st.warning(f"NLP endpoint unavailable, continuing without it. ({e})")

    if nlp_payload:
        render_goals(nlp_payload)

    # 2) Planner call
    st.subheader("Plan")
    profile = {
        "age": int(age),
        "income_monthly_aed": float(income),
        "expenses_monthly_aed": float(expenses),
        "expected_return_annual": 0.07,
        "inflation_annual": 0.03,
        "risk_tolerance": rt_opt,
    }
    plan_payload = {
        "profile": profile,
        "goals_text": txt,
        "frs_score": 65,
        "mode_per_goal": "assumed_return",
        "monthly_budget_map": None,
        "lstm_symbols": universe
    }
    plan_resp = None
    try:
        r = requests.post(f"{API}/plan/from-text-advanced", json=plan_payload, timeout=60)
        r.raise_for_status()
        plan_resp = r.json()
    except Exception as e:
        st.error(f"Planner call failed: {e}")

    if plan_resp:
        render_plan(plan_resp)

    # 3) Recommender (12m)
    st.subheader("Recommended Stocks (12m view)")
    try:
        rec_payload = {"risk_level": risk, "candidate_universe": universe}
        rec = requests.post(f"{API}/recommend/stocks", json=rec_payload, timeout=30)
        rec.raise_for_status()
        rec = rec.json()

        if not rec.get("picks"):
            missing = rec.get("missing") or []
            st.info("No picks — either forecasts aren’t generated yet or tickers don’t match your files.")
            if missing:
                st.caption(f"Missing forecast files for: {', '.join(missing)}")
        else:
            for p in rec["picks"]:
                st.markdown(f"### **{p['ticker']}** — Score: **{p['score']:.3f}**")
                st.caption(p.get("reason", ""))

                # Chart 12m bands
                try:
                    fres = requests.get(f"{API}/forecast/{p['ticker']}", timeout=30)
                    fres.raise_for_status()
                    fres = fres.json()
                    bands = fres.get("bands")
                    if bands:
                        df = pd.DataFrame(bands).rename(columns={"Date":"date"})
                        if "date" in df.columns:
                            df["date"] = pd.to_datetime(df["date"])
                            df = df.set_index("date")[["P10","P50","P90"]]
                            st.line_chart(df)
                        else:
                            st.caption("No date column in bands response.")
                    else:
                        st.caption("No 12m bands available.")
                except Exception as e:
                    st.caption(f"Chart load failed for {p['ticker']}: {e}")
                st.divider()
    except Exception as e:
        st.error(f"Recommender call failed: {e}")

st.caption("Not financial advice. Educational demo.")