import os, requests, streamlit as st, pandas as pd

API = os.environ.get("PLANNER_API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Financial Planner (MVP)", page_icon="📈", layout="centered")
st.title("📈 Financial Planner — UAE (MVP)")

with st.form("goals"):
    st.write("Describe your goals. Your existing FastAPI NLP+planner stays in charge.")
    txt = st.text_area("Goal text", height=120,
        placeholder="Example: I’m saving for a down payment in 5 years, medium risk, no leverage, prefer dividends.")
    risk = st.selectbox("Risk level (used for weighting picks)", ["low","medium","high"], index=1)
    universe = st.multiselect(
        "Candidate stocks",
        ["EMAAR.AE","ALDAR.AE","ADNOCDRILL.AE","EAND.AE","DIB.AE","DEWA.DU","FAB.AE","EMIRATESNBD.AE","AIRARABIA.AE"],
        default=["EMAAR.AE","ALDAR.AE","ADNOCDRILL.AE","EAND.AE","DIB.AE"]
    )
    submitted = st.form_submit_button("Plan")

if submitted:
    # 1) call YOUR existing NLP endpoint (adjust path if different)
    try:
        nlp_res = requests.post(f"{API}/nlp/parse", json={"text": txt}, timeout=30)
        nlp_res.raise_for_status()
        goals = nlp_res.json()
        st.subheader("Structured Goals")
        st.json(goals)
    except Exception as e:
        st.warning(f"NLP endpoint unavailable, proceeding with UI risk only. ({e})")
        goals = {"risk_level": risk}

    # (Optional) call YOUR existing planner endpoint here if you want to show plan numbers
    # plan_res = requests.post(f"{API}/planner/plan", json={"goals": goals}).json()
    # st.subheader("Plan")
    # st.json(plan_res)

    # 2) get stock picks using our NEW endpoint
    rec_payload = {
        "risk_level": goals.get("risk_level", risk),
        "candidate_universe": universe
    }
    rec = requests.post(f"{API}/recommend/stocks", json=rec_payload, timeout=30).json()
    st.info("recs = ")
    st.json(rec)

    st.subheader("Recommended Stocks")
    if not rec["picks"]:
        st.info("No picks — run the refresh job to generate forecasts first.")
    for p in rec["picks"]:
        st.markdown(f"### **{p['ticker']}** — Score: **{p['score']:.3f}**")
        st.caption(p["reason"])

        fres = requests.get(f"{API}/forecast/{p['ticker']}", timeout=30).json()
        bands = fres.get("bands")
        if bands:
            df = pd.DataFrame(bands).rename(columns={"Date":"date"}).set_index("date")[["P10","P50","P90"]]
            st.line_chart(df)
        st.divider()

st.caption("Not financial advice. Educational demo.")