import re
import json
import os
from typing import Optional, Tuple, Dict, Any, List

# ==============================
# NLP Goal Parser — v1.7 (refined)
# ==============================
SCHEMA_VERSION = "1.7"

CURRENCY_MAP = {
    "aed": "AED", "dhs": "AED", "dh": "AED", "dirham": "AED", "dirhams": "AED", "د.إ": "AED",
    "usd": "USD", "dollar": "USD", "dollars": "USD", "$": "USD",
    "inr": "INR", "rs": "INR", "rupee": "INR", "rupees": "INR", "₹": "INR",
    "eur": "EUR", "€": "EUR", "gbp": "GBP", "£": "GBP"
}
UAE_HINTS = {
    "dubai","abu dhabi","sharjah","uae","united arab emirates",
    "ajman","ras al khaimah","rak","fujairah","umm al quwain"
}

GOAL_KEYWORDS = {
    "house purchase": [
        r"\b(build|buy|purchase)\b.*\b(house|home|apartment|flat)\b",
        r"\b(3bhk|2bhk)\b",
        r"\b(house|home|apartment|flat)\b"
    ],
    "wedding": [r"\b(wedding|marriage)\b"],
    "education": [r"\b(college|school|university|tuition|fee)\b"],
    "retirement": [r"\b(retire|retirement)\b"],
    "vehicle purchase": [r"\b(car|bike)\b"],
    "luxury purchase": [r"\b(lamborghini|ferrari|rolex|luxury)\b"],
    "business": [
        r"\b(start|open|launch)\b.*\b(business|restaurant|company|startup)\b",
        r"\b(business|restaurant)\b"
    ],
    "travel": [r"\b(travel|travelling|trip|world tour)\b"],
    "real estate investment": [
        r"\breal estate\b.*\b(invest|portfolio)\b",
        r"\bproperty\b.*\bportfolio\b",
        r"\b(real estate|property)\b.*\b(invest|income)\b"
    ],
    "passive income": [r"\bpassive income\b", r"\bmonthly income\b", r"\bincome of\b"],
    "gift": [r"\bgift\b.*\b(apartment|house|car|watch)\b"]
}

def clean_number_str(s: str) -> str:
    return re.sub(r"[,\s]", "", s)

def normalize_currency(raw):
    if not raw:
        return None
    raw_l = str(raw).strip().lower()
    return CURRENCY_MAP.get(raw_l, CURRENCY_MAP.get(raw, None))

def has_uae_context(text: str) -> bool:
    tl = text.lower()
    return any(h in tl for h in UAE_HINTS)

def score_confidence(found: Dict[str, Any]) -> float:
    score = 0.0
    score += 0.8 if found.get("goal_type_source") == "keyword" else (0.5 if found.get("goal_type") else 0.0)
    if found.get("amount") is not None:
        score += 0.9 if found.get("amount_source") in ("symbol_pair","word_pair","scaled_word","scaled_suffix") else 0.6
    if found.get("currency"):
        score += 0.6
    elif found.get("currency_defaulted"):
        score += 0.3
    if found.get("timeframe_years") is not None or found.get("timeframe_months") is not None:
        score += 0.6
    if found.get("location"):
        score += 0.6 if found.get("location_source") == "explicit" else 0.4
    return max(0.0, min(1.0, round(score/3.2, 2)))

def find_amount_and_currency(text: str):
    notes: List[str] = []
    # remove BHK and age-like "I am 30"
    safe_text = re.sub(r"\b\d+\s*bhk\b", "", text, flags=re.I)
    safe_text = re.sub(r"\bI\s*am\s*\d+\b", "", safe_text, flags=re.I)

    # 80k / 2.5m
    m = re.search(r"\b(?P<num>\d+(?:\.\d+)?)\s*(?P<suf>[kKmM])\b(?:\s*(?P<cur>aed|usd|inr|rs|د\.إ|\$|₹))?", safe_text)
    if m:
        n = float(m.group("num"))
        mult = 1_000 if m.group("suf").lower() == "k" else 1_000_000
        amount = n * mult
        cur = normalize_currency(m.group("cur")) if m.group("cur") else None
        return amount, cur, "scaled_suffix", notes

    # one/half million/thousand
    m = re.search(
        r"\b(?:(?P<mult>half|one|two|three|four|five|six|seven|eight|nine|ten)\s+)?(?:a\s*)?"
        r"(?P<unit>million|thousand)\b(?:\s*(?:of)?\s*(?P<cur>aed|usd|inr|rs|د\.إ|\$|₹))?",
        safe_text, flags=re.I
    )
    if m:
        mult_map = {"half":0.5,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}
        mult = mult_map.get((m.group("mult") or "").lower(), 1)
        unit = m.group("unit").lower()
        amount = mult * (1_000_000 if unit == "million" else 1_000)
        cur = normalize_currency(m.group("cur")) if m.group("cur") else None
        return float(amount), cur, "scaled_word", notes

    # $ 1000 / ₹ 500,000
    m = re.search(r"(?P<cur>[$₹])\s*(?P<number>[\d,]+(?:\.\d+)?)", safe_text)
    if m:
        num = float(clean_number_str(m.group("number")))
        return num, normalize_currency(m.group("cur")), "symbol_pair", notes

    # AED 150,000
    m = re.search(r"\b(?P<cur>aed|usd|inr|rs|د\.إ)\b[\s:,-]*?(?P<number>[\d,]+(?:\.\d+)?)", safe_text, flags=re.I)
    if m:
        num = float(clean_number_str(m.group("number")))
        return num, normalize_currency(m.group("cur")), "word_pair", notes

    # 150,000 AED
    m = re.search(r"(?P<number>[\d,]+(?:\.\d+)?)\s*(?P<cur>aed|usd|inr|rs|د\.إ|\$|₹)\b", safe_text, flags=re.I)
    if m:
        num = float(clean_number_str(m.group("number")))
        return num, normalize_currency(m.group("cur")), "word_pair", notes

    # Fallback: largest standalone number not tied to time or per-year
    candidates = []
    for nm in re.finditer(r"[\d,]+(?:\.\d+)?", safe_text):
        val = float(clean_number_str(nm.group(0)))
        after = safe_text[nm.end(): nm.end()+12].lower()
        if re.match(r"\s*(years?|yrs?|months?|/year|per\s+year)\b", after):
            continue
        candidates.append(val)
    if candidates:
        return max(candidates), None, "fallback", notes

    return None, None, "none", notes

def find_timeframe(text: str):
    tl = text.lower()
    years = None
    months = None
    notes: List[str] = []

    m = re.search(r"(\d+)\s*(?:years?|yrs?)\b", tl)
    if m:
        years = int(m.group(1))
    m2 = re.search(r"(\d+)\s*(?:months?|mos?)\b", tl)
    if m2:
        months = int(m2.group(1))

    if "next year" in tl and years is None:
        years = 1
        notes.append("Interpreted 'next year' as 1 year.")
    if any(k in tl for k in ("asap", "immediately", "right away", "as soon as possible")):
        years = 0; months = 0
        notes.append("Interpreted ASAP/Immediate as 0 years, 0 months.")

    return years, months, notes

def timeframe_months_total(years, months):
    if years is None and months is None:
        return None
    return (years or 0) * 12 + (months or 0)

def classify_goal(text: str):
    tl = text.lower()
    for goal, patterns in GOAL_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, tl, flags=re.I):
                return goal, "keyword"
    if "invest" in tl:
        return "investment", "heuristic"
    return None, "none"

def _tidy_place(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    small = {"of","and","the","in","at","on","for","to","by"}
    parts = s.split()
    out = []
    for i, w in enumerate(parts):
        ww = w.lower()
        out.append(ww if (i > 0 and ww in small) else ww.capitalize())
    return " ".join(out)

def extract_location(text: str):
    # stop before time phrases & urgency; trim punctuation; reject obvious non-places
    lookahead = (
        r"(?:"
        r",|\.|;|:|\(|\)"                                  # punctuation
        r"|\bto\b|\band\b|\bwhich\b|\bthat\b"              # conjunctions
        r"|\bby\b\s+\d+\s+(?:years?|months?)"              # 'by 5 years'
        r"|\bin\s+\d+\s+(?:years?|months?)"                # 'in 5 years'
        r"|\bin\s+the\s+next\s+\d+\s+(?:years?|months?)"   # 'in the next 5 years'
        r"|\bas\s+soon\s+as\s+possible\b|\basap\b|\bimmediately\b|\bright\s+away\b"
        r"|$"
        r")"
    )
    leaders = r"(?:\bin\b|\bat\b|\bfrom\b|\bto\s+settle\s+in\b|\bsettle\s+in\b)"
    pat = rf"{leaders}\s+([A-Za-z][A-Za-z0-9\s\-\&']+?)(?={lookahead})"
    m = re.search(pat, text, flags=re.IGNORECASE)
    if m:
        cand = m.group(1).strip()
        if re.match(r"the\s+next\b", cand, flags=re.IGNORECASE):
            cand = ""
        cand = re.sub(r"[\.]+$", "", cand).strip()
        cand = re.sub(r"\s+", " ", cand)

        bad_phrases = [
            r"^my\s+account\b",
            r"^my\s+home\s+country\b",
            r"^my\s+job\b",
            r"^my\s+life\b",
            r"^the\s+world\b",
        ]
        for bp in bad_phrases:
            if re.search(bp, cand, flags=re.IGNORECASE):
                cand = ""
                break

        cand = re.sub(r"^(my\s+hometown)\b.*$", r"\1", cand, flags=re.IGNORECASE)

        if cand:
            return _tidy_place(cand), "explicit"

    tl = text.lower()
    for hint in UAE_HINTS:
        if hint in tl:
            return _tidy_place(" ".join(w.capitalize() for w in hint.split())), "hint"

    m2 = re.search(r"\b(kerala|india|america|usa|uk|canada|qatar|saudi arabia)\b", tl, flags=re.IGNORECASE)
    if m2:
        return _tidy_place(m2.group(1)), "hint"

    return None, "none"

def parse_single(text: str):
    amount, currency, amount_source, amt_notes = find_amount_and_currency(text)
    years, months, tf_notes = find_timeframe(text)
    goal_type, goal_source = classify_goal(text)
    location, loc_source = extract_location(text)

    assumptions: List[str] = []
    currency_defaulted = False
    if amount is not None and not currency and has_uae_context(text):
        currency = "AED"
        currency_defaulted = True
        assumptions.append("Defaulted currency to AED due to UAE context.")

    tm_total = timeframe_months_total(years, months)

    found: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "original_text": text,
        "goal_type": goal_type,
        "goal_type_source": goal_source,
        "amount": amount,
        "amount_source": amount_source,
        "currency": currency,
        "currency_defaulted": currency_defaulted,
        "timeframe_years": years,
        "timeframe_months": months,
        "timeframe_months_total": tm_total,
        "location": location,
        "location_source": loc_source,
        "assumptions": assumptions,
        "extraction_notes": list(filter(None, amt_notes + tf_notes))
    }
    found["confidence"] = score_confidence(found)
    return found

def split_into_units(text: str):
    parts = re.split(r"(?<=[\.\?\!])\s+|\s*(?<=\b)(?:;)\s*", text.strip())
    units: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if " and " in p.lower() and "be able to" not in p.lower() and re.search(r"\b(buy|build|save|invest|retire|start|gift|travel)\b", p, re.I):
            subs = re.split(r"\s+\band\b\s+", p, flags=re.I)
            for s in subs:
                s = s.strip().strip(",")
                if s:
                    units.append(s)
        else:
            units.append(p)
    return units

def parse_text(text: str):
    units = split_into_units(text)
    goals = [parse_single(u) for u in units]
    return {"schema_version": SCHEMA_VERSION, "input_text": text, "goals": goals}

# ===============================
# Runner: prints & saves outputs
# ===============================
if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("Please install pandas: pip install pandas")

    samples = [
        "I want to save 300,000 AED by the next 10 years to build a house in Kerala.",
        "My daughter's wedding is planned in the next 3 years, how do I get 150,000 AED for it?",
        "I want to build a real estate investment portfolio in Dubai and be able to live off of the income generated by it.",
        "I want to buy a car worth 80k AED next year. How do I save or invest for it?",
        "My son will start college in 2 years, and I have to pay a fee of 25000AED per year for 4 years for it.",
        "I am renting an apartment right now in Dubai, and I want to buy my dream apartment of 3BHK in Damac Hills by 5 years.",
        "I am 30 now and want to retire in 20 years and live peacefully in my home country without worrying about money.",
        "I want to quit my job and start travelling the world in 10 years time.",
        "I want to start a restaurant business in Dubai in 5 years which would have an initial cost of 300,000 AED.",
        "I want to retire in 5 years with 1 million AED in my account.",
        "I want to make a passive income of 100,000 AED yearly.",
        "I want to buy a Lamborghini in Dubai in the next 5 years.",
        "I want to make enough money to retire and settle in America.",
        "I want to leave Dubai and start a business in my hometown as soon as possible. It might cost around 200,000 AED.",
        "I want to gift my wife a 2BHK apartment in Dubai Marina."
    ]

    rows: List[Dict[str, Any]] = []
    for s in samples:
        out = parse_text(s)
        for g in out["goals"]:
            rows.append({
                "original_text": g["original_text"],
                "goal_type": g["goal_type"],
                "amount": g["amount"],
                "currency": g["currency"],
                "timeframe_years": g["timeframe_years"],
                "timeframe_months": g["timeframe_months"],
                "timeframe_months_total": g["timeframe_months_total"],
                "location": g["location"],
                "confidence": g["confidence"],
                "assumptions": "; ".join(g.get("assumptions", [])),
                "notes": "; ".join(g.get("extraction_notes", [])),
            })

    df = pd.DataFrame(rows, columns=[
        "original_text","goal_type","amount","currency",
        "timeframe_years","timeframe_months","timeframe_months_total",
        "location","confidence","assumptions","notes"
    ])
    print(df.to_string(index=False))

    # CHANGE THIS PATH IF NEEDED
    OUTPUT_DIR = r"C:/Users/shezan.shabeer/AI/financial-planner-ai"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "parsed_goals.csv")
    json_path = os.path.join(OUTPUT_DIR, "parsed_goals.json")

    df.to_csv(csv_path, index=False, encoding="utf-8")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("Wrote:", csv_path)
    print("Wrote:", json_path)
