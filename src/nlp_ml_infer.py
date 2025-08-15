# src/nlp_ml_infer.py
# Inference pipeline: sentence split -> mask (amt+cur only) -> classifier -> spaCy NER -> postprocess -> structured JSON
# This version adds robust fallbacks for TIME and LOC (and amt/cur if NER misses them).

import re
import math
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy

# ----------------------------
# Config / Paths
# ----------------------------
CLS_DIR = "artifacts/goal_type/best"          # HF classifier dir
NER_DIR = "artifacts/ner/model-last"          # spaCy NER dir
MAX_LEN = 128
LOW_CONF = 0.55                               # threshold for light rules

# masking: match your training
MASK_AMT_CUR = True
MASK_TIME    = False
MASK_LOC     = False

CUR_WORDS = r"(?:aed|usd|inr|eur|gbp|dhs|dirhams?|د\.إ|\$|₹|€|£)"
TIME_PAT  = r"(?:\b\d+\s+(?:years?|yrs?|months?|mos?)\b|\bnext\s+year\b|\bnext\s+\d+\s+(?:years?|months?)\b)"
LOC_HINTS = [
    "dubai","abu dhabi","sharjah","uae","united arab emirates","ajman",
    "ras al khaimah","rak","fujairah","umm al quwain","deira","mirdif",
    "damac hills","jlt","jbr","downtown dubai","dubai marina","al barsha",
    "kerala","america","usa","uk","canada","qatar","saudi arabia","dubai hills"
]
LOC_RE = re.compile(r"\b(" + "|".join(re.escape(x) for x in LOC_HINTS) + r")\b", flags=re.I)

# keywords for low-confidence nudges
VEHICLE_WORDS = re.compile(r"\b(car|bike|suv|sedan|mercedes|nissan|toyota|ferrari|lamborghini)\b", re.I)
BUSINESS_WORDS= re.compile(r"\b(start|open|launch|restaurant|cafe|startup|trading company|hotel|business)\b", re.I)
WEDDING_WORDS = re.compile(r"\b(wedding|marriage)\b", re.I)
EDU_WORDS     = re.compile(r"\b(school|college|university|tuition|fee|overseas education)\b", re.I)
RETIRE_WORDS  = re.compile(r"\b(retire|retirement)\b", re.I)
LUX_WORDS     = re.compile(r"\b(rolex|luxury|lamborghini|ferrari)\b", re.I)
TRAVEL_WORDS  = re.compile(r"\b(travel|trip|world tour|europe)\b", re.I)
REI_WORDS     = re.compile(r"\b(real\s*estate|rental income|portfolio)\b", re.I)
PASSIVE_WORDS = re.compile(r"\b(passive income|monthly income|income of)\b", re.I)

CURRENCY_MAP = {
    "aed": "AED", "dhs": "AED", "dh": "AED", "dirham": "AED", "dirhams": "AED", "د.إ": "AED",
    "usd": "USD", "dollar": "USD", "dollars": "USD", "$": "USD",
    "inr": "INR", "rs": "INR", "rupee": "INR", "rupees": "INR", "₹": "INR",
    "eur": "EUR", "€": "EUR", "gbp": "GBP", "£": "GBP"
}

# ----------------------------
# Utils
# ----------------------------
def mask_text(t: str) -> str:
    s = t
    if MASK_AMT_CUR:
        s = re.sub(r"\b\d+(?:\.\d+)?\s*[kKmM]\b", "<AMOUNT>", s)
        s = re.sub(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b", "<AMOUNT>", s)
        s = re.sub(r"\b\d+(?:\.\d+)?\b", "<AMOUNT>", s)
        s = re.sub(rf"\b{CUR_WORDS}\b", "<CUR>", s, flags=re.I)
    if MASK_TIME:
        s = re.sub(TIME_PAT, "<TIME>", s, flags=re.I)
    if MASK_LOC:
        s = LOC_RE.sub("<LOC>", s)
    s = re.sub(r"\s+", " ", s).strip()
    if MASK_AMT_CUR:
        s = s.replace("<AMOUNT>", "<amount>").replace("<CUR>", "<cur>")
    if MASK_TIME:
        s = s.replace("<TIME>", "<time>")
    if MASK_LOC:
        s = s.replace("<LOC>", "<loc>")
    return s

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

def normalize_currency(raw: str | None) -> str | None:
    if not raw: return None
    r = raw.strip().lower()
    return CURRENCY_MAP.get(r, CURRENCY_MAP.get(raw, None))

def clean_number_str(s: str) -> str:
    return re.sub(r"[,\s]", "", s)

def parse_amount(text: str) -> float | None:
    # handles "80k", "2.5m", "300,000", "150000"
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*([kKmM])\b", text)
    if m:
        n = float(m.group(1))
        mult = 1_000 if m.group(2).lower() == "k" else 1_000_000
        return n * mult
    m = re.search(r"\b([\d,]+(?:\.\d+)?)\b", text)
    if m:
        return float(clean_number_str(m.group(1)))
    return None

def parse_timeframe(text: str) -> Tuple[int|None, int|None, int|None]:
    # returns years, months, total_months
    y = None; mo = None
    m = re.search(r"\b(\d+)\s*(years?|yrs?)\b", text, flags=re.I)
    if m: y = int(m.group(1))
    m = re.search(r"\b(\d+)\s*(months?|mos?)\b", text, flags=re.I)
    if m: mo = int(m.group(1))
    if y is None and re.search(r"\bnext\s+year\b", text, flags=re.I):
        y = 1
    total = None
    if y is not None or mo is not None:
        total = (y or 0) * 12 + (mo or 0)
    return y, mo, total

def sentencize(nlp, text: str) -> List[str]:
    # use spaCy sentencizer to stay dependency-light
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return [s.text.strip() for s in nlp(text).sents if s.text.strip()]

# ----------------------------
# Fallbacks (NEW)
# ----------------------------
UAE_HINTS_RE = LOC_RE  # reuse compiled regex above

TIME_PATTERNS = [
    re.compile(r"\bnext\s+(\d+)\s*(years?|yrs?)\b", re.I),
    re.compile(r"\bin\s+(\d+)\s*(years?|yrs?)\b", re.I),
    re.compile(r"\bnext\s+(\d+)\s*(months?|mos?)\b", re.I),
    re.compile(r"\bin\s+(\d+)\s*(months?|mos?)\b", re.I),
    re.compile(r"\b(\d+)\s*(years?|yrs?)\b", re.I),
    re.compile(r"\b(\d+)\s*(months?|mos?)\b", re.I),
]

def fallback_time(text: str):
    # returns (years, months, months_total) or (None,None,None)
    for pat in TIME_PATTERNS:
        m = pat.search(text)
        if m:
            val = int(m.group(1))
            unit = m.group(2).lower()
            if unit.startswith("year") or unit.startswith("yr"):
                return val, None, val*12
            else:
                return None, val, val
    if "next year" in text.lower():
        return 1, None, 12
    return None, None, None

def fallback_loc(text: str):
    m = UAE_HINTS_RE.search(text)
    if m:
        return " ".join(w.capitalize() for w in m.group(1).split())
    return None

def fallback_amount_currency(text: str):
    # amount
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*([kKmM])\b", text)
    if m:
        n = float(m.group(1))
        mult = 1000 if m.group(2).lower()=="k" else 1_000_000
        amt = n * mult
    else:
        m2 = re.search(r"\b(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\b", text)
        amt = float(m2.group(1).replace(",","")) if m2 else None
    # currency
    cur = None
    m3 = re.search(r"\b(aed|usd|inr|eur|gbp|dhs|dirhams?|د\.إ|\$|₹|€|£)\b", text, re.I)
    if m3:
        cur = CURRENCY_MAP.get(m3.group(1).lower(), None)
    return amt, cur

# ----------------------------
# Load models
# ----------------------------
_cls_tokenizer = AutoTokenizer.from_pretrained(CLS_DIR)
_cls_model     = AutoModelForSequenceClassification.from_pretrained(CLS_DIR)
_cls_model.eval()

_ner_nlp = spacy.load(NER_DIR)  # your trained NER with AMOUNT/CURRENCY/TIME/LOC
if "sentencizer" not in _ner_nlp.pipe_names and "senter" not in _ner_nlp.pipe_names and "parser" not in _ner_nlp.pipe_names:
    _ner_nlp.add_pipe("sentencizer")

# build label maps from classifier config
id2label = _cls_model.config.id2label
label2id = _cls_model.config.label2id

# ----------------------------
# Low-confidence nudge
# ----------------------------
def nudge_label_if_needed(text: str, pred_label: str, conf: float) -> str:
    if conf >= LOW_CONF:
        return pred_label
    t = text.lower()
    # order matters: try the most concrete catches first
    if VEHICLE_WORDS.search(t): return "vehicle purchase"
    if WEDDING_WORDS.search(t): return "wedding"
    if EDU_WORDS.search(t):     return "education"
    if PASSIVE_WORDS.search(t): return "passive income"
    if REI_WORDS.search(t):     return "real estate investment"
    if BUSINESS_WORDS.search(t):return "business"
    if RETIRE_WORDS.search(t):  return "retirement"
    if LUX_WORDS.search(t):     return "luxury purchase"
    if TRAVEL_WORDS.search(t):  return "travel"
    return pred_label

# ----------------------------
# NER extraction helpers
# ----------------------------
def ner_extract(sentence: str) -> Dict[str, Any]:
    doc = _ner_nlp(sentence)
    spans = {"amount_text": None, "currency_text": None, "time_text": None, "loc_text": None}
    for ent in doc.ents:
        label = ent.label_.upper()
        if label in ("AMOUNT",) and spans["amount_text"] is None:
            spans["amount_text"] = ent.text
        elif label in ("CURRENCY",) and spans["currency_text"] is None:
            spans["currency_text"] = ent.text
        elif label in ("TIME",) and spans["time_text"] is None:
            spans["time_text"] = ent.text
        elif label in ("LOC", "GPE") and spans["loc_text"] is None:
            spans["loc_text"] = ent.text

    # Clean/normalize
    amount_val = parse_amount(spans["amount_text"] or "") if spans["amount_text"] else None
    currency   = normalize_currency(spans["currency_text"]) if spans["currency_text"] else None
    y, mo, total_m = parse_timeframe(spans["time_text"] or "") if spans["time_text"] else (None, None, None)
    loc = spans["loc_text"]
    return {
        "amount_text": spans["amount_text"],
        "amount": amount_val,
        "currency": currency,
        "time_text": spans["time_text"],
        "timeframe_years": y,
        "timeframe_months": mo,
        "timeframe_months_total": total_m,
        "location": loc,
    }

# ----------------------------
# Classify + score
# ----------------------------
def classify(sentences: List[str]) -> List[Dict[str, Any]]:
    masked = [mask_text(s) for s in sentences] if MASK_AMT_CUR or MASK_TIME or MASK_LOC else sentences
    batch = _cls_tokenizer(
        masked,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = _cls_model(**{k: batch[k] for k in ["input_ids", "attention_mask"]}).logits
    logits_np = logits.cpu().numpy()
    probs = np.apply_along_axis(softmax, 1, logits_np)
    pred_ids = probs.argmax(axis=1)
    results = []
    for s, m, pid, p in zip(sentences, masked, pred_ids, probs):
        label = id2label[int(pid)]
        conf  = float(p[pid])
        label2 = nudge_label_if_needed(s, label, conf)
        results.append({
            "text": s,
            "masked": m,
            "pred_label": label2,
            "raw_pred_label": label,
            "confidence": conf
        })
    return results

# ----------------------------
# Public API
# ----------------------------
def parse_goals(text: str) -> Dict[str, Any]:
    # sentence split with spaCy (re-use NER nlp to avoid extra load)
    sents = sentencize(_ner_nlp, text)
    if not sents:
        sents = [text.strip()]

    cls = classify(sents)
    out_goals = []
    for i, sent in enumerate(sents):
        ner = ner_extract(sent)
        # After ner = ner_extract(sent)
        if ner["currency"] is None:
    # optional: only do this if we see UAE hints
            if re.search(r"\b(ae|uae|dubai|abu dhabi|sharjah|ajman|fujairah|ras al khaimah|umm al quwain)\b", sent, re.I):
                ner["currency"] = "AED"
            else:
                # or always default:
                ner["currency"] = "AED"


        # --- FALLBACKS if NER missed something ---
        if ner["timeframe_years"] is None and ner["timeframe_months"] is None:
            fy, fm, ft = fallback_time(sent)
            ner["timeframe_years"] = fy
            ner["timeframe_months"] = fm
            ner["timeframe_months_total"] = ft

        if not ner["location"]:
            loc_fb = fallback_loc(sent)
            if loc_fb:
                ner["location"] = loc_fb

        if ner["amount"] is None or ner["currency"] is None:
            a_fb, c_fb = fallback_amount_currency(sent)
            if ner["amount"] is None and a_fb is not None:
                ner["amount"] = a_fb
                ner["amount_text"] = ner["amount_text"] or str(int(a_fb))
            if ner["currency"] is None and c_fb:
                ner["currency"] = c_fb

        goal_type = cls[i]["pred_label"]
        conf      = cls[i]["confidence"]
        out_goals.append({
            "original_text": sent,
            "goal_type": goal_type,
            "classifier_confidence": round(conf, 3),
            "amount": ner["amount"],
            "currency": ner["currency"],
            "timeframe_years": ner["timeframe_years"],
            "timeframe_months": ner["timeframe_months"],
            "timeframe_months_total": ner["timeframe_months_total"],
            "location": ner["location"],
            "notes": {
                "amount_text": ner["amount_text"],
                "time_text": ner["time_text"]
            }
        })

    return {
        "schema_version": "ml-infer-1.0",
        "input_text": text,
        "goals": out_goals
    }

if __name__ == "__main__":
    demo = """I want to save 200,000 AED in the next 4 years to open a small café in Sharjah.
My daughter's wedding is in 3 years; I need 150k AED.
I want 300,000 AED in 10 years to buy a house in Ajman. Also need 80k for a car next year."""
    print(parse_goals(demo))