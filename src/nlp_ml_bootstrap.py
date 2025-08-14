import json, re
from pathlib import Path
from typing import List, Dict, Any
from nlp_parser import parse_text  # your rule-based parser

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
NER_LABELS = ["O","B-AMOUNT","I-AMOUNT","B-CURRENCY","B-TIME","I-TIME","B-LOC","I-LOC"]

def simple_tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)

def label_ner(tokens: List[str], spans: Dict[str, List[str]]) -> List[str]:
    tags = ["O"] * len(tokens)
    def mark(span_text: str, tag: str):
        if not span_text: return
        span_toks = simple_tokenize(span_text)
        if not span_toks: return
        L = len(span_toks)
        low_tokens = [t.lower() for t in tokens]
        low_span   = [t.lower() for t in span_toks]
        for i in range(len(tokens) - L + 1):
            if low_tokens[i:i+L] == low_span:
                tags[i] = f"B-{tag}"
                for j in range(1, L):
                    tags[i+j] = f"I-{tag}"
    for tag in ["AMOUNT","CURRENCY","TIME","LOC"]:
        val = spans.get(tag)
        if not val: continue
        vals = val if isinstance(val, list) else [val]
        for v in vals: mark(v, tag)
    return tags

def extract_spans_rule(goal: Dict[str, Any]) -> Dict[str, Any]:
    spans: Dict[str, Any] = {}
    s = goal.get("original_text", "") or ""
    # amount (prefer original surface form)
    m = re.search(r"\b\d[\d,]*(?:\.\d+)?\s*[kKmM]?\b", s)
    if m: spans["AMOUNT"] = m.group(0)
    else:
        amt = goal.get("amount")
        if isinstance(amt, (int,float)): spans["AMOUNT"] = str(int(amt)) if float(amt).is_integer() else str(float(amt))
    # currency
    cur = goal.get("currency")
    if cur:
        m2 = re.search(r"\b(aed|usd|inr|rs|د\.إ|\$|₹|dhs|dirham|dirhams)\b", s, flags=re.I)
        spans["CURRENCY"] = m2.group(0) if m2 else cur
    # time phrase
    m3 = re.search(r"\b(?:in|by|within)\s+\d+\s+(?:years?|months?)\b|next\s+\d+\s+years?|next\s+year|\b\d+\s+(?:years?|months?)\b", s, flags=re.I)
    if m3: spans["TIME"] = m3.group(0)
    # location
    if goal.get("location"): spans["LOC"] = goal["location"]
    return spans

def to_conll_item(text: str, spans: Dict[str, Any]) -> Dict[str, Any]:
    tokens = simple_tokenize(text)
    tags = label_ner(tokens, spans)
    return {"tokens": tokens, "ner_tags": tags}

def main(infile="data/nlp/sentences.txt"):
    lines = [ln.strip() for ln in Path(infile).read_text(encoding="utf-8").splitlines() if ln.strip()]
    cls_rows, ner_rows = [], []
    for s in lines:
        parsed = parse_text(s)
        for g in parsed["goals"]:
            text = g["original_text"]
            cls_rows.append({"text": text, "label": g.get("goal_type") or "other"})
            spans = extract_spans_rule(g)
            ner_rows.append(to_conll_item(text, spans))

    out_dir = Path("data/nlp")
    out_dir.mkdir(parents=True, exist_ok=True)

    # train/val split 85/15
    cut_cls = max(1, int(0.80*len(cls_rows)))
    cut_ner = max(1, int(0.80*len(ner_rows)))
    tr_cls, va_cls = cls_rows[:cut_cls], cls_rows[cut_cls:]
    tr_ner, va_ner = ner_rows[:cut_ner], ner_rows[cut_ner:]

    (out_dir/"goal_type_train.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in tr_cls), encoding="utf-8")
    (out_dir/"goal_type_val.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in va_cls), encoding="utf-8")

    (out_dir/"ner_train.json").write_text(json.dumps({"tokens":[r["tokens"] for r in tr_ner],
                                                      "ner_tags":[r["ner_tags"] for r in tr_ner]}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir/"ner_val.json").write_text(json.dumps({"tokens":[r["tokens"] for r in va_ner],
                                                    "ner_tags":[r["ner_tags"] for r in va_ner]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote datasets to data/nlp/. Now hand-fix and add more examples for better accuracy.")

if __name__ == "__main__":
    main()