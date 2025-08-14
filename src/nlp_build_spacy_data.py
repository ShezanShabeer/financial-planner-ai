import json
from typing import List, Tuple
import os
import spacy
from spacy.tokens import Doc, Span
from spacy.tokens import DocBin

# -------- Paths (root-relative) --------
BASE = "data/nlp/"
NER_TRAIN_JSON = os.path.join(BASE, "ner_train.json")
NER_DEV_JSON   = os.path.join(BASE, "ner_val.json")   # using your test as dev for training eval
OUT_DIR = os.path.join(BASE, "corpus")
OUT_TRAIN = os.path.join(OUT_DIR, "train.spacy")
OUT_DEV   = os.path.join(OUT_DIR, "dev.spacy")

os.makedirs(OUT_DIR, exist_ok=True)

# -------- Tag mapping (must match ner_* files) --------
ID2TAG = {
    0: "O",
    1: "B-GOAL_TYPE",
    2: "I-GOAL_TYPE",
    3: "B-AMOUNT",
    4: "I-AMOUNT",
    5: "B-CURRENCY",
    6: "I-CURRENCY",
    7: "B-DURATION",
    8: "I-DURATION",
    9: "B-LOCATION",
    10: "I-LOCATION",
}
BIO2LABEL = {
    "GOAL_TYPE": "GOAL_TYPE",
    "AMOUNT": "AMOUNT",
    "CURRENCY": "CURRENCY",
    "DURATION": "DURATION",
    "LOCATION": "LOCATION",
}

def bio_to_spans(tag_ids: List[int]) -> List[Tuple[int,int,str]]:
    spans = []
    start = None
    cur_label = None

    def flush(end_i):
        nonlocal start, cur_label
        if start is not None and cur_label is not None:
            spans.append((start, end_i, cur_label))
        start, cur_label = None, None

    for i, tid in enumerate(tag_ids):
        tag = ID2TAG.get(tid, "O")
        if tag == "O":
            flush(i); continue
        bio, _, label = tag.partition("-")
        label = BIO2LABEL.get(label, None)
        if label is None:
            flush(i); continue
        if bio == "B":
            flush(i)
            start, cur_label = i, label
        elif bio == "I":
            if cur_label != label:
                flush(i)
                start, cur_label = i, label
        else:
            flush(i)
    flush(len(tag_ids))
    return spans

def build_spacy_bin(in_path: str, out_path: str):
    nlp = spacy.blank("en")
    db = DocBin(store_user_data=True)
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        tokens = item["tokens"]
        tags = item["ner_tags"]
        spaces = [True] * len(tokens)
        if spaces: spaces[-1] = False
        doc = Doc(nlp.vocab, words=tokens, spaces=spaces)

        spans = []
        for start, end, label in bio_to_spans(tags):
            try:
                span = Span(doc, start, end, label=label)
                spans.append(span)
            except Exception:
                pass
        doc.ents = spans
        db.add(doc)

    db.to_disk(out_path)
    print(f"Wrote {out_path} ({len(db)} docs)")

if __name__ == "__main__":
    build_spacy_bin(NER_TRAIN_JSON, OUT_TRAIN)
    build_spacy_bin(NER_DEV_JSON, OUT_DEV)
    print("Done.")