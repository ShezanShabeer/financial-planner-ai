# src/nlp_train_goaltype.py
# Train a goal-type classifier (ML) on goal_type_train/val.jsonl
# Works across a range of transformers versions (handles arg compatibility)

import os
import json
import random
import inspect
from typing import Dict, List, Tuple

import numpy as np
import datasets
from datasets import load_dataset, DatasetDict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# -----------------------
# Config
# -----------------------
TRAIN_PATH = "data/nlp/goal_type_train.jsonl"
VAL_PATH   = "data/nlp/goal_type_val.jsonl"   # or goal_type_test.jsonl
MODEL_NAME = "distilbert-base-uncased"
OUT_DIR    = "artifacts/goal_type"
SEED       = 42
MAX_LEN    = 128
BATCH_SIZE = 16
EPOCHS     = 5
LR         = 5e-5

# -----------------------
# Utils
# -----------------------
def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _supports_arg(cls, arg_name: str) -> bool:
    """Check if a dataclass/constructor supports an argument by name."""
    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):
        # If cls is a dataclass (like TrainingArguments), inspect its __init__
        try:
            sig = inspect.signature(cls.__init__)
        except Exception:
            return False
    return arg_name in sig.parameters

def _load_jsonl_dataset(train_path: str, val_path: str) -> DatasetDict:
    """
    Load JSONL (one object per line). Expects {"text": str, "label": str}
    If "label" is a list, we keep the first.
    """
    ds_train = load_dataset("json", data_files=train_path, split="train")
    ds_val   = load_dataset("json", data_files=val_path,   split="train")

    def _normalize(batch):
        text = batch.get("text")
        label = batch.get("label")
        if isinstance(label, list):
            # keep the first if it's a list (some tools export like this)
            label = label[0] if label else None
        return {"text": text, "label": label}

    ds_train = ds_train.map(_normalize)
    ds_val   = ds_val.map(_normalize)

    # Filter rows with missing text/label
    def _ok(example):
        return (example.get("text") is not None) and (example.get("label") is not None)
    ds_train = ds_train.filter(_ok)
    ds_val   = ds_val.filter(_ok)

    # Small prints to verify distributions
    train_labels = ds_train["label"]
    val_labels   = ds_val["label"]
    print(f"[INFO] Loaded {train_path}: {len(ds_train)}/{len(ds_train)} usable rows")
    print(f"[INFO] Loaded {val_path}: {len(ds_val)}/{len(ds_val)} usable rows")

    return DatasetDict(train=ds_train, validation=ds_val)

def build_datasets() -> Tuple[DatasetDict, Dict[str, int], Dict[int, str]]:
    datasets_dict = _load_jsonl_dataset(TRAIN_PATH, VAL_PATH)

    # Build consistent label space across splits
    labels = sorted(list(set(datasets_dict["train"]["label"]) | set(datasets_dict["validation"]["label"])))
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}

    # Show label distribution
    def _dist(lbls):
        d = {}
        for x in lbls:
            d[x] = d.get(x, 0) + 1
        return d

    print("[INFO] Train label distribution:", _dist(datasets_dict["train"]["label"]))
    print("[INFO]  Val  label distribution:", _dist(datasets_dict["validation"]["label"]))

    # Tokenizer + encode + map labels to ids
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def _tok(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN
        )
        enc["labels"] = [label2id[l] for l in batch["label"]]
        return enc

    encoded = DatasetDict(
        train=datasets_dict["train"].map(_tok, batched=True, remove_columns=datasets_dict["train"].column_names),
        validation=datasets_dict["validation"].map(_tok, batched=True, remove_columns=datasets_dict["validation"].column_names),
    )

    encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return encoded, label2id, id2label

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1m = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1m}

# -----------------------
# Main
# -----------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_all_seeds(SEED)

    encoded, label2id, id2label = build_datasets()

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    # Build TrainingArguments with version-compat handling
    ta_kwargs = dict(
        output_dir=OUT_DIR,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        seed=SEED,
        logging_steps=50,
        save_total_limit=2,
        weight_decay=0.01,
        load_best_model_at_end=True,        # we want the best checkpoint loaded at end
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to=[],                       # avoid wandb if present
    )

    # Force both strategies to 'epoch' iff supported, to satisfy HF constraint:
    # load_best_model_at_end requires evaluation_strategy == save_strategy
    if _supports_arg(TrainingArguments, "evaluation_strategy"):
        ta_kwargs["evaluation_strategy"] = "epoch"
    if _supports_arg(TrainingArguments, "save_strategy"):
        ta_kwargs["save_strategy"] = "epoch"
    # If some older version doesn't accept those args, strip them & also disable load_best_model_at_end
    if not _supports_arg(TrainingArguments, "evaluation_strategy") or not _supports_arg(TrainingArguments, "save_strategy"):
        # remove incompatible fields
        ta_kwargs.pop("load_best_model_at_end", None)
        ta_kwargs.pop("metric_for_best_model", None)
        ta_kwargs.pop("greater_is_better", None)

    args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME),
        compute_metrics=compute_metrics,
    )

    print("\n[INFO] Starting training…")
    trainer.train()

    print("\n[INFO] Evaluating on validation set…")
    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2))

    # Save model + label maps
    print(f"\n[INFO] Saving model to: {OUT_DIR}")
    trainer.save_model(OUT_DIR)
    with open(os.path.join(OUT_DIR, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_DIR, "id2label.json"), "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)

    # Quick preview: show a few predictions on the val set
    print("\n[INFO] Preview predictions (first 10 val rows):")
    raw_val = load_dataset("json", data_files=VAL_PATH, split="train")
    texts = raw_val["text"][:10]
    enc_tok = AutoTokenizer.from_pretrained(MODEL_NAME)(
        texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**{k: v for k, v in enc_tok.items() if k in ["input_ids", "attention_mask"]}).logits
        pred_ids = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
    for t, pid in zip(texts, pred_ids):
        print(f"- {t}\n  -> {id2label[pid]}")

if __name__ == "__main__":
    main()
