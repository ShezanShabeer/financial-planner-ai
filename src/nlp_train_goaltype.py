# src/nlp_train_goaltype.py
# Improved: entity masking + class weights + early stopping (Transformers 4.55+ compatible)

import os
import re
import json
import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# Config toggles
# -------------------------
USE_ENTITY_MASKING = True   # turn masking on/off
MASK_LOWERCASE     = True   # masks appear as <amount>/<cur>/<time>/<loc>

# simple UAE-aware masking
CUR_WORDS = r"(?:aed|usd|inr|eur|gbp|dhs|dirhams?|د\.إ|\$|₹|€|£)"
TIME_PAT  = r"(?:\b\d+\s+(?:years?|yrs?|months?|mos?)\b|\bnext\s+year\b|\bnext\s+\d+\s+(?:years?|months?)\b)"
LOC_HINTS = [
    "dubai","abu dhabi","sharjah","uae","united arab emirates","ajman",
    "ras al khaimah","rak","fujairah","umm al quwain","deira","mirdif",
    "damac hills","jlt","jbr","downtown dubai","dubai marina","al barsha"
]
LOC_RE = re.compile(r"\b(" + "|".join(re.escape(x) for x in LOC_HINTS) + r")\b", flags=re.I)

# toggle what to mask
MASK_AMT_CUR = True
MASK_TIME    = False   # <— change to False
MASK_LOC     = False   # <— change to False

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


# -------------------------
# Data loading / cleaning
# -------------------------
def _coerce_label_to_str(label_value: Any) -> str | None:
    if label_value is None:
        return None
    if isinstance(label_value, str):
        s = label_value.strip()
        return s if s else None
    if isinstance(label_value, (list, tuple)) and len(label_value) > 0:
        first = label_value[0]
        if isinstance(first, str):
            s = first.strip()
            return s if s else None
    return None


def load_jsonl_datasets(train_path: str, val_path: str) -> tuple[DatasetDict, list[str], dict[str,int], dict[int,str]]:
    ds_train = load_dataset("json", data_files=train_path, split="train")
    ds_val   = load_dataset("json", data_files=val_path,   split="train")

    def _normalize(ex):
        ex["label"] = _coerce_label_to_str(ex.get("label"))
        return ex

    ds_train = ds_train.map(_normalize)
    ds_val   = ds_val.map(_normalize)

    def _ok(ex):
        return isinstance(ex.get("text"), str) and isinstance(ex.get("label"), str)

    ds_train = ds_train.filter(_ok)
    ds_val   = ds_val.filter(_ok)

    if len(ds_val) == 0:
        raise SystemExit(
            "[FATAL] Validation set has 0 usable rows after normalization.\n"
            "Check goal_type_val.jsonl labels (should be strings, not lists)."
        )

    labels = sorted(list(set(ds_train["label"]) | set(ds_val["label"])))
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}

    def _map_label(ex):
        ex["label_id"] = label2id[ex["label"]]
        return ex

    # optional masking column
    if USE_ENTITY_MASKING:
        ds_train = ds_train.map(lambda ex: {"text_masked": mask_text(ex["text"])})
        ds_val   = ds_val.map(  lambda ex: {"text_masked": mask_text(ex["text"])})
        text_key = "text_masked"
    else:
        text_key = "text"

    ds_train = ds_train.map(_map_label)
    ds_val   = ds_val.map(_map_label)

    print(f"[INFO] Train size: {len(ds_train)} | Val size: {len(ds_val)} | #labels: {len(labels)}")
    print(f"[INFO] Labels: {labels}")
    return DatasetDict(train=ds_train, validation=ds_val), labels, label2id, id2label, text_key


def build_tokenizer_and_model(model_name: str, num_labels: int, id2label: Dict[int, str], label2id: Dict[str, int]):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    return tok, mdl


def tokenize_function(examples, tokenizer, text_key="text", max_length=128):
    return tokenizer(examples[text_key], truncation=True, max_length=max_length)


def compute_metrics_builder():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        pr, rc, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        return {
            "accuracy": acc,
            "f1_macro": f1,           # metric_for_best_model uses this
            "precision_macro": pr,
            "recall_macro": rc,
        }
    return compute_metrics


# -------------------------
# Weighted loss Trainer
# -------------------------
class WeightedCELossTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = None
        if class_weights is not None:
            self._class_weights = torch.tensor(class_weights, dtype=torch.float)
        self._loss_fct = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        logits = outputs.get("logits")

        if self._loss_fct is None:
            if self._class_weights is not None:
                self._class_weights = self._class_weights.to(logits.device)
                self._loss_fct = nn.CrossEntropyLoss(weight=self._class_weights)
            else:
                self._loss_fct = nn.CrossEntropyLoss()

        loss = self._loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# -------------------------
# Reporting helpers
# -------------------------
def save_report_and_confusion(y_true: List[int], y_pred: List[int], id2label: Dict[int, str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    labels_sorted = sorted(id2label.keys())
    target_names = [id2label[i] for i in labels_sorted]

    rep = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    csv_path = os.path.join(out_dir, "classification_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["label", "precision", "recall", "f1", "support"])
        for name in target_names + ["macro avg", "weighted avg", "accuracy"]:
            if name == "accuracy":
                total = sum(rep[n]["support"] for n in target_names)
                w.writerow([name, "", "", rep[name], total])
            else:
                r = rep[name]
                w.writerow([name, r["precision"], r["recall"], r["f1-score"], r["support"]])

    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(target_names)),
        yticks=np.arange(len(target_names)),
        xticklabels=target_names,
        yticklabels=target_names,
        ylabel="True",
        xlabel="Predicted",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    png_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(png_path, dpi=180)
    plt.close(fig)
    print(f"[SAVED] {csv_path}")
    print(f"[SAVED] {png_path}")


# -------------------------
# Main
# -------------------------
def main():
    print("[INFO] transformers version:", __import__("transformers").__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/nlp/goal_type_train.jsonl")
    parser.add_argument("--val_path",   type=str, default="data/nlp/goal_type_val.jsonl")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--out_dir",    type=str, default="artifacts/goal_type")
    parser.add_argument("--epochs",     type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr",         type=float, default=3e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--patience",   type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    datasets, labels, label2id, id2label, text_key = load_jsonl_datasets(args.train_path, args.val_path)

    tokenizer, model = build_tokenizer_and_model(args.model_name, len(labels), id2label, label2id)

    def _tok(b):
        t = tokenize_function(b, tokenizer, text_key=text_key, max_length=args.max_length)
        t["labels"] = b["label_id"]
        return t

    tokenized_train = datasets["train"].map(_tok, batched=True, remove_columns=datasets["train"].column_names)
    tokenized_val   = datasets["validation"].map(_tok, batched=True, remove_columns=datasets["validation"].column_names)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # class weights from train labels
    y_train = np.array(tokenized_train["labels"])
    classes, counts = np.unique(y_train, return_counts=True)
    total = counts.sum()
    weights = {int(c): float(total / (len(classes) * cnt)) for c, cnt in zip(classes, counts)}
    class_weights = np.array([weights.get(i, 1.0) for i in range(len(id2label))], dtype=np.float32)
    print("[INFO] Class weights:", {id2label[i]: round(float(w), 3) for i, w in enumerate(class_weights)})

    compute_metrics = compute_metrics_builder()

    # 4.55.x uses eval_strategy/save_strategy
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=50,
        save_total_limit=2,
        report_to=[],
    )

    trainer = WeightedCELossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    print("[INFO] Starting training…")
    trainer.train()  # best model auto-loaded at end

    print("[INFO] Evaluating on validation set…")
    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2))

    # detailed report
    preds_logits = trainer.predict(tokenized_val).predictions
    y_pred = np.argmax(preds_logits, axis=-1)
    y_true = tokenized_val["labels"]
    save_report_and_confusion(y_true, y_pred, id2label, args.out_dir)

    # save final + best
    best_dir = os.path.join(args.out_dir, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"[SAVED] Best model to: {best_dir}")

    # quick preview with padding to avoid tensor shape errors
    raw_val = load_dataset("json", data_files=args.val_path, split="train")
    texts = raw_val["text"][:10]
    texts_masked = [mask_text(t) for t in texts] if USE_ENTITY_MASKING else texts
    batch = tokenizer(texts_masked, truncation=True, padding=True, max_length=args.max_length, return_tensors="pt")
    with torch.no_grad():
        out = model(**{k: v for k, v in batch.items() if k in ["input_ids", "attention_mask"]}).logits.argmax(-1).tolist()
    print("\n[INFO] Preview predictions:")
    for t, m, pid in zip(texts, texts_masked, out):
        print(f"- {t}\n  [masked] {m}\n  -> {id2label[pid]}")


if __name__ == "__main__":
    main()
