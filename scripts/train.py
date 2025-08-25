
# =====================
# scripts/train.py
# =====================
import argparse
import os
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer, TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils import ensure_dir, resolve_model, set_global_seed
import yaml


def load_config(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main(args):
    cfg = load_config(args.config)
    set_global_seed(cfg.get("seed", 42))

    processed_dir = cfg["data"]["processed_dir"]
    models_dir = cfg["models_dir"]
    results_dir = cfg["results_dir"]
    run_name = cfg.get("run_name", "run")

    ensure_dir(models_dir)
    ensure_dir(results_dir)

    # Load label maps
    with open(os.path.join(processed_dir, "label2id.json"), "r", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {i: lbl for lbl, i in label2id.items()}
    num_labels = len(label2id)

    # Load dataset
    data_files = {
        "train": os.path.join(processed_dir, "train.jsonl"),
        "validation": os.path.join(processed_dir, "val.jsonl"),
    }
    ds = load_dataset("json", data_files=data_files, split={
                      "train": "train", "validation": "validation"})

    # Use a better model and more realistic hyperparameters for improved results
    model_name = resolve_model(cfg.get("model_name", "roberta-base"))
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Use a longer max_seq_length from config, or fallback to 256
    max_seq_length = cfg.get("max_seq_length", 256)

    def tok(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_seq_length)

    ds = {k: v.map(tok, batched=True).rename_column(
        "label", "labels").with_format("torch") for k, v in ds.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # Use the full dataset for training and validation
    train_dataset = ds["train"]
    val_dataset = ds["validation"]

    # Use more realistic training arguments for better results
    args_tr = TrainingArguments(
        output_dir=os.path.join(models_dir, run_name),
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=cfg.get("learning_rate", 2e-5),
        per_device_train_batch_size=cfg.get("batch_size", 16),
        per_device_eval_batch_size=cfg.get("batch_size", 16),
        num_train_epochs=cfg.get("num_epochs", 4),
        weight_decay=cfg.get("weight_decay", 0.01),
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        eval_strategy="epoch",
        report_to=["none"],
        run_name=run_name,
        disable_tqdm=False,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Final eval & save metrics
    eval_metrics = trainer.evaluate()
    with open(os.path.join(results_dir, f"{run_name}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)

    # Save model
    trainer.save_model(os.path.join(models_dir, run_name))

    print({"run_name": run_name, **eval_metrics})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_config.yaml")
    args = parser.parse_args()
    main(args)
