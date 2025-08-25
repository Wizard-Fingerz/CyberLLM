
# =====================
# scripts/evaluate.py
# =====================
import argparse, os, json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main(args):
    model_dir = args.model_dir
    processed_dir = args.processed_dir

    with open(os.path.join(processed_dir, "label2id.json"), "r", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {i: lbl for lbl, i in label2id.items()}

    data_files = {
        "validation": os.path.join(processed_dir, "val.jsonl"),
    }
    ds = load_dataset("json", data_files=data_files)["validation"]

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    def tok(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=args.max_seq_length)

    ds = ds.map(tok, batched=True).rename_column("label", "labels").with_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    args_tr = TrainingArguments(output_dir=os.path.join(model_dir, "eval_tmp"), per_device_eval_batch_size=args.batch_size)

    trainer = Trainer(model=model, args=args_tr, eval_dataset=ds, tokenizer=tokenizer, compute_metrics=compute_metrics)
    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_seq_length", type=int, default=256)
    args = p.parse_args()
    main(args)

