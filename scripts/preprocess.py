# =====================
# scripts/preprocess.py
# =====================
import argparse, os, json
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import ensure_dir
import yaml


def load_config(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_text(row, text_fields):
    parts = []
    for f in text_fields:
        val = str(row.get(f, "")).strip()
        if val and val.lower() != "nan":
            parts.append(f"{f}: {val}")
    return " \n".join(parts)


def main(args):
    cfg = load_config(args.config)
    raw_csv = cfg["data"]["raw_csv"]
    processed_dir = cfg["data"]["processed_dir"]
    text_fields = cfg["text_fields"]
    label_field = cfg["label_field"]
    train_size = cfg.get("train_size", 0.8)
    stratify = cfg.get("stratify", True)
    remove_duplicates = cfg.get("remove_duplicates", True)
    min_class_count = cfg.get("min_class_count", 2)  # default to 2

    ensure_dir(processed_dir)

    df = pd.read_csv(raw_csv)

    # Basic cleaning
    if remove_duplicates:
        df = df.drop_duplicates()

    # Drop rows missing the label
    df = df.dropna(subset=[label_field])

    # Merge rare classes into "Other"
    label_counts = df[label_field].value_counts()
    rare_classes = label_counts[label_counts < min_class_count].index
    if len(rare_classes) > 0:
        print(f"Merging rare classes into 'Other': {list(rare_classes)}")
        df[label_field] = df[label_field].apply(
            lambda x: "Other" if x in rare_classes else x
        )

    # Create model input text
    df["text"] = df.apply(lambda r: build_text(r, text_fields), axis=1)

    # Label encoding
    labels = sorted(df[label_field].astype(str).unique())
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    df["label"] = df[label_field].astype(str).map(label2id)

    # Train/validation split
    stratify_vals = df["label"] if stratify else None
    train_df, val_df = train_test_split(
        df[["text", "label"]],
        train_size=train_size,
        random_state=42,
        stratify=stratify_vals,
    )

    # Save processed data
    train_path = os.path.join(processed_dir, "train.jsonl")
    val_path = os.path.join(processed_dir, "val.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for _, row in train_df.iterrows():
            f.write(json.dumps({"text": row["text"], "label": int(row["label"])}) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for _, row in val_df.iterrows():
            f.write(json.dumps({"text": row["text"], "label": int(row["label"])}) + "\n")

    # Save label maps
    with open(os.path.join(processed_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2)
    with open(os.path.join(processed_dir, "id2label.json"), "w", encoding="utf-8") as f:
        json.dump(id2label, f, indent=2)

    print({
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "num_labels": len(label2id),
        "labels": labels,
        "train_path": train_path,
        "val_path": val_path,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_config.yaml")
    args = parser.parse_args()
    main(args)
