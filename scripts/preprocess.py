# =====================
# scripts/preprocess.py
# =====================
import argparse, os, json
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import ensure_dir
import yaml

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

def select_text_fields_by_pca(df, text_fields, label_field, n_components=0.95):
    """
    Use PCA to select text fields that most contribute to variance in the data.
    This is a heuristic: we vectorize each text field (using simple length/counts),
    then use PCA to see which fields contribute most to the principal components.
    """
    # For each text field, compute a simple numeric feature (length, word count)
    features = []
    feature_names = []
    for f in text_fields:
        # Use both char length and word count as features
        features.append(df[f].astype(str).apply(len))
        features.append(df[f].astype(str).apply(lambda x: len(str(x).split())))
        feature_names.append(f"{f}_len")
        feature_names.append(f"{f}_wc")
    X = pd.concat(features, axis=1)
    X.columns = feature_names

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_scaled)
    # Get the absolute value of the components for each original feature
    importance = pd.DataFrame(
        abs(pca.components_).sum(axis=0),
        index=feature_names,
        columns=["importance"]
    )
    # For each text field, sum the importances of its features
    field_importance = {}
    for f in text_fields:
        field_importance[f] = (
            importance.loc[f"{f}_len", "importance"] +
            importance.loc[f"{f}_wc", "importance"]
        )
    # Sort fields by importance
    sorted_fields = sorted(field_importance.items(), key=lambda x: x[1], reverse=True)
    # Select fields with above-median importance (or at least 1)
    median_imp = pd.Series([v for k, v in sorted_fields]).median()
    selected_fields = [k for k, v in sorted_fields if v >= median_imp]
    if not selected_fields:
        selected_fields = [sorted_fields[0][0]]  # fallback: at least one
    print(f"Selected text fields by PCA: {selected_fields}")
    return selected_fields

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
    max_instances = 5000  # Limit to 5000 instances

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

    # --- PCA-based text field selection ---
    # Only use text fields that contribute most to variance
    selected_text_fields = select_text_fields_by_pca(df, text_fields, label_field)
    # Create model input text using only selected fields
    df["text"] = df.apply(lambda r: build_text(r, selected_text_fields), axis=1)

    # Label encoding
    labels = sorted(df[label_field].astype(str).unique())
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    df["label"] = df[label_field].astype(str).map(label2id)

    # Reduce to max_instances
    if len(df) > max_instances:
        # Stratified sample if possible
        if stratify and df["label"].nunique() > 1:
            # For each class, determine how many samples to take
            class_counts = df["label"].value_counts()
            # Compute per-class sample sizes, but do not exceed the available samples in each class
            per_class_n = {
                label: min(count, int(round(max_instances * count / len(df))))
                for label, count in class_counts.items()
            }
            # Adjust total if rounding error causes sum to differ from max_instances
            total = sum(per_class_n.values())
            diff = max_instances - total
            if diff != 0:
                # Adjust the largest class to absorb the difference
                largest_class = max(per_class_n, key=lambda k: per_class_n[k])
                per_class_n[largest_class] += diff
                # But do not exceed the available samples
                per_class_n[largest_class] = min(per_class_n[largest_class], class_counts[largest_class])
            # Now sample per class
            sampled = []
            for label, n in per_class_n.items():
                group = df[df["label"] == label]
                if n > len(group):
                    n = len(group)
                if n > 0:
                    sampled.append(group.sample(n=n, random_state=42, replace=False))
            df = pd.concat(sampled, axis=0)
            # If still too many due to rounding, downsample
            if len(df) > max_instances:
                df = df.sample(n=max_instances, random_state=42)
            print(f"Reduced dataset to {len(df)} instances.")
        else:
            df = df.sample(n=max_instances, random_state=42)
            print(f"Reduced dataset to {len(df)} instances.")

    # Train/validation split
    stratify_vals = df["label"] if stratify and df["label"].nunique() > 1 else None
    try:
        train_df, val_df = train_test_split(
            df[["text", "label"]],
            train_size=train_size,
            random_state=42,
            stratify=stratify_vals,
        )
    except ValueError as e:
        print(f"Warning: train_test_split failed with stratify due to: {e}")
        print("Falling back to random split without stratification.")
        train_df, val_df = train_test_split(
            df[["text", "label"]],
            train_size=train_size,
            random_state=42,
            stratify=None,
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
        "selected_text_fields": selected_text_fields,
        "total_instances": len(df)
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_config.yaml")
    args = parser.parse_args()
    main(args)
