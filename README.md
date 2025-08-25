
# =====================
# README.md
# =====================
# Cyber LLM Project — Starter

This repo fine-tunes Transformer models for cybersecurity classification using your structured dataset (e.g., predicting **Attack Type** from `Scenario Description` + `Tools Used`). It’s modular so you can swap models easily (BERT/RoBERTa/DeBERTa/Longformer/etc.).

## Quickstart

```bash
# 0) Create folders and place your CSV
mkdir -p data/raw data/processed models results configs scripts
# Save your dataset as: data/raw/cyber_attacks.csv

# 1) Install deps
pip install -r requirements.txt

# 2) Configure
# Edit configs/model_config.yaml (choose model_name, label_field, text_fields)

# 3) Preprocess
python scripts/preprocess.py --config configs/model_config.yaml

# 4) Train
python scripts/train.py --config configs/model_config.yaml

# 5) Evaluate a saved model
python scripts/evaluate.py --model_dir models/roberta_attack_type_v1 --processed_dir data/processed
```

## Switching Models
- In `configs/model_config.yaml`, set `model_name` to one of:
  - `bert-base-uncased`, `roberta-base`, `distilbert-base-uncased`, `microsoft/deberta-v3-base`, `allenai/longformer-base-4096`, or any Hugging Face model id.
- Or use shorthand keys by editing `scripts/utils.py` mapping (e.g., `"deberta" -> microsoft/deberta-v3-base`).

## Changing the Prediction Target
- Set `label_field` in `model_config.yaml` to: `Attack Type`, `Category`, `MITRE Technique`, `Impact`, etc.
- Adjust `text_fields` to include the columns you want (e.g., `Scenario Description`, `Tools Used`, `Detection Method`).

## Notes
- For **Longformer**, increase `max_seq_length` to 1024–4096 and ensure GPU memory is sufficient.
- Metric reported is **macro F1** by default (via Trainer’s compute_metrics).
- Class imbalance: consider weighted loss or augmentation later.

## Next Steps
- Add **multi-task** heads (predict `Attack Type` and `MITRE Technique` jointly).
- Add **NER** model for extracting `Tools Used`, `Vulnerability`, etc.
- Add **inference script** with pretty JSON outputs for a single scenario.



# Note

Got it — here’s your **summarised methodology** formatted so it can slot neatly into your `README.md`:

---

## 🛠 Methodology

Our phishing detection system follows a structured pipeline to ensure accurate, real-time detection:

1. **Data Collection**

   * Gathered phishing and legitimate website/email datasets from public sources such as PhishTank, UCI ML Repository, and Kaggle.

2. **Data Preprocessing**

   * Removed duplicates and irrelevant entries.
   * Extracted and normalized features such as URL length, presence of special characters, domain age, SSL certificate validity, and email header patterns.

3. **Feature Engineering**

   * Applied domain-specific knowledge to create meaningful features.
   * Used **TF-IDF vectorization** for textual content and **One-Hot Encoding** for categorical features.

4. **Model Selection & Training**

   * Experimented with various algorithms:

     * Logistic Regression
     * Random Forest
     * Gradient Boosting (XGBoost, LightGBM)
     * Support Vector Machines (SVM)
     * Neural Networks (MLP)
   * Selected **Random Forest + XGBoost ensemble** for optimal accuracy and low false positives.

5. **Model Evaluation**

   * Evaluated models using **accuracy, precision, recall, F1-score**, and **ROC-AUC** metrics.
   * Performed **cross-validation** to prevent overfitting.

6. **Deployment Pipeline**

   * Packaged the model with **Django REST Framework** for API serving.
   * Integrated with real-time URL/email scanning endpoints.
   * Designed for scalability, supporting future integration with browser extensions and email gateways.

7. **Continuous Learning**

   * Periodic retraining with new phishing patterns.
   * Incremental updates without full retraining to adapt to emerging threats.

---

If you want, I can also **add an architecture diagram** to your README so it’s visually appealing. That would make the methodology more digestible.
