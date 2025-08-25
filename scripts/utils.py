
# =====================
# scripts/utils.py
# =====================
import os, json, random, numpy as np, torch
from typing import Dict

MODEL_CATALOG: Dict[str, str] = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "distilbert": "distilbert-base-uncased",
    "deberta": "microsoft/deberta-v3-base",
    "longformer": "allenai/longformer-base-4096",
    # Small/quick baselines
    "bert-tiny": "prajjwal1/bert-tiny",
}

def resolve_model(name_or_key: str) -> str:
    return MODEL_CATALOG.get(name_or_key.lower(), name_or_key)


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
