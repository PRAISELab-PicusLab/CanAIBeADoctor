"""
Adds a per-row dominant-emotion column for every text column in EMOTION_TARGETS,
using GoEmotions (SamLowe/roberta-base-go_emotions), majority vote over
512-token chunks -- same method as compute_emotions_iclinicqa.py, generalized
here for the two extended datasets (config_medquad_ext / config_medredqa_ext).

Must run on a GPU compute node (works on CPU too, just much slower):
    srun --partition=boost_usr_prod --gres=gpu:1 --time=01:00:00 --ntasks=1 --cpus-per-task=8 --mem=32G \
        python3 compute_emotions_ext.py --dataset medquad_ext [--smoke-test]

Writes "metrics/<dataset>_emotion.csv" (question + "{col}_emotion" columns) --
NOT cfg.DATA_PATH directly, so this can run in parallel with
compute_similarity.py/compute_sentiment.py on the same dataset without a
read-modify-write race on the shared main CSV. merge_metrics.py folds all
three metric files back into cfg.DATA_PATH afterward.
"""
import argparse
import os
from collections import Counter

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(__file__), ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODEL_NAME = "SamLowe/roberta-base-go_emotions"
MAX_TOKENS = 512


def load_config(dataset):
    if dataset == "medquad_ext":
        import config_medquad_ext as cfg
    elif dataset == "medredqa_ext":
        import config_medredqa_ext as cfg
    elif dataset == "ablation_medquad":
        import config_ablation_medquad as cfg
    elif dataset == "ablation_medredqa":
        import config_ablation_medredqa as cfg
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    return cfg


def chunk_text(text, tokenizer, max_tokens=MAX_TOKENS):
    if not isinstance(text, str) or not text.strip():
        return []
    ids = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
    return [tokenizer.decode(ids[i:i + max_tokens], skip_special_tokens=True)
            for i in range(0, len(ids), max_tokens)]


def majority_emotion(text, tokenizer, pipe):
    chunks = chunk_text(text, tokenizer)
    if not chunks:
        return None
    labels = []
    for chunk in chunks:
        try:
            labels.append(pipe(chunk, truncation=True, max_length=MAX_TOKENS)[0]["label"])
        except Exception as e:
            print(f"  [warn] chunk failed: {e}")
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["medquad_ext", "medredqa_ext", "ablation_medquad", "ablation_medredqa"])
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.dataset)

    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading {MODEL_NAME} (device={device}) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
    print("Loaded.")

    df = pd.read_csv(cfg.DATA_PATH)
    if args.smoke_test:
        df = df.head(3).copy()
    print(f"Loaded {cfg.DATA_PATH}: {df.shape}")

    seen = set()
    for text_col, emotion_col in cfg.EMOTION_TARGETS:
        if text_col in seen or text_col not in df.columns:
            continue
        seen.add(text_col)
        print(f"Computing {emotion_col} ({df[text_col].notna().sum()} rows) ...", flush=True)
        df[emotion_col] = df[text_col].apply(lambda t: majority_emotion(t, tokenizer, pipe))

    emotion_cols = [c for _, c in cfg.EMOTION_TARGETS if c in df.columns]
    if args.smoke_test:
        print(df[emotion_cols[:3]])
        print("Smoke test complete, nothing saved.")
        return

    out_dir = os.path.join(os.path.dirname(__file__), "metrics")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.dataset}_emotion.csv")
    df[["question"] + emotion_cols].to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
