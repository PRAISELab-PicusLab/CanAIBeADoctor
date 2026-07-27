"""
One-off step for the Iclinicqa dataset: adds a per-row dominant-emotion column
for every model/condition text column (GoEmotions, majority vote over 512-token
chunks -- same method as MedQuAD's precomputed emotion columns used, so the two
datasets' emotion labels stay comparable).

Must run on a compute node (torch inference), not the login node:
    srun --partition=boost_usr_prod --time=00:30:00 --ntasks=1 --cpus-per-task=8 \
        python3 compute_emotions_iclinicqa.py [--smoke-test]

Writes config_iclinicqa.DATA_PATH (source CSV + new "<col>_emotion" columns).
"""

import argparse
import os
from collections import Counter

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(__file__), ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from config_iclinicqa import RAW_DATA_PATH, DATA_PATH, EMOTION_TARGETS

MODEL_NAME = "SamLowe/roberta-base-go_emotions"
MAX_TOKENS = 512


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
    parser.add_argument("--smoke-test", action="store_true", help="only process first 3 rows, no save")
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading {MODEL_NAME} (device={device})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
    print("Model loaded.")

    df = pd.read_csv(RAW_DATA_PATH)
    if args.smoke_test:
        df = df.head(3).copy()

    for text_col, out_col in EMOTION_TARGETS:
        if text_col not in df.columns:
            print(f"[skip] missing text column: {text_col}")
            continue
        print(f"Processing {text_col} -> {out_col} ({df[text_col].notna().sum()} rows)")
        df[out_col] = df[text_col].apply(lambda t: majority_emotion(t, tokenizer, pipe))

    if args.smoke_test:
        print(df[[c for _, c in EMOTION_TARGETS if c in df.columns]])
        print("Smoke test complete, nothing saved.")
        return

    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {DATA_PATH}")


if __name__ == "__main__":
    main()
