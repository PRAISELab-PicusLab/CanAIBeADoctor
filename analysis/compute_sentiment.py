"""
Computes a 5-class sentiment label for every text column in SENTIMENT_TARGETS,
using tabularisai/robust-sentiment-analysis with the exact chunked weighted-vote
scheme from Generazione_insight.ipynb: split into 510-token chunks, classify
each chunk, weight each chunk's confidence by SENTIMENT_WEIGHTS, sum per label
across chunks. Final label is Neutral only if EVERY chunk was Neutral;
otherwise Neutral is excluded from the running tally and the argmax over the
remaining labels wins (final score = summed weighted score / n_chunks).

Must run on a GPU compute node (works on CPU too, just much slower):
    srun --partition=boost_usr_prod --gres=gpu:1 --time=01:00:00 --ntasks=1 --cpus-per-task=8 --mem=32G \
        python3 compute_sentiment.py --dataset medquad_ext [--smoke-test]

Writes "metrics/<dataset>_sentiment.csv" (question + "{col}_sentiment" +
"{col}_sentiment_score" columns) -- NOT cfg.DATA_PATH directly, so this can
run in parallel with compute_similarity.py/compute_emotions_ext.py on the
same dataset without a read-modify-write race on the shared main CSV.
merge_metrics.py folds all three metric files back into cfg.DATA_PATH
afterward.
"""
import argparse
import os
from collections import defaultdict

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(__file__), ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer

MODEL_NAME = "tabularisai/robust-sentiment-analysis"
MAX_TOKENS = 510

SENTIMENT_WEIGHTS = {
    "Very Negative": 2.0, "Negative": 1.0, "Neutral": 0.2, "Positive": 1.0, "Very Positive": 2.0,
}


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
    ids = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [ids[i:i + max_tokens] for i in range(0, len(ids), max_tokens)]
    return [tokenizer.decode(c, skip_special_tokens=True) for c in chunks]


def sentiment_full_text_weighted(text, tokenizer, pipe):
    if not isinstance(text, str) or not text.strip():
        return None, None
    chunks = chunk_text(text, tokenizer)
    if not chunks:
        return None, None

    scores = defaultdict(float)
    labels_seen = []
    for chunk in chunks:
        result = pipe(chunk, truncation=True, max_length=MAX_TOKENS)[0]
        label, score = result["label"], result["score"]
        labels_seen.append(label)
        scores[label] += score * SENTIMENT_WEIGHTS[label]

    if set(labels_seen) == {"Neutral"}:
        return "Neutral", scores["Neutral"] / len(chunks)

    scores.pop("Neutral", None)
    final_label = max(scores, key=scores.get)
    return final_label, scores[final_label] / len(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["medquad_ext", "medredqa_ext", "ablation_medquad", "ablation_medredqa"])
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.dataset)

    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading {MODEL_NAME} (device={device}) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    pipe = pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=tokenizer, device=device)
    print("Loaded.")

    df = pd.read_csv(cfg.DATA_PATH)
    if args.smoke_test:
        df = df.head(3).copy()
    print(f"Loaded {cfg.DATA_PATH}: {df.shape}")

    seen = set()
    for text_col, sentiment_col in cfg.SENTIMENT_TARGETS:
        if text_col in seen or text_col not in df.columns:
            continue
        seen.add(text_col)
        score_col = f"{text_col}_sentiment_score"
        print(f"Computing {sentiment_col} ({df[text_col].notna().sum()} rows) ...", flush=True)
        labels, scores = [], []
        for text in df[text_col]:
            label, score = sentiment_full_text_weighted(text, tokenizer, pipe)
            labels.append(label)
            scores.append(score)
        df[sentiment_col] = labels
        df[score_col] = scores

    new_cols = []
    for text_col, sentiment_col in cfg.SENTIMENT_TARGETS:
        if sentiment_col in df.columns:
            new_cols += [sentiment_col, f"{text_col}_sentiment_score"]

    if args.smoke_test:
        print(df[new_cols[:6]])
        print("Smoke test complete, nothing saved.")
        return

    out_dir = os.path.join(os.path.dirname(__file__), "metrics")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.dataset}_sentiment.csv")
    df[["question"] + new_cols].to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
