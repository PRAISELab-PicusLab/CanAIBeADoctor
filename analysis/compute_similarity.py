"""
Computes semantic-fidelity cosine similarity between each model/condition answer
and the physician reference answer, for all pairs in SIMILARITY_TARGETS.

Model: sentence-transformers/all-MiniLM-L6-v2 (matches Generazione_insight.ipynb's
actual code -- the paper text names a different BioBERT encoder, but the notebook
is what was actually run; per explicit user instruction, follow the notebook).

Handles arbitrarily long answers via TOKEN-based chunking (not the notebook's
180-WORD heuristic): all-MiniLM-L6-v2 truncates at 256 tokens
(model.max_seq_length == tokenizer.model_max_length == 256), and 180 words can
exceed that for verbose text, silently losing content. Instead, each text is
split into token chunks sized to the model's own tokenizer (256 - 2 margin for
special tokens), each chunk embedded, and the chunk embeddings mean-pooled --
this guarantees every token of every answer contributes to the final
embedding regardless of length.

Must run on a GPU compute node:
    srun --partition=boost_usr_prod --gres=gpu:1 --time=00:30:00 --ntasks=1 --cpus-per-task=8 --mem=32G \
        python3 compute_similarity.py --dataset medquad_ext [--smoke-test]

Writes "metrics/<dataset>_similarity.csv" (question + "sim_medico_{col}" columns)
-- NOT cfg.DATA_PATH directly, so this can run in parallel with
compute_sentiment.py/compute_emotions_ext.py on the same dataset without a
read-modify-write race on the shared main CSV. merge_metrics.py folds all
three metric files back into cfg.DATA_PATH afterward.
"""
import argparse
import os

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(__file__), ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


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


def chunk_by_tokens(text, tokenizer, max_tokens):
    ids = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
    if not ids:
        return []
    chunks = [ids[i:i + max_tokens] for i in range(0, len(ids), max_tokens)]
    return [tokenizer.decode(c, skip_special_tokens=True) for c in chunks]


def embed_long_text(text, model, tokenizer, max_tokens, device):
    if not isinstance(text, str) or not text.strip():
        return None
    chunks = chunk_by_tokens(text, tokenizer, max_tokens)
    if not chunks:
        return None
    with torch.no_grad():
        embeddings = model.encode(chunks, convert_to_tensor=True, device=device, show_progress_bar=False)
    return embeddings.mean(dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["medquad_ext", "medredqa_ext", "ablation_medquad", "ablation_medredqa"])
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.dataset)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} (device={device}) ...")
    model = SentenceTransformer(MODEL_NAME, cache_folder=os.path.join(os.path.dirname(__file__), ".hf_cache_st"))
    model = model.to(device)
    max_tokens = model.max_seq_length - 2
    tokenizer = model.tokenizer
    print(f"Loaded. max_seq_length={model.max_seq_length}, chunking at {max_tokens} tokens.")

    df = pd.read_csv(cfg.DATA_PATH)
    if args.smoke_test:
        df = df.head(3).copy()
    print(f"Loaded {cfg.DATA_PATH}: {df.shape}")

    ref_col = cfg.REFERENCE_TEXT_COL
    emb_cache = {}

    def get_ref_embedding(idx, text):
        if idx not in emb_cache:
            emb_cache[idx] = embed_long_text(text, model, tokenizer, max_tokens, device)
        return emb_cache[idx]

    for text_col, sim_col in cfg.SIMILARITY_TARGETS:
        if text_col not in df.columns:
            print(f"  [skip] missing column: {text_col}")
            continue
        print(f"Computing {sim_col} ({df[text_col].notna().sum()} rows) ...", flush=True)
        sims = []
        for idx, row in df.iterrows():
            emb_medico = get_ref_embedding(idx, row[ref_col])
            emb_model = embed_long_text(row[text_col], model, tokenizer, max_tokens, device)
            if emb_medico is None or emb_model is None:
                sims.append(None)
                continue
            sims.append(util.cos_sim(emb_medico, emb_model).item())
        df[sim_col] = sims

    sim_cols = [c for _, c in cfg.SIMILARITY_TARGETS if c in df.columns]
    if args.smoke_test:
        print(df[[cfg.MODEL_COLUMNS["Physicians"]["text"]] + sim_cols[:3]])
        print("Smoke test complete, nothing saved.")
        return

    out_dir = os.path.join(os.path.dirname(__file__), "metrics")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.dataset}_similarity.csv")
    df[["question"] + sim_cols].to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
