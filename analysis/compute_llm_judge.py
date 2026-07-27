"""
LLM-as-Judge pass over a dataset's model answers, using Qwen3.6-27B (local
weights on scratch, see config_llm_judge.MODEL_PATH) as an independent
evaluator against the physician reference answer.

For every model/condition column (everything in MODEL_COLUMNS except the
Physicians reference itself) and every row, runs the four independent
per-axis judgments defined in config_llm_judge.AXES (factual accuracy,
completeness, clinical coherence, safety) and writes them as new columns:
"<text_col>_judge_factual", "..._judge_completeness", "..._judge_coherence",
"..._judge_unsafe".

Must run on a GPU compute node with 2 GPUs (the 32B model in bf16 doesn't fit
comfortably on a single 64GB A100 and accelerate falls back to slow CPU
offload with just one):
    srun --partition=boost_usr_prod --gres=gpu:2 --time=04:00:00 \
        --ntasks=1 --cpus-per-task=16 --mem=150G \
        python3 compute_llm_judge.py --dataset iclinicqa [--smoke-test]

Writes "<dataset>_with_judge.csv" next to this script, checkpointing after
every axis so a crash partway through only loses the in-flight axis.

Pass --axis <factual|completeness|coherence|safety> to restrict a run to one
axis (used for job-array sharding, see run_llm_judge_array.sbatch); this
writes "<dataset>_with_judge_<axis>.csv" instead, to be combined afterwards
with merge_judge_shards.py.
"""

import argparse
import json
import os
import re

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(__file__), ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config_llm_judge import (
    MODEL_PATH, GENERATION_KWARGS, ENABLE_THINKING, JUDGE_SYSTEM_PROMPT, AXES, similarity_bin,
)

HERE = os.path.dirname(__file__)


def load_config(dataset):
    if dataset == "medquad":
        import config_medquad as cfg
    elif dataset == "iclinicqa":
        import config_iclinicqa as cfg
    elif dataset == "medquad_ext":
        import config_medquad_ext as cfg
    elif dataset == "medredqa_ext":
        import config_medredqa_ext as cfg
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    return cfg


def build_prompt(tokenizer, axis_key, question, reference, candidate, similarity=None):
    axis = AXES[axis_key]
    if similarity is not None and pd.notna(similarity):
        similarity_pct = similarity * 100
        similarity_str = f"{similarity_pct:.0f}/100"
        anchor_str = str(similarity_bin(similarity_pct))
    else:
        similarity_str = "not available"
        anchor_str = "not available"
    user_content = axis["prompt"].format(
        question=str(question) if pd.notna(question) else "(question not available)",
        reference=str(reference) if pd.notna(reference) else "(reference not available)",
        candidate=str(candidate),
        similarity=similarity_str,
        anchor=anchor_str,
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=ENABLE_THINKING,
    )


def parse_judge_output(text, axis_key):
    """Strip a leading <think>...</think> block if present, then pull the first {...} JSON object."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if axis_key == "safety":
        return obj.get("unsafe"), obj.get("reason")
    return obj.get("score"), obj.get("reason")


def run_axis_batch(model, tokenizer, prompts, batch_size, device):
    """Greedy-generate completions for a list of already-templated prompts, batched."""
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, **GENERATION_KWARGS, pad_token_id=tokenizer.pad_token_id)
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        outputs.extend(decoded)
        print(f"  batch {i // batch_size + 1}/{(len(prompts) - 1) // batch_size + 1} done", flush=True)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["medquad", "iclinicqa", "medquad_ext", "medredqa_ext"])
    parser.add_argument("--smoke-test", action="store_true", help="only first 2 rows, first model column, first axis")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--axis", choices=list(AXES.keys()), default=None,
                         help="restrict to a single axis (for job-array sharding); writes a "
                              "per-axis shard file instead of the combined output")
    args = parser.parse_args()

    cfg = load_config(args.dataset)
    out_path = os.path.join(
        HERE,
        f"{args.dataset}_with_judge.csv" if args.axis is None
        else f"{args.dataset}_with_judge_{args.axis}.csv",
    )

    print(f"Loading tokenizer/model from {MODEL_PATH} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    device = next(model.parameters()).device
    print("Model loaded.", flush=True)

    df = pd.read_csv(cfg.DATA_PATH)
    if args.smoke_test:
        df = df.head(2).copy()
    elif os.path.exists(out_path):
        # Resume: fold in whatever judge columns were already computed in a
        # previous (possibly interrupted, or partially-unparseable) run of
        # this exact axis/dataset, so rows_to_do below only re-attempts rows
        # still missing a score -- this script used to always start fresh
        # from cfg.DATA_PATH, silently redoing everything on every re-run.
        prior = pd.read_csv(out_path)
        judge_cols = [c for c in prior.columns if "judge" in c]
        key = "question" if "question" in df.columns else df.columns[0]
        prior_key = prior[key].astype(str).str.strip()
        df_key = df[key].astype(str).str.strip()
        prior_indexed = prior.set_index(prior_key)[judge_cols]
        for c in judge_cols:
            df[c] = df_key.map(prior_indexed[c])
        print(f"Resuming from {out_path}: {len(judge_cols)} judge columns loaded", flush=True)

    question_col = "question" if "question" in df.columns else None
    ref_col = cfg.MODEL_COLUMNS[cfg.REFERENCE_LABEL]["text"]

    labels = [l for l in cfg.MODEL_ORDER if l != cfg.REFERENCE_LABEL and l in cfg.MODEL_COLUMNS]
    axis_keys = [args.axis] if args.axis is not None else list(AXES.keys())
    if args.smoke_test:
        labels = labels[:1]
        axis_keys = axis_keys[:1]

    for axis_key in axis_keys:
        axis = AXES[axis_key]
        for label in labels:
            text_col = cfg.MODEL_COLUMNS[label]["text"]
            if text_col not in df.columns:
                print(f"[skip] missing column: {text_col}")
                continue
            sim_col = cfg.MODEL_COLUMNS[label]["sim"]
            score_col = f"{text_col}_{axis['column_suffix']}"
            reason_col = f"{score_col}_reason"
            if score_col not in df.columns:
                df[score_col] = None
                df[reason_col] = None

            rows_to_do = df[df[text_col].notna() & df[score_col].isna()]
            if rows_to_do.empty:
                continue
            print(f"[{axis_key}] {label} ({text_col}): {len(rows_to_do)} rows", flush=True)

            prompts = [
                build_prompt(
                    tokenizer, axis_key,
                    row[question_col] if question_col else None,
                    row[ref_col],
                    row[text_col],
                    row[sim_col] if sim_col and sim_col in df.columns else None,
                )
                for _, row in rows_to_do.iterrows()
            ]
            completions = run_axis_batch(model, tokenizer, prompts, args.batch_size, device)

            for idx, completion in zip(rows_to_do.index, completions):
                parsed = parse_judge_output(completion, axis_key)
                if parsed is None:
                    print(f"  [warn] unparseable output at row {idx}: {completion[:200]!r}")
                    continue
                value, reason = parsed
                df.at[idx, score_col] = value
                df.at[idx, reason_col] = reason

        if not args.smoke_test:
            df.to_csv(out_path, index=False)
            print(f"Checkpoint saved after axis '{axis_key}' -> {out_path}", flush=True)

    if args.smoke_test:
        preview_cols = [c for c in df.columns if "judge" in c]
        print(df[preview_cols])
        print("Smoke test complete, nothing saved.")
        return

    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
