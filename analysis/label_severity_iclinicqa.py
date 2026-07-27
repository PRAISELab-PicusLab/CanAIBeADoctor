"""
Severity (triage) labeling for the iCliniqQAs pool beyond the existing top-50,
using MedGemma-27b-it as a documented, reproducible substitute for the
paper's undocumented "PalMed-2" (flagged by reviewers R2.3/R3.6 as
unreproducible/likely a typo for Med-PaLM 2, which is not publicly
accessible).

5-level scale, matching the exact label strings already used in
top50_Iclinicqa_with_emotions.csv (Severity_code 1-5 <-> Severity_label):
  1 = white        trivial / non-urgent, informational or minor self-limiting concern
  2 = light green   low urgency, routine primary-care matter
  3 = green        moderate, warrants medical evaluation soon but not emergent
  4 = yellow       urgent, should seek care promptly / could worsen without timely care
  5 = red          emergent / potentially life-threatening, needs immediate attention

Each label requires a one-line justification (auditable, unlike the paper's
opaque classification) so a clinician can spot-check a sample afterward.

Must run on a GPU compute node (MedGemma-27b-it, ~54GB bf16):
    srun --partition=boost_usr_prod --gres=gpu:2 --time=04:00:00 \
        --ntasks=1 --cpus-per-task=16 --mem=150G \
        python3 label_severity_iclinicqa.py [--smoke-test]

Writes "iclinicqa_severity_pool.csv" next to this script, checkpointing every
batch (resumable).
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

HERE = os.path.dirname(__file__)
MODEL_PATH = "/leonardo_scratch/large/userexternal/vmoscato/hf_models/medgemma-27b-it"
ICLINIQ_JSON = os.path.join(os.path.dirname(HERE), "icliniqQAs.json")
EXISTING_TOP50 = os.path.join(HERE, "top50_Iclinicqa_with_emotions.csv")
OUT_PATH = os.path.join(HERE, "iclinicqa_severity_pool.csv")

GENERATION_KWARGS = dict(do_sample=False, max_new_tokens=256, temperature=None, top_p=None, top_k=None)

SEVERITY_LABELS = {1: "white", 2: "light green", 3: "green", 4: "yellow", 5: "red"}

SYSTEM_PROMPT = (
    "You are an experienced triage physician. Given a patient's own description of their "
    "problem, assign a clinical urgency level using this fixed 5-level scale:\n"
    "1 (white) = trivial or non-urgent, purely informational or a minor self-limiting concern.\n"
    "2 (light green) = low urgency, a routine primary-care matter.\n"
    "3 (green) = moderate, warrants medical evaluation soon but is not emergent.\n"
    "4 (yellow) = urgent, should seek care promptly or the condition could worsen without "
    "timely care.\n"
    "5 (red) = emergent or potentially life-threatening, needs immediate medical attention.\n"
    "Base your judgment only on what the patient describes. Respond with ONLY a single JSON "
    "object, no extra text."
)

USER_TEMPLATE = (
    "Patient's description:\n{question_text}\n\n"
    'Respond with JSON: {{"level": <integer 1-5>, "reason": "<one sentence>"}}'
)


def build_prompt(tokenizer, question_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(question_text=question_text)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_output(text):
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    level = obj.get("level")
    if level not in (1, 2, 3, 4, 5):
        return None
    return level, obj.get("reason")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    with open(ICLINIQ_JSON) as f:
        all_rows = json.load(f)
    full = pd.DataFrame(all_rows)
    full["question"] = full["question"].astype(str).str.strip()

    existing = pd.read_csv(EXISTING_TOP50)
    existing_questions = set(existing["question"].astype(str).str.strip())
    pool = full[~full["question"].isin(existing_questions)].reset_index(drop=True)
    print(f"Pool to label: {len(pool)} rows (excluded {len(existing_questions)} already in top-50)")

    if args.smoke_test:
        pool = pool.head(2).copy()

    if os.path.exists(OUT_PATH) and not args.smoke_test:
        done = pd.read_csv(OUT_PATH)
        done_qs = set(done["question"].astype(str).str.strip())
        pool = pool[~pool["question"].isin(done_qs)].reset_index(drop=True)
        print(f"Resuming: {len(pool)} rows remaining")

    if pool.empty:
        print("Nothing to do.")
        return

    print(f"Loading tokenizer/model from {MODEL_PATH} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    device = next(model.parameters()).device
    print("Model loaded.", flush=True)

    results = []
    for i in range(0, len(pool), args.batch_size):
        batch = pool.iloc[i:i + args.batch_size]
        text_field = batch["question_text"].where(batch["question_text"].notna(), batch["question"])
        prompts = [build_prompt(tokenizer, str(t)) for t in text_field]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, **GENERATION_KWARGS, pad_token_id=tokenizer.pad_token_id)
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for (_, row), completion in zip(batch.iterrows(), decoded):
            parsed = parse_output(completion)
            if parsed is None:
                print(f"  [warn] unparseable output: {completion[:200]!r}")
                continue
            level, reason = parsed
            results.append({
                "question": row["question"],
                "question_text": row.get("question_text"),
                "answer": row["answer"],
                "tags": row.get("tags"),
                "Severity_code": float(level),
                "Severity_label": SEVERITY_LABELS[level],
                "Severity_reason": reason,
            })
        print(f"  batch {i // args.batch_size + 1}/{(len(pool) - 1) // args.batch_size + 1} done", flush=True)

        if not args.smoke_test:
            df_out = pd.DataFrame(results)
            if os.path.exists(OUT_PATH):
                df_out = pd.concat([pd.read_csv(OUT_PATH), df_out], ignore_index=True)
                df_out = df_out.drop_duplicates(subset=["question"])
            df_out.to_csv(OUT_PATH, index=False)

    if args.smoke_test:
        print(pd.DataFrame(results))
        print("Smoke test complete, nothing saved.")
        return

    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
