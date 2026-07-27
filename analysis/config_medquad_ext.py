"""
Column mapping for the EXTENDED MedQuAD dataset (231 rows, df_top200_MedQuAD.csv),
generated fresh in this session with 5 locally-hosted/API models, symmetric
Base/Empathy/Rephrase for every model (unlike the legacy 13-model dataset, which
had gaps -- see config_medquad.py). Chart labels per explicit user decision
2026-07-23 (only the underlying generator differs from a "real" model name for
medgemma/gptoss/gemma4, kept simple/plain for gptoss and gemma4, and NOT mapped
to the old legacy "MedPaLM" name for medgemma):
  mixtral -> "Mixtral", medgemma -> "MedGemma", gptoss -> "GPT",
  gemma4 -> "Gemini", claude -> "Claude".

Metric column naming (produced by compute_readability.py / compute_similarity.py
/ compute_sentiment.py / compute_emotions_ext.py, all in this directory):
  "{col}_FleschKincaid", "{col}_GunningFog", "{col}_sentiment",
  "sim_medico_{col}", "{col}_emotion".
"""

DATASET_NAME = "medquad_ext"
DATA_PATH = "/leonardo_work/CESMA_leonardo/Can_AI_Be_a_Doctor/df_top200_MedQuAD.csv"
OUTPUT_DIR = "/leonardo_work/CESMA_leonardo/Can_AI_Be_a_Doctor/insights/outputs/medquad_ext"

REFERENCE_LABEL = "Physicians"
REFERENCE_TEXT_COL = "Answer_medico"

MODELS = ["mixtral", "medgemma", "gptoss", "gemma4", "claude"]
MODEL_LABELS = {"mixtral": "Mixtral", "medgemma": "MedGemma", "gptoss": "GPT", "gemma4": "Gemini", "claude": "Claude"}
CONDITIONS = ["base", "empathy", "rephrase"]
CONDITION_LABELS = {"base": "Base", "empathy": "Empathy", "rephrase": "Rephrase"}

MODEL_COLUMNS = {
    "Physicians": {
        "text": REFERENCE_TEXT_COL, "fk": f"{REFERENCE_TEXT_COL}_FleschKincaid",
        "gf": f"{REFERENCE_TEXT_COL}_GunningFog", "sentiment": f"{REFERENCE_TEXT_COL}_sentiment",
        "sim": None, "emotion": f"{REFERENCE_TEXT_COL}_emotion",
    },
}
for m in MODELS:
    for c in CONDITIONS:
        col = f"{m}_{c}"
        label = f"{MODEL_LABELS[m]}\n{CONDITION_LABELS[c]}"
        MODEL_COLUMNS[label] = {
            "text": col, "fk": f"{col}_FleschKincaid", "gf": f"{col}_GunningFog",
            "sentiment": f"{col}_sentiment", "sim": f"sim_medico_{col}", "emotion": f"{col}_emotion",
        }

# Gemini (gemma4) Base/Empathy excluded from all analyses per explicit user
# request 2026-07-24 -- mirrors the legacy pipeline's pattern (Gemini only ever
# had a Rephrase condition there too). Data stays in MODEL_COLUMNS/the CSV,
# just dropped from the plotted/analyzed order.
EXCLUDED_LABELS = {"Gemini\nBase", "Gemini\nEmpathy"}
MODEL_ORDER = ["Physicians"] + [
    f"{MODEL_LABELS[m]}\n{CONDITION_LABELS[c]}" for m in MODELS for c in CONDITIONS
    if f"{MODEL_LABELS[m]}\n{CONDITION_LABELS[c]}" not in EXCLUDED_LABELS
]

# No severity data for MedQuAD.
SEVERITY_COL = None

# All (text_col, emotion_col) pairs needing emotion computation.
EMOTION_TARGETS = [(c["text"], c["emotion"]) for c in MODEL_COLUMNS.values()]
# All (text_col, sentiment_col) pairs needing sentiment computation.
SENTIMENT_TARGETS = [(c["text"], c["sentiment"]) for c in MODEL_COLUMNS.values()]
# All (text_col, fk_col, gf_col) needing readability computation.
READABILITY_TARGETS = [(c["text"], c["fk"], c["gf"]) for c in MODEL_COLUMNS.values()]
# All (text_col, sim_col) needing similarity computation (excludes reference).
SIMILARITY_TARGETS = [(c["text"], c["sim"]) for c in MODEL_COLUMNS.values() if c["sim"]]
