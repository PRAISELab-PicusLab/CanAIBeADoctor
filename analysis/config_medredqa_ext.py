"""
Column mapping for the MedRedQA dataset (200 rows, df_top200_MedRedQA.csv, the new
third dataset added this session). Same 5-model x 3-condition symmetric roster
and metric-column naming convention as config_medquad_ext.py -- see that file's
docstring for the label-mapping rationale.
"""

DATASET_NAME = "medredqa_ext"
DATA_PATH = "/leonardo_work/CESMA_leonardo/Can_AI_Be_a_Doctor/df_top200_MedRedQA.csv"
OUTPUT_DIR = "/leonardo_work/CESMA_leonardo/Can_AI_Be_a_Doctor/insights/outputs/medredqa_ext"

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

# No severity data for MedRedQA.
SEVERITY_COL = None

EMOTION_TARGETS = [(c["text"], c["emotion"]) for c in MODEL_COLUMNS.values()]
SENTIMENT_TARGETS = [(c["text"], c["sentiment"]) for c in MODEL_COLUMNS.values()]
READABILITY_TARGETS = [(c["text"], c["fk"], c["gf"]) for c in MODEL_COLUMNS.values()]
SIMILARITY_TARGETS = [(c["text"], c["sim"]) for c in MODEL_COLUMNS.values() if c["sim"]]
