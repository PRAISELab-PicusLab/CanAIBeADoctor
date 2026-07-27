"""
Column mapping for the MedQuAD prompt-sensitivity ablation (Phase 5): Main
(v2 production prompt) vs Variant B vs Variant C, for Mixtral and MedGemma
only, all 3 conditions. See build_ablation_dataset.py for how
df_ablation_MedQuAD.csv is assembled.
"""

DATASET_NAME = "ablation_medquad"
DATA_PATH = "/leonardo_work/CESMA_leonardo/Can_AI_Be_a_Doctor/df_ablation_MedQuAD.csv"
OUTPUT_DIR = "/leonardo_work/CESMA_leonardo/Can_AI_Be_a_Doctor/insights_ablation/medquad"

REFERENCE_LABEL = "Physicians"
REFERENCE_TEXT_COL = "Answer_medico"

MODELS = ["mixtral", "medgemma"]
MODEL_LABELS = {"mixtral": "Mixtral", "medgemma": "MedGemma"}
CONDITIONS = ["base", "empathy", "rephrase"]
CONDITION_LABELS = {"base": "Base", "empathy": "Empathy", "rephrase": "Rephrase"}
VERSIONS = ["Main", "VariantB", "VariantC"]
VERSION_SUFFIX = {"Main": "", "VariantB": "_variantB", "VariantC": "_variantC"}
VERSION_LABEL = {"Main": "Main", "VariantB": "Var.B", "VariantC": "Var.C"}

MODEL_COLUMNS = {
    "Physicians": {
        "text": REFERENCE_TEXT_COL, "fk": f"{REFERENCE_TEXT_COL}_FleschKincaid",
        "gf": f"{REFERENCE_TEXT_COL}_GunningFog", "sentiment": f"{REFERENCE_TEXT_COL}_sentiment",
        "sim": None, "emotion": f"{REFERENCE_TEXT_COL}_emotion",
    },
}
for m in MODELS:
    for c in CONDITIONS:
        for v in VERSIONS:
            col = f"{m}_{c}{VERSION_SUFFIX[v]}"
            label = f"{MODEL_LABELS[m]}\n{CONDITION_LABELS[c]} ({VERSION_LABEL[v]})"
            MODEL_COLUMNS[label] = {
                "text": col, "fk": f"{col}_FleschKincaid", "gf": f"{col}_GunningFog",
                "sentiment": f"{col}_sentiment", "sim": f"sim_medico_{col}", "emotion": f"{col}_emotion",
            }

MODEL_ORDER = ["Physicians"] + [
    f"{MODEL_LABELS[m]}\n{CONDITION_LABELS[c]} ({VERSION_LABEL[v]})"
    for m in MODELS for c in CONDITIONS for v in VERSIONS
]

SEVERITY_COL = None

EMOTION_TARGETS = [(c["text"], c["emotion"]) for c in MODEL_COLUMNS.values()]
SENTIMENT_TARGETS = [(c["text"], c["sentiment"]) for c in MODEL_COLUMNS.values()]
READABILITY_TARGETS = [(c["text"], c["fk"], c["gf"]) for c in MODEL_COLUMNS.values()]
SIMILARITY_TARGETS = [(c["text"], c["sim"]) for c in MODEL_COLUMNS.values() if c["sim"]]
