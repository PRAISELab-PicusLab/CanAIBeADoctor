"""
Column mapping for the Iclinicqa dataset. All FK/GunningFog/sentiment/similarity
columns are already precomputed in the source CSV; only per-row dominant emotion
is missing and gets added by compute_emotions_iclinicqa.py into DATA_PATH below.
"""

DATASET_NAME = "iclinicqa"
RAW_DATA_PATH = "/leonardo_work/CESMA_leonardo/Can_AI_Be_a_Doctor/top50_Iclinicqa.csv"
DATA_PATH = "/leonardo_work/CESMA_leonardo/Can_AI_Be_a_Doctor/insights/top50_Iclinicqa_with_emotions.csv"
OUTPUT_DIR = "/leonardo_work/CESMA_leonardo/Can_AI_Be_a_Doctor/insights/outputs/iclinicqa"

REFERENCE_LABEL = "Physicians"

# label -> {text, fk, gf, sentiment, sim, emotion}
# "sim"/"emotion" left unset (None) for the physicians' own reference answer.
MODEL_COLUMNS = {
    "Physicians": {
        "text": "answer", "fk": "answer_FK", "gf": "answer_GunningFog",
        "sentiment": "answer_sentiment", "sim": None, "emotion": "answer_emotion",
    },
    "Mixtral\nBase": {
        "text": "answer_llm", "fk": "answer_llm_FK", "gf": "answer_llm_GunningFog",
        "sentiment": "answer_llm_sentiment", "sim": "sim_medico_answer_llm", "emotion": "answer_llm_emotion",
    },
    "Mixtral\nEmpathy": {
        "text": "llm_caring_answer", "fk": "llm_caring_answer_FK", "gf": "llm_caring_answer_GunningFog",
        "sentiment": "llm_caring_answer_sentiment", "sim": "sim_medico_llm_caring_answer", "emotion": "llm_caring_answer_emotion",
    },
    "Mixtral\nRephrase": {
        "text": "llm_rep_answer", "fk": "llm_rep_answer_FK", "gf": "llm_rep_answer_GunningFog",
        "sentiment": "llm_rep_answer_sentiment", "sim": "sim_medico_llm_rep_answer", "emotion": "llm_rep_answer_emotion",
    },
    "MedPaLM\nBase": {
        "text": "palmed_answer", "fk": "palmed_answer_FK", "gf": "palmed_answer_GunningFog",
        "sentiment": "palmed_answer_sentiment", "sim": "sim_medico_palmed_answer", "emotion": "palmed_answer_emotion",
    },
    "MedPaLM\nEmpathy": {
        "text": "palmed_caring", "fk": "palmed_caring_FK", "gf": "palmed_caring_GunningFog",
        "sentiment": "palmed_caring_sentiment", "sim": "sim_medico_palmed_caring", "emotion": "palmed_caring_emotion",
    },
    "MedPaLM\nRephrase": {
        "text": "palmed_rep", "fk": "palmed_rep_FK", "gf": "palmed_rep_GunningFog",
        "sentiment": "palmed_rep_sentiment", "sim": "sim_medico_palmed_rep", "emotion": "palmed_rep_emotion",
    },
    "GPT\nBase": {
        "text": "gpt_answer", "fk": "gpt_answer_FK", "gf": "gpt_answer_GunningFog",
        "sentiment": "gpt_answer_sentiment", "sim": "sim_medico_gpt_answer", "emotion": "gpt_answer_emotion",
    },
    "GPT\nEmpathy": {
        "text": "gpt_caring", "fk": "gpt_caring_FK", "gf": "gpt_caring_GunningFog",
        "sentiment": "gpt_caring_sentiment", "sim": "sim_medico_gpt_caring", "emotion": "gpt_caring_emotion",
    },
    "GPT\nRephrase": {
        "text": "gpt_rep", "fk": "gpt_rep_FK", "gf": "gpt_rep_GunningFog",
        "sentiment": "gpt_rep_sentiment", "sim": "sim_medico_gpt_rep", "emotion": "gpt_rep_emotion",
    },
    "Gemini\nRephrase": {
        "text": "gemini_answer_rep", "fk": "gemini_answer_rep_FK", "gf": "gemini_answer_rep_GunningFog",
        "sentiment": "gemini_answer_rep_sentiment", "sim": "sim_medico_gemini_answer_rep", "emotion": "gemini_answer_rep_emotion",
    },
    "Claude\nBase": {
        "text": "Claude_answer", "fk": "Claude_answer_FK", "gf": "Claude_answer_GunningFog",
        "sentiment": "Claude_answer_sentiment", "sim": "sim_medico_Claude_answer", "emotion": "Claude_answer_emotion",
    },
    "Claude\nEmpathy": {
        "text": "Claude_caring_answer", "fk": "Claude_caring_answer_FK", "gf": "Claude_caring_answer_GunningFog",
        "sentiment": "Claude_caring_answer_sentiment", "sim": "sim_medico_Claude_caring_answer", "emotion": "Claude_caring_answer_emotion",
    },
    "Claude\nRephrase": {
        "text": "Claude_rep_answer", "fk": "Claude_rep_answer_FK", "gf": "Claude_rep_answer_GunningFog",
        "sentiment": "Claude_rep_answer_sentiment", "sim": "sim_medico_Claude_rep_answer", "emotion": "Claude_rep_answer_emotion",
    },
}

MODEL_ORDER = [
    "Physicians",
    "Mixtral\nBase", "Mixtral\nEmpathy", "Mixtral\nRephrase",
    "MedPaLM\nBase", "MedPaLM\nEmpathy", "MedPaLM\nRephrase",
    "GPT\nBase", "GPT\nEmpathy", "GPT\nRephrase",
    "Gemini\nRephrase",
    "Claude\nBase", "Claude\nEmpathy", "Claude\nRephrase",
]

# Iclinicqa has Severity_code (1-5) -> the severity-clustering scatter plot applies here.
SEVERITY_COL = "Severity_code"

# Columns that still need per-row dominant-emotion analysis (GoEmotions), used by
# compute_emotions_iclinicqa.py. Kept as (source_text_col, output_emotion_col) pairs.
EMOTION_TARGETS = [(c["text"], c["emotion"]) for c in MODEL_COLUMNS.values()]
