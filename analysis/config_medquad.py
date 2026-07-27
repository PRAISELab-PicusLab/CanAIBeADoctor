"""
Column mapping for the MedQuAD dataset. Naming here predates the cleaner
snake_case scheme used for Iclinicqa (mix of spaces/CamelCase), and several
model/condition combinations were only ever computed for a subset of columns
(e.g. Gemini Base/prompt2 raw text exists but was never scored -- only
Gemini Rephrase has FK/sentiment/similarity/emotion, so only that condition
is included below, matching the Iclinicqa dataset's Gemini-Rephrase-only pattern).

Decisions made resolving ambiguous/legacy columns (confirm and adjust if wrong):
- "prompt2" columns are treated as the Empathy condition (per user confirmation).
- "Answer_medico_caring_LLM(_PALMYRA)" are treated as the Rephrase condition for
  Mixtral/MedPaLM (per user confirmation) -- an older naming predating the
  "rephrase" label used for GPT/Gemini/Claude, not a second empathy attempt.
  This makes all five models symmetric: Base + Empathy(prompt2) + Rephrase.
- "GPT4o base_FleschKincaid/GunningFog" are leftover columns with no matching
  raw-text column (no "GPT4o base" answer column exists) -- dropped as stale.
- "Sentiment_gpt_base" (lowercase, legacy) is dropped in favor of the current
  "Sentiment_GPT5_BASE".
- MedQuAD has no severity labels, so the severity-clustering scatter plot
  (generated for Iclinicqa) does not apply here.
"""

DATASET_NAME = "medquad"
DATA_PATH = "/leonardo_work/CESMA_leonardo/Can_AI_Be_a_Doctor/df_top50_ MedQuAD.csv"
OUTPUT_DIR = "/leonardo_work/CESMA_leonardo/Can_AI_Be_a_Doctor/insights/outputs/medquad"

REFERENCE_LABEL = "Physicians"

# label -> {text, fk, gf, sentiment, sim, emotion_raw}
# "emotion_raw" holds the precomputed stringified score-dict column; common.parse_emotion_dict
# turns it into a per-row dominant-emotion label at load time (see generate_insights.py).
MODEL_COLUMNS = {
    "Physicians": {
        "text": "Answer_medico", "fk": "Answer_medico_FleschKincaid", "gf": "Answer_medico_GunningFog",
        "sentiment": "Sentiment_medico", "sim": None, "emotion_raw": "Answer_medico_emotions",
    },
    "Mixtral\nBase": {
        "text": "Answer_LLM", "fk": "Answer_LLM_FleschKincaid", "gf": "Answer_LLM_GunningFog",
        "sentiment": "Sentiment_LLM", "sim": "sim_medico_llm", "emotion_raw": "Answer_LLM_emotions",
    },
    "Mixtral\nEmpathy": {
        "text": "Answer_LLM_prompt2", "fk": "Answer_LLM_prompt2_FleschKincaid", "gf": "Answer_LLM_prompt2_GunningFog",
        "sentiment": "Sentiment_LLM_prompt2", "sim": "sim_medico_llm_prompt2", "emotion_raw": "Answer_LLM_prompt2_emotions",
    },
    "MedPaLM\nBase": {
        "text": "Answer_LLM_PALMYRA", "fk": "Answer_LLM_PALMYRA_FleschKincaid", "gf": "Answer_LLM_PALMYRA_GunningFog",
        "sentiment": "Sentiment_LLM_PALMYRA", "sim": "sim_medico_palmyra", "emotion_raw": "Answer_LLM_PALMYRA_emotions",
    },
    "MedPaLM\nEmpathy": {
        "text": "Answer_LLM_PALMYRA_prompt2", "fk": "Answer_LLM_PALMYRA_prompt2_FleschKincaid",
        "gf": "Answer_LLM_PALMYRA_prompt2_GunningFog", "sentiment": "Sentiment_LLM_PALMYRA_prompt2",
        "sim": "sim_medico_palmyra_prompt2", "emotion_raw": "Answer_LLM_PALMYRA_prompt2_emotions",
    },
    "Mixtral\nRephrase": {
        "text": "Answer_medico_caring_LLM", "fk": "Answer_medico_caring_LLM_FleschKincaid",
        "gf": "Answer_medico_caring_LLM_GunningFog", "sentiment": "Sentiment_CARING_LLM",
        "sim": "sim_medico_caring_llm", "emotion_raw": "Answer_medico_caring_LLM_emotions",
    },
    "MedPaLM\nRephrase": {
        "text": "Answer_medico_caring_LLM_PALMYRA", "fk": "Answer_medico_caring_LLM_PALMYRA_FleschKincaid",
        "gf": "Answer_medico_caring_LLM_PALMYRA_GunningFog", "sentiment": "Sentiment_CARING_PALMYRA",
        "sim": "sim_medico_caring_palmyra", "emotion_raw": "Answer_medico_caring_LLM_PALMYRA_emotions",
    },
    "GPT\nBase": {
        "text": "GPT5 BASE", "fk": "GPT5 BASE_FleschKincaid", "gf": "GPT5 BASE_GunningFog",
        "sentiment": "Sentiment_GPT5_BASE", "sim": "sim_medico_GPT5_BASE", "emotion_raw": "GPT5_BASE_emotions",
    },
    "GPT\nEmpathy": {
        "text": "GPT5_prompt2", "fk": "GPT5_prompt2_FleschKincaid", "gf": "GPT5_prompt2_GunningFog",
        "sentiment": "Sentiment_GPT5_prompt2", "sim": "sim_medico_GPT5_prompt2", "emotion_raw": "GPT5_prompt2_emotions",
    },
    "GPT\nRephrase": {
        "text": "GPT5_rephrase", "fk": "GPT5_rephrase_FleschKincaid", "gf": "GPT5_rephrase_GunningFog",
        "sentiment": "Sentiment_GPT5_rephrase", "sim": "sim_medico_GPT5_rephrase", "emotion_raw": "GPT5_rephrase_emotions",
    },
    "Gemini\nRephrase": {
        "text": "Gemini_rephrase", "fk": "Gemini_rephrase_FleschKincaid", "gf": "Gemini_rephrase_GunningFog",
        "sentiment": "Sentiment_Gemini_rephrase", "sim": "sim_medico_Gemini_rephrase", "emotion_raw": "Gemini_rephrase_emotions",
    },
    "Claude\nBase": {
        "text": "Claude BASE", "fk": "Claude BASE_FleschKincaid", "gf": "Claude BASE_GunningFog",
        "sentiment": "Sentiment_Claude_BASE", "sim": "sim_medico_Claude_BASE", "emotion_raw": "Claude_BASE_emotions",
    },
    "Claude\nEmpathy": {
        "text": "Claude_prompt2", "fk": "Claude_prompt2_FleschKincaid", "gf": "Claude_prompt2_GunningFog",
        "sentiment": "Sentiment_Claude_prompt2", "sim": "sim_medico_Claude_prompt2", "emotion_raw": "Claude_prompt2_emotions",
    },
    "Claude\nRephrase": {
        "text": "Claude_rephrase", "fk": "Claude_rephrase_FleschKincaid", "gf": "Claude_rephrase_GunningFog",
        "sentiment": "Sentiment_Claude_rephrase", "sim": "sim_medico_Claude_rephrase", "emotion_raw": "Claude_rephrase_emotions",
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

# No severity data available for MedQuAD.
SEVERITY_COL = None
