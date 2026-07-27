"""
One-off patch: removes the degenerate MedQuAD row ("How to prevent
Acanthamoeba - Granulomatous Amebic Encephalitis (GAE); Keratitis ?", whose
Answer_medico is literally just the scraped placeholder "Topics") from
df_top200_MedQuAD.csv, and replaces it with one new row for the same
"prevention" category, selected via the same readability-clustering
criterion as build_extended_medquad.py (FKGL/GFI/TF-IDF/length, IQR filter,
k-means with k=1 among "prevention" candidates not already used).

Adds a minimum-word-count guard (>=20 words) on top of the existing IQR
filter, since the degenerate row slipped through precisely because it was
part of the original fixed top-50 (predates this session's clustering
pipeline, which the IQR filter alone might not have been enough for --
"Topics" is short enough it likely would have been an FK/GF outlier anyway,
but no reason to rely on that alone).

Run on the login node (CPU only, textstat/sklearn, same as
build_extended_medquad.py -- no GPU needed).
"""

import os

import numpy as np
import pandas as pd
import textstat
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(HERE)
HF_CSV_URL = "https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset/resolve/main/medDataset_processed.csv"
MEDQUAD_PATH = os.path.join(PROJECT_ROOT, "df_top200_MedQuAD.csv")

BAD_QUESTION = "How to prevent Acanthamoeba - Granulomatous Amebic Encephalitis (GAE); Keratitis ?"
CATEGORY = "prevention"
MIN_WORDS = 20


def main():
    current = pd.read_csv(MEDQUAD_PATH)
    current["question"] = current["question"].astype(str).str.strip()
    before = len(current)
    current = current[current["question"] != BAD_QUESTION.strip()]
    print(f"Removed bad row: {before} -> {len(current)} rows")

    print("Loading full MedQuAD corpus ...")
    full = pd.read_csv(HF_CSV_URL)
    full = full.rename(columns={"Question": "question", "Answer": "Answer_medico", "qtype": "context"})
    full["question"] = full["question"].astype(str).str.strip()
    full = full.dropna(subset=["question", "Answer_medico", "context"])
    full = full.drop_duplicates(subset=["question"])

    used_questions = set(current["question"])
    pool = full[(full["context"] == CATEGORY) & (~full["question"].isin(used_questions))].copy()
    pool = pool[pool["question"] != BAD_QUESTION.strip()]

    word_count = pool["Answer_medico"].astype(str).str.split().str.len()
    pool = pool[word_count >= MIN_WORDS]
    print(f"Candidate pool for '{CATEGORY}' after used/bad/length filters: {len(pool)}")

    pool["_fk"] = pool["Answer_medico"].apply(lambda t: textstat.flesch_kincaid_grade(str(t)))
    pool["_gf"] = pool["Answer_medico"].apply(lambda t: textstat.gunning_fog(str(t)))
    pool["_len"] = pool["Answer_medico"].apply(lambda t: len(str(t).split()))

    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    X_tfidf = tfidf.fit_transform(pool["Answer_medico"].astype(str))
    centroid = np.asarray(X_tfidf.mean(axis=0))
    pool["_lexrep"] = cosine_similarity(X_tfidf, centroid).ravel()

    for col in ["_fk", "_gf"]:
        q1, q3 = pool[col].quantile(0.25), pool[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        pool = pool[pool[col].between(lo, hi)]
    print(f"After IQR filtering: {len(pool)}")

    features = ["_fk", "_gf", "_lexrep", "_len"]
    X = StandardScaler().fit_transform(pool[features].values)
    km = KMeans(n_clusters=1, random_state=42, n_init=20).fit(X)
    dist = km.transform(X)[:, 0]
    pool = pool.assign(_dist=dist).sort_values("_dist")
    new_row = pool.iloc[[0]][["context", "question", "Answer_medico"]]
    print("\nSelected replacement row:")
    print(new_row.to_string())

    keep_cols = ["context", "question", "Answer_medico"]
    out = pd.concat([current[keep_cols], new_row], ignore_index=True)
    out.to_csv(MEDQUAD_PATH, index=False)
    print(f"\nSaved {len(out)} rows -> {MEDQUAD_PATH}")


if __name__ == "__main__":
    main()
