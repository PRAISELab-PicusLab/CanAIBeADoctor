"""
Extend the MedQuAD subset from 50 to >=200 questions.

Replicates the paper's readability-driven clustering protocol (Section 3.2,
"MedQuAD Subset Construction"): FKGL, GFI, TF-IDF lexical representativeness,
and answer length -> IQR outlier removal on FKGL/GFI -> z-score normalize ->
k-means -> representative = closest to each cluster's centroid. Stratified
per `qtype` category (this HF source has 16 qtypes, not the 37 the paper's
text mentions for the original full corpus -- reviewers' "5-10 items per
category" suggestion is applied against the categories actually present).

The existing 50 rows (df_top50_ MedQuAD.csv) are kept fixed in the output so
prior generations/metrics for those 50 stay valid; new representatives fill
the remaining budget per category.

Usage: python3 build_extended_medquad.py [--target 220]
"""

import argparse
import os

import numpy as np
import pandas as pd
import textstat
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(__file__)
HF_CSV_URL = "https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset/resolve/main/medDataset_processed.csv"
EXISTING_TOP50_PATH = "/leonardo_work/CESMA_leonardo/Can_AI_Be_a_Doctor/df_top50_ MedQuAD.csv"
OUT_PATH = os.path.join(os.path.dirname(HERE), "df_top200_MedQuAD.csv")


def iqr_filter(df, cols):
    mask = pd.Series(True, index=df.index)
    for c in cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask &= df[c].between(lo, hi)
    return df[mask]


def allocate_per_category(counts, target_total):
    """Stratified allocation capped by availability, shortfall redistributed
    proportionally among non-capped categories until target is met or
    exhausted."""
    cats = list(counts.keys())
    base_k = max(1, round(target_total / len(cats)))
    alloc = {c: min(base_k, counts[c]) for c in cats}
    for _ in range(50):
        total = sum(alloc.values())
        if total >= target_total:
            break
        room = {c: counts[c] - alloc[c] for c in cats if counts[c] > alloc[c]}
        if not room:
            break
        room_total = sum(room.values())
        shortfall = target_total - total
        added_any = False
        for c, r in room.items():
            add = max(1, round(shortfall * r / room_total))
            add = min(add, r)
            if add > 0:
                alloc[c] += add
                added_any = True
        if not added_any:
            break
    return alloc


def select_representatives(df, features, k):
    if len(df) <= k:
        return df.copy()
    X = StandardScaler().fit_transform(df[features].values)
    km = KMeans(n_clusters=k, random_state=42, n_init=20).fit(X)
    dist = km.transform(X)
    df = df.copy()
    df["_cluster"] = km.labels_
    df["_dist"] = [dist[i, km.labels_[i]] for i in range(len(df))]
    reps = df.sort_values("_dist").groupby("_cluster").head(1)
    return reps.drop(columns=["_cluster", "_dist"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=220)
    args = parser.parse_args()

    print(f"Loading full corpus from {HF_CSV_URL} ...")
    full = pd.read_csv(HF_CSV_URL)
    full = full.rename(columns={"Question": "question", "Answer": "Answer_medico", "qtype": "context"})
    full["question"] = full["question"].astype(str).str.strip()
    full = full.dropna(subset=["question", "Answer_medico", "context"])
    full = full.drop_duplicates(subset=["question"])
    print("Full corpus:", full.shape, "| categories:", full["context"].nunique())

    existing = pd.read_csv(EXISTING_TOP50_PATH)
    existing_questions = set(existing["question"].astype(str).str.strip())
    print("Existing top-50 rows:", len(existing_questions))

    print("Computing FKGL/GFI/length ...")
    full["_fk"] = full["Answer_medico"].apply(lambda t: textstat.flesch_kincaid_grade(str(t)))
    full["_gf"] = full["Answer_medico"].apply(lambda t: textstat.gunning_fog(str(t)))
    full["_len"] = full["Answer_medico"].apply(lambda t: len(str(t).split()))

    print("Computing TF-IDF lexical representativeness ...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    X_tfidf = tfidf.fit_transform(full["Answer_medico"].astype(str))
    centroid = np.asarray(X_tfidf.mean(axis=0))
    full["_lexrep"] = cosine_similarity(X_tfidf, centroid).ravel()

    filtered = iqr_filter(full, ["_fk", "_gf"])
    print("After IQR filtering:", filtered.shape)

    remaining_budget = args.target - len(existing_questions)
    pool = filtered[~filtered["question"].isin(existing_questions)]
    counts = pool["context"].value_counts().to_dict()
    alloc = allocate_per_category(counts, remaining_budget)
    print("Per-category allocation (new rows):", alloc)

    features = ["_fk", "_gf", "_lexrep", "_len"]
    reps = []
    for cat, k in alloc.items():
        df_cat = pool[pool["context"] == cat]
        reps.append(select_representatives(df_cat, features, k))
    new_rows = pd.concat(reps, ignore_index=True) if reps else pool.iloc[0:0]

    keep_cols = ["context", "question", "Answer_medico"]
    existing_slim = existing[keep_cols].copy()
    new_slim = new_rows[keep_cols].copy()
    out = pd.concat([existing_slim, new_slim], ignore_index=True).drop_duplicates(subset=["question"])

    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(out)} rows -> {OUT_PATH}")
    print("Category coverage:")
    print(out["context"].value_counts())


if __name__ == "__main__":
    main()
