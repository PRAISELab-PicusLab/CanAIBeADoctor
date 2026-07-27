"""
Extend the iCliniqQAs subset from 50 to >=200 questions.

Mirrors build_extended_medquad.py's readability-driven clustering protocol
(FKGL, GFI, TF-IDF lexical representativeness, answer length -> IQR outlier
removal on FKGL/GFI -> z-score normalize -> k-means -> representative =
closest to each cluster's centroid), but stratified per `Severity_code`
(1-5, from MedGemma-27b-it triage labeling, see label_severity_iclinicqa.py)
instead of per MedQuAD's `qtype`, since severity is this dataset's analogue
category axis -- the existing top-50 is itself perfectly balanced at 10
rows/level, so this keeps that balance as the dataset grows.

The existing 50 rows (top50_Iclinicqa_with_emotions.csv) are kept fixed in
the output so prior generations/metrics for those 50 stay valid; new
representatives fill the remaining budget per severity level, drawn from the
MedGemma-labeled pool (iclinicqa_severity_pool.csv, produced by
label_severity_iclinicqa.py -- must be run first).

Usage: python3 build_extended_iclinicqa.py [--target 220]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import textstat
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(__file__)
SEVERITY_POOL_PATH = os.path.join(HERE, "iclinicqa_severity_pool.csv")
EXISTING_TOP50_PATH = os.path.join(HERE, "top50_Iclinicqa_with_emotions.csv")
ICLINIQ_JSON = os.path.join(os.path.dirname(HERE), "icliniqQAs.json")
OUT_PATH = os.path.join(os.path.dirname(HERE), "df_top200_Iclinicqa.csv")

KEEP_COLS = ["question", "question_text", "answer", "tags", "url", "Severity_code", "Severity_label"]


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

    if not os.path.exists(SEVERITY_POOL_PATH):
        raise SystemExit(
            f"{SEVERITY_POOL_PATH} not found -- run label_severity_iclinicqa.py first "
            "(sbatch run_severity_labeling.sbatch)."
        )

    pool_full = pd.read_csv(SEVERITY_POOL_PATH)
    pool_full["question"] = pool_full["question"].astype(str).str.strip()
    print("Labeled severity pool:", pool_full.shape)

    # "url" wasn't carried through by label_severity_iclinicqa.py -- recover it
    # from the original JSON pool by question text.
    with open(ICLINIQ_JSON) as f:
        raw = json.load(f)
    url_by_question = {str(r["question"]).strip(): r.get("url") for r in raw}
    pool_full["url"] = pool_full["question"].map(url_by_question)

    existing = pd.read_csv(EXISTING_TOP50_PATH)
    existing_questions = set(existing["question"].astype(str).str.strip())
    print("Existing top-50 rows:", len(existing_questions))

    pool = pool_full[~pool_full["question"].isin(existing_questions)].copy()

    print("Computing FKGL/GFI/length ...")
    pool["_fk"] = pool["answer"].apply(lambda t: textstat.flesch_kincaid_grade(str(t)))
    pool["_gf"] = pool["answer"].apply(lambda t: textstat.gunning_fog(str(t)))
    pool["_len"] = pool["answer"].apply(lambda t: len(str(t).split()))

    print("Computing TF-IDF lexical representativeness ...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    X_tfidf = tfidf.fit_transform(pool["answer"].astype(str))
    centroid = np.asarray(X_tfidf.mean(axis=0))
    pool["_lexrep"] = cosine_similarity(X_tfidf, centroid).ravel()

    filtered = iqr_filter(pool, ["_fk", "_gf"])
    print("After IQR filtering:", filtered.shape)

    remaining_budget = args.target - len(existing_questions)
    counts = filtered["Severity_code"].value_counts().to_dict()
    alloc = allocate_per_category(counts, remaining_budget)
    print("Per-severity-level allocation (new rows):", alloc)

    features = ["_fk", "_gf", "_lexrep", "_len"]
    reps = []
    for level, k in alloc.items():
        df_level = filtered[filtered["Severity_code"] == level]
        reps.append(select_representatives(df_level, features, k))
    new_rows = pd.concat(reps, ignore_index=True) if reps else filtered.iloc[0:0]

    existing_slim = existing[KEEP_COLS].copy()
    new_slim = new_rows[KEEP_COLS].copy()
    out = pd.concat([existing_slim, new_slim], ignore_index=True).drop_duplicates(subset=["question"])

    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(out)} rows -> {OUT_PATH}")
    print("Severity-level coverage:")
    print(out["Severity_code"].value_counts().sort_index())


if __name__ == "__main__":
    main()
