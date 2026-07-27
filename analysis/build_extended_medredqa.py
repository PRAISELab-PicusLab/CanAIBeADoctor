"""
Build the MedRedQA subset (third dataset): 200 question/answer pairs from
/leonardo_work/CESMA_leonardo/Can_AI_Be_a_Doctor/medredqa/medredqa_train.csv,
selected with the same readability-driven clustering criterion as the
MedQuAD subset (build_extended_medquad.py): FKGL, GFI, TF-IDF lexical
representativeness, and answer length -> IQR outlier removal on FKGL/GFI ->
z-score normalize -> k-means -> representative = closest to each cluster's
centroid. No qtype-like category exists in this dataset, so clustering runs
over the whole filtered pool (no per-category stratification), i.e. the
k-means k IS the 200 target directly.

Priority filter (per explicit requirement): keep only rows where `Response`
is a genuine, substantial answer -- not a doctor's clarifying question back
to the patient, and not a throwaway one-liner.
  - min length: >= MIN_WORDS words (default 50, ~70th percentile of the raw
    40.8k-row corpus, so "long enough" without shrinking the pool too much).
  - not a follow-up question: drop rows whose Response ends with "?", or
    that contain >= MAX_QMARKS question marks anywhere (a proxy for
    "several clarifying questions instead of an answer").

Output columns are named to match the other two datasets' convention
(`question`/`question_text`/`Answer_medico`), keeping `Response Score` and
`Occupation` as extra metadata (analogous role to iCliniqQAs' `tags`).

Usage: python3 build_extended_medredqa.py [--target 200]
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
DATA_PATH = "/leonardo_work/CESMA_leonardo/Can_AI_Be_a_Doctor/medredqa/medredqa_train.csv"
OUT_PATH = os.path.join(os.path.dirname(HERE), "df_top200_MedRedQA.csv")

MIN_WORDS = 50
MAX_QMARKS = 2


def iqr_filter(df, cols):
    mask = pd.Series(True, index=df.index)
    for c in cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask &= df[c].between(lo, hi)
    return df[mask]


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
    parser.add_argument("--target", type=int, default=200)
    args = parser.parse_args()

    print(f"Loading {DATA_PATH} ...")
    full = pd.read_csv(DATA_PATH)
    full = full.rename(columns={"Title": "question", "Body": "question_text", "Response": "Answer_medico"})
    full["question"] = full["question"].astype(str).str.strip()
    full = full.dropna(subset=["question", "Answer_medico"])
    full = full.drop_duplicates(subset=["question", "Answer_medico"])
    print("Full corpus:", full.shape)

    resp = full["Answer_medico"].astype(str)
    word_count = resp.str.split().str.len()
    ends_with_q = resp.str.strip().str.endswith("?")
    qmark_count = resp.str.count(r"\?")

    keep = (word_count >= MIN_WORDS) & (~ends_with_q) & (qmark_count < MAX_QMARKS)
    full = full[keep].copy()
    print(f"After long-genuine-answer filter (>= {MIN_WORDS} words, not a follow-up question): {full.shape}")

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

    features = ["_fk", "_gf", "_lexrep", "_len"]
    reps = select_representatives(filtered, features, args.target)

    keep_cols = ["question", "question_text", "Answer_medico", "Response Score", "Occupation"]
    out = reps[keep_cols].copy().drop_duplicates(subset=["question"])

    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(out)} rows -> {OUT_PATH}")
    print("Word-count stats of selected answers:")
    print(out["Answer_medico"].apply(lambda t: len(str(t).split())).describe())


if __name__ == "__main__":
    main()
