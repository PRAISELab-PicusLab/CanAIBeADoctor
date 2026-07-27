"""
Statistical comparison of the 3 ablation prompt versions -- Main (v2, the
production prompt used in the main run) vs Variant B vs Variant C -- scoped
to each (model, condition) pair individually (not mixed across models/
conditions like the full ablation figures in insights_ablation/). Answers:
"is the initial (Main) prompt statistically different from the other two?"

For each dataset (medquad, medredqa) x model (mixtral, medgemma) x condition
(base, empathy, rephrase) -- 12 combos total -- runs:
  - paired t-test (Main vs VarB, Main vs VarC, VarB vs VarC), FDR-corrected,
    on similarity / Flesch-Kincaid / Gunning-Fog (reuses the exact
    common.pairwise_ttest_heatmap logic already used for model comparisons)
  - chi-square / Cramer's V on sentiment distribution
  - the same 8-figure battery as generate_insights.py, but with only the 3
    prompt versions as "models" being compared
  - a CSV dump of the raw pairwise test statistics (t/chi2, p, p_fdr, effect)
    for both continuous and categorical axes

Also builds one "combined" figure set per dataset with all 2 models x 3
conditions x 3 versions (+ Physicians reference) together in a single plot
per analysis type -- same layout as the original insights_ablation/ figures,
but colored by prompt VERSION (not model/condition) so Main Prompt vs
VariantA vs VariantB pops out visually across every group at a glance.

Output: insights/ablation_prompt/<dataset>/<model>_<condition>/*.pdf +
        insights/ablation_prompt/<dataset>/<model>_<condition>/stats.csv
        insights/ablation_prompt/<dataset>/combined/*.pdf (all arms, one plot per analysis)
        insights/ablation_prompt/prompt_version_stats_summary.csv (all combos)

Usage: python3 compare_prompt_versions.py
"""
import os

import pandas as pd
from scipy.stats import ttest_rel, chi2_contingency
from statsmodels.stats.multitest import multipletests
from itertools import combinations

import common

HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(HERE)
OUT_ROOT = os.path.join(HERE, "ablation_prompt")

DATASETS = {
    "medquad": os.path.join(PROJECT_ROOT, "df_ablation_MedQuAD.csv"),
    "medredqa": os.path.join(PROJECT_ROOT, "df_ablation_MedRedQA.csv"),
}

MODELS = ["mixtral", "medgemma"]
MODEL_LABELS = {"mixtral": "Mixtral", "medgemma": "MedGemma"}
CONDITIONS = ["base", "empathy", "rephrase"]
CONDITION_LABELS = {"base": "Base", "empathy": "Empathy", "rephrase": "Rephrase"}
VERSIONS = ["Main Prompt", "VariantA", "VariantB"]
VERSION_SUFFIX = {"Main Prompt": "", "VariantA": "_variantB", "VariantB": "_variantC"}
SENTIMENT_ORDER = common.SENTIMENT_ORDER

# common.build_palette() only knows model-family/condition substrings, so
# "Main Prompt"/"VariantA"/"VariantB" all fall back to its gray default --
# use an explicit, distinct color per prompt version instead.
VERSION_PALETTE = {
    "Main Prompt": "#4C72B0",
    "VariantA": "#DD8452",
    "VariantB": "#55A868",
}


def continuous_pairwise_stats(df, col_map, order, metric_name):
    """Same paired-t-test + FDR logic as common.pairwise_ttest_heatmap, but
    returns the raw results table instead of only plotting it."""
    labels = [l for l in order if l in col_map and col_map[l] in df.columns]
    rows = []
    for m1, m2 in combinations(labels, 2):
        c1, c2 = col_map[m1], col_map[m2]
        sub = df[[c1, c2]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < 2:
            continue
        t, p = ttest_rel(sub[c1], sub[c2])
        rows.append({
            "metric": metric_name, "version_1": m1, "version_2": m2,
            "n": len(sub), "mean_1": sub[c1].mean(), "mean_2": sub[c2].mean(),
            "mean_diff": sub[c1].mean() - sub[c2].mean(), "t_stat": t, "p_raw": p,
        })
    if not rows:
        return pd.DataFrame()
    res = pd.DataFrame(rows)
    _, p_fdr, _, _ = multipletests(res["p_raw"], method="fdr_bh")
    res["p_fdr"] = p_fdr
    res["significant_fdr_0.05"] = res["p_fdr"] < 0.05
    return res


def sentiment_pairwise_stats(df, col_map, order):
    labels = [l for l in order if l in col_map and col_map[l] in df.columns and df[col_map[l]].notna().any()]
    rows = []

    def counts(series):
        return series.value_counts().reindex(SENTIMENT_ORDER, fill_value=0)

    for m1, m2 in combinations(labels, 2):
        c1, c2 = col_map[m1], col_map[m2]
        sub = df[[c1, c2]].dropna()
        if len(sub) == 0:
            continue
        vals = pd.concat([counts(sub[c1]), counts(sub[c2])], axis=1).values.T
        vals = vals[:, vals.sum(axis=0) > 0]
        if vals.shape[1] < 2:
            continue
        chi2, p, dof, _ = chi2_contingency(vals)
        v = (chi2 / (len(sub) * dof)) ** 0.5 if dof > 0 else float("nan")
        rows.append({
            "metric": "sentiment", "version_1": m1, "version_2": m2,
            "n": len(sub), "chi2": chi2, "cramers_v": v, "p_raw": p,
        })
    if not rows:
        return pd.DataFrame()
    res = pd.DataFrame(rows)
    _, p_fdr, _, _ = multipletests(res["p_raw"], method="fdr_bh")
    res["p_fdr"] = p_fdr
    res["significant_fdr_0.05"] = res["p_fdr"] < 0.05
    return res


REFERENCE_LABEL = "Physicians"
REFERENCE_TEXT_COL = "Answer_medico"
REFERENCE_COLOR = "#333333"


def combined_label(model, condition, version):
    return f"{MODEL_LABELS[model]}\n{CONDITION_LABELS[condition]} ({version})"


def build_combined_columns():
    """All 18 model x condition x version arms + Physicians reference, for the
    single all-in-one-plot view (mirrors config_ablation_medquad.py's 18-arm
    layout, but colored by prompt VERSION instead of model/condition so Main
    Prompt vs VariantA vs VariantB pops out visually across every group)."""
    columns = {
        REFERENCE_LABEL: {
            "text": REFERENCE_TEXT_COL, "fk": f"{REFERENCE_TEXT_COL}_FleschKincaid",
            "gf": f"{REFERENCE_TEXT_COL}_GunningFog", "sentiment": f"{REFERENCE_TEXT_COL}_sentiment",
            "sim": None, "emotion": f"{REFERENCE_TEXT_COL}_emotion",
        },
    }
    for m in MODELS:
        for c in CONDITIONS:
            for v in VERSIONS:
                col = f"{m}_{c}{VERSION_SUFFIX[v]}"
                label = combined_label(m, c, v)
                columns[label] = {
                    "text": col, "fk": f"{col}_FleschKincaid", "gf": f"{col}_GunningFog",
                    "sentiment": f"{col}_sentiment", "sim": f"sim_medico_{col}", "emotion": f"{col}_emotion",
                }
    order = [REFERENCE_LABEL] + [
        combined_label(m, c, v) for m in MODELS for c in CONDITIONS for v in VERSIONS
    ]
    return columns, order


def build_combined_palette(order):
    palette = {}
    for label in order:
        if label == REFERENCE_LABEL:
            palette[label] = REFERENCE_COLOR
        else:
            for v, color in VERSION_PALETTE.items():
                if f"({v})" in label:
                    palette[label] = color
                    break
    return palette


def generate_combined_figures(dataset, df):
    """One figure per analysis, all 2 models x 3 conditions x 3 versions (+
    Physicians) together -- "same as the base experiments", per explicit
    user request, colored by prompt version so Main/VariantA/VariantB are
    directly comparable across every model/condition group at a glance."""
    columns, order = build_combined_columns()
    palette = build_combined_palette(order)
    combo_dir = os.path.join(OUT_ROOT, dataset, "combined")
    os.makedirs(combo_dir, exist_ok=True)
    out = lambda name: os.path.join(combo_dir, name)

    sim_map = {l: c["sim"] for l, c in columns.items() if c.get("sim")}
    fk_map = {l: c["fk"] for l, c in columns.items()}
    gf_map = {l: c["gf"] for l, c in columns.items()}
    sent_map = {l: c["sentiment"] for l, c in columns.items()}
    emo_map = {l: c["emotion"] for l, c in columns.items()}

    common.pairwise_ttest_heatmap(
        df, sim_map, [l for l in order if l in sim_map], out("heatmap_similarity.pdf"),
        title="All Models/Conditions x Prompt Version -- Similarity",
        cmap_colors=("#FFFFFF", "#D6E9FC", "#A9D0FA", "#7BB7F7", "#4F9EF4"),
        cbar_label="Mean Difference (Semantic Fidelity)", center=None,
    )
    common.pairwise_ttest_heatmap(
        df, fk_map, order, out("heatmap_flesch_kincaid_diff.pdf"),
        title="All Models/Conditions x Prompt Version -- Flesch-Kincaid",
    )
    common.pairwise_ttest_heatmap(
        df, gf_map, order, out("heatmap_gunning_fog_diff.pdf"),
        title="All Models/Conditions x Prompt Version -- Gunning-Fog",
    )
    common.sentiment_agreement_heatmap(
        df, sent_map, order, out("heatmap_sentiment_agreement.pdf"), highlight_label=REFERENCE_LABEL,
    )
    common.similarity_violin_plot(
        df, sim_map, [l for l in order if l in sim_map], out("violin_similarity.pdf"), palette=palette,
    )
    common.readability_barplot(
        df, fk_map, order, out("bar_flesch_kincaid.pdf"),
        title="Flesch-Kincaid Grade Level -- All Models/Conditions x Prompt Version", palette=palette,
    )
    common.readability_barplot(
        df, gf_map, order, out("bar_gunning_fog.pdf"),
        title="Gunning Fog Index -- All Models/Conditions x Prompt Version", palette=palette,
    )
    common.sentiment_distribution_barplot(
        df, sent_map, order, out("bar_sentiment_distribution.pdf"), palette=palette,
    )
    common.top5_emotion_barplot(
        df, emo_map, order, out("bar_top5_emotions.pdf"), palette=palette,
    )
    print(f"  [combined] figures saved under {combo_dir}")


def main():
    common.apply_style()
    os.makedirs(OUT_ROOT, exist_ok=True)
    all_stats = []

    for dataset, data_path in DATASETS.items():
        df = pd.read_csv(data_path)
        print(f"\n=== {dataset}: {df.shape} ===")

        generate_combined_figures(dataset, df)

        for model in MODELS:
            for condition in CONDITIONS:
                combo_dir = os.path.join(OUT_ROOT, dataset, f"{model}_{condition}")
                os.makedirs(combo_dir, exist_ok=True)
                out = lambda name: os.path.join(combo_dir, name)

                col_map = {}
                text_map, fk_map, gf_map, sent_map, sim_map, emo_map = {}, {}, {}, {}, {}, {}
                for v in VERSIONS:
                    col = f"{model}_{condition}{VERSION_SUFFIX[v]}"
                    text_map[v] = col
                    fk_map[v] = f"{col}_FleschKincaid"
                    gf_map[v] = f"{col}_GunningFog"
                    sent_map[v] = f"{col}_sentiment"
                    sim_map[v] = f"sim_medico_{col}"
                    emo_map[v] = f"{col}_emotion"

                label_prefix = f"{MODEL_LABELS[model]} / {CONDITION_LABELS[condition]}"
                print(f"  -- {label_prefix} --")

                # Statistical tests (raw tables)
                stats_frames = [
                    continuous_pairwise_stats(df, sim_map, VERSIONS, "similarity"),
                    continuous_pairwise_stats(df, fk_map, VERSIONS, "flesch_kincaid"),
                    continuous_pairwise_stats(df, gf_map, VERSIONS, "gunning_fog"),
                ]
                sent_stats = sentiment_pairwise_stats(df, sent_map, VERSIONS)
                combo_stats = pd.concat(stats_frames, ignore_index=True)
                if not combo_stats.empty:
                    combo_stats.insert(0, "condition", CONDITION_LABELS[condition])
                    combo_stats.insert(0, "model", MODEL_LABELS[model])
                    combo_stats.insert(0, "dataset", dataset)
                if not sent_stats.empty:
                    sent_stats.insert(0, "condition", CONDITION_LABELS[condition])
                    sent_stats.insert(0, "model", MODEL_LABELS[model])
                    sent_stats.insert(0, "dataset", dataset)
                combo_all = pd.concat([combo_stats, sent_stats], ignore_index=True)
                combo_all.to_csv(out("stats.csv"), index=False)
                all_stats.append(combo_all)
                for _, r in combo_all.iterrows():
                    sig = "***" if r.get("significant_fdr_0.05") else "ns"
                    print(f"    [{r['metric']:>15}] {r['version_1']} vs {r['version_2']}: "
                          f"p_fdr={r['p_fdr']:.4f} ({sig})")

                # Figures (same battery as generate_insights.py, scoped to 3 versions)
                palette = VERSION_PALETTE
                common.pairwise_ttest_heatmap(
                    df, sim_map, VERSIONS, out("heatmap_similarity.pdf"),
                    title=f"{label_prefix} -- Similarity by Prompt Version",
                    cmap_colors=("#FFFFFF", "#D6E9FC", "#A9D0FA", "#7BB7F7", "#4F9EF4"),
                    cbar_label="Mean Difference (Semantic Fidelity)", center=None,
                )
                common.pairwise_ttest_heatmap(
                    df, fk_map, VERSIONS, out("heatmap_flesch_kincaid_diff.pdf"),
                    title=f"{label_prefix} -- Flesch-Kincaid by Prompt Version",
                )
                common.pairwise_ttest_heatmap(
                    df, gf_map, VERSIONS, out("heatmap_gunning_fog_diff.pdf"),
                    title=f"{label_prefix} -- Gunning-Fog by Prompt Version",
                )
                common.sentiment_agreement_heatmap(
                    df, sent_map, VERSIONS, out("heatmap_sentiment_agreement.pdf"),
                )
                common.similarity_violin_plot(
                    df, sim_map, VERSIONS, out("violin_similarity.pdf"), palette=palette,
                )
                common.readability_barplot(
                    df, fk_map, VERSIONS, out("bar_flesch_kincaid.pdf"),
                    title=f"{label_prefix} -- Flesch-Kincaid Grade Level", palette=palette,
                )
                common.readability_barplot(
                    df, gf_map, VERSIONS, out("bar_gunning_fog.pdf"),
                    title=f"{label_prefix} -- Gunning Fog Index", palette=palette,
                )
                common.sentiment_distribution_barplot(
                    df, sent_map, VERSIONS, out("bar_sentiment_distribution.pdf"), palette=palette,
                )
                common.top5_emotion_barplot(
                    df, emo_map, VERSIONS, out("bar_top5_emotions.pdf"), palette=palette,
                )

    summary = pd.concat(all_stats, ignore_index=True)
    summary_path = os.path.join(OUT_ROOT, "prompt_version_stats_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nAll figures + per-combo stats.csv saved under {OUT_ROOT}/<dataset>/<model>_<condition>/")
    print(f"Combined stats summary -> {summary_path}")

    n_sig = summary["significant_fdr_0.05"].sum()
    print(f"\n{n_sig}/{len(summary)} pairwise comparisons significant at FDR<0.05")


if __name__ == "__main__":
    main()
