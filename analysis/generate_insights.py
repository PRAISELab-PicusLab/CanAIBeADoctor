"""
Generates every figure for one dataset ("medquad" or "iclinicqa") from its
already-analyzed CSV (similarity/readability/sentiment precomputed upstream;
emotion precomputed for medquad, computed separately for iclinicqa via
compute_emotions_iclinicqa.py). Pure plotting/aggregation, no model inference,
safe to run on the login node.

Usage:
    python3 generate_insights.py --dataset medquad
    python3 generate_insights.py --dataset iclinicqa
"""

import argparse
import copy
import os

import pandas as pd

import common


def load_config(dataset):
    if dataset == "medquad":
        import config_medquad as cfg
    elif dataset == "iclinicqa":
        import config_iclinicqa as cfg
    elif dataset == "medquad_ext":
        import config_medquad_ext as cfg
    elif dataset == "medredqa_ext":
        import config_medredqa_ext as cfg
    elif dataset == "ablation_medquad":
        import config_ablation_medquad as cfg
    elif dataset == "ablation_medredqa":
        import config_ablation_medredqa as cfg
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    return cfg


def resolve_emotion_columns(df, columns):
    """For datasets storing emotion as a raw score-dict string (medquad), derive
    a per-row dominant-emotion column once and point columns[label]['emotion'] at it."""
    columns = copy.deepcopy(columns)
    for label, cols in columns.items():
        if cols.get("emotion"):
            continue
        raw_col = cols.get("emotion_raw")
        if raw_col and raw_col in df.columns:
            derived_col = f"{raw_col}__dominant"
            df[derived_col] = df[raw_col].apply(common.parse_emotion_dict)
            cols["emotion"] = derived_col
    return columns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=[
        "medquad", "iclinicqa", "medquad_ext", "medredqa_ext", "ablation_medquad", "ablation_medredqa",
    ])
    args = parser.parse_args()

    cfg = load_config(args.dataset)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    common.apply_style()

    if not os.path.exists(cfg.DATA_PATH):
        raise FileNotFoundError(
            f"{cfg.DATA_PATH} not found. "
            "For iclinicqa, run compute_emotions_iclinicqa.py on a compute node first."
        )

    df = pd.read_csv(cfg.DATA_PATH)
    print(f"Loaded {cfg.DATA_PATH}: {df.shape}")

    columns = resolve_emotion_columns(df, cfg.MODEL_COLUMNS)
    order = cfg.MODEL_ORDER
    ref = cfg.REFERENCE_LABEL
    palette = common.build_palette(order)
    out = lambda name: os.path.join(cfg.OUTPUT_DIR, name)

    # 1. Similarity-to-doctor heatmap (models with a sim column only, no reference)
    sim_map = {l: c["sim"] for l, c in columns.items() if c.get("sim")}
    common.pairwise_ttest_heatmap(
        df, sim_map, [l for l in order if l in sim_map],
        out("heatmap_similarity.pdf"),
        title="Pairwise Model Comparison (Semantic Fidelity)",
        cmap_colors=("#FFFFFF", "#D6E9FC", "#A9D0FA", "#7BB7F7", "#4F9EF4"),
        cbar_label="Mean Difference (Semantic Fidelity)", center=None,
    )

    # 2. Sentiment agreement heatmap (Cramer's V), includes reference
    sentiment_map = {l: c["sentiment"] for l, c in columns.items() if c.get("sentiment")}
    common.sentiment_agreement_heatmap(
        df, sentiment_map, order, out("heatmap_sentiment_agreement.pdf"), highlight_label=ref,
    )

    # 3. Readability pairwise-diff heatmaps (FK, Gunning Fog), includes reference
    fk_map = {l: c["fk"] for l, c in columns.items() if c.get("fk")}
    gf_map = {l: c["gf"] for l, c in columns.items() if c.get("gf")}
    common.pairwise_ttest_heatmap(
        df, fk_map, order, out("heatmap_flesch_kincaid_diff.pdf"),
        title="Pairwise Model Comparison (Flesch-Kincaid)",
    )
    common.pairwise_ttest_heatmap(
        df, gf_map, order, out("heatmap_gunning_fog_diff.pdf"),
        title="Pairwise Model Comparison (Gunning-Fog)",
    )

    # 4. Similarity violin plot
    common.similarity_violin_plot(
        df, sim_map, [l for l in order if l in sim_map],
        out("violin_similarity.pdf"), palette=palette,
    )

    # 5. Readability bar plots (FK, Gunning Fog), includes reference
    common.readability_barplot(
        df, fk_map, order, out("bar_flesch_kincaid.pdf"),
        title="Flesch-Kincaid Grade Level", palette=palette,
    )
    common.readability_barplot(
        df, gf_map, order, out("bar_gunning_fog.pdf"),
        title="Gunning Fog Index", palette=palette,
    )

    # 6. Sentiment distribution bar plot, includes reference
    table = common.sentiment_distribution_barplot(
        df, sentiment_map, order, out("bar_sentiment_distribution.pdf"), palette=palette,
    )
    if table is not None:
        print("\nSentiment distribution (%) per model:")
        print(table)

    # 7. Top-5 dominant emotions bar plot, includes reference
    emotion_map = {l: c["emotion"] for l, c in columns.items() if c.get("emotion")}
    common.top5_emotion_barplot(
        df, emotion_map, order, out("bar_top5_emotions.pdf"), palette=palette,
    )

    # 8. Severity clustering scatter (iclinicqa only)
    if getattr(cfg, "SEVERITY_COL", None) and cfg.SEVERITY_COL in df.columns:
        common.severity_cluster_scatter(
            df, columns[ref]["fk"], columns[ref]["gf"], cfg.SEVERITY_COL,
            out("scatter_severity_clustering.pdf"),
        )

    print(f"\nAll figures saved to {cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
