"""
Shared plotting/analysis helpers for the "Can AI Be a Doctor" insight figures.

Ported and de-duplicated from Generazione_insight.ipynb: the notebook had the same
heatmap/violin/bar plot logic copy-pasted (with small drift) across several cells,
run repeatedly while iterating. This module keeps one version of each plot type,
parameterized by a per-dataset column mapping (see config_*.py), so both datasets
share the exact same figure code and only differ in which columns/models feed it.

All style knobs (fonts, sizes, colors) live in STYLE / FAMILY_COLORS below --
edit here once instead of hunting through per-plot rcParams blocks.
"""

import re
import ast
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import ttest_rel, chi2_contingency
from statsmodels.stats.multitest import multipletests

# ============================================================
# STYLE (edit these to change fonts/sizes across every figure)
# ============================================================

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "axes.titlesize": 20,
}

HEATMAP_ANNOT_SIZE = 22
HEATMAP_FIGSIZE = (20, 15)
HEATMAP_DPI = 400

# Base color per "model family" -- looked up by substring match against each
# model's display label, so it works for either dataset's label set.
FAMILY_COLORS = {
    "Physicians": "#1f77b4",
    "Mixtral":    "#ff7f0e",
    "MedPaLM":    "#2ca02c",
    "Med-PaLM":   "#2ca02c",
    "PALMYRA":    "#2ca02c",
    "GPT":        "#c5b0d5",
    "Gemini":     "#9edae5",
    "Claude":     "#bcbd22",
}
CONDITION_SHADE = {
    "Base": 0.0,
    "Empathy": 0.35,
    "Rephrase": -0.35,
}

SENTIMENT_ORDER = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]


def apply_style():
    plt.rcParams.update(STYLE)
    mpl.rcParams["axes.unicode_minus"] = False


def stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _shade(hex_color, amount):
    """amount>0 lightens, amount<0 darkens (simple linear blend toward white/black)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    if amount >= 0:
        r, g, b = (int(c + (255 - c) * amount) for c in (r, g, b))
    else:
        r, g, b = (int(c * (1 + amount)) for c in (r, g, b))
    return f"#{max(0,min(255,r)):02x}{max(0,min(255,g)):02x}{max(0,min(255,b)):02x}"


def build_palette(order, family_colors=FAMILY_COLORS, condition_shade=CONDITION_SHADE):
    """Assign each label in `order` a color based on the model family + condition
    substrings it contains, so every dataset's labels get a consistent look
    without hand-listing every model/condition combination."""
    palette = {}
    for label in order:
        base = "#7f7f7f"
        for fam, color in family_colors.items():
            if fam in label:
                base = color
                break
        amount = 0.0
        for cond, shade in condition_shade.items():
            if cond in label:
                amount = shade
                break
        palette[label] = _shade(base, amount)
    return palette


def parse_emotion_dict(raw):
    """Parse a stringified emotion-score dict like
    "{'neutral': np.float64(0.81), 'approval': np.float64(0.09), ...}"
    (as stored precomputed in the MedQuAD CSV) into a real dict, and return the
    dominant (argmax) label. Returns None for missing/unparseable values."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        cleaned = re.sub(r"np\.float64\(([^)]+)\)", r"\1", raw)
        d = ast.literal_eval(cleaned)
        if not d:
            return None
        return max(d, key=d.get)
    except (ValueError, SyntaxError):
        return None


# ============================================================
# 1. PAIRWISE NUMERIC HEATMAP (t-test + FDR), e.g. similarity or readability
# ============================================================

def pairwise_ttest_heatmap(df, col_map, order, filename, title="",
                            cmap_colors=("#B2182B", "#FFFFF0", "#4CAF50"),
                            cbar_label="Mean Difference", center=0.0,
                            figsize=HEATMAP_FIGSIZE, dpi=HEATMAP_DPI):
    """col_map: {label -> column name}. order: labels to include, in display order.
    Draws the lower-triangle heatmap of pairwise mean differences with FDR-corrected
    significance stars, mirroring notebook cells 23/25."""
    labels = [l for l in order if l in col_map and col_map[l] in df.columns]
    for l in labels:
        df[col_map[l]] = pd.to_numeric(df[col_map[l]], errors="coerce")

    results = []
    for m1, m2 in combinations(labels, 2):
        c1, c2 = col_map[m1], col_map[m2]
        sub = df[[c1, c2]].dropna()
        if len(sub) < 2:
            continue
        t, p = ttest_rel(sub[c1], sub[c2])
        results.append({"m1": m1, "m2": m2, "p": p, "diff": sub[c1].mean() - sub[c2].mean()})

    if not results:
        print(f"[SKIP] {filename}: no valid pairwise comparisons")
        return

    res = pd.DataFrame(results)
    rej, p_corr, _, _ = multipletests(res["p"], method="fdr_bh")
    res["p_fdr"] = p_corr

    diff = pd.DataFrame(0.0, index=labels, columns=labels)
    pmat = pd.DataFrame(1.0, index=labels, columns=labels)
    for _, r in res.iterrows():
        diff.loc[r["m1"], r["m2"]] = r["diff"]
        diff.loc[r["m2"], r["m1"]] = -r["diff"]
        pmat.loc[r["m1"], r["m2"]] = r["p_fdr"]
        pmat.loc[r["m2"], r["m1"]] = r["p_fdr"]

    annot = diff.copy().astype(str)
    for i in labels:
        for j in labels:
            annot.loc[i, j] = "" if i == j else f"{diff.loc[i,j]:+.2f}\n{stars(pmat.loc[i,j])}"

    cmap = LinearSegmentedColormap.from_list("diverging", list(cmap_colors))
    mask = np.triu(np.ones_like(diff, dtype=bool), k=1)

    plt.figure(figsize=figsize, dpi=dpi)
    sns.heatmap(
        diff, mask=mask, annot=annot, fmt="", cmap=cmap, center=center,
        linewidths=0.6, linecolor="white",
        cbar_kws={"shrink": 0.7, "label": cbar_label},
        annot_kws={"size": HEATMAP_ANNOT_SIZE, "weight": "bold"},
    )
    if title:
        plt.title(title, fontweight="bold")
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {filename}")


# ============================================================
# 2. SENTIMENT AGREEMENT HEATMAP (chi2 + Cramer's V), notebook cell 24
# ============================================================

def sentiment_agreement_heatmap(df, col_map, order, filename,
                                 sentiment_order=SENTIMENT_ORDER,
                                 highlight_label=None,
                                 cmap_colors=("#FEE0D0", "#FEE0D2", "#FC9272", "#FB6A4A", "#CB181D"),
                                 figsize=HEATMAP_FIGSIZE, dpi=HEATMAP_DPI):
    labels = [l for l in order if l in col_map and col_map[l] in df.columns and df[col_map[l]].notna().any()]

    def counts(series):
        return series.value_counts().reindex(sentiment_order, fill_value=0)

    results = []
    for m1, m2 in combinations(labels, 2):
        c1, c2 = col_map[m1], col_map[m2]
        sub = df[[c1, c2]].dropna()
        if len(sub) == 0:
            continue
        vals = np.vstack([counts(sub[c1]).values, counts(sub[c2]).values])
        vals = vals[:, vals.sum(axis=0) > 0]
        if vals.shape[1] < 2:
            continue
        chi2, p, dof, _ = chi2_contingency(vals)
        v = np.sqrt(chi2 / (len(sub) * dof)) if dof > 0 else np.nan
        results.append({"m1": m1, "m2": m2, "p": p, "v": v})

    if not results:
        print(f"[SKIP] {filename}: no valid pairwise comparisons")
        return

    res = pd.DataFrame(results)
    rej, p_corr, _, _ = multipletests(res["p"], method="fdr_bh")
    res["p_fdr"] = p_corr

    V = pd.DataFrame(0.0, index=labels, columns=labels)
    P = pd.DataFrame(1.0, index=labels, columns=labels)
    for _, r in res.iterrows():
        V.loc[r["m1"], r["m2"]] = V.loc[r["m2"], r["m1"]] = r["v"]
        P.loc[r["m1"], r["m2"]] = P.loc[r["m2"], r["m1"]] = r["p_fdr"]

    annot = V.copy().astype(str)
    for i in labels:
        for j in labels:
            annot.loc[i, j] = "" if i == j else f"{V.loc[i,j]:.2f}\n{stars(P.loc[i,j])}"

    cmap = LinearSegmentedColormap.from_list("agreement", list(cmap_colors))
    mask = np.triu(np.ones_like(V, dtype=bool), k=1)

    plt.figure(figsize=figsize, dpi=dpi)
    ax = sns.heatmap(
        V, mask=mask, annot=annot, fmt="", cmap=cmap, vmin=0, vmax=max(V.values.max(), 1e-9),
        linewidths=0.6, linecolor="white",
        cbar_kws={"shrink": 0.7, "label": "Cramer's V (Effect Size)"},
        annot_kws={"size": HEATMAP_ANNOT_SIZE, "weight": "bold"},
    )
    if highlight_label and highlight_label in labels:
        for i, mi in enumerate(labels):
            for j, mj in enumerate(labels):
                if mask[i, j] or mi == mj:
                    continue
                if highlight_label in (mi, mj):
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color="#BFDBFE", alpha=0.35, lw=0))
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {filename}")


# ============================================================
# 3. VIOLIN PLOT -- normalized similarity-to-doctor across models, cell 27
# ============================================================

def similarity_violin_plot(df, sim_col_map, order, filename, palette=None,
                            ylabel="Cosine Similarity (normalized)",
                            title="Cosine Similarity Across Models",
                            figsize=(18, 8), dpi=300):
    available = {l: c for l, c in sim_col_map.items() if l in order and c in df.columns}
    order = [l for l in order if l in available]
    if not order:
        print(f"[SKIP] {filename}: no similarity columns found")
        return
    palette = palette or build_palette(order)

    melted = df[list(available.values())].rename(columns={v: k for k, v in available.items()})
    melted = melted.melt(var_name="Model", value_name="Similarity").dropna()
    melted["Similarity"] = pd.to_numeric(melted["Similarity"], errors="coerce")
    melted = melted.dropna()

    lo, hi = melted["Similarity"].min(), melted["Similarity"].max()
    melted["Similarity"] = 0.5 if lo == hi else (melted["Similarity"] - lo) / (hi - lo)

    plt.figure(figsize=figsize, dpi=dpi)
    sns.set_style("whitegrid")
    ax = sns.violinplot(x="Model", y="Similarity", data=melted, order=order, palette=palette, inner=None, linewidth=1.2)

    # The KDE tails overshoot past the data's true [0,1] range (a normal violinplot
    # artifact). Rather than truncating the estimate (which flattens the tips), rescale
    # every already-drawn violin shape with one single linear transform so the whole
    # plot -- tips included, shape untouched -- fits exactly inside [0,1].
    violin_paths = [path for coll in ax.collections for path in coll.get_paths()]
    all_y = np.concatenate([path.vertices[:, 1] for path in violin_paths])
    y_min, y_max = all_y.min(), all_y.max()
    span = y_max - y_min

    def rescale(y):
        return (np.asarray(y) - y_min) / span if span > 0 else np.asarray(y)

    for path in violin_paths:
        path.vertices[:, 1] = rescale(path.vertices[:, 1])

    stats = melted.groupby("Model")["Similarity"].agg(["mean", "std"]).reindex(order)
    mean_r = rescale(stats["mean"].values)
    lower_r = rescale((stats["mean"] - stats["std"]).values)
    upper_r = rescale((stats["mean"] + stats["std"]).values)
    plt.errorbar(x=np.arange(len(order)), y=mean_r, yerr=[mean_r - lower_r, upper_r - mean_r], fmt="o",
                 color="black", capsize=4, markersize=5, label="Mean +/- Std")

    plt.xticks(rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.title(title)
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {filename}")


# ============================================================
# 4. GROUPED BAR -- readability mean+-std per model, cell 28
# ============================================================

def readability_barplot(df, metric_col_map, order, filename, ylabel="Readability Score",
                         title="", palette=None, figsize=(24, 9), dpi=300):
    labels = [l for l in order if l in metric_col_map and metric_col_map[l] in df.columns]
    if not labels:
        print(f"[SKIP] {filename}: no readability columns found")
        return
    palette = palette or build_palette(labels)

    means, stds = [], []
    for l in labels:
        col = pd.to_numeric(df[metric_col_map[l]], errors="coerce")
        means.append(col.mean())
        stds.append(col.std())

    x = np.arange(len(labels))
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    for i in range(len(labels)):
        ax.axvspan(i - 0.5, i + 0.5, color="#f7f7f7" if i % 2 == 0 else "#ececec", alpha=0.5, zorder=0)
    ax.bar(x, means, yerr=stds, capsize=5, color=[palette[l] for l in labels], edgecolor="black", zorder=1)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {filename}")


# ============================================================
# 5. SENTIMENT DISTRIBUTION BARPLOT, cell 29
# ============================================================

def sentiment_distribution_barplot(df, col_map, order, filename, palette=None,
                                    sentiment_order=SENTIMENT_ORDER, figsize=(22, 10), dpi=300):
    labels = [l for l in order if l in col_map and col_map[l] in df.columns]
    if not labels:
        print(f"[SKIP] {filename}: no sentiment columns found")
        return
    palette = palette or build_palette(labels)

    contingency = pd.DataFrame({
        l: df[col_map[l]].value_counts().reindex(sentiment_order, fill_value=0) for l in labels
    })
    contingency_pct = contingency.div(contingency.sum(axis=0), axis=1) * 100

    melted = contingency_pct.reset_index().melt(id_vars="index", var_name="Model", value_name="Percentage")
    melted.rename(columns={"index": "Sentiment"}, inplace=True)
    melted["Model"] = pd.Categorical(melted["Model"], categories=labels, ordered=True)

    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    for i in range(len(sentiment_order)):
        ax.axvspan(i - 0.5, i + 0.5, color="#f6f6f6" if i % 2 == 0 else "#ffffff", alpha=0.6, zorder=0)

    sns.barplot(data=melted, x="Sentiment", y="Percentage", hue="Model", order=sentiment_order,
                hue_order=labels, palette=palette, edgecolor="black", linewidth=1.0)

    plt.ylabel("Percentage (%)")
    plt.xlabel("")
    plt.title("Sentiment Distribution Across Models")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend(title="Models", loc="upper right", frameon=True, edgecolor="black", ncol=2)
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {filename}")
    return contingency_pct.round(2)


# ============================================================
# 6. TOP-5 DOMINANT EMOTIONS BARPLOT, cells 31/33/34
# ============================================================

def top5_emotion_barplot(df, col_map, order, filename, palette=None,
                          excluded_emotions=("neutral",), figsize=(18, 10), dpi=300):
    """col_map: {label -> column name holding a per-row dominant-emotion string}."""
    labels = [l for l in order if l in col_map and col_map[l] in df.columns]
    if not labels:
        print(f"[SKIP] {filename}: no emotion columns found")
        return
    palette = palette or build_palette(labels)

    records = []
    for l in labels:
        for emo in df[col_map[l]].dropna():
            records.append({"Emotion": emo, "Model": l})
    long_df = pd.DataFrame(records)
    if long_df.empty:
        print(f"[SKIP] {filename}: no emotion values present")
        return

    counts = long_df.value_counts(["Model", "Emotion"]).reset_index(name="Count")
    totals = long_df.groupby("Model").size().reset_index(name="Total")
    pct = counts.merge(totals, on="Model")
    pct["Percent"] = 100 * pct["Count"] / pct["Total"]

    top5 = (
        pct[~pct["Emotion"].isin(excluded_emotions)]
        .groupby("Emotion")["Percent"].sum()
        .sort_values(ascending=False).head(5).index
    )
    plot_df = pct[pct["Emotion"].isin(top5)].copy()
    plot_df["Model"] = pd.Categorical(plot_df["Model"], categories=labels, ordered=True)
    plot_df["Emotion"] = pd.Categorical(plot_df["Emotion"], categories=top5, ordered=True)
    plot_df = plot_df.sort_values(["Emotion", "Model"])

    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    for i in range(len(top5)):
        ax.axvspan(i - 0.5, i + 0.5, color="#f7f7f7" if i % 2 == 0 else "#ececec", alpha=0.55, zorder=0)

    sns.barplot(data=plot_df, x="Emotion", y="Percent", hue="Model", order=list(top5),
                hue_order=labels, palette=palette, edgecolor="black")

    plt.title("Top 5 Dominant Emotions Across Models")
    plt.xlabel("Emotion")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Model", loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {filename}")


# ============================================================
# 7. SEVERITY CLUSTERING SCATTER, cell 20 (Iclinicqa only: needs Severity_code)
# ============================================================

def severity_cluster_scatter(df, fk_col, gf_col, severity_col, filename,
                              n_clusters=10, severity_colors=None, severity_names=None,
                              figsize=(8, 6), dpi=300):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    severity_colors = severity_colors or {1: "#E6E6E6", 2: "#2ca02c", 3: "#f1c40f", 4: "#ff7f0e", 5: "#d62728"}
    severity_names = severity_names or {1: "White", 2: "Green", 3: "Yellow", 4: "Orange", 5: "Red"}

    d = df.dropna(subset=[fk_col, gf_col, severity_col]).copy()
    d[severity_col] = d[severity_col].astype(int)

    scaler = StandardScaler()
    reps = []
    for sev in sorted(d[severity_col].unique()):
        d_sev = d[d[severity_col] == sev].copy()
        if len(d_sev) < n_clusters:
            print(f"  severity {sev}: only {len(d_sev)} rows, skipping cluster-rep selection")
            continue
        X = scaler.fit_transform(d_sev[[fk_col, gf_col]].values)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        clusters = km.fit_predict(X)
        d_sev["cluster"] = clusters
        dist = km.transform(X)
        d_sev["cluster_distance"] = [dist[i, clusters[i]] for i in range(len(clusters))]
        reps.append(d_sev.sort_values("cluster_distance").groupby("cluster").head(1))

    final = pd.concat(reps).reset_index(drop=True) if reps else pd.DataFrame(columns=d.columns)

    plt.figure(figsize=figsize, dpi=dpi)
    for sev in sorted(severity_colors):
        subset = d[d[severity_col] == sev]
        plt.scatter(subset[fk_col], subset[gf_col], alpha=0.15, s=25, color=severity_colors[sev])
    for _, row in final.iterrows():
        plt.scatter(row[fk_col], row[gf_col], s=110, color=severity_colors[row[severity_col]],
                    edgecolor="black", linewidth=1.2, zorder=3)

    plt.xlabel("Flesch-Kincaid")
    plt.ylabel("Gunning-Fog")
    legend_elements = [
        Line2D([0], [0], marker="o", linestyle="", label=f"Severity {sev} ({severity_names[sev]})",
               markerfacecolor=severity_colors[sev], markeredgecolor="black", markersize=12)
        for sev in sorted(severity_colors)
    ]
    plt.legend(handles=legend_elements, frameon=True, framealpha=1, edgecolor="black")
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {filename}")
