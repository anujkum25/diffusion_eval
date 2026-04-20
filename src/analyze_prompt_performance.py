"""
Research-grade analysis for prompt performance across image-generation models.

This script is intentionally plain Python so you can edit the analysis in VS Code.
It reads the manual evaluation CSV, infers metric families from column names, and
exports tables plus publication-ready PNG/PDF visualizations.

Example:
    python src/analyze_prompt_performance.py ^
      --csv "C:/Users/anujq/OneDrive/Documents/Amrita/curated_dataset_diffusion/diffusion_evaluation/manual_eval_with_com_columns.csv" ^
      --out outputs/research_grade_analysis
"""

from __future__ import annotations

import argparse
import itertools
import re
import textwrap
from collections import Counter
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
except ImportError as exc:
    missing = str(exc).split("No module named ")[-1].strip("'")
    raise SystemExit(
        "\nMissing dependency: "
        f"{missing}\n\n"
        "Install dependencies in your VS Code environment:\n"
        "  python -m pip install -r requirements.txt\n"
    ) from exc


DEFAULT_CSV = (
    "C:/Users/anujq/OneDrive/Documents/Amrita/curated_dataset_diffusion/"
    "diffusion_evaluation/manual_eval_with_com_columns.csv"
)

DEFAULT_OUT = "outputs/research_grade_analysis"
RANDOM_SEED = 42


MODEL_RENAMES = {
    "Stable Diffusion 3.5 large": "Stable Diffusion 3.5 Large",
    "SD_api_stable-diffusion-xl-1024-v1-0": "sd_api_stable-diffusion-xl-1024-v1-0",
}

STOPWORDS = {
    "the", "and", "with", "not", "are", "but", "for", "has", "have", "that",
    "this", "looks", "look", "like", "shown", "image", "there", "properly",
    "does", "missing", "acceptable", "good", "bad", "very", "is", "in",
    "of", "to", "a", "an", "it", "as", "on", "at", "from", "by", "or",
    "be", "was", "were", "no", "yes",
}


def configure_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="DejaVu Sans",
        rc={
            "figure.dpi": 150,
            "savefig.dpi": 320,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )


def save_figure(fig: plt.Figure, path_without_ext: Path) -> None:
    path_without_ext.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path_without_ext.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_without_ext.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def clean_metric_name(col: str) -> str:
    name = re.sub(r"^r\d+_(?:d\d+|com)_", "", col)
    return name.replace("_", " ").replace("mutli", "multi").title()


def infer_columns(df: pd.DataFrame) -> dict[str, str | None]:
    lower_to_original = {c.lower(): c for c in df.columns}
    candidates = {
        "dataset": ["dataset_name", "dataset"],
        "model": ["model_name", "model"],
        "prompt": ["file_name", "prompt", "prompt_text"],
        "category": ["category"],
        "super_category": ["super_category", "supercategory"],
    }
    found = {}
    for role, names in candidates.items():
        found[role] = next((lower_to_original[n] for n in names if n in lower_to_original), None)
    return found


def detect_metric_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        match = re.match(r"^r\d+_(d\d+|com)_", col)
        if match:
            groups.setdefault(match.group(1), []).append(col)
    return dict(sorted(groups.items()))


def add_composites(df: pd.DataFrame, metric_groups: dict[str, list[str]]) -> pd.DataFrame:
    df = df.copy()
    for group, cols in metric_groups.items():
        df[f"{group}_mean_score"] = df[cols].mean(axis=1, skipna=True)
    dataset_specific_cols = [
        col for group, cols in metric_groups.items() if group != "com" for col in cols
    ]
    if dataset_specific_cols:
        df["dataset_specific_mean_score"] = df[dataset_specific_cols].mean(axis=1, skipna=True)
    composite_parts = [
        col for col in ["com_mean_score", "dataset_specific_mean_score"] if col in df.columns
    ]
    df["overall_mean_score"] = df[composite_parts].mean(axis=1, skipna=True)
    return df


def bootstrap_ci(values: pd.Series, n_boot: int = 2000, seed: int = RANDOM_SEED) -> tuple[float, float, float]:
    x = values.dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(x))
    if len(x) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    boot = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean, float(lo), float(hi)


def cliffs_delta(a: pd.Series, b: pd.Series) -> float:
    x = a.dropna().to_numpy(dtype=float)
    y = b.dropna().to_numpy(dtype=float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    diffs = x[:, None] - y[None, :]
    return float((np.sum(diffs > 0) - np.sum(diffs < 0)) / diffs.size)


def grouped_scores(df: pd.DataFrame, group_cols: list[str], score_col: str, n_boot: int) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        mean, lo, hi = bootstrap_ci(sub[score_col], n_boot=n_boot)
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "score_mean": mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "score_median": sub[score_col].median(),
                "score_std": sub[score_col].std(ddof=1),
                "n": int(sub[score_col].notna().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("score_mean", ascending=False)


def metric_summary(df: pd.DataFrame, metric_groups: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for group, cols in metric_groups.items():
        for col in cols:
            s = df[col].dropna()
            rows.append(
                {
                    "metric_group": group,
                    "metric": col,
                    "metric_label": clean_metric_name(col),
                    "n": int(s.size),
                    "mean": s.mean(),
                    "std": s.std(ddof=1),
                    "median": s.median(),
                    "q1": s.quantile(0.25),
                    "q3": s.quantile(0.75),
                    "min": s.min(),
                    "max": s.max(),
                    "missing_rate": df[col].isna().mean(),
                }
            )
    return pd.DataFrame(rows)


def data_dictionary(df: pd.DataFrame, metric_groups: dict[str, list[str]]) -> pd.DataFrame:
    col_to_group = {c: g for g, cols in metric_groups.items() for c in cols}
    rows = []
    for col in df.columns:
        rows.append(
            {
                "column": col,
                "role": "metric" if col in col_to_group else "metadata_or_annotation",
                "metric_group": col_to_group.get(col, ""),
                "dtype": str(df[col].dtype),
                "non_missing": int(df[col].notna().sum()),
                "missing_rate": df[col].isna().mean(),
                "unique_values": int(df[col].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def pairwise_model_table(df: pd.DataFrame, model_col: str, score_col: str) -> pd.DataFrame:
    rows = []
    models = sorted(df[model_col].dropna().unique())
    for model_a, model_b in itertools.combinations(models, 2):
        a = df.loc[df[model_col] == model_a, score_col]
        b = df.loc[df[model_col] == model_b, score_col]
        statistic, p_value = stats.mannwhitneyu(a.dropna(), b.dropna(), alternative="two-sided")
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "mean_a": a.mean(),
                "mean_b": b.mean(),
                "mean_diff_a_minus_b": a.mean() - b.mean(),
                "cliffs_delta_a_vs_b": cliffs_delta(a, b),
                "mannwhitney_u": statistic,
                "p_value_uncorrected": p_value,
                "n_a": int(a.notna().sum()),
                "n_b": int(b.notna().sum()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_value_bh_fdr"] = benjamini_hochberg(out["p_value_uncorrected"])
    return out.sort_values("mean_diff_a_minus_b", ascending=False)


def benjamini_hochberg(p_values: pd.Series) -> np.ndarray:
    p = p_values.to_numpy(dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    cumulative = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        original_rank = n - rank + 1
        cumulative = min(cumulative, p[idx] * n / original_rank)
        adjusted[idx] = cumulative
    return np.clip(adjusted, 0, 1)


def word_frequency_table(df: pd.DataFrame) -> pd.DataFrame:
    text_cols = [c for c in df.columns if c.lower().startswith("manual_eval")]
    counter: Counter[str] = Counter()
    for col in text_cols:
        text = " ".join(df[col].dropna().astype(str))
        words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text.lower())
        counter.update(w for w in words if w not in STOPWORDS)
    return pd.DataFrame(counter.most_common(100), columns=["word", "count"])


def plot_model_ranking(model_scores: pd.DataFrame, model_col: str, out_path: Path) -> None:
    data = model_scores.sort_values("score_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(9.5, max(5, 0.38 * len(data))))
    xerr = np.vstack(
        [
            data["score_mean"] - data["ci95_low"],
            data["ci95_high"] - data["score_mean"],
        ]
    )
    ax.errorbar(
        data["score_mean"],
        data[model_col],
        xerr=xerr,
        fmt="o",
        color="#b4235a",
        ecolor="#2b6cb0",
        elinewidth=2,
        capsize=3,
    )
    ax.set_xlabel("Mean common-metric composite score (95% bootstrap CI)")
    ax.set_ylabel("")
    ax.set_xlim(0.8, 5.1)
    ax.set_title("Model Ranking by Common-Metric Composite")
    save_figure(fig, out_path)


def plot_common_metric_heatmap(df: pd.DataFrame, model_col: str, common_cols: list[str], out_path: Path) -> None:
    matrix = df.groupby(model_col)[common_cols].mean()
    matrix.columns = [clean_metric_name(c) for c in common_cols]
    matrix = matrix.loc[matrix.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(9.5, max(5.0, 0.38 * len(matrix))))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=1,
        vmax=5,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Mean rating"},
        ax=ax,
    )
    ax.set_title("Common Metrics by Model")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_figure(fig, out_path)


def plot_dataset_model_heatmap(model_dataset: pd.DataFrame, dataset_col: str, model_col: str, out_path: Path) -> None:
    matrix = model_dataset.pivot(index=model_col, columns=dataset_col, values="score_mean")
    matrix = matrix.loc[matrix.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(8.5, max(5, 0.38 * len(matrix))))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="mako",
        vmin=1,
        vmax=5,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Mean composite"},
        ax=ax,
    )
    ax.set_title("Dataset x Model Composite Performance")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_figure(fig, out_path)


def plot_metric_distributions(df: pd.DataFrame, metric_groups: dict[str, list[str]], out_path: Path) -> None:
    records = []
    for group, cols in metric_groups.items():
        for col in cols:
            records.append(
                pd.DataFrame(
                    {
                        "metric_group": group,
                        "metric": clean_metric_name(col),
                        "score": df[col],
                    }
                )
            )
    long = pd.concat(records, ignore_index=True).dropna()
    fig, ax = plt.subplots(figsize=(11, max(6, 0.28 * long["metric"].nunique())))
    sns.violinplot(
        data=long,
        y="metric",
        x="score",
        hue="metric_group",
        dodge=False,
        inner="quartile",
        cut=0,
        linewidth=0.7,
        palette="Set2",
        ax=ax,
    )
    ax.set_title("Metric Score Distributions")
    ax.set_xlabel("Manual rating")
    ax.set_ylabel("")
    ax.set_xlim(-0.2, 5.2)
    ax.legend(title="Metric family", loc="lower right")
    save_figure(fig, out_path)


def plot_likert_common(df: pd.DataFrame, common_cols: list[str], out_path: Path) -> None:
    rows = []
    for col in common_cols:
        counts = df[col].dropna().round().astype(int).value_counts(normalize=True)
        for score in range(1, 6):
            rows.append(
                {
                    "metric": clean_metric_name(col),
                    "score": score,
                    "percent": 100 * counts.get(score, 0.0),
                }
            )
    data = pd.DataFrame(rows)
    pivot = data.pivot(index="metric", columns="score", values="percent").fillna(0)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    colors = ["#b91c1c", "#f97316", "#facc15", "#38bdf8", "#15803d"]
    fig, ax = plt.subplots(figsize=(9.5, max(3.5, 0.42 * len(pivot))))
    left = np.zeros(len(pivot))
    y = np.arange(len(pivot))
    for idx, score in enumerate(range(1, 6)):
        values = pivot[score].to_numpy()
        ax.barh(y, values, left=left, color=colors[idx], label=f"Score {score}")
        left += values
    ax.set_yticks(y, pivot.index)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percent of ratings")
    ax.set_ylabel("")
    ax.set_title("Common Metric Likert Distribution")
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    save_figure(fig, out_path)


def plot_category_heatmap(
    df: pd.DataFrame,
    category_col: str,
    model_col: str,
    out_path: Path,
    top_n_categories: int = 14,
) -> None:
    top_categories = df[category_col].value_counts().head(top_n_categories).index
    matrix = (
        df[df[category_col].isin(top_categories)]
        .pivot_table(index=category_col, columns=model_col, values="overall_mean_score", aggfunc="mean")
    )
    fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(matrix))))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="crest",
        vmin=1,
        vmax=5,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Mean composite"},
        ax=ax,
    )
    ax.set_title("Top Prompt Categories x Models")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_figure(fig, out_path)


def plot_dataset_specific_heatmaps(
    df: pd.DataFrame,
    model_col: str,
    metric_groups: dict[str, list[str]],
    figures_dir: Path,
) -> None:
    for group, cols in metric_groups.items():
        if group == "com":
            continue
        matrix = df.groupby(model_col)[cols].mean()
        matrix.columns = [clean_metric_name(c) for c in cols]
        matrix = matrix.loc[matrix.mean(axis=1).sort_values(ascending=False).index]
        fig, ax = plt.subplots(figsize=(11, max(5, 0.38 * len(matrix))))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="flare",
            vmin=0 if matrix.min().min() == 0 else 1,
            vmax=5,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Mean rating"},
            ax=ax,
        )
        ax.set_title(f"{group.upper()} Dataset-Specific Metrics by Model")
        ax.set_xlabel("")
        ax.set_ylabel("")
        save_figure(fig, figures_dir / f"{group}_specific_metric_heatmap")


def plot_top_model_radar(df: pd.DataFrame, model_scores: pd.DataFrame, model_col: str, common_cols: list[str], out_path: Path) -> None:
    top_models = model_scores.head(5)[model_col].tolist()
    profile = df[df[model_col].isin(top_models)].groupby(model_col)[common_cols].mean()
    profile = profile.loc[[m for m in top_models if m in profile.index]]
    labels = [clean_metric_name(c) for c in common_cols]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    for model in profile.index:
        values = profile.loc[model].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.08)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 5)
    ax.set_title("Top Model Profiles Across Common Metrics", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.08), frameon=False)
    save_figure(fig, out_path)


def plot_word_frequencies(words: pd.DataFrame, out_path: Path) -> None:
    data = words.head(25).sort_values("count", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sns.barplot(data=data, y="word", x="count", color="#0f766e", ax=ax)
    ax.set_title("Most Frequent Manual Comment Terms")
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    save_figure(fig, out_path)


def write_summary(
    out_path: Path,
    csv_path: Path,
    df: pd.DataFrame,
    metric_groups: dict[str, list[str]],
    model_scores: pd.DataFrame,
    model_col: str,
    dataset_col: str,
) -> None:
    top = model_scores.head(8)
    lines = [
        "# Prompt Performance Analysis Summary",
        "",
        f"Input CSV: `{csv_path}`",
        f"Rows: {len(df):,}",
        f"Models: {df[model_col].nunique():,}",
        f"Datasets: {df[dataset_col].nunique():,}",
        f"Metric groups: {', '.join(metric_groups.keys())}",
        "",
        "## Top Models By Common Composite",
        "",
        "| Rank | Model | Mean | 95% CI | N |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        lines.append(
            f"| {rank} | {row[model_col]} | {row['score_mean']:.3f} | "
            f"{row['ci95_low']:.3f}-{row['ci95_high']:.3f} | {int(row['n'])} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `com_mean_score` is the row-wise mean of all `*_com_*` metric columns.",
            "- `dataset_specific_mean_score` is the row-wise mean of all `*_d*_*` metric columns.",
            "- `overall_mean_score` averages common and dataset-specific composites when available.",
            "- Confidence intervals are non-parametric bootstrap intervals over row-level scores.",
            "- Pairwise model comparisons use Mann-Whitney U tests and Cliff's delta.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze prompt performance from manual evaluation CSV.")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to manual evaluation CSV.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output directory.")
    parser.add_argument("--n-bootstrap", type=int, default=2000, help="Bootstrap samples for CIs.")
    args = parser.parse_args()

    configure_style()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    ids = infer_columns(df)
    if not ids["dataset"] or not ids["model"]:
        raise ValueError("Could not infer dataset/model columns. Expected dataset_name and model_name.")

    dataset_col = ids["dataset"]
    model_col = ids["model"]
    prompt_col = ids["prompt"]
    category_col = ids["category"]
    super_col = ids["super_category"]

    df[model_col] = df[model_col].astype(str).str.strip().replace(MODEL_RENAMES)
    metric_groups = detect_metric_groups(df)
    df = add_composites(df, metric_groups)

    common_score_col = "com_mean_score" if "com_mean_score" in df.columns else "overall_mean_score"
    model_scores = grouped_scores(df, [model_col], common_score_col, args.n_bootstrap)
    model_dataset_scores = grouped_scores(df, [dataset_col, model_col], "overall_mean_score", args.n_bootstrap)

    tables = {
        "cleaned_rows_with_composites.csv": df,
        "data_dictionary.csv": data_dictionary(df, metric_groups),
        "metric_summary.csv": metric_summary(df, metric_groups),
        "model_common_scores.csv": model_scores,
        "model_dataset_scores.csv": model_dataset_scores,
        "pairwise_model_comparisons.csv": pairwise_model_table(df, model_col, common_score_col),
        "manual_comment_word_frequencies.csv": word_frequency_table(df),
    }

    if category_col:
        tables["category_model_scores.csv"] = grouped_scores(
            df.dropna(subset=[category_col]), [category_col, model_col], "overall_mean_score", args.n_bootstrap
        )
    if super_col:
        tables["super_category_model_scores.csv"] = grouped_scores(
            df.dropna(subset=[super_col]), [super_col, model_col], "overall_mean_score", args.n_bootstrap
        )
    if prompt_col:
        prompt_cols = [c for c in [dataset_col, model_col, prompt_col, category_col, super_col] if c]
        tables["prompt_level_scores.csv"] = df[prompt_cols + ["com_mean_score", "dataset_specific_mean_score", "overall_mean_score"]]

    for filename, table in tables.items():
        table.to_csv(tables_dir / filename, index=False)

    plot_model_ranking(model_scores, model_col, figures_dir / "model_ranking_common_composite")
    plot_dataset_model_heatmap(model_dataset_scores, dataset_col, model_col, figures_dir / "dataset_model_composite_heatmap")
    plot_metric_distributions(df, metric_groups, figures_dir / "metric_score_distributions")
    plot_dataset_specific_heatmaps(df, model_col, metric_groups, figures_dir)

    if "com" in metric_groups:
        plot_common_metric_heatmap(df, model_col, metric_groups["com"], figures_dir / "common_metric_heatmap")
        plot_likert_common(df, metric_groups["com"], figures_dir / "common_metric_likert_distribution")
        plot_top_model_radar(df, model_scores, model_col, metric_groups["com"], figures_dir / "top_model_common_metric_radar")

    if category_col and df[category_col].notna().any():
        plot_category_heatmap(df.dropna(subset=[category_col]), category_col, model_col, figures_dir / "category_model_heatmap")

    words = tables["manual_comment_word_frequencies.csv"]
    if not words.empty:
        plot_word_frequencies(words, figures_dir / "manual_comment_word_frequencies")

    write_summary(
        out_dir / "analysis_summary.md",
        csv_path,
        df,
        metric_groups,
        model_scores,
        model_col,
        dataset_col,
    )

    print(f"Done. Outputs written to: {out_dir.resolve()}")
    print(f"Tables: {tables_dir.resolve()}")
    print(f"Figures: {figures_dir.resolve()}")


if __name__ == "__main__":
    main()
