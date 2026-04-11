"""
Statistically grounded classification-based evaluation for summary utility.

This script extends downstream-task evaluation with repeated train/test splits,
confidence intervals, paired statistical tests, effect sizes, and publication-
ready result tables/figures for Section 5.4 (Statistical Validation of
Performance Differences).

What it does:
1. Loads generated summaries from model-specific folders
2. Maps summaries to class labels using the source CSV dataset
3. Builds repeated train/test splits by ORIGINAL resume ID to avoid leakage
4. Trains the same TF-IDF + Logistic Regression classifier for every model
5. Collects per-run metrics (accuracy, weighted F1, macro F1)
6. Computes 95% confidence intervals for each model
7. Performs omnibus and pairwise statistical testing across repeated runs
8. Exports reviewer-friendly CSV tables and plots

Recommended manuscript use:
- Use mean ± std and 95% CI from repeated runs in the main results table
- Use Friedman test for omnibus significance across multiple models
- Use Wilcoxon signed-rank + Holm correction for pairwise post-hoc comparison
- Report effect size alongside p-values

Notes:
- This script treats each repeated split as one paired observation across models.
- It is appropriate for statistical comparison of model-level downstream utility.
- If you need per-instance significance testing on one fixed test set, add
  McNemar tests using exported predictions.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare, rankdata, sem, t, wilcoxon
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import warnings


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SUMMARY_BASE_DIR = Path("data/summaries")
CSV_DATA_PATH = Path("data/resumes_dataset.csv")

ID_COLUMN = "ID"
CATEGORY_COLUMN = "Category"
GENERATED_SUFFIX = "_summary.txt"


USE_REPEATED_KFOLD = True
N_SPLITS = 5
N_REPEATS = 5
TEST_SET_SIZE = 0.25 
RANDOM_STATE = 42


TFIDF_MAX_DF = 0.9
TFIDF_MIN_DF = 3
TFIDF_NGRAM_RANGE = (1, 2)
LOGREG_MAX_ITER = 1000
LOGREG_SOLVER = "liblinear"
LOGREG_CLASS_WEIGHT = "balanced"


PRIMARY_METRIC = "f1_weighted"
ALPHA = 0.05
CONFIDENCE_LEVEL = 0.95
P_VALUE_DECIMALS = 6


SAVE_PLOTS = True
PLOT_OUTPUT_DIR = Path("evaluation_plots")
PLOT_DPI = 300
RESULTS_OUTPUT_DIR = Path("evaluation_results")

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ------------------------------------------------------------------
# Helpers: file loading
# ------------------------------------------------------------------
def load_text(file_path: Path) -> str | None:
    """Safely load text with fallback encodings."""
    if not file_path.is_file():
        return None

    encodings_to_try = ["utf-8", "latin-1", "cp1252"]
    for encoding in encodings_to_try:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read().strip()
            return content if content else None
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"[WARNING] Failed to load {file_path}: {e}")
            return None
    return None


def load_category_map(csv_path: Path, id_col: str, category_col: str) -> dict | None:
    """Load mapping from resume ID to category label."""
    if not csv_path.is_file():
        print(f"[ERROR] CSV file not found: {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=[id_col, category_col]).copy()
        df[id_col] = df[id_col].astype(str)
        df[category_col] = df[category_col].astype(str)
        category_map = pd.Series(df[category_col].values, index=df[id_col]).to_dict()
        print(f"[INFO] Loaded category map for {len(category_map)} items")
        return category_map
    except KeyError as e:
        print(f"[ERROR] Missing required CSV column: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        return None


# ------------------------------------------------------------------
# Helpers: data collection
# ------------------------------------------------------------------
def extract_resume_id_from_summary(summary_file: Path) -> str:
    """Recover original resume ID from summary filename."""
    stem = summary_file.stem
    suffix_without_ext = GENERATED_SUFFIX.replace(".txt", "")
    return stem[: -len(suffix_without_ext)] if stem.endswith(suffix_without_ext) else stem


def collect_summary_records(summary_base_dir: Path, category_map: dict) -> Tuple[pd.DataFrame, List[str]]:
    """Collect all valid summary records across model folders."""
    if not summary_base_dir.is_dir():
        print(f"[ERROR] Summary directory not found: {summary_base_dir}")
        return pd.DataFrame(), []

    model_dirs = [d for d in summary_base_dir.iterdir() if d.is_dir()]
    if not model_dirs:
        print(f"[ERROR] No model directories found in {summary_base_dir}")
        return pd.DataFrame(), []

    records = []
    all_categories = set()

    print(f"[INFO] Scanning summaries in: {summary_base_dir}")
    for model_dir in tqdm(model_dirs, desc="Scanning models"):
        model_name = model_dir.name
        summary_files = list(model_dir.rglob(f"*{GENERATED_SUFFIX}"))

        for summary_file in summary_files:
            resume_id = extract_resume_id_from_summary(summary_file)
            if resume_id not in category_map:
                continue

            summary_text = load_text(summary_file)
            if not summary_text:
                continue

            category = category_map[resume_id]
            records.append(
                {
                    "model": model_name,
                    "resume_id": resume_id,
                    "summary": summary_text,
                    "category": category,
                }
            )
            all_categories.add(category)

    df = pd.DataFrame(records)
    if df.empty:
        return df, []

    # Keep only resume IDs present for ALL models so comparisons are fair and paired.
    model_counts = df.groupby("resume_id")["model"].nunique()
    required_model_count = df["model"].nunique()
    complete_ids = model_counts[model_counts == required_model_count].index.tolist()
    df = df[df["resume_id"].isin(complete_ids)].copy()

    return df, sorted(all_categories)


def validate_dataset_coverage(all_df: pd.DataFrame) -> bool:
    """Ensure each model has the same resume IDs after filtering."""
    if all_df.empty:
        return False

    coverage = all_df.groupby("model")["resume_id"].nunique().sort_values()
    print("[INFO] Resume coverage per model after intersection filtering:")
    print(coverage.to_string())

    return coverage.nunique() == 1


# ------------------------------------------------------------------
# Helpers: repeated split design
# ------------------------------------------------------------------
def build_resume_level_table(all_df: pd.DataFrame) -> pd.DataFrame:
    """Construct resume-level table for split generation."""
    resume_df = all_df[["resume_id", "category"]].drop_duplicates(subset=["resume_id"]).copy()
    resume_df.sort_values("resume_id", inplace=True)
    resume_df.reset_index(drop=True, inplace=True)
    return resume_df


def generate_splits(resume_df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """Generate repeated paired splits over original resume IDs."""
    X = np.arange(len(resume_df))
    y = resume_df["category"].values
    splits = []

    class_counts = pd.Series(y).value_counts()
    min_class_count = int(class_counts.min()) if not class_counts.empty else 0

    if USE_REPEATED_KFOLD:
        if min_class_count < N_SPLITS:
            print(
                f"[WARNING] Smallest class has only {min_class_count} samples; "
                f"switching from repeated stratified k-fold to repeated stratified shuffle split."
            )
            splitter = StratifiedShuffleSplit(
                n_splits=N_SPLITS * N_REPEATS,
                test_size=TEST_SET_SIZE,
                random_state=RANDOM_STATE,
            )
        else:
            splitter = RepeatedStratifiedKFold(
                n_splits=N_SPLITS,
                n_repeats=N_REPEATS,
                random_state=RANDOM_STATE,
            )
    else:
        splitter = StratifiedShuffleSplit(
            n_splits=N_SPLITS * N_REPEATS,
            test_size=TEST_SET_SIZE,
            random_state=RANDOM_STATE,
        )

    for split_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        splits.append((train_idx, test_idx, split_idx))

    print(f"[INFO] Generated {len(splits)} repeated paired split(s)")
    return splits


# ------------------------------------------------------------------
# Helpers: model evaluation
# ------------------------------------------------------------------
def make_classifier_pipeline() -> Pipeline:
    """Build the downstream classifier pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_df=TFIDF_MAX_DF,
                    min_df=TFIDF_MIN_DF,
                    ngram_range=TFIDF_NGRAM_RANGE,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    solver=LOGREG_SOLVER,
                    random_state=RANDOM_STATE,
                    max_iter=LOGREG_MAX_ITER,
                    class_weight=LOGREG_CLASS_WEIGHT,
                ),
            ),
        ]
    )


def evaluate_single_model_on_split(
    model_df: pd.DataFrame,
    train_resume_ids: np.ndarray,
    test_resume_ids: np.ndarray,
    category_labels: List[str],
) -> Tuple[dict, List[dict], dict]:
    """Train/evaluate one model on one paired split."""
    train_df = model_df[model_df["resume_id"].isin(train_resume_ids)]
    test_df = model_df[model_df["resume_id"].isin(test_resume_ids)]

    if train_df.empty or test_df.empty:
        raise ValueError("Insufficient data after split")

    X_train, y_train = train_df["summary"], train_df["category"]
    X_test, y_test = test_df["summary"], test_df["category"]

    clf_pipeline = make_classifier_pipeline()
    clf_pipeline.fit(X_train, y_train)
    y_pred = clf_pipeline.predict(X_test)

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
        labels=category_labels,
    )

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": report_dict["weighted avg"]["f1-score"],
        "f1_macro": report_dict["macro avg"]["f1-score"],
    }

    per_category_rows = []
    for label, metrics in report_dict.items():
        if label in category_labels:
            per_category_rows.append(
                {
                    "category": label,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1-score"],
                    "support": metrics["support"],
                }
            )

    prediction_payload = {
        "y_true": np.array(y_test),
        "y_pred": np.array(y_pred),
    }
    return results, per_category_rows, prediction_payload


# ------------------------------------------------------------------
# Helpers: statistics
# ------------------------------------------------------------------
def compute_confidence_interval(values: List[float], confidence_level: float = CONFIDENCE_LEVEL) -> Tuple[float, float]:
    """Compute t-based confidence interval for repeated-run metric values."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]

    if len(arr) == 0:
        return np.nan, np.nan
    if len(arr) == 1:
        return arr[0], arr[0]

    mean_val = arr.mean()
    interval = sem(arr) * t.ppf((1 + confidence_level) / 2.0, len(arr) - 1)
    return mean_val - interval, mean_val + interval


def cliffs_delta_from_paired_differences(differences: np.ndarray) -> float:
    """
    Simple paired effect-size proxy using standardized signed-rank style magnitude.

    For paired repeated-run comparisons, this returns a rank-biserial-like quantity
    in [-1, 1], interpretable as effect direction and strength.
    """
    diffs = np.asarray(differences, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    diffs = diffs[diffs != 0]

    if len(diffs) == 0:
        return 0.0

    ranks = rankdata(np.abs(diffs))
    pos_sum = ranks[diffs > 0].sum()
    neg_sum = ranks[diffs < 0].sum()
    total = ranks.sum()

    if total == 0:
        return 0.0
    return float((pos_sum - neg_sum) / total)


def holm_correction(p_values: List[float]) -> List[float]:
    """Apply Holm step-down correction."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)

    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p_values[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)

    return adjusted.tolist()


def run_omnibus_friedman(run_metrics_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Run Friedman omnibus test across all successfully evaluated models."""
    pivot_df = run_metrics_df.pivot(index="split_id", columns="model", values=metric)
    pivot_df = pivot_df.dropna(axis=0, how="any")

    if pivot_df.shape[1] < 3 or pivot_df.shape[0] < 2:
        return pd.DataFrame(
            [
                {
                    "metric": metric,
                    "test": "friedman",
                    "n_models": pivot_df.shape[1],
                    "n_splits": pivot_df.shape[0],
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "significant": False,
                    "note": "Need at least 3 models and 2 paired runs for Friedman test",
                }
            ]
        )

    model_vectors = [pivot_df[col].values for col in pivot_df.columns]
    stat, p_value = friedmanchisquare(*model_vectors)

    return pd.DataFrame(
        [
            {
                "metric": metric,
                "test": "friedman",
                "n_models": pivot_df.shape[1],
                "n_splits": pivot_df.shape[0],
                "statistic": stat,
                "p_value": p_value,
                "significant": bool(p_value < ALPHA),
                "note": "Omnibus repeated-measures comparison across summarization models",
            }
        ]
    )


def run_pairwise_wilcoxon(run_metrics_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Run pairwise Wilcoxon signed-rank tests with Holm correction."""
    pivot_df = run_metrics_df.pivot(index="split_id", columns="model", values=metric)
    pivot_df = pivot_df.dropna(axis=0, how="any")
    models = list(pivot_df.columns)

    rows = []
    raw_p_values = []
    pair_meta = []

    for model_a, model_b in combinations(models, 2):
        paired = pivot_df[[model_a, model_b]].dropna()
        vals_a = paired[model_a].values
        vals_b = paired[model_b].values
        diffs = vals_a - vals_b

        mean_diff = float(np.mean(diffs)) if len(diffs) else np.nan
        effect_size = cliffs_delta_from_paired_differences(diffs)

        try:
            if len(diffs) < 2 or np.allclose(diffs, 0):
                stat, p_value = np.nan, 1.0
                note = "Insufficient non-zero paired differences for Wilcoxon"
            else:
                stat, p_value = wilcoxon(vals_a, vals_b, zero_method="wilcox", correction=False)
                note = "Paired post-hoc comparison"
        except ValueError:
            stat, p_value = np.nan, 1.0
            note = "Wilcoxon could not be computed"

        row = {
            "metric": metric,
            "model_a": model_a,
            "model_b": model_b,
            "n_paired_runs": len(diffs),
            "mean_model_a": float(np.mean(vals_a)) if len(vals_a) else np.nan,
            "mean_model_b": float(np.mean(vals_b)) if len(vals_b) else np.nan,
            "mean_difference_a_minus_b": mean_diff,
            "wilcoxon_statistic": stat,
            "p_value_raw": p_value,
            "effect_size_rank_biserial": effect_size,
            "note": note,
        }
        rows.append(row)
        raw_p_values.append(p_value)
        pair_meta.append((model_a, model_b))

    if not rows:
        return pd.DataFrame()

    adjusted = holm_correction(raw_p_values)
    for row, p_adj in zip(rows, adjusted):
        row["p_value_holm"] = p_adj
        row["significant_at_alpha"] = bool(p_adj < ALPHA)

    return pd.DataFrame(rows).sort_values(
        by=["significant_at_alpha", "p_value_holm", "mean_difference_a_minus_b"],
        ascending=[False, True, False],
    )


def build_summary_statistics(run_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated-run metrics into mean/std/CI summary table."""
    rows = []

    for model_name, sub_df in run_metrics_df.groupby("model"):
        row = {"model": model_name, "n_runs": len(sub_df)}

        for metric in ["accuracy", "f1_weighted", "f1_macro"]:
            values = sub_df[metric].astype(float).tolist()
            row[f"{metric}_mean"] = float(np.mean(values)) if values else np.nan
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ci_low, ci_high = compute_confidence_interval(values)
            row[f"{metric}_ci_low"] = ci_low
            row[f"{metric}_ci_high"] = ci_high

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df.sort_values(by=f"{PRIMARY_METRIC}_mean", ascending=False, inplace=True)
    return summary_df


def build_average_ranks(run_metrics_df: pd.DataFrame, metric: str = PRIMARY_METRIC) -> pd.DataFrame:
    """Compute average model ranks across repeated runs."""
    pivot_df = run_metrics_df.pivot(index="split_id", columns="model", values=metric)
    pivot_df = pivot_df.dropna(axis=0, how="any")

    if pivot_df.empty:
        return pd.DataFrame(columns=["model", "average_rank"])

    ranks_per_split = pivot_df.apply(lambda row: rankdata(-row.values, method="average"), axis=1, result_type="expand")
    ranks_per_split.columns = pivot_df.columns

    avg_ranks = ranks_per_split.mean(axis=0).sort_values()
    return pd.DataFrame({"model": avg_ranks.index, "average_rank": avg_ranks.values})


# ------------------------------------------------------------------
# Helpers: plots
# ------------------------------------------------------------------
def plot_summary_metric_bars(summary_df: pd.DataFrame, metric: str, save_path: Path | None = None):
    """Plot model mean performance with CI error bars."""
    mean_col = f"{metric}_mean"
    low_col = f"{metric}_ci_low"
    high_col = f"{metric}_ci_high"

    if summary_df.empty or mean_col not in summary_df.columns:
        return

    plot_df = summary_df.sort_values(mean_col, ascending=False).copy()
    x = np.arange(len(plot_df))
    y = plot_df[mean_col].values
    yerr = np.vstack([
        y - plot_df[low_col].values,
        plot_df[high_col].values - y,
    ])

    plt.figure(figsize=(max(8, len(plot_df) * 0.8), 6))
    plt.bar(x, y)
    plt.errorbar(x, y, yerr=yerr, fmt="none", capsize=5)
    plt.xticks(x, plot_df["model"], rotation=45, ha="right")
    plt.ylabel(metric.replace("_", " ").title())
    plt.xlabel("Summarization Model")
    plt.title(f"Repeated-Split Classification Performance ({metric.replace('_', ' ').title()})")

    for xi, yi in zip(x, y):
        plt.annotate(f"{yi:.3f}", (xi, yi), ha="center", va="bottom", xytext=(0, 6), textcoords="offset points")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")


def plot_average_ranks(avg_rank_df: pd.DataFrame, save_path: Path | None = None):
    """Plot average ranks from repeated runs."""
    if avg_rank_df.empty:
        return

    plot_df = avg_rank_df.sort_values("average_rank", ascending=True)
    plt.figure(figsize=(max(8, len(plot_df) * 0.8), 5.5))
    sns.barplot(data=plot_df, x="model", y="average_rank")
    plt.gca().invert_yaxis()
    plt.title(f"Average Model Ranks Across Repeated Runs ({PRIMARY_METRIC})")
    plt.xlabel("Summarization Model")
    plt.ylabel("Average Rank (1 = best)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")


def plot_per_category_heatmap(per_category_avg_df: pd.DataFrame, metric: str = "f1_score", save_path: Path | None = None):
    """Plot average per-category metric across repeated runs."""
    if per_category_avg_df.empty:
        return

    pivot_df = per_category_avg_df.pivot(index="model", columns="category", values=metric)
    if pivot_df.empty:
        return

    plt.figure(figsize=(max(10, pivot_df.shape[1] * 0.9), max(6, pivot_df.shape[0] * 0.6)))
    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="viridis", linewidths=0.5, cbar=True)
    plt.title(f"Average Per-Category {metric.replace('_', ' ').title()} Across Repeated Runs")
    plt.xlabel("Category")
    plt.ylabel("Model")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")


def plot_confusion_matrix_figure(cm, categories, model_name: str, save_path: Path | None = None):
    """Plot aggregated confusion matrix for one model."""
    categories = list(categories)
    if len(categories) == 0:
        return

    df_cm = pd.DataFrame(cm, index=categories, columns=categories)
    plt.figure(figsize=(max(8, len(categories) * 0.6), max(6, len(categories) * 0.5)))
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", linewidths=0.5, cbar=True)
    except ValueError:
        heatmap = sns.heatmap(df_cm, annot=True, fmt=".0f", cmap="Blues", linewidths=0.5, cbar=True)

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha="right")
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=30, ha="right")
    plt.ylabel("True Category")
    plt.xlabel("Predicted Category")
    plt.title(f"Aggregated Confusion Matrix: {model_name}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")


# ------------------------------------------------------------------
# Helpers: saving
# ------------------------------------------------------------------
def save_dataframe(df: pd.DataFrame, output_path: Path, index: bool = False):
    """Save a DataFrame with consistent float formatting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=index, float_format="%.6f")
    print(f"[INFO] Saved: {output_path.resolve()}")


# ------------------------------------------------------------------
# Main runner
# ------------------------------------------------------------------
def run_classification_evaluation():
    """Run repeated-split evaluation and statistical validation."""
    if not SUMMARY_BASE_DIR.is_dir():
        print(f"[ERROR] Summary directory not found: {SUMMARY_BASE_DIR}")
        return

    category_map = load_category_map(CSV_DATA_PATH, ID_COLUMN, CATEGORY_COLUMN)
    if category_map is None:
        print("[ERROR] Evaluation aborted: label mapping unavailable")
        return

    all_df, category_labels = collect_summary_records(SUMMARY_BASE_DIR, category_map)
    if all_df.empty:
        print("[ERROR] No valid summary/category pairs collected")
        return
    if not category_labels:
        print("[ERROR] No category labels found")
        return

    n_models = all_df["model"].nunique()
    n_resumes = all_df["resume_id"].nunique()
    print(f"[INFO] Collected {len(all_df)} valid summary records")
    print(f"[INFO] Models included: {n_models}")
    print(f"[INFO] Resume IDs retained for fair paired comparison: {n_resumes}")
    print(f"[INFO] Categories found: {len(category_labels)}")

    if not validate_dataset_coverage(all_df):
        print("[ERROR] Models do not have identical resume coverage after filtering")
        return

    resume_df = build_resume_level_table(all_df)
    splits = generate_splits(resume_df)
    if not splits:
        print("[ERROR] No valid splits generated")
        return

    RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_PLOTS:
        PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_metric_rows = []
    per_category_rows = []
    aggregated_confusions: Dict[str, np.ndarray] = {
        model_name: np.zeros((len(category_labels), len(category_labels)), dtype=int)
        for model_name in sorted(all_df["model"].unique())
    }

    model_names = sorted(all_df["model"].unique())
    print(f"[INFO] Evaluating models: {model_names}")

    for train_idx, test_idx, split_id in tqdm(splits, desc="Repeated paired splits"):
        train_resume_ids = resume_df.iloc[train_idx]["resume_id"].values
        test_resume_ids = resume_df.iloc[test_idx]["resume_id"].values

        for model_name in model_names:
            model_df = all_df[all_df["model"] == model_name]
            try:
                split_results, split_per_category, prediction_payload = evaluate_single_model_on_split(
                    model_df=model_df,
                    train_resume_ids=train_resume_ids,
                    test_resume_ids=test_resume_ids,
                    category_labels=category_labels,
                )

                run_metric_rows.append(
                    {
                        "split_id": split_id,
                        "model": model_name,
                        **split_results,
                    }
                )

                for row in split_per_category:
                    per_category_rows.append({"split_id": split_id, "model": model_name, **row})

                cm = confusion_matrix(
                    prediction_payload["y_true"],
                    prediction_payload["y_pred"],
                    labels=category_labels,
                )
                aggregated_confusions[model_name] += cm

            except Exception as e:
                print(f"[WARNING] Evaluation failed for model={model_name}, split={split_id}: {e}")

    run_metrics_df = pd.DataFrame(run_metric_rows)
    if run_metrics_df.empty:
        print("[ERROR] No run-level metrics produced")
        return

    per_category_df = pd.DataFrame(per_category_rows)
    summary_df = build_summary_statistics(run_metrics_df)
    avg_ranks_df = build_average_ranks(run_metrics_df, metric=PRIMARY_METRIC)

    # Aggregate per-category results across repeated runs
    if not per_category_df.empty:
        per_category_avg_df = (
            per_category_df.groupby(["model", "category"], as_index=False)[["precision", "recall", "f1_score", "support"]]
            .mean()
            .sort_values(["model", "category"])
        )
    else:
        per_category_avg_df = pd.DataFrame(columns=["model", "category", "precision", "recall", "f1_score", "support"])

    # Statistics
    omnibus_df = run_omnibus_friedman(run_metrics_df, metric=PRIMARY_METRIC)
    pairwise_df = run_pairwise_wilcoxon(run_metrics_df, metric=PRIMARY_METRIC)

    # Save tables
    save_dataframe(run_metrics_df.sort_values(["split_id", "model"]), RESULTS_OUTPUT_DIR / "classification_run_metrics.csv")
    save_dataframe(summary_df, RESULTS_OUTPUT_DIR / "classification_summary_statistics.csv")
    save_dataframe(avg_ranks_df, RESULTS_OUTPUT_DIR / "classification_average_ranks.csv")
    save_dataframe(omnibus_df, RESULTS_OUTPUT_DIR / f"statistical_validation_omnibus_{PRIMARY_METRIC}.csv")
    if not pairwise_df.empty:
        save_dataframe(pairwise_df, RESULTS_OUTPUT_DIR / f"statistical_validation_pairwise_{PRIMARY_METRIC}.csv")
    if not per_category_df.empty:
        save_dataframe(per_category_df.sort_values(["split_id", "model", "category"]), RESULTS_OUTPUT_DIR / "classification_per_category_by_run.csv")
        save_dataframe(per_category_avg_df, RESULTS_OUTPUT_DIR / "classification_per_category_average.csv")

    # Console summary
    print("\n[INFO] Repeated-run summary statistics:")
    display_cols = [
        "model",
        "n_runs",
        "accuracy_mean",
        "accuracy_std",
        "f1_weighted_mean",
        "f1_weighted_std",
        "f1_macro_mean",
        "f1_macro_std",
    ]
    print(summary_df[display_cols].to_markdown(index=False, floatfmt=".4f"))

    print("\n[INFO] Average ranks:")
    if not avg_ranks_df.empty:
        print(avg_ranks_df.to_markdown(index=False, floatfmt=".4f"))

    print("\n[INFO] Omnibus statistical test:")
    print(omnibus_df.to_markdown(index=False, floatfmt=".6f"))

    if not pairwise_df.empty:
        print("\n[INFO] Pairwise post-hoc comparisons:")
        preview_cols = [
            "model_a",
            "model_b",
            "mean_difference_a_minus_b",
            "p_value_raw",
            "p_value_holm",
            "effect_size_rank_biserial",
            "significant_at_alpha",
        ]
        print(pairwise_df[preview_cols].to_markdown(index=False, floatfmt=".6f"))

    # Plots
    if SAVE_PLOTS:
        for metric in ["accuracy", "f1_weighted", "f1_macro"]:
            plot_summary_metric_bars(
                summary_df,
                metric=metric,
                save_path=PLOT_OUTPUT_DIR / f"repeated_split_{metric}_with_ci.png",
            )

        plot_average_ranks(
            avg_ranks_df,
            save_path=PLOT_OUTPUT_DIR / f"average_ranks_{PRIMARY_METRIC}.png",
        )

        if not per_category_avg_df.empty:
            plot_per_category_heatmap(
                per_category_avg_df,
                metric="f1_score",
                save_path=PLOT_OUTPUT_DIR / "per_category_average_f1_heatmap.png",
            )

        for model_name, cm in aggregated_confusions.items():
            plot_confusion_matrix_figure(
                cm,
                category_labels,
                model_name,
                save_path=PLOT_OUTPUT_DIR / f"aggregated_confusion_matrix_{model_name}.png",
            )

        plt.close("all")
        print(f"\n[INFO] Plots saved to: {PLOT_OUTPUT_DIR.resolve()}")

    print(f"\n[INFO] Statistical evaluation complete. Results saved to: {RESULTS_OUTPUT_DIR.resolve()}")
    print("[INFO] Recommended Section 5.4 source files:")
    print(f"  - {RESULTS_OUTPUT_DIR / 'classification_summary_statistics.csv'}")
    print(f"  - {RESULTS_OUTPUT_DIR / f'statistical_validation_omnibus_{PRIMARY_METRIC}.csv'}")
    print(f"  - {RESULTS_OUTPUT_DIR / f'statistical_validation_pairwise_{PRIMARY_METRIC}.csv'}")


if __name__ == "__main__":
    run_classification_evaluation()
