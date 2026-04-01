"""
Classification-based evaluation for summary utility.

This script:
1. Loads generated summaries
2. Maps summaries to labels using a dataset CSV
3. Performs a consistent train/test split by original resume ID
4. Trains a TF-IDF + Logistic Regression classifier per summarization model
5. Saves overall metrics, per-category metrics, confusion matrices, and plots

Designed for reproducible downstream-task evaluation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from tqdm import tqdm


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SUMMARY_BASE_DIR = Path("data/summaries")
CSV_DATA_PATH = Path("data/resumes_dataset.csv")

ID_COLUMN = "ID"
CATEGORY_COLUMN = "Category"
GENERATED_SUFFIX = "_summary.txt"

TEST_SET_SIZE = 0.25
RANDOM_STATE = 42

SAVE_PLOTS = True
PLOT_OUTPUT_DIR = Path("evaluation_plots")
PLOT_DPI = 300

RESULTS_OUTPUT_DIR = Path("evaluation_results")


# ------------------------------------------------------------------
# Data Loading Helpers
# ------------------------------------------------------------------
def load_text(file_path: Path):
    """
    Safely load text from a file using multiple fallback encodings.

    Args:
        file_path (Path): Input file path

    Returns:
        str | None: File content, or None if loading fails
    """
    if not file_path.is_file():
        return None

    encodings_to_try = ["utf-8", "latin-1", "cp1252"]

    try:
        for encoding in encodings_to_try:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read().strip()
                return content if content else None
            except UnicodeDecodeError:
                continue
    except Exception as e:
        print(f"[WARNING] Failed to load file {file_path}: {e}")

    return None


def load_category_map(csv_path: Path, id_col: str, category_col: str):
    """
    Load mapping from document ID to category label.

    Args:
        csv_path (Path): CSV file path
        id_col (str): ID column name
        category_col (str): Category column name

    Returns:
        dict | None: ID-to-category mapping
    """
    if not csv_path.is_file():
        print(f"[ERROR] CSV file not found: {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=[id_col, category_col])
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
# Plotting Helpers
# ------------------------------------------------------------------
def plot_overall_performance(df: pd.DataFrame, metric: str, title_suffix: str, save_path: Path | None = None):
    """
    Plot overall model performance for a selected metric.
    """
    if df.empty or metric not in df.columns:
        print(f"[INFO] Skipping overall performance plot for {metric}")
        return

    plot_df = df.sort_values(metric, ascending=False)

    plt.figure(figsize=(max(8, len(plot_df.index) * 0.8), 6))
    ax = sns.barplot(data=plot_df, x=plot_df.index, y=metric, palette="viridis")

    plt.title(f"Overall Classification Performance ({title_suffix})")
    plt.xlabel("Summarization Model")
    plt.ylabel(title_suffix)
    plt.xticks(rotation=45, ha="right")

    for patch in ax.patches:
        ax.annotate(
            f"{patch.get_height():.3f}",
            (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
            ha="center",
            va="center",
            xytext=(0, 9),
            textcoords="offset points",
        )

    y_max = plot_df[metric].max()
    plt.ylim(0, 1.05 * y_max if y_max > 0 else 1.05)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
        print(f"[INFO] Saved plot: {save_path}")


def plot_per_category_grouped_bar(df: pd.DataFrame, metric: str = "f1_score", save_path: Path | None = None):
    """
    Plot per-category performance as grouped bars.
    """
    if df.empty:
        print("[INFO] Skipping grouped bar plot: no per-category data")
        return

    n_categories = df["category"].nunique()
    n_models = df["model"].nunique()

    plt.figure(figsize=(max(12, n_categories * 1.2), max(6, n_models * 0.5)))

    try:
        sns.barplot(data=df, x="category", y=metric, hue="model", palette="tab10")
        plt.title(f"Per-Category Classification {metric.replace('_', ' ').title()}")
        plt.xlabel("Category")
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.ylim(0, 1.05)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
            print(f"[INFO] Saved plot: {save_path}")

    except Exception as e:
        print(f"[WARNING] Could not generate grouped bar plot: {e}")


def plot_per_category_heatmap(df: pd.DataFrame, metric: str = "f1_score", save_path: Path | None = None):
    """
    Plot per-category performance as a heatmap.
    """
    if df.empty:
        print("[INFO] Skipping heatmap: no per-category data")
        return

    try:
        pivot_df = df.pivot_table(index="model", columns="category", values=metric, fill_value=0)

        if pivot_df.empty:
            print("[INFO] Skipping heatmap: pivot table is empty")
            return

        n_rows = len(pivot_df.index)
        n_cols = len(pivot_df.columns)

        plt.figure(figsize=(max(10, n_cols * 0.8), max(6, n_rows * 0.6)))
        sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="viridis", linewidths=0.5, cbar=True)

        plt.title(f"Per-Category Classification {metric.replace('_', ' ').title()} Heatmap")
        plt.xlabel("Category")
        plt.ylabel("Model")
        plt.yticks(rotation=0)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
            print(f"[INFO] Saved plot: {save_path}")

    except Exception as e:
        print(f"[WARNING] Could not generate heatmap: {e}")


def plot_confusion_matrix_figure(cm, categories, model_name: str, save_path: Path | None = None):
    """
    Plot confusion matrix for a single summarization model.
    """
    categories = list(categories)
    if not categories:
        print(f"[INFO] Skipping confusion matrix for {model_name}: no categories")
        return

    df_cm = pd.DataFrame(cm, index=categories, columns=categories)

    plt.figure(figsize=(max(8, len(categories) * 0.6), max(6, len(categories) * 0.5)))

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", linewidths=0.5, cbar=True)
    except ValueError:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="s", cmap="Blues", linewidths=0.5, cbar=True)

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha="right")
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=30, ha="right")

    plt.ylabel("True Category")
    plt.xlabel("Predicted Category")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
        print(f"[INFO] Saved plot: {save_path}")


# ------------------------------------------------------------------
# Core Collection and Evaluation Helpers
# ------------------------------------------------------------------
def extract_resume_id_from_summary(summary_file: Path) -> str:
    """
    Recover original resume ID from summary filename.
    """
    stem = summary_file.stem
    suffix_without_ext = GENERATED_SUFFIX.replace(".txt", "")

    if stem.endswith(suffix_without_ext):
        return stem[: -len(suffix_without_ext)]

    return stem


def collect_summary_records(summary_base_dir: Path, category_map: dict):
    """
    Collect summary texts and labels across all model directories.

    Returns:
        tuple[pd.DataFrame, list[str]]
    """
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
                    "unique_id": f"{model_name}__{resume_id}",
                    "model": model_name,
                    "resume_id": resume_id,
                    "summary": summary_text,
                    "category": category,
                }
            )
            all_categories.add(category)

    df = pd.DataFrame(records)
    return df, sorted(all_categories)


def build_train_test_split(all_df: pd.DataFrame):
    """
    Create a consistent split based on original resume IDs.
    """
    unique_resume_ids = all_df["resume_id"].unique()

    if len(unique_resume_ids) < 2:
        raise ValueError("Need at least two unique resume IDs for train/test split")

    dedup_df = (
        all_df[["resume_id", "category"]]
        .drop_duplicates(subset=["resume_id"])
        .set_index("resume_id")
    )

    stratify_array = None

    try:
        aligned_categories = dedup_df.loc[unique_resume_ids, "category"]
        category_counts = aligned_categories.value_counts()

        if all(count >= 2 for count in category_counts):
            stratify_array = aligned_categories
            print("[INFO] Using stratified train/test split")
        else:
            print("[WARNING] Some classes have fewer than 2 samples; using non-stratified split")
    except Exception as e:
        print(f"[WARNING] Could not prepare stratified split: {e}")

    train_ids, test_ids = train_test_split(
        unique_resume_ids,
        test_size=TEST_SET_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_array,
    )

    print(f"[INFO] Train/Test split: {len(train_ids)} train, {len(test_ids)} test")
    return train_ids, test_ids


def evaluate_single_model(model_df: pd.DataFrame, train_ids, test_ids, category_labels):
    """
    Train and evaluate TF-IDF + Logistic Regression for one summarization model.
    """
    train_df = model_df[model_df["resume_id"].isin(train_ids)]
    test_df = model_df[model_df["resume_id"].isin(test_ids)]

    if train_df.empty or test_df.empty:
        return {"error": "Insufficient data for split"}, None, None

    X_train, y_train = train_df["summary"], train_df["category"]
    X_test, y_test = test_df["summary"], test_df["category"]

    clf_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.9, min_df=3, ngram_range=(1, 2))),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    clf_pipeline.fit(X_train, y_train)
    y_pred = clf_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
        labels=category_labels,
    )

    results = {
        "accuracy": accuracy,
        "f1_weighted": report_dict["weighted avg"]["f1-score"],
        "f1_macro": report_dict["macro avg"]["f1-score"],
        "error": None,
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

    predictions = {"y_true": y_test, "y_pred": y_pred}

    return results, per_category_rows, predictions


# ------------------------------------------------------------------
# Save / Reporting Helpers
# ------------------------------------------------------------------
def save_results(success_df: pd.DataFrame, error_df: pd.DataFrame, per_category_df: pd.DataFrame):
    """
    Save evaluation tables to CSV.
    """
    RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not success_df.empty:
        sort_col = "f1_weighted" if "f1_weighted" in success_df.columns else "accuracy"
        success_path = RESULTS_OUTPUT_DIR / "classification_overall_success.csv"
        success_df.sort_values(by=sort_col, ascending=False).to_csv(
            success_path,
            index=True,
            float_format="%.4f",
        )
        print(f"[INFO] Saved overall success results: {success_path.resolve()}")

    if not error_df.empty:
        error_path = RESULTS_OUTPUT_DIR / "classification_overall_errors.csv"
        error_df.to_csv(error_path, index=True)
        print(f"[INFO] Saved error results: {error_path.resolve()}")

    if not per_category_df.empty:
        per_category_path = RESULTS_OUTPUT_DIR / "classification_per_category.csv"
        per_category_df.sort_values(by=["model", "category"]).to_csv(
            per_category_path,
            index=False,
            float_format="%.4f",
        )
        print(f"[INFO] Saved per-category results: {per_category_path.resolve()}")


def generate_plots(success_df: pd.DataFrame, per_category_df: pd.DataFrame, predictions_dict: dict, category_labels):
    """
    Generate and save evaluation plots.
    """
    if success_df.empty:
        print("[INFO] Skipping plots: no successful model evaluations")
        return

    if SAVE_PLOTS:
        PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if "accuracy" in success_df.columns:
        plot_overall_performance(
            success_df,
            metric="accuracy",
            title_suffix="Accuracy",
            save_path=PLOT_OUTPUT_DIR / "overall_accuracy.png" if SAVE_PLOTS else None,
        )

    if "f1_weighted" in success_df.columns:
        plot_overall_performance(
            success_df,
            metric="f1_weighted",
            title_suffix="Weighted F1-Score",
            save_path=PLOT_OUTPUT_DIR / "overall_f1_weighted.png" if SAVE_PLOTS else None,
        )

    if "f1_macro" in success_df.columns:
        plot_overall_performance(
            success_df,
            metric="f1_macro",
            title_suffix="Macro F1-Score",
            save_path=PLOT_OUTPUT_DIR / "overall_f1_macro.png" if SAVE_PLOTS else None,
        )

    if not per_category_df.empty:
        plot_per_category_grouped_bar(
            per_category_df,
            metric="f1_score",
            save_path=PLOT_OUTPUT_DIR / "per_category_f1_grouped_bar.png" if SAVE_PLOTS else None,
        )
        plot_per_category_heatmap(
            per_category_df,
            metric="f1_score",
            save_path=PLOT_OUTPUT_DIR / "per_category_f1_heatmap.png" if SAVE_PLOTS else None,
        )

    for model_name in success_df.index:
        if model_name not in predictions_dict:
            continue

        preds = predictions_dict[model_name]
        cm = confusion_matrix(preds["y_true"], preds["y_pred"], labels=category_labels)

        plot_confusion_matrix_figure(
            cm,
            category_labels,
            model_name,
            save_path=PLOT_OUTPUT_DIR / f"confusion_matrix_{model_name}.png" if SAVE_PLOTS else None,
        )

    if not SAVE_PLOTS:
        plt.show()
    else:
        print(f"[INFO] Plots saved to: {PLOT_OUTPUT_DIR.resolve()}")
        plt.close("all")


# ------------------------------------------------------------------
# Main Evaluation Runner
# ------------------------------------------------------------------
def run_classification_evaluation():
    """
    Run full evaluation pipeline.
    """
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

    print(f"[INFO] Collected {len(all_df)} valid summary records")
    print(f"[INFO] Found {len(category_labels)} categories")

    try:
        train_ids, test_ids = build_train_test_split(all_df)
    except ValueError as e:
        print(f"[ERROR] Split failed: {e}")
        return

    results = {}
    predictions_dict = {}
    per_category_rows = []

    model_names = sorted(all_df["model"].unique())
    print(f"[INFO] Evaluating models: {model_names}")

    for model_name in tqdm(model_names, desc="Evaluating models"):
        model_df = all_df[all_df["model"] == model_name]

        try:
            model_results, model_per_category, model_predictions = evaluate_single_model(
                model_df,
                train_ids,
                test_ids,
                category_labels,
            )

            results[model_name] = model_results

            if model_predictions is not None:
                predictions_dict[model_name] = model_predictions

            if model_per_category:
                for row in model_per_category:
                    row["model"] = model_name
                    per_category_rows.append(row)

        except Exception as e:
            print(f"[WARNING] Evaluation failed for {model_name}: {e}")
            results[model_name] = {"error": str(e)}

    overall_results_df = pd.DataFrame.from_dict(results, orient="index")

    if overall_results_df.empty:
        print("[ERROR] No evaluation results produced")
        return

    success_mask = overall_results_df["error"].isna()
    success_df = overall_results_df[success_mask].copy()
    error_df = overall_results_df[~success_mask].copy()

    if "error" in success_df.columns:
        success_df.drop(columns=["error"], inplace=True)

    per_category_df = pd.DataFrame(per_category_rows)

    if not success_df.empty:
        display_cols = [col for col in ["accuracy", "f1_weighted", "f1_macro"] if col in success_df.columns]
        display_df = success_df[display_cols].copy()

        sort_col = "f1_weighted" if "f1_weighted" in display_df.columns else "accuracy"
        if sort_col in display_df.columns:
            display_df.sort_values(by=sort_col, ascending=False, inplace=True)

        print("\n[INFO] Successful evaluations:")
        print(display_df.to_markdown(floatfmt=".4f"))
    else:
        print("[INFO] No models were successfully evaluated")

    if not error_df.empty and "error" in error_df.columns:
        print("\n[INFO] Models with evaluation errors:")
        print(error_df[["error"]].to_markdown())

    successful_models = success_df.index.tolist() if not success_df.empty else []
    if not per_category_df.empty:
        per_category_df = per_category_df[per_category_df["model"].isin(successful_models)].copy()

    save_results(success_df, error_df, per_category_df)
    generate_plots(success_df, per_category_df, predictions_dict, category_labels)

    print(f"\n[INFO] Evaluation complete. Tables saved to: {RESULTS_OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    run_classification_evaluation()
