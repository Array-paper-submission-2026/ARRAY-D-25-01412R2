"""
Main pipeline for resume summarization with efficiency tracking.

This script:
1. Loads configuration
2. Finds resume files recursively
3. Generates summaries using selected models
4. Saves outputs
5. Reports runtime efficiency per model

Designed for reproducible benchmarking and performance analysis.
"""

from pathlib import Path
import os
import time

import pandas as pd
import yaml
from tqdm import tqdm

from preprocessing import ResumePreprocessor
from modeling_v2 import SummarizationModels


DEFAULT_CONFIG = {
    "raw_data_dir": "data/raw",
    "output_dir": "data/summaries",
    "results_output_dir": "evaluation_results",
    "models_to_run": ["luhn", "textrank"],
    "extractive_sentences": 3,
    "abstractive_max_length": 150,
    "abstractive_input_max_length": 1024,
    "device": "cpu",
    "preserve_subfolder_structure": True,
}


class Pipeline:
    """End-to-end pipeline with summarization and efficiency reporting."""

    def __init__(self, config_path: str = "config/params.yaml"):
        self.config = self._load_config(config_path)

        self.raw_dir = Path(self.config["raw_data_dir"])
        self.output_dir = Path(self.config["output_dir"])
        self.results_dir = Path(self.config["results_output_dir"])

        self.models_to_run = self.config.get("models_to_run", ["luhn"])
        self.preserve_structure = self.config.get("preserve_subfolder_structure", True)

        self.preprocessor = ResumePreprocessor()
        self.models = SummarizationModels(device=self.config.get("device", "cpu"))

        self._ensure_directories()
        self._print_setup()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config if config else DEFAULT_CONFIG.copy()
        except FileNotFoundError:
            print(f"[WARNING] Config not found: {config_path}")
            print("[INFO] Using default configuration.")
            return DEFAULT_CONFIG.copy()
        except Exception as e:
            print(f"[ERROR] Failed to load config: {e}")
            raise

    def _ensure_directories(self):
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _print_setup(self):
        print(f"[INFO] Raw data: {self.raw_dir.resolve()}")
        print(f"[INFO] Output dir: {self.output_dir.resolve()}")
        print(f"[INFO] Results dir: {self.results_dir.resolve()}")

    # ------------------------------------------------------------------
    # File Discovery
    # ------------------------------------------------------------------
    def find_resume_files(self):
        supported_extensions = [".pdf", ".docx", ".txt"]

        if not self.raw_dir.is_dir():
            print(f"[ERROR] Raw directory not found: {self.raw_dir}")
            return []

        print(f"[INFO] Scanning for files in {self.raw_dir}")

        files = []
        for ext in supported_extensions:
            files.extend(self.raw_dir.glob(f"**/*{ext}"))

        files = sorted(set(files))

        print(f"[INFO] Found {len(files)} files")
        for f in files[: min(5, len(files))]:
            print(f"  - {f}")

        return files

    # ------------------------------------------------------------------
    # Core Helpers
    # ------------------------------------------------------------------
    def _build_output_path(self, file_path: Path, model_name: str) -> Path:
        model_dir = self.output_dir / model_name
        relative_path = file_path.relative_to(self.raw_dir)

        if self.preserve_structure:
            output_dir = model_dir / relative_path.parent
            filename = f"{relative_path.stem}_summary.txt"
        else:
            output_dir = model_dir
            prefix = str(relative_path.parent).replace(os.sep, "_")
            filename = (
                f"{prefix}_{relative_path.stem}_summary.txt"
                if prefix and prefix != "."
                else f"{relative_path.stem}_summary.txt"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / filename

    def _summarize(self, text: str, model_name: str) -> str:
        if model_name in self.models.extractive_models:
            return self.models.extractive_summarize(
                text,
                model_name,
                sentences=self.config.get("extractive_sentences", 3),
            )

        if model_name in self.models.abstractive_models:
            return self.models.abstractive_summarize(
                text,
                model_name,
                input_max_length=self.config.get("abstractive_input_max_length", 1024),
                summary_max_length=self.config.get("abstractive_max_length", 150),
            )

        raise ValueError(f"Unknown model: {model_name}")

    def _save_summary(self, summary: str, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(summary)

    # ------------------------------------------------------------------
    # Efficiency Reporting
    # ------------------------------------------------------------------
    def _build_efficiency_table(self, timings, counts):
        rows = []

        for model, durations in timings.items():
            count = counts.get(model, 0)
            model_type = (
                "abstractive" if model in self.models.abstractive_models else "extractive"
            )

            if count > 0:
                total = sum(durations)
                avg = total / count
                min_t = min(durations)
                max_t = max(durations)
            else:
                total = 0.0
                avg = float("nan")
                min_t = float("nan")
                max_t = float("nan")

            rows.append(
                {
                    "Model": model,
                    "Type": model_type,
                    "Files Processed": count,
                    "Total Time (s)": total,
                    "Avg Time/File (s)": avg,
                    "Min Time (s)": min_t,
                    "Max Time (s)": max_t,
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            df.sort_values(by="Avg Time/File (s)", inplace=True)
        return df

    # ------------------------------------------------------------------
    # Main Execution
    # ------------------------------------------------------------------
    def run(self):
        files = self.find_resume_files()
        if not files:
            print("[INFO] No files found. Exiting.")
            return

        available_models = self.models.get_available_models()
        models = [m for m in self.models_to_run if m in available_models]

        if not models:
            print("[ERROR] No valid models available.")
            return

        print(f"[INFO] Running models: {models}")

        timings = {m: [] for m in models}
        counts = {m: 0 for m in models}

        for model_name in tqdm(models, desc="Models"):
            print(f"[INFO] Processing with {model_name}")

            for file_path in tqdm(files, desc=model_name, leave=False):
                text = self.preprocessor.convert_to_text(str(file_path))
                if not text:
                    continue

                output_path = self._build_output_path(file_path, model_name)

                start = time.time()
                try:
                    summary = self._summarize(text, model_name)
                    elapsed = time.time() - start

                    timings[model_name].append(elapsed)
                    counts[model_name] += 1

                    self._save_summary(summary, output_path)

                except Exception as e:
                    print(f"[WARNING] Failed: {file_path.name} ({model_name}) → {e}")
                    continue

        print(f"[INFO] Summaries saved to: {self.output_dir.resolve()}")

        # Efficiency report
        df = self._build_efficiency_table(timings, counts)

        if df.empty:
            print("[INFO] No timing data collected.")
            return

        print("\n[INFO] Efficiency Summary:")
        print(df.to_markdown(index=False, floatfmt=".3f"))

        output_file = self.results_dir / "efficiency_summary.csv"
        try:
            df.to_csv(output_file, index=False, float_format="%.3f")
            print(f"[INFO] Saved: {output_file.resolve()}")
        except Exception as e:
            print(f"[WARNING] Could not save efficiency results: {e}")

        print("\n[INFO] Notes:")
        print("- Times reflect summarization step only")
        print("- Excludes preprocessing, I/O, and model loading")
        print("- Hardware differences (CPU/GPU) affect results")


if __name__ == "__main__":
    Pipeline().run()
