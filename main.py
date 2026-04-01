"""
Main pipeline for recursive resume summarization.

This script:
1. Loads configuration
2. Finds supported resume files recursively
3. Converts resumes to text
4. Generates summaries with selected models
5. Saves summaries to model-specific output directories

Designed for reproducible benchmarking workflows.
"""

from pathlib import Path
import os

import yaml
from tqdm import tqdm

from preprocessing import ResumePreprocessor
from modeling import SummarizationModels


DEFAULT_CONFIG = {
    "raw_data_dir": "data/raw",
    "output_dir": "data/summaries",
    "models_to_run": ["luhn", "textrank"],
    "extractive_sentences": 3,
    "abstractive_max_length": 150,
    "abstractive_input_max_length": 1024,
    "device": "cpu",
    "preserve_subfolder_structure": True,
}


class Pipeline:
    """End-to-end pipeline for recursive summary generation."""

    def __init__(self, config_path: str = "config/params.yaml"):
        self.config = self._load_config(config_path)

        self.raw_dir = Path(self.config["raw_data_dir"])
        self.output_dir = Path(self.config["output_dir"])
        self.models_to_run = self.config.get("models_to_run", ["luhn"])
        self.preserve_structure = self.config.get("preserve_subfolder_structure", True)

        self.preprocessor = ResumePreprocessor()
        self.models = SummarizationModels(device=self.config.get("device", "cpu"))

        self._ensure_directories()
        self._print_setup_summary()

    def _load_config(self, config_path: str) -> dict:
        """
        Load YAML configuration. If unavailable, fall back to defaults.
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config if config else DEFAULT_CONFIG.copy()
        except FileNotFoundError:
            print(f"[WARNING] Configuration file not found: {config_path}")
            print("[INFO] Falling back to default configuration.")
            return DEFAULT_CONFIG.copy()
        except Exception as e:
            print(f"[ERROR] Failed to load configuration: {e}")
            raise

    def _ensure_directories(self) -> None:
        """Create required directories if they do not exist."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _print_setup_summary(self) -> None:
        """Print a short setup summary."""
        print(f"[INFO] Raw data directory: {self.raw_dir.resolve()}")
        print(f"[INFO] Output directory: {self.output_dir.resolve()}")

    def find_resume_files(self):
        """
        Recursively find supported resume files in the raw data directory.

        Returns:
            list[Path]: Sorted list of input file paths.
        """
        supported_extensions = [".pdf", ".docx", ".txt"]

        if not self.raw_dir.is_dir():
            print(f"[ERROR] Raw data directory not found: {self.raw_dir}")
            return []

        print(f"[INFO] Scanning {self.raw_dir} for supported files: {supported_extensions}")

        files = []
        for ext in supported_extensions:
            files.extend(self.raw_dir.glob(f"**/*{ext}"))

        unique_files = sorted(set(files))

        print(f"[INFO] Found {len(unique_files)} resume file(s).")
        if unique_files:
            preview_count = min(5, len(unique_files))
            print("[INFO] Example files:")
            for file_path in unique_files[:preview_count]:
                print(f"  - {file_path}")

        return unique_files

    def _build_output_path(self, file_path: Path, model_name: str) -> Path:
        """
        Build output path for a generated summary.

        Args:
            file_path (Path): Original resume file
            model_name (str): Summarization model name

        Returns:
            Path: Output file path
        """
        model_output_dir = self.output_dir / model_name
        relative_path = file_path.relative_to(self.raw_dir)

        if self.preserve_structure:
            output_sub_dir = model_output_dir / relative_path.parent
            summary_filename = f"{relative_path.stem}_summary.txt"
        else:
            output_sub_dir = model_output_dir
            parent_prefix = str(relative_path.parent).replace(os.sep, "_")
            if parent_prefix and parent_prefix != ".":
                summary_filename = f"{parent_prefix}_{relative_path.stem}_summary.txt"
            else:
                summary_filename = f"{relative_path.stem}_summary.txt"

        output_sub_dir.mkdir(parents=True, exist_ok=True)
        return output_sub_dir / summary_filename

    def _summarize_text(self, text: str, model_name: str) -> str:
        """
        Route input text to the appropriate summarization method.
        """
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

        raise ValueError(f"Unknown or unavailable model: {model_name}")

    def _save_summary(self, summary: str, output_path: Path) -> None:
        """
        Save generated summary to disk.
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary)

    def run(self) -> None:
        """Run the recursive summarization pipeline."""
        resume_files = self.find_resume_files()
        if not resume_files:
            print("[INFO] No input files found. Exiting.")
            return

        available_models = self.models.get_available_models()
        models_to_process = [m for m in self.models_to_run if m in available_models]

        if not models_to_process:
            print("[ERROR] None of the requested models are available.")
            return

        print(f"[INFO] Models selected for processing: {models_to_process}")

        for model_name in tqdm(models_to_process, desc="Models"):
            print(f"[INFO] Processing with model: {model_name}")

            for file_path in tqdm(resume_files, desc=f"{model_name}", leave=False):
                raw_text = self.preprocessor.convert_to_text(str(file_path))
                if not raw_text:
                    continue

                output_path = self._build_output_path(file_path, model_name)

                try:
                    summary = self._summarize_text(raw_text, model_name)

                    if not summary:
                        print(f"[WARNING] Empty summary generated for {file_path.name} with '{model_name}'")

                    self._save_summary(summary, output_path)

                except Exception as e:
                    print(f"[WARNING] Failed on {file_path.name} with model '{model_name}': {e}")
                    continue

        print(f"[INFO] Pipeline complete. Summaries saved to: {self.output_dir.resolve()}")


if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run()
