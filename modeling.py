"""
Summarization Models Module

Provides a unified interface for extractive and abstractive summarization
methods used in benchmarking pipelines.

Supports:
- Extractive: Luhn, LSA, LexRank, TextRank
- Abstractive: BART, PEGASUS, T5

Designed for reproducible experimentation and controlled evaluation.
"""

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer

# Transformer models
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:
    raise ImportError(
        "Transformers library not found. Install with: "
        "pip install transformers sentencepiece torch"
    )

import torch
import nltk
import traceback


# Ensure NLTK tokenizer is available
def _ensure_nltk_resources():
    try:
        nltk.data.find("tokenizers/punkt")
    except (LookupError, nltk.downloader.DownloadError):
        nltk.download("punkt", quiet=True)


_ensure_nltk_resources()


class SummarizationModels:
    """
    Wrapper class for extractive and abstractive summarization models.
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize models and load available summarizers.

        Args:
            device (str): 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        print(f"[INFO] Using device: {self.device}")

        # Extractive models
        self.extractive_models = {
            "luhn": LuhnSummarizer(),
            "lsa": LsaSummarizer(),
            "lexrank": LexRankSummarizer(),
            "textrank": TextRankSummarizer(),
        }

        # Abstractive models (loaded dynamically)
        self.abstractive_models = {}

        self._load_abstractive_models()

    def _load_abstractive_models(self):
        """Load pretrained abstractive summarization models."""
        models_to_load = {
            "bart": "facebook/bart-large-cnn",
            "pegasus": "google/pegasus-xsum",
            "t5": "t5-base",
        }

        for name, model_id in models_to_load.items():
            try:
                print(f"[INFO] Loading {name.upper()} ({model_id})...")
                self.abstractive_models[name] = {
                    "model": AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device),
                    "tokenizer": AutoTokenizer.from_pretrained(model_id),
                }
            except Exception as e:
                print(f"[WARNING] Failed to load {name}: {e}")

    # ------------------------------------------------------------------
    # Extractive Summarization
    # ------------------------------------------------------------------
    def extractive_summarize(self, text: str, model_name: str, sentences: int = 5) -> str:
        """
        Generate extractive summary.

        Args:
            text (str): Input document
            model_name (str): Extractive model key
            sentences (int): Number of sentences

        Returns:
            str: Summary text
        """
        if not text or model_name not in self.extractive_models:
            return ""

        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))

            if len(parser.document.sentences) == 0:
                return ""

            summarizer = self.extractive_models[model_name]
            summary = summarizer(parser.document, sentences)

            return " ".join(str(sentence) for sentence in summary)

        except Exception:
            print(f"[ERROR] Extractive summarization failed ({model_name})")
            print(traceback.format_exc())
            return ""

    # ------------------------------------------------------------------
    # Abstractive Summarization
    # ------------------------------------------------------------------
    def abstractive_summarize(
        self,
        text: str,
        model_name: str,
        input_max_length: int = 1024,
        summary_max_length: int = 150,
    ) -> str:
        """
        Generate abstractive summary.

        Args:
            text (str): Input document
            model_name (str): Model key
            input_max_length (int): Max input tokens
            summary_max_length (int): Max output tokens

        Returns:
            str: Summary text
        """
        if not text or model_name not in self.abstractive_models:
            return ""

        try:
            config = self.abstractive_models[model_name]

            # T5 requires task prefix
            input_text = f"summarize: {text}" if model_name.startswith("t5") else text

            inputs = config["tokenizer"](
                input_text,
                max_length=input_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            if inputs.input_ids.shape[1] == 0:
                return ""

            summary_ids = config["model"].generate(
                inputs["input_ids"],
                max_length=summary_max_length,
                min_length=max(10, int(summary_max_length * 0.1)),
                num_beams=4,
                early_stopping=True,
            )

            return config["tokenizer"].decode(
                summary_ids[0],
                skip_special_tokens=True,
            ).strip()

        except Exception as e:
            print(f"[ERROR] Abstractive summarization failed ({model_name})")

            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()

            print(traceback.format_exc())
            return ""

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def hybrid_summarize(self, text: str) -> str:
        """
        Placeholder for hybrid summarization strategy.
        """
        raise NotImplementedError("Hybrid summarization is not implemented.")

    def get_available_models(self):
        """
        Returns list of available model names.
        """
        return list(self.extractive_models.keys()) + list(self.abstractive_models.keys())
