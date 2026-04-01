# ARRAY-D-25-01412R2
Anonymous code and supplementary materials for double-blind review.

Resume Summarization Benchmark for Downstream Classification
A reproducible pipeline for benchmarking extractive vs. abstractive summarization on resumes, evaluated via downstream classification utility rather than surface-level text quality.

🚀 Key Idea
Instead of evaluating summaries using ROUGE or fluency metrics, this project measures:
How well summaries preserve information needed for decision-making tasks (e.g., job-category classification).

🧩 Features
📄 Resume ingestion from PDF, DOCX, TXT
🧹 Text cleaning and section-aware preprocessing
✂️ Multiple extractive summarizers
🤖 Multiple abstractive transformer models
📊 Extrinsic evaluation via classification
⚡ Efficiency benchmarking (runtime per model)
📈 Automatic plotting and result export


📁 Project Structure
. 
├── preprocessing.py              # Resume parsing, cleaning, segmentation 
├── modeling.py                  # Core summarization models 
├── modeling_v2.py               # Extended models (more baselines) 
├── main.py                      # Pipeline + timing analysis 
├── main_v2.py                   # Recursive pipeline variant 
├── evaluate_results_classification_idea_with_figures_v3.py        # Classification-based evaluation 
├── requirements.txt             # Dependencies 
├── config/ 
│   └── params.yaml              # Configuration file 
├── data/ 
│   ├── raw/                     # Input resumes 
│   └── summaries/               # Generated summaries 
└── evaluation_results/          # Outputs (metrics, plots) 


🧠 Supported Models
✂️ Extractive
Luhn
LSA
LexRank
TextRank
KL (v2)
Reduction (v2)
Random (v2)


🤖 Abstractive
BART (facebook/bart-large-cnn)
PEGASUS (google/pegasus-xsum)
T5 (base, small, large)
FLAN-T5


⚙️ Installation
git clone <anonymous-repo-url>
cd <repo>
pip install -r requirements.txt


📦 Main dependencies include:
transformers, torch
sumy
scikit-learn
spacy, nltk
pandas, matplotlib, seaborn


📥 Additional Setup
Some resources are downloaded automatically on first run:
spaCy model: en_core_web_lg
NLTK tokenizer: punkt


⚙️ Configuration

Edit:
config/params.yaml

Example:

raw_data_dir: data/raw
output_dir: data/summaries
results_output_dir: evaluation_results

models_to_run:
  - luhn
  - textrank
  - bart

extractive_sentences: 3
abstractive_input_max_length: 1024
abstractive_max_length: 150

device: cpu
preserve_subfolder_structure: true

▶️ Running the Pipeline

1. Generate Summaries
python main.py

or:
python main_v2.py

✔️ What happens:
Recursively loads resumes
Converts to text
Runs selected summarization models
Saves summaries per model

2. Run Evaluation
python evaluate_results_classification_idea_with_figures_v3.py

✔️ Outputs:
Accuracy, F1 (macro & weighted)
Per-category performance
Confusion matrices
Plots and CSV files


📊 Evaluation Methodology
Summaries are evaluated via:

TF-IDF + Logistic Regression classifier

This serves as a controlled proxy for measuring:
Information preservation
Feature-level distortion
Task-specific utility

📈 Output Examples
Saved Results
evaluation_results/
├── classification_overall_success.csv
├── classification_per_category.csv
└── efficiency_summary.csv

Generated Plots
Overall accuracy / F1
Per-category performance (bar + heatmap)
Confusion matrices


⚡ Efficiency Reporting
The pipeline reports:

⏱ Total runtime per model
⚖️ Average time per resume
📉 Min / Max processing time

⚠️ Note: Timing includes summarization only, excluding preprocessing and I/O.

🧪 Design Philosophy

This project emphasizes:
Extrinsic evaluation over intrinsic metrics
Task utility over fluency
Lightweight baselines vs heavy models
Reproducibility and transparency


⚠️ Limitations
Resume segmentation is rule-based
Abstractive models may:
hallucinate
omit critical keywords
Performance depends on:
dataset structure
domain mismatch
Evaluation is limited to classification downstream task


🔁 Reproducibility
For consistent results, fix:
Random seed (RANDOM_STATE = 42)
Dataset splits
Hardware (CPU/GPU)
Model versions
Config parameters

⚖️ Ethics & Use
This system is intended for research purposes only.

Automated resume processing may introduce:
bias amplification
loss of critical candidate information
unfair filtering risks

Use responsibly in high-stakes applications.

📜 License
To be added upon de-anonymization.

📌 Citation
A citation will be provided after the review process.

🔒 Anonymity Notice
This repository is shared in anonymous form for peer review.
No author identities included
No institutional references
Dataset access details omitted

These will be added after acceptance.
