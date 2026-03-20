# 🎓 WH-Question Generation from English and Ukrainian Texts

**Diploma Project**: Generating WH-questions using Transformer Language Models (T5, BART, mT5)  
**Institution**: National University "Lviv Polytechnic"  
**Author**: Anastasiia DUNAS (Group SHI-42)  
**Supervisor**: Oleksiy SHAMURATOV  
**Year**: 2025

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.30+-yellow.svg)](https://huggingface.co/transformers/)

## 📖 Overview

This project implements a bilingual (English/Ukrainian) system for automatic WH-question generation (QG) using state-of-the-art transformer models. The research focuses on the comparative analysis of **Answer-Aware** and **Answer-Agnostic** generation modes, ensuring high factual accuracy and grammatical correctness.

### 🎯 Research Goals
- **ROUGE-L Improvement**: Achieve ≥ 5% gain compared to baseline configurations.
- **QG→QA Success Rate**: Increase the "answerability" of generated questions by ≥ 5 percentage points.
- **Bilingual Portability**: Validate the transferability of QG methods between English and Ukrainian corpora.

### 🛠 Methodology
The system supports two primary generation modes:
1. **Answer-Aware**: Generates questions targeting a specific answer span highlighted in the context.
2. **Answer-Agnostic**: Generates questions based solely on the context, automatically identifying significant facts.

### 🌟 Key Features

- ✅ **Bilingual Support**: Full pipeline for English (SQuAD 2.0) and Ukrainian datasets.
- ✅ **Multiple Architectures**: Optimized T5, BART, and mT5 models.
- ✅ **PEFT/LoRA Integration**: Efficient fine-tuning using Parameter-Efficient Fine-Tuning.
- ✅ **QG→QA Validation**: Integrated pipeline that verifies if a generated question can be correctly answered by a separate QA model (ROBerta/XLM-R).
- ✅ **Comprehensive Metrics**: Automatic evaluation using ROUGE (1/2/L), BLEU, BERTScore, EM, F1, and QG→QA Pass-Rate.
- ✅ **WH-Type Analysis**: Automatic classification and distribution analysis of question types (Who, What, Where, When, Why, How).
- ✅ **Interactive UI**: Gradio-based web interface for real-time testing and visualization.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 16GB+ RAM
- 50GB+ disk space for datasets and models

### Installation

#### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd diploma

# Build Docker image
docker-compose build
```

#### Option 2: Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd diploma

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Docker Usage (Recommended - with CUDA GPU)

```bash
# 1. Download dataset
docker compose --profile download up download-production

# 2. List available models
docker compose --profile utils run --rm models

# 3. Train specific model on 20% dataset (fast test)
MODEL_NAME=t5_base_en_agnostic DATASET_PERCENT=20 docker compose --profile train up train

# 4. Train on full dataset (production)
MODEL_NAME=t5_base_en_agnostic DATASET_PERCENT=100 docker compose --profile train up train

# 5. Run UI
docker compose up ui
# Access at http://localhost:7860
```

#### Local Usage

```bash
# 1. Download dataset
python main.py download-production

# 2. List available models
python main.py models

# 3. Train specific model on 20% dataset (fast test)
python main.py train --model t5_base_en_agnostic --dataset 20

# 4. Train on full dataset (production)
python main.py train --model t5_base_en_agnostic --dataset 100

# 5. Run UI
python main.py ui
```

## 📋 New Flexible Training Commands

### Train Specific Model

Train any model on any percentage of the dataset:

```bash
# General format
python main.py train --model <MODEL_NAME> --dataset <PERCENT>

# Examples
python main.py train --model t5_base_en_agnostic --dataset 20
python main.py train --model mt5_base_ua_agnostic --dataset 100
```

### Available Models

| Model Name | Architecture | Language | Mode | Status |
|------------|-------------|----------|------|--------|
| `t5_base_en_aware` | T5 Base | English | Answer-Aware | ✅ Trained |
| `t5_base_en_agnostic` | T5 Base | English | Answer-Agnostic | ✅ Trained |
| `bart_base_en_aware` | BART Base | English | Answer-Aware | ✅ Trained |
| `bart_base_en_agnostic` | BART Base | English | Answer-Agnostic | ✅ Trained |
| `mt5_base_ua_aware` | mT5 Base | Ukrainian | Answer-Aware | ✅ Trained |
| `mt5_base_ua_agnostic` | mT5 Base | Ukrainian | Answer-Agnostic | ✅ Trained |

### List Models

```bash
python main.py models
```

### Dataset Options

- **5-10%**: Quick testing (15-30 min)
- **20%**: Fast experimentation (1-2 hours)
- **50%**: Medium training (3-4 hours)
- **100%**: Full production training (6-8 hours per model)

**Note**: Full training can take 12-24 hours depending on hardware.

### 5. Run UI

Launches the interactive web interface:

```bash
python main.py run_ui
```

Then open http://localhost:7860 in your browser.

### 6. Check Status

View current state of datasets and models:

```bash
python main.py status
```

## 🎯 Project Structure

```
diploma/
├── main.py                      # Main CLI interface
├── src/
│   ├── data/
│   │   ├── prepare_datasets.py  # Dataset download and preparation
│   │   ├── dataset_loader.py    # Dataset loading utilities
│   │   ├── normalizer.py        # Text normalization
│   │   └── preprocessor.py      # Data preprocessing
│   ├── models/
│   │   ├── qg_model.py          # Question generation model
│   │   └── qa_model.py          # Question answering model (for validation)
│   ├── evaluation/
│   │   ├── metrics.py           # Evaluation metrics
│   │   └── visualizer.py        # Results visualization
│   ├── utils/
│   │   ├── config.py            # Configuration management
│   │   └── seed.py              # Random seed utilities
│   ├── train.py                 # Training script
│   └── ui.py                    # Gradio UI
├── configs/
│   ├── models/                  # Model configurations
│   │   ├── t5_base_en_aware.yaml
│   │   ├── bart_base_en_aware.yaml
│   │   └── mt5_base_ua_aware.yaml
│   └── train_*.yaml             # Training configurations
├── data/                        # Datasets (auto-created)
│   ├── squad_v2/               # English dataset
│   └── ukrainian_qa.jsonl      # Ukrainian dataset
├── checkpoints/                 # Trained models (auto-created)
│   ├── t5_base_en_aware/
│   ├── bart_base_en_aware/
│   └── mt5_base_ua_aware/
└── README.md
```

## 🏗 System Architecture

The system is designed with a three-layer architecture for robust question generation and validation:

1.  **Generation Layer**: Powered by PyTorch and Hugging Face Transformers. Utilizes **LoRA (Low-Rank Adaptation)** and **PEFT (Parameter-Efficient Fine-Tuning)** for efficient model adaptation on bilingual datasets (EN/UA). Supports T5, BART, and mT5 architectures.
2.  **Validation Layer**: Implements a strict **QG→QA pipeline**. Each generated question is passed to a secondary, SQuAD-compatible QA model (e.g., `deepset/roberta-base-squad2` for EN or `xlm-roberta-large-squad2` for UA/Bilingual). The question is considered valid only if the QA model can recover the original answer from the context with high confidence (F1 ≥ 0.8, Confidence ≥ 0.35).
3.  **Visualization & Logging Layer**: Provides structured JSON logging of all experiments, including metrics, WH-type distributions, and generation examples. Results are visualized through a Gradio-based dashboard and interactive UI.

## 📊 Results & Metrics

The system evaluates generated questions using a comprehensive suite of lexical and semantic metrics:

### 📈 Goal Metrics
| Metric | Goal Target | Purpose |
| :--- | :--- | :--- |
| **ROUGE-L** | +5% Improvement | Lexical similarity & structural correctness |
| **QG→QA Pass-Rate** | +5 p.p. Increase | Factual accuracy & answerability |
| **BERTScore** | ≥ 0.85 | Semantic alignment with human references |

### 🔍 Evaluation Metrics
- **ROUGE-1/2/L**: Measures n-gram overlap with ground-truth questions.
- **BLEU**: Evaluates precision of generated n-grams.
- **BERTScore**: Uses contextual embeddings to measure semantic similarity.
- **EM (Exact Match)** & **F1 Score**: Standard QA metrics used to validate the QG output.
- **Pass-Rate**: The percentage of generated questions that pass the QG→QA validation threshold.
- **WH-Type Distribution**: Analysis of question variety (What, Who, Where, etc.).

## 🔬 Technical Details

### 🤖 Models & Architectures
- **T5 (Text-to-Text Transfer Transformer)**: Unified seq2seq approach, excellent for English QG.
- **BART (Bidirectional and Auto-Regressive Transformers)**: Denoising pre-training, effective for summarization-like QG.
- **mT5 (Multilingual T5)**: Specifically tuned for Ukrainian and cross-lingual tasks.
- **LoRA/PEFT**: Used to reduce memory footprint and training time while maintaining high quality.

### ⚙️ Training & Inference
- **Max Input Length**: 512 context tokens.
- **Max Question Length**: 48-64 tokens (generation limit).
- **Beam Search**: num_beams=5, early_stopping=True.
- **Length Penalty**: 1.0 - 1.2 (to control question verbosity).
- **No Repeat N-gram**: 3 (to avoid repetitive phrases).
- **Batch Size**: 8-32 (with gradient accumulation).
- **Precision**: FP16/BF16 mixed-precision (on CUDA).
- **Learning Rate**: 3e-5 to 5e-5 (Cosine decay).

## 📚 Datasets

### English (SQuAD 2.0)

- Source: Stanford Question Answering Dataset v2.0
- Size: ~150K question-answer pairs
- License: CC BY-SA 4.0
- Reference: Rajpurkar et al. (2018)

### Ukrainian

- Source: Custom curated dataset
- Size: ~5K examples (production), ~20 examples (demo)
- Topics: History, geography, culture, science
- Format: SQuAD-compatible JSONL

## 🤝 Contributing

This is a diploma project. For questions or suggestions, please contact the author.

## 📄 License

This project is part of academic work at Lviv Polytechnic National University.

## 👤 Author

**Anastasiia Dunas**
- Student Group: ШІ-42
- University: Lviv Polytechnic National University
- Supervisor: Oleksii Shamuratov

## 🙏 Acknowledgments

- Hugging Face for Transformers library
- Stanford NLP for SQuAD dataset
- Gradio team for the UI framework

## 📖 References

1. T5: Raffel et al. (2020) - "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
2. BART: Lewis et al. (2020) - "BART: Denoising Sequence-to-Sequence Pre-training"
3. mT5: Xue et al. (2021) - "mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer"
4. SQuAD 2.0: Rajpurkar et al. (2018) - "Know What You Don't Know: Unanswerable Questions for SQuAD"
5. ROUGE: Lin (2004) - "ROUGE: A Package for Automatic Evaluation of Summaries"

---

**Built with** ❤️ **using PyTorch, Hugging Face Transformers, and Gradio**
