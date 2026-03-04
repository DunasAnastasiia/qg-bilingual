# Question Generation Training

Bilingual (EN/UA) question generation using T5, BART, and mT5 models.

## Setup

1. Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```

2. Build and run training:
```bash
docker-compose up --build
```

## Training

Train all 6 models (T5, BART, mT5 for EN/UA with aware/agnostic modes):
```bash
docker-compose up trainer
```

Or train a single model:
```bash
python3 src/train.py --config configs/train_t5_base.yaml
```

## Models

- T5-base (English, answer-aware/agnostic)
- BART-base (English, answer-aware/agnostic)
- mT5-base (Ukrainian, answer-aware/agnostic)

Checkpoints saved to `./checkpoints/`

## Config Files

- `configs/train_t5_base.yaml` - T5 configuration
- `configs/train_bart_base_aware_en.yaml` - BART aware configuration
- `configs/train_bart_base_agnostic_en.yaml` - BART agnostic configuration
- `configs/train_mt5_base_aware_ua.yaml` - mT5 configuration

## Metrics

After training, each model produces:
- ROUGE-L, BLEU scores
- QA-F1, QA-EM metrics
- Pass-rate statistics
- Training logs in WandB (if configured)
