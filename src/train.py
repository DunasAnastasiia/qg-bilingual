import os
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from datasets import DatasetDict
from transformers import Seq2SeqTrainer

import wandb
from src.data.dataset_loader import DatasetLoader
from src.data.normalizer import TextNormalizer
from src.data.preprocessor import QGPreprocessor
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.visualizer import MetricsVisualizer
from src.models.qg_model import QGModel
from src.utils.config import Config
from src.utils.seed import set_seed


def compute_metrics(eval_preds, tokenizer, metrics_calc, config):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    lang = config.get("language", "en")
    rouge_metrics = metrics_calc.compute_rouge(decoded_preds, decoded_labels, lang=lang)

    return {
        "rouge-l": rouge_metrics["rouge-l"],
        "rouge-1": rouge_metrics["rouge-1"],
        "rouge-2": rouge_metrics["rouge-2"],
    }


def train(config_path: str, mode_override: str = None, dataset_percent: int = 100):
    config = Config(config_path)
    if mode_override:
        config.config["mode"] = mode_override
    set_seed(config.get("seed", 42))

    wandb.init(
        entity=os.getenv("WANDB_ENTITY", None),
        project="wh-question-generation",
        name=f"{config.get('model_name', 'model')}-{config.get('language', 'en')}-{config.get('mode', 'aware')}",
        config=config.config,
    )

    normalizer = TextNormalizer(language=config.get("language", "en"))
    dataset_loader = DatasetLoader(config.config, normalizer)

    if config.get("language", "en") == "en":
        agnostic = config.get("mode") == "answer_agnostic"
        dataset = dataset_loader.load_squad_v2(
            filter_unanswerable=agnostic, deduplicate_by_context=agnostic
        )
    else:
        dataset_path = Path(config.data_dir) / "ukrainian_qa.jsonl"
        agnostic = config.get("mode") == "answer_agnostic"
        raw_dataset = dataset_loader.load_ukrainian_dataset(
            dataset_path, filter_unanswerable=agnostic, deduplicate_by_context=agnostic
        )
        dataset = dataset_loader.stratified_split(
            raw_dataset,
            config["data"]["train_split"],
            config["data"]["val_split"],
            config.get("seed", 42),
        )

    dataset = DatasetDict(
        {
            split: dataset_loader.filter_by_length(dataset[split], config["data"])
            for split in dataset.keys()
        }
    )

    if dataset_percent < 100:
        print(f"\n{'='*60}")
        print(f"SUBSAMPLING DATASET TO {dataset_percent}%")
        print(f"{'='*60}")
        for split in dataset.keys():
            original_size = len(dataset[split])
            n_samples = max(1, int(original_size * dataset_percent / 100))
            dataset[split] = (
                dataset[split]
                .shuffle(seed=config.get("seed", 42))
                .select(range(n_samples))
            )
            print(
                f"{split:12} | {original_size:6} → {n_samples:6} samples ({dataset_percent}%)"
            )
        print(f"{'='*60}\n")

    for split_name, split_data in dataset.items():
        if len(split_data) == 0:
            print(f"ERROR: {split_name} split is empty after filtering!")
            print(
                f"Consider relaxing length constraints in config or increasing dataset size."
            )
            raise ValueError(f"{split_name} split has no samples after filtering")
        print(f"{split_name} split: {len(split_data)} samples")

    qg_model = QGModel(
        config.get("model_name", "t5-base"),
        config.config,
        device=config.get("device", "cuda"),
    )
    preprocessor = QGPreprocessor(
        qg_model.tokenizer,
        mode=config.get("mode", "answer_aware"),
        max_source_length=config["data"]["max_context_len"],
        max_target_length=config["data"]["max_question_len"],
    )

    tokenized_dataset = dataset.map(
        preprocessor.preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=4,
    )

    metrics_calc = MetricsCalculator()

    output_dir_from_config = config.get("training.output_dir", "./checkpoints/model")
    model_folder_name = Path(output_dir_from_config).name
    output_dir = config.checkpoint_dir / model_folder_name
    training_args = qg_model.get_training_args(str(output_dir))

    callbacks = []
    if config.get("training.early_stopping_patience"):
        from transformers import EarlyStoppingCallback

        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.get("training.early_stopping_patience")
            )
        )

    trainer = Seq2SeqTrainer(
        model=qg_model.model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=qg_model.get_data_collator(),
        compute_metrics=lambda eval_preds: compute_metrics(
            eval_preds,
            qg_model.tokenizer,
            None,
            dataset["validation"],
            metrics_calc,
            config.config,
        ),
        callbacks=callbacks,
    )

    trainer.train()

    if "test" in tokenized_dataset:
        test_results = trainer.evaluate(tokenized_dataset["test"])
        print(f"Test results: {test_results}")

    qg_model.save(str(output_dir / "final_model"))

    visualizer = MetricsVisualizer(output_dir / "visualizations")
    train_logs = trainer.state.log_history
    train_losses = [log["loss"] for log in train_logs if "loss" in log]
    val_losses = [log["eval_loss"] for log in train_logs if "eval_loss" in log]
    val_rouge_scores = [
        log["eval_rouge-l"] for log in train_logs if "eval_rouge-l" in log
    ]
    visualizer.plot_training_curves(train_losses, val_losses, val_rouge_scores)

    wandb.finish()
    print(f"Training complete. Model saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--mode",
        type=str,
        required=False,
        help="Override mode (answer_aware or answer_agnostic)",
    )
    parser.add_argument(
        "--dataset_percent",
        type=int,
        default=100,
        help="Percentage of dataset to use (1-100, default: 100)",
    )
    args = parser.parse_args()

    if args.dataset_percent < 1 or args.dataset_percent > 100:
        parser.error("--dataset_percent must be between 1 and 100")

    train(args.config, args.mode, args.dataset_percent)
