# Standard library imports
import sys
import os
from pathlib import Path

# Add project root and src to sys.path before internal imports
# This ensures imports work regardless of how/where the script is launched
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
if str(project_root) not in sys.path:
    sys.path.insert(1, str(project_root))

# Third-party imports
import torch
import numpy as np
from transformers import Seq2SeqTrainer
from datasets import DatasetDict
import wandb

from utils.config import Config
from utils.seed import set_seed
from data.dataset_loader import DatasetLoader
from data.normalizer import TextNormalizer
from data.preprocessor import QGPreprocessor
from models.qg_model import QGModel
from models.qa_model import QAModel
from evaluation.metrics import MetricsCalculator
from evaluation.visualizer import MetricsVisualizer

def compute_metrics(eval_preds, tokenizer, qa_model, eval_dataset, metrics_calc, config):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    contexts = [ex['context'] for ex in eval_dataset]
    gold_answers = [ex['answer'] for ex in eval_dataset]

    metrics = metrics_calc.compute_all_metrics(
        decoded_preds, decoded_labels, contexts, gold_answers, qa_model,
        lang=config.get('language', 'en'), config=config['evaluation']
    )

    return metrics

def train(config_path: str, mode_override: str = None):
    config = Config(config_path)
    if mode_override:
        config.config['mode'] = mode_override
    set_seed(config['seed'])

    wandb.init(
        entity=os.getenv('WANDB_ENTITY', None),
        project='wh-question-generation',
        name=f"{config['model_name']}-{config['language']}-{config['mode']}",
        config=config.config
    )

    normalizer = TextNormalizer(language=config['language'])
    dataset_loader = DatasetLoader(config.config, normalizer)

    if config['language'] == 'en':
        dataset = dataset_loader.load_squad_v2()
    else:
        dataset_path = Path(config.data_dir) / 'ukrainian_qa.jsonl'
        raw_dataset = dataset_loader.load_ukrainian_dataset(dataset_path)
        dataset = dataset_loader.stratified_split(
            raw_dataset, config['data']['train_split'],
            config['data']['val_split'], config['seed']
        )

    dataset = DatasetDict({
        split: dataset_loader.filter_by_length(dataset[split], config['data'])
        for split in dataset.keys()
    })

    # Check if filtering left us with any data
    for split_name, split_data in dataset.items():
        if len(split_data) == 0:
            print(f"ERROR: {split_name} split is empty after filtering!")
            print(f"Consider relaxing length constraints in config or increasing dataset size.")
            raise ValueError(f"{split_name} split has no samples after filtering")
        print(f"{split_name} split: {len(split_data)} samples")

    qg_model = QGModel(config['model_name'], config.config, device=config.get('device', 'cuda'))
    preprocessor = QGPreprocessor(
        qg_model.tokenizer, mode=config['mode'],
        max_source_length=config['data']['max_context_len'],
        max_target_length=config['data']['max_question_len']
    )

    tokenized_dataset = dataset.map(
        preprocessor.preprocess_function, batched=True,
        remove_columns=dataset['train'].column_names, num_proc=4
    )

    qa_model = QAModel(device=config.get('device', 'cuda'))
    metrics_calc = MetricsCalculator()

    output_dir = config.checkpoint_dir / config['training']['output_dir'].split('/')[-1]
    training_args = qg_model.get_training_args(str(output_dir))

    trainer = Seq2SeqTrainer(
        model=qg_model.model, args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        data_collator=qg_model.get_data_collator(),
        compute_metrics=lambda eval_preds: compute_metrics(
            eval_preds, qg_model.tokenizer, qa_model,
            dataset['validation'], metrics_calc, config.config
        )
    )

    trainer.train()

    if 'test' in tokenized_dataset:
        test_results = trainer.evaluate(tokenized_dataset['test'])
        print(f"Test results: {test_results}")

    qg_model.save(str(output_dir / 'final_model'))

    visualizer = MetricsVisualizer(output_dir / 'visualizations')
    train_logs = trainer.state.log_history
    train_losses = [log['loss'] for log in train_logs if 'loss' in log]
    val_losses = [log['eval_loss'] for log in train_logs if 'eval_loss' in log]
    val_rouge_scores = [log['eval_rouge-l'] for log in train_logs if 'eval_rouge-l' in log]
    visualizer.plot_training_curves(train_losses, val_losses, val_rouge_scores)

    wandb.finish()
    print(f"Training complete. Model saved to {output_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--mode', type=str, required=False)
    args = parser.parse_args()
    train(args.config, args.mode)
