from datasets import load_dataset, Dataset, DatasetDict
from typing import Dict, Optional
import json
import os
from pathlib import Path
from .normalizer import TextNormalizer

class DatasetLoader:
    def __init__(self, config: dict, normalizer: TextNormalizer):
        self.config = config
        self.normalizer = normalizer
        self.dataset_limit = int(os.getenv('DATASET_LIMIT', '0'))
        self.train_limit = int(os.getenv('TRAIN_LIMIT', '0'))
        self.val_limit = int(os.getenv('VAL_LIMIT', '0'))
        self.test_limit = int(os.getenv('TEST_LIMIT', '0'))

    def load_squad_v2(self) -> DatasetDict:
        dataset = load_dataset('squad_v2')
        dataset = dataset.map(self._process_squad_example, remove_columns=['id', 'title'], num_proc=4)
        dataset = self._apply_limits(dataset)
        return dataset

    def _process_squad_example(self, example: Dict) -> Dict:
        context = self.normalizer.normalize(example['context'])
        question = self.normalizer.normalize(example['question'])
        if example['answers']['text']:
            answer = self.normalizer.normalize(example['answers']['text'][0])
            answer_start = example['answers']['answer_start'][0]
            unanswerable = False
        else:
            answer = ''
            answer_start = -1
            unanswerable = True
        return {'context': context, 'question': question, 'answer': answer, 'answer_start': answer_start, 'unanswerable': unanswerable, 'lang': 'en'}

    def load_ukrainian_dataset(self, file_path: Path) -> Dataset:
        examples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                example = self._process_ukrainian_example(data)
                if example:
                    examples.append(example)
        return Dataset.from_list(examples)

    def _process_ukrainian_example(self, data: Dict) -> Optional[Dict]:
        context = self.normalizer.normalize(data['context'])
        question = self.normalizer.normalize(data['question'])
        answer = self.normalizer.normalize(data.get('answer', ''))
        if not answer or not self.normalizer.verify_answer_span(context, answer):
            if 'gold_answer' in data:
                answer = self.normalizer.normalize(data['gold_answer'])
                if not self.normalizer.verify_answer_span(context, answer):
                    return None
            else:
                return None
        answer_start, answer_end = self.normalizer.find_answer_span(context, answer)
        return {'context': context, 'question': question, 'answer': answer, 'answer_start': answer_start, 'unanswerable': data.get('unanswerable', False), 'lang': 'ua'}

    def filter_by_length(self, dataset: Dataset, config: dict) -> Dataset:
        def length_filter(example):
            ctx_len = len(example['context'].split())
            q_len = len(example['question'].split())
            ans_len = len(example['answer'].split()) if example['answer'] else 0
            return (config['min_context_len'] <= ctx_len <= config['max_context_len'] and
                    config['min_question_len'] <= q_len <= config['max_question_len'] and
                    (example['unanswerable'] or config['min_answer_len'] <= ans_len <= config['max_answer_len']))
        return dataset.filter(length_filter, num_proc=4)

    def remove_duplicates(self, dataset: Dataset) -> Dataset:
        seen = set()
        indices_to_keep = []
        for idx, example in enumerate(dataset):
            key = (example['context'][:100], example['question'])
            if key not in seen:
                seen.add(key)
                indices_to_keep.append(idx)
        return dataset.select(indices_to_keep)

    def stratified_split(self, dataset: Dataset, train_ratio: float, val_ratio: float, seed: int) -> DatasetDict:
        total_size = len(dataset)

        # Handle very small datasets
        if total_size < 3:
            print(f"Warning: Dataset too small ({total_size} samples). Using all data for train, validation, and test.")
            return DatasetDict({
                'train': dataset,
                'validation': dataset,
                'test': dataset
            })

        # Calculate minimum viable split sizes
        min_train_size = max(1, int(total_size * train_ratio))
        min_val_size = max(1, int(total_size * val_ratio))
        min_test_size = max(1, total_size - min_train_size - min_val_size)

        # Adjust if total doesn't match
        if min_train_size + min_val_size + min_test_size != total_size:
            # Prioritize train, then val, then test
            remaining = total_size - min_train_size - min_val_size
            min_test_size = max(1, remaining)

        # If still can't split properly, use simple approach
        if min_train_size < 1 or min_val_size < 1 or min_test_size < 1:
            print(f"Warning: Cannot properly split {total_size} samples. Duplicating data across splits.")
            train_size = max(1, int(total_size * 0.8))
            dataset = dataset.train_test_split(test_size=total_size - train_size, seed=seed)
            return DatasetDict({
                'train': dataset['train'],
                'validation': dataset['test'],
                'test': dataset['test']
            })

        # Normal splitting
        dataset = dataset.train_test_split(test_size=1-train_ratio, seed=seed)
        val_test_ratio = val_ratio / (1 - train_ratio)

        # Check if we can do the second split
        test_size = len(dataset['test'])
        if test_size < 2:
            # Not enough samples to split further
            return DatasetDict({
                'train': dataset['train'],
                'validation': dataset['test'],
                'test': dataset['test']
            })

        val_test = dataset['test'].train_test_split(test_size=1-val_test_ratio, seed=seed)
        result = DatasetDict({
            'train': dataset['train'],
            'validation': val_test['train'],
            'test': val_test['test']
        })

        result = self._apply_limits(result)
        return result

    def _apply_limits(self, dataset: DatasetDict) -> DatasetDict:
        """Apply dataset size limits from environment variables."""
        limited_dataset = {}

        for split_name, split_data in dataset.items():
            # Determine which limit to use
            if self.dataset_limit > 0:
                # Global limit applies to all splits
                limit = self.dataset_limit
            elif split_name == 'train' and self.train_limit > 0:
                limit = self.train_limit
            elif split_name == 'validation' and self.val_limit > 0:
                limit = self.val_limit
            elif split_name == 'test' and self.test_limit > 0:
                limit = self.test_limit
            else:
                # No limit, use full dataset
                limited_dataset[split_name] = split_data
                continue

            # Apply the limit
            current_size = len(split_data)
            if limit < current_size:
                print(f"Limiting {split_name} split from {current_size} to {limit} examples")
                limited_dataset[split_name] = split_data.select(range(limit))
            else:
                limited_dataset[split_name] = split_data

        return DatasetDict(limited_dataset)
