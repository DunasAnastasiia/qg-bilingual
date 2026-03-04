from datasets import load_dataset, Dataset, DatasetDict
from typing import Dict, Optional
import json
from pathlib import Path
from .normalizer import TextNormalizer

class DatasetLoader:
    def __init__(self, config: dict, normalizer: TextNormalizer):
        self.config = config
        self.normalizer = normalizer

    def load_squad_v2(self) -> DatasetDict:
        dataset = load_dataset('squad_v2')
        dataset = dataset.map(self._process_squad_example, remove_columns=['id', 'title'], num_proc=4)
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
        dataset = dataset.train_test_split(test_size=1-train_ratio, seed=seed)
        val_test_ratio = val_ratio / (1 - train_ratio)
        val_test = dataset['test'].train_test_split(test_size=1-val_test_ratio, seed=seed)
        return DatasetDict({'train': dataset['train'], 'validation': val_test['train'], 'test': val_test['test']})
