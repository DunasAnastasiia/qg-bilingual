from transformers import PreTrainedTokenizer
from typing import Dict

class QGPreprocessor:
    def __init__(self, tokenizer: PreTrainedTokenizer, mode: str = 'answer_aware', max_source_length: int = 512, max_target_length: int = 48):
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def preprocess_function(self, examples: Dict) -> Dict:
        inputs = []
        for i in range(len(examples['context'])):
            if self.mode == 'answer_aware':
                inputs.append(f"generate question: context: {examples['context'][i]} answer: {examples['answer'][i]}")
            else:
                inputs.append(f"generate question: context: {examples['context'][i]}")
        model_inputs = self.tokenizer(inputs, max_length=self.max_source_length, truncation=True)
        labels = self.tokenizer(examples['question'], max_length=self.max_target_length, truncation=True)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
