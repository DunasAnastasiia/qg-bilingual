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
            context = examples['context'][i]
            # Heuristic: if context has many Cyrillic characters, use Ukrainian prefixes
            cyrillic_chars = sum(1 for c in context if 'а' <= c.lower() <= 'я' or c.lower() in 'ґєії')
            is_ua = cyrillic_chars > 10 # reasonable threshold to avoid random matches
            
            if is_ua:
                q_prefix = "генерувати питання"
                c_prefix = "контекст"
                a_prefix = "відповідь"
            else:
                q_prefix = "generate question"
                c_prefix = "context"
                a_prefix = "answer"

            if self.mode == 'answer_aware':
                inputs.append(f"{q_prefix}: {c_prefix}: {context} {a_prefix}: {examples['answer'][i]}")
            else:
                inputs.append(f"{q_prefix}: {c_prefix}: {context}")
        model_inputs = self.tokenizer(inputs, max_length=self.max_source_length, truncation=True)
        labels = self.tokenizer(examples['question'], max_length=self.max_target_length, truncation=True)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
