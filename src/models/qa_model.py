import re
from typing import Dict, Tuple

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


class QAModel:
    def __init__(
        self, model_name: str = "deepset/xlm-roberta-large-squad2", device: str = "cuda"
    ):
        self.device = device

        print(f"QA Model using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def answer_question(self, question: str, context: str) -> Dict:
        return self.answer_question_batch([question], [context])[0]

    def answer_question_batch(
        self, questions: list[str], contexts: list[str]
    ) -> list[Dict]:
        if not questions:
            return []

        if len(contexts) == 1 and len(questions) > 1:
            contexts = contexts * len(questions)

        inputs = self.tokenizer(
            questions,
            contexts,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = []
        for i in range(len(questions)):
            answer_start = torch.argmax(outputs.start_logits[i])
            answer_end = torch.argmax(outputs.end_logits[i])
            start_score = outputs.start_logits[i, answer_start].item()
            end_score = outputs.end_logits[i, answer_end].item()
            confidence = (start_score + end_score) / 2

            if answer_end < answer_start:
                results.append(
                    {"answer": "", "confidence": 0.0, "start": -1, "end": -1}
                )
                continue

            answer_tokens = inputs["input_ids"][i][answer_start : answer_end + 1]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
            results.append(
                {
                    "answer": answer,
                    "confidence": confidence,
                    "start": answer_start.item(),
                    "end": answer_end.item(),
                }
            )
        return results

    def compute_em_f1(self, predicted: str, gold: str) -> Tuple[float, float]:
        pred_tokens = self._normalize_answer(predicted).split()
        gold_tokens = self._normalize_answer(gold).split()

        if len(pred_tokens) == 0 and len(gold_tokens) == 0:
            return 1.0, 1.0

        if len(pred_tokens) == 0 or len(gold_tokens) == 0:
            return 0.0, 0.0

        em = float(pred_tokens == gold_tokens)

        common = set(pred_tokens) & set(gold_tokens)
        num_common = len(common)

        if num_common == 0:
            return em, 0.0

        precision = num_common / len(pred_tokens)
        recall = num_common / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return em, f1

    def _normalize_answer(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"\b(a|an|the)\b", " ", text)

        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
