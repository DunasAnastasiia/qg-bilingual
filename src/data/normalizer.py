import re


class TextNormalizer:
    def __init__(self, language: str = "en"):
        self.language = language

    def normalize(self, text: str) -> str:
        if text is None:
            return ""
        text = str(text)
        text = re.sub(r'[""‟„]', '"', text)
        text = re.sub(r"[''‛‚]", "'", text)
        text = re.sub(r"[\u2018\u2019]", "'", text)
        text = re.sub(r"[–—―]", "-", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\u00a0", " ", text)
        text = re.sub(r"(\d),(\d)", r"\1\2", text)
        return text.strip()

    def verify_answer_span(self, context: str, answer: str) -> bool:
        norm_context = self.normalize(context.lower())
        norm_answer = self.normalize(answer.lower())
        return norm_answer in norm_context

    def find_answer_span(self, context: str, answer: str) -> tuple:
        norm_context = self.normalize(context)
        norm_answer = self.normalize(answer)
        start_idx = norm_context.lower().find(norm_answer.lower())
        if start_idx == -1:
            return (-1, -1)
        end_idx = start_idx + len(norm_answer)
        return (start_idx, end_idx)
