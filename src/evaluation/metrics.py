from typing import List, Dict
import numpy as np
import evaluate
from collections import defaultdict

class MetricsCalculator:
    def __init__(self):
        self.rouge = evaluate.load('rouge')
        self.bleu = evaluate.load('sacrebleu')
        self.bertscore = evaluate.load('bertscore')

    def compute_rouge(self, predictions: List[str], references: List[str]) -> Dict:
        try:
            results = self.rouge.compute(predictions=predictions, references=references, use_stemmer=True)
            return {'rouge-1': results['rouge1'], 'rouge-2': results['rouge2'], 'rouge-l': results['rougeL']}
        except Exception as e:
            print(f"Warning: ROUGE computation failed: {e}. Skipping ROUGE.")
            return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}

    def compute_bleu(self, predictions: List[str], references: List[List[str]]) -> float:
        try:
            result = self.bleu.compute(predictions=predictions, references=references)
            return result['score']
        except Exception as e:
            print(f"Warning: BLEU computation failed: {e}. Skipping BLEU.")
            return 0.0

    def compute_bertscore(self, predictions: List[str], references: List[str], lang: str = 'en') -> Dict:
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_type = 'microsoft/deberta-xlarge-mnli' if lang == 'en' else 'xlm-roberta-large'
            
            # Filter empty sequences to avoid issues with some models
            filtered_preds, filtered_refs = [], []
            for p, r in zip(predictions, references):
                if p.strip() and r.strip():
                    filtered_preds.append(p)
                    filtered_refs.append(r)
            
            if not filtered_preds:
                return {'bertscore-precision': 0.0, 'bertscore-recall': 0.0, 'bertscore-f1': 0.0}

            results = self.bertscore.compute(
                predictions=filtered_preds, 
                references=filtered_refs, 
                lang=lang, 
                model_type=model_type,
                device=device
            )
            
            # Ensure results are numpy arrays or lists before calling mean
            def to_numpy_list(val):
                if isinstance(val, (list, tuple)):
                    return np.array(val)
                if hasattr(val, 'cpu'): # It's a tensor
                    return val.detach().cpu().numpy()
                return val

            precision = to_numpy_list(results['precision'])
            recall = to_numpy_list(results['recall'])
            f1 = to_numpy_list(results['f1'])
            
            return {
                'bertscore-precision': float(np.mean(precision)), 
                'bertscore-recall': float(np.mean(recall)), 
                'bertscore-f1': float(np.mean(f1))
            }
        except (OverflowError, Exception) as e:
            # We don't want to crash the whole training if BERTScore fails
            # But we should log the issue if it's persistent
            import traceback
            traceback_str = traceback.format_exc()
            print(f"Warning: BERTScore computation failed with error: {e}. Skipping BERTScore.")
            # Only print traceback if it's a new or critical issue, for now we keep it quiet to not spam logs
            # unless it's the specific "int too big to convert" error which we want to debug
            if "int too big to convert" in str(e):
                print(f"Detailed BERTScore error:\n{traceback_str}")
            return {'bertscore-precision': 0.0, 'bertscore-recall': 0.0, 'bertscore-f1': 0.0}

    def compute_qg_qa_metrics(self, predictions: List[str], references: List[str], contexts: List[str], gold_answers: List[str], qa_model, f1_threshold: float = 0.8, conf_threshold: float = 0.35) -> Dict:
        try:
            em_scores, f1_scores, confidences, pass_count = [], [], [], 0
            for pred_q, ref_q, context, gold_ans in zip(predictions, references, contexts, gold_answers):
                try:
                    qa_result = qa_model.answer_question(pred_q, context)
                    em, f1 = qa_model.compute_em_f1(qa_result['answer'], gold_ans)
                    em_scores.append(em)
                    f1_scores.append(f1)
                    confidences.append(qa_result['confidence'])
                    if f1 >= f1_threshold and qa_result['confidence'] >= conf_threshold:
                        pass_count += 1
                except Exception as e:
                    print(f"Warning: QA metric computation failed for one example: {e}")
                    em_scores.append(0.0)
                    f1_scores.append(0.0)
                    confidences.append(0.0)
            return {'qa_em': np.mean(em_scores) if em_scores else 0.0, 'qa_f1': np.mean(f1_scores) if f1_scores else 0.0, 'qa_conf': np.mean(confidences) if confidences else 0.0, 'qa_pass_rate': pass_count / len(predictions) if predictions else 0.0, 'qa_pass_count': pass_count}
        except Exception as e:
            print(f"Warning: QA metrics computation failed: {e}. Skipping QA metrics.")
            return {'qa_em': 0.0, 'qa_f1': 0.0, 'qa_conf': 0.0, 'qa_pass_rate': 0.0, 'qa_pass_count': 0}

    def compute_all_metrics(self, predictions: List[str], references: List[str], contexts: List[str], gold_answers: List[str], qa_model, lang: str = 'en', config: Dict = None) -> Dict:
        if config is None:
            config = {'qa_f1_threshold': 0.8, 'qa_conf_threshold': 0.35}
        metrics = {}
        metrics.update(self.compute_rouge(predictions, references))
        metrics['bleu'] = self.compute_bleu(predictions, [[ref] for ref in references])
        metrics.update(self.compute_bertscore(predictions, references, lang))
        qa_metrics = self.compute_qg_qa_metrics(predictions, references, contexts, gold_answers, qa_model, config['qa_f1_threshold'], config['qa_conf_threshold'])
        metrics.update(qa_metrics)
        return metrics

    def analyze_wh_types(self, questions: List[str], lang: str = 'en') -> Dict:
        wh_words = ['what', 'when', 'where', 'who', 'why', 'how', 'which', 'whose'] if lang == 'en' else ['що', 'коли', 'де', 'хто', 'чому', 'як', 'який', 'чий']
        wh_counts = defaultdict(int)
        for question in questions:
            question_lower = question.lower()
            for wh in wh_words:
                if question_lower.startswith(wh):
                    wh_counts[wh] += 1
                    break
            else:
                wh_counts['other'] += 1
        total = len(questions)
        return {k: v / total for k, v in wh_counts.items()}
