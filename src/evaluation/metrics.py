from typing import List, Dict, Any
import numpy as np
import evaluate
from collections import defaultdict

class MetricsCalculator:
    def __init__(self):
        self._rouge = None
        self._bleu = None
        self._bertscore = None

    @property
    def rouge(self):
        if self._rouge is None:
            self._rouge = evaluate.load('rouge')
        return self._rouge

    @property
    def bleu(self):
        if self._bleu is None:
            self._bleu = evaluate.load('sacrebleu')
        return self._bleu

    @property
    def bertscore(self):
        if self._bertscore is None:
            # We use BERTScorer directly to handle some special cases like OverflowError
            from bert_score import BERTScorer
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # Default to English model for English, and multilingual for others
            # microsoft/deberta-xlarge-mnli is a good default for English
            model_type = 'microsoft/deberta-xlarge-mnli'
            
            self._bertscore = BERTScorer(model_type=model_type, lang='en', device=device)
            # Fix for OverflowError in Deberta: cap model_max_length
            if self._bertscore._tokenizer.model_max_length > 1e10:
                self._bertscore._tokenizer.model_max_length = 512
                
        return self._bertscore

    def _normalize_references(self, references: Any) -> List[List[str]]:
        """Ensure references are in List[List[str]] format."""
        if not references:
            return []
        
        norm_refs = []
        for refs in references:
            if isinstance(refs, str):
                # Single reference as a string -> convert to [ref]
                norm_refs.append([refs])
            elif isinstance(refs, (list, tuple)):
                # Already a list/tuple -> ensure it's a list of strings
                if not refs:
                    norm_refs.append([""])
                else:
                    norm_refs.append([str(r) for r in refs])
            else:
                # Other types -> convert to [str(ref)]
                norm_refs.append([str(refs)])
        return norm_refs

    def compute_rouge(self, predictions: List[str], references: Any, lang: str = 'en') -> Dict:
        try:
            if not predictions:
                return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}
            
            # Normalize references to List[List[str]]
            norm_refs = self._normalize_references(references)
            
            # evaluate('rouge') supports multi-reference natively (List[List[str]])
            # and takes the max score per prediction.
            # Use a custom tokenizer that supports Unicode (Cyrillic) word characters
            # to avoid the default English-centric tokenizer that strips non-ASCII.
            import re
            
            def stem_ukrainian(word):
                """Покращений стеммер для української мови"""
                if len(word) <= 3:
                    return word
                
                # 1. Спроба видалити довгі закінчення
                new_word = re.sub(r'(ами|ями|иму|ими|ому|ові|еві|ого|ої|ій|ий|ям|ам|ах|ях|ів|ей|ою|ею|ий|их|іх)$', '', word)
                if len(new_word) < 3:
                    word = word
                else:
                    word = new_word
                    
                # 2. Спроба видалити короткі закінчення
                new_word = re.sub(r'(а|я|о|е|и|і|у|ю)$', '', word)
                if len(new_word) < 3:
                    word = word
                else:
                    word = new_word
                    
                # 3. Дієслівні закінчення (тільки якщо слово ще достатньо довге)
                if len(word) > 4:
                    word = re.sub(r'(ться|лись|всь|ти|ла|ло|ли|в|ш|те|мо)$', '', word)
                    
                return word

            def unicode_tokenizer(text):
                # Standardize and tokenize while keeping Unicode characters
                # \w in Python 3 matches Unicode characters including Cyrillic
                tokens = re.findall(r'\w+', text.lower(), re.UNICODE)
                if lang == 'ua':
                    return [stem_ukrainian(t) for t in tokens]
                return tokens
                
            # Use custom tokenizer for non-English languages to preserve Unicode characters
            tokenizer_arg = unicode_tokenizer if lang != 'en' else None
            # If we use our own stemmer for UA, don't use default stemmer (which is English-only)
            use_stemmer = (lang == 'en')
            
            res = self.rouge.compute(
                predictions=predictions, 
                references=norm_refs, 
                use_stemmer=use_stemmer,
                tokenizer=tokenizer_arg
            )
            
            return {
                'rouge-1': res['rouge1'],
                'rouge-2': res['rouge2'],
                'rouge-l': res['rougeL']
            }
        except Exception as e:
            print(f"Warning: ROUGE computation failed: {e}. Skipping ROUGE.")
            return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}

    def compute_bleu(self, predictions: List[str], references: Any) -> float:
        try:
            if not predictions:
                return 0.0
            
            # Normalize references to List[List[str]]
            norm_refs_per_sample = self._normalize_references(references)
            
            # SacreBLEU expects List[List[str]] but it must be padded to the same number of references.
            # norm_refs_per_sample is List[num_preds][num_refs]
            max_refs = max(len(refs) for refs in norm_refs_per_sample) if norm_refs_per_sample else 1
            
            final_refs = []
            for rlist in norm_refs_per_sample:
                if len(rlist) < max_refs:
                    rlist = rlist + [rlist[-1]] * (max_refs - len(rlist))
                final_refs.append(rlist)
            
            # Pass as List[num_preds][num_refs] (Nested format)
            result = self.bleu.compute(predictions=predictions, references=final_refs)
            return result['score']
        except Exception as e:
            print(f"Warning: BLEU computation failed: {e}. Skipping BLEU.")
            return 0.0

    def compute_bertscore(self, predictions: List[str], references: Any, lang: str = 'en') -> Dict:
        try:
            if not predictions:
                return {'bertscore-precision': 0.0, 'bertscore-recall': 0.0, 'bertscore-f1': 0.0, 'bertscore': 0.0}

            # Normalize references to List[List[str]]
            norm_refs = self._normalize_references(references)

            # Filter empty sequences to avoid issues with some models
            filtered_preds, filtered_refs = [], []
            for p, r_list in zip(predictions, norm_refs):
                if p.strip() and r_list and any(r.strip() for r in r_list):
                    filtered_preds.append(p)
                    filtered_refs.append([r for r in r_list if r.strip()])

            if not filtered_preds:
                return {'bertscore-precision': 0.0, 'bertscore-recall': 0.0, 'bertscore-f1': 0.0, 'bertscore': 0.0}

            # We need to handle the case where the scorer was initialized for a different language
            # For now, we assume it's English by default as per __init__ logic
            # If lang is not 'en', we might need to re-initialize or have a second scorer
            if lang != 'en' and not hasattr(self, '_bertscore_multilang'):
                from bert_score import BERTScorer
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self._bertscore_multilang = BERTScorer(model_type='xlm-roberta-large', lang=lang, device=device)
                if self._bertscore_multilang._tokenizer.model_max_length > 1e10:
                    self._bertscore_multilang._tokenizer.model_max_length = 512
            
            scorer = self.bertscore if lang == 'en' else self._bertscore_multilang
            
            # Compute BERTScore
            P, R, F1 = scorer.score(filtered_preds, filtered_refs)
            
            # Ensure results are numpy arrays or lists before calling mean
            def to_numpy_list(val):
                if isinstance(val, (list, tuple)):
                    return np.array(val)
                if hasattr(val, 'cpu'): # It's a tensor
                    return val.detach().cpu().numpy()
                return val

            precision = to_numpy_list(P)
            recall = to_numpy_list(R)
            f1 = to_numpy_list(F1)
            
            mean_f1 = float(np.mean(f1))
            return {
                'bertscore-precision': float(np.mean(precision)), 
                'bertscore-recall': float(np.mean(recall)), 
                'bertscore-f1': mean_f1,
                'bertscore': mean_f1  # For convenience
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

    def compute_qg_qa_metrics(self, predictions: List[str], contexts: List[str], gold_answers: List[List[str]], qa_model, f1_threshold: float = 0.8, conf_threshold: float = 0.35) -> Dict:
        try:
            em_scores, f1_scores, confidences, pass_count = [], [], [], 0
            for pred_q, context, gold_ans_list in zip(predictions, contexts, gold_answers):
                try:
                    qa_result = qa_model.answer_question(pred_q, context)
                    
                    # Compute EM/F1 against ALL gold answers for this context and take the best
                    best_em, best_f1 = 0.0, 0.0
                    for gold_ans in gold_ans_list:
                        em, f1 = qa_model.compute_em_f1(qa_result['answer'], gold_ans)
                        best_em = max(best_em, em)
                        best_f1 = max(best_f1, f1)
                    
                    em_scores.append(best_em)
                    f1_scores.append(best_f1)
                    confidences.append(qa_result['confidence'])
                    
                    if best_f1 >= f1_threshold and qa_result['confidence'] >= conf_threshold:
                        pass_count += 1
                except Exception as e:
                    print(f"Warning: QA metric computation failed for one example: {e}")
                    em_scores.append(0.0)
                    f1_scores.append(0.0)
                    confidences.append(0.0)
            return {
                'qa_em': np.mean(em_scores) if em_scores else 0.0, 
                'qa_f1': np.mean(f1_scores) if f1_scores else 0.0, 
                'qa_conf': np.mean(confidences) if confidences else 0.0, 
                'qa_pass_rate': pass_count / len(predictions) if predictions else 0.0, 
                'qa_pass_count': pass_count,
                'qa_total': len(predictions)
            }
        except Exception as e:
            print(f"Warning: QA metrics computation failed: {e}. Skipping QA metrics.")
            return {'qa_em': 0.0, 'qa_f1': 0.0, 'qa_conf': 0.0, 'qa_pass_rate': 0.0, 'qa_pass_count': 0, 'qa_total': 0}

    def compute_all_metrics(self, predictions: List[str], references: List[List[str]], contexts: List[str], gold_answers: List[List[str]], qa_model, lang: str = 'en', config: Dict = None) -> Dict:
        if config is None:
            config = {'qa_f1_threshold': 0.8, 'qa_conf_threshold': 0.35}
        metrics = {}
        metrics.update(self.compute_rouge(predictions, references, lang))
        metrics['bleu'] = self.compute_bleu(predictions, references)
        metrics.update(self.compute_bertscore(predictions, references, lang))
        qa_metrics = self.compute_qg_qa_metrics(predictions, contexts, gold_answers, qa_model, config['qa_f1_threshold'], config['qa_conf_threshold'])
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
