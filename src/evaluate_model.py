#!/usr/bin/env python3
"""
Evaluation Script for Question Generation Models

Computes comprehensive metrics (ROUGE, BLEU, BERTScore, EM, F1, QG→QA pass-rate)
and displays them with goal values from the project requirements.

Usage:
    python src/evaluate_model.py --checkpoint checkpoints/t5_base_en_aware/final_model --config configs/models/t5_base_en_aware.yaml
    python src/evaluate_model.py --checkpoint checkpoints/mt5_base_ua_aware/final_model --config configs/models/mt5_base_ua_aware.yaml
"""

import os
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.data.dataset_loader import DatasetLoader
from src.data.normalizer import TextNormalizer
from src.evaluation.metrics import MetricsCalculator
from src.models.qa_model import QAModel
from src.utils.config import Config

GOAL_VALUES = {
    "rouge-l": {
        "goal": 0.05,  # ≥5% improvement (absolute: ≥5 percentage points)
        "description": "ROUGE-L improvement ≥5% vs baseline",
        "baseline": 0.40,  # Assumed baseline ~40%
        "target": 0.45,  # 40% + 5% = 45%
    },
    "bleu": {
        "goal": "no_degradation",
        "description": "BLEU should not degrade",
        "baseline": 0.30,
    },
    "em": {
        "goal": 0.05,  # ≥5 p.p. improvement
        "description": "EM improvement ≥5 p.p. in QG→QA",
        "baseline": 0.45,
        "target": 0.50,
    },
    "f1": {
        "goal": 0.05,  # ≥5 p.p. improvement
        "description": "F1 improvement ≥5 p.p. in QG→QA",
        "baseline": 0.55,
        "target": 0.60,
    },
    "pass_rate": {
        "goal": 0.70,  # ≥70%
        "description": "QG→QA pass-rate ≥70% (F1≥0.8, conf≥0.35)",
        "target": 0.70,
    },
    "mos": {
        "goal": 4.0,  # ≥4.0
        "description": "MOS ≥4.0 (human evaluation, scale 1-5)",
        "target": 4.0,
    },
}


def load_model(checkpoint_path: str, device: str = "cuda"):
    print(f"\n📦 Loading model from: {checkpoint_path}")

    import os

    from peft import PeftConfig, PeftModel

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")

    if os.path.exists(adapter_config_path):
        print("   Detected PEFT/LoRA checkpoint, loading with adapters...")

        peft_config = PeftConfig.from_pretrained(checkpoint_path)
        base_model_name = peft_config.base_model_name_or_path

        print(f"   Base model: {base_model_name}")

        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

        model = PeftModel.from_pretrained(base_model, checkpoint_path)

        print("   Merging LoRA adapters into base model...")
        model = model.merge_and_unload()
    else:
        print("   Loading full model checkpoint...")
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)

    model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully on {device}")
    return model, tokenizer


def generate_questions(
    model,
    tokenizer,
    dataset,
    config,
    device="cuda",
    max_samples=None,
    qa_model: "QAModel" = None,
):
    """Generate questions for dataset"""
    print("\n🔮 Generating questions...")

    mode = config.get("mode", "answer_aware")

    if mode == "answer_agnostic":
        context_groups = defaultdict(lambda: {"questions": [], "answers": []})
        for example in dataset:
            ctx = example["context"]
            context_groups[ctx]["questions"].append(example["question"])

            if "all_answers" in example:
                for ans in example["all_answers"]:
                    if ans not in context_groups[ctx]["answers"]:
                        context_groups[ctx]["answers"].append(ans)
            else:
                if example["answer"] not in context_groups[ctx]["answers"]:
                    context_groups[ctx]["answers"].append(example["answer"])

        unique_contexts = list(context_groups.keys())
        if max_samples:
            unique_contexts = unique_contexts[:max_samples]

        eval_items = []
        for ctx in unique_contexts:
            eval_items.append(
                {
                    "context": ctx,
                    "gold_questions": context_groups[ctx]["questions"],
                    "gold_answers": context_groups[ctx]["answers"],
                    "answer": None,
                }
            )
    else:
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        eval_items = []
        for example in dataset:
            eval_items.append(
                {
                    "context": example["context"],
                    "gold_questions": [example["question"]],
                    "gold_answers": example.get("all_answers", [example["answer"]]),
                    "answer": example["answer"],
                }
            )

    generated_questions = []
    gold_questions_list = []
    contexts = []
    gold_answers_list = []

    def _post_process(q: str):
        q = (q or "").strip()
        if not q:
            return q
        if not q.endswith("?"):
            q = q.rstrip(".!") + "?"
        return q

    for item in tqdm(eval_items, desc="Generating"):
        context = item["context"]
        answer = item["answer"]

        cyrillic_chars = sum(
            1 for c in context if "а" <= c.lower() <= "я" or c.lower() in "ґєії"
        )
        is_ua = cyrillic_chars > 10

        if is_ua:
            q_prefix = "генерувати питання"
            c_prefix = "контекст"
            a_prefix = "відповідь"
        else:
            q_prefix = "generate question"
            c_prefix = "context"
            a_prefix = "answer"

        if mode == "answer_aware":
            prompt = f"{q_prefix}: {c_prefix}: {context} {a_prefix}: {answer}"
        else:
            prompt = f"{q_prefix}: {c_prefix}: {context}"

        inputs = tokenizer(
            prompt,
            max_length=config.get("data.max_context_len", 512),
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            do_sample = config.get("generation.do_sample", False)
            num_return = int(config.get("generation.num_return_sequences", 1))
            num_beams = int(config.get("generation.num_beams", 1))
            if not do_sample and num_beams < num_return:
                num_beams = num_return

            num_beam_groups = int(config.get("generation.num_beam_groups", 1))
            diversity_penalty = float(config.get("generation.diversity_penalty", 0.0))
            if num_beam_groups > 1 and num_beams % num_beam_groups != 0:
                num_beams = (num_beams // num_beam_groups + 1) * num_beam_groups
            outputs = model.generate(
                **inputs,
                num_beams=num_beams,
                num_beam_groups=num_beam_groups if num_beam_groups > 1 else 1,
                diversity_penalty=diversity_penalty if num_beam_groups > 1 else 0.0,
                length_penalty=config.get("generation.length_penalty", 1.1),
                no_repeat_ngram_size=config.get("generation.no_repeat_ngram_size", 3),
                repetition_penalty=config.get("generation.repetition_penalty", 1.1),
                min_new_tokens=config.get("generation.min_new_tokens", 5),
                max_new_tokens=config.get("generation.max_new_tokens", 50),
                do_sample=do_sample,
                top_p=config.get("generation.top_p", 0.9),
                top_k=config.get("generation.top_k", 50),
                temperature=config.get("generation.temperature", 1.0),
                num_return_sequences=num_return,
                trust_remote_code=True,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        candidates = [_post_process(d) for d in decoded]

        selected_q = candidates[0] if candidates else ""
        if candidates:
            lang = config.get("language", "en").lower()
            if lang == "en":
                wh_regex = re.compile(
                    r"^(who|what|when|where|why|how|which|whose|whom|is|are|do|does|did|can|could|will|would|should|may|might|must|has|have|had)\b",
                    re.I,
                )
            elif lang == "ua":
                wh_regex = re.compile(
                    r"^(хто|що|де|коли|куди|звідки|як|чому|навіщо|який|чий|скільки|чи|яка|яке|які|чий|чия|чиє|чиї)\b",
                    re.I,
                )
            else:
                wh_regex = None

            best_score = -1.0
            gold_ans = item.get("gold_answers", []) or []

            qa_results = []
            if qa_model is not None and gold_ans:
                qa_results = qa_model.answer_question_batch(
                    candidates, [context] * len(candidates)
                )

            for idx, cand in enumerate(candidates):
                try:
                    if qa_results and idx < len(qa_results):
                        qa_out = qa_results[idx]
                        f1s = [
                            qa_model.compute_em_f1(qa_out.get("answer", ""), ga)[1]
                            for ga in gold_ans
                        ]
                        cand_f1 = max(f1s) if f1s else 0.0
                        qa_conf = qa_out.get("confidence", 0.0)
                    else:
                        cand_f1 = 0.0
                        qa_conf = 0.0

                    wh_bonus = 1.0 if wh_regex and wh_regex.match(cand) else 0.0

                    cand_words = cand.lower().split()
                    ctx_words = set(context.lower().split())
                    overlap = (
                        sum(1 for w in cand_words if w in ctx_words) / len(cand_words)
                        if cand_words
                        else 1.0
                    )
                    copy_penalty = (
                        -1.0 if overlap > 0.85 else (-0.4 if overlap > 0.7 else 0.0)
                    )

                    ans_in_q_penalty = 0.0
                    if qa_results and idx < len(qa_results):
                        qa_ans = qa_results[idx].get("answer", "")
                        if (
                            qa_ans
                            and len(qa_ans) > 3
                            and qa_ans.lower() in cand.lower()
                        ):
                            ans_in_q_penalty = -1.0
                    pass_bonus_val = 2.0 if mode == "answer_aware" else 10.0
                    pass_bonus = (
                        pass_bonus_val if cand_f1 >= 0.8 and qa_conf >= 0.35 else 0.0
                    )
                    score = (
                        (cand_f1 * 2.0)
                        + pass_bonus
                        + wh_bonus
                        + copy_penalty
                        + ans_in_q_penalty
                        + (qa_conf * 0.5)
                    )
                except Exception:
                    score = -1.0

                if score > best_score:
                    best_score = score
                    selected_q = cand

        generated_questions.append(selected_q)
        gold_questions_list.append(item["gold_questions"])
        contexts.append(context)
        gold_answers_list.append(item["gold_answers"])

    print(f"✓ Generated {len(generated_questions)} questions")
    return generated_questions, gold_questions_list, contexts, gold_answers_list


def compute_all_metrics(
    generated_questions,
    gold_questions,
    contexts,
    gold_answers,
    qa_model,
    metrics_calc,
    config,
):
    print("\n📊 Computing metrics...")

    lang = config.get("language", "en")
    eval_config = config["evaluation"]

    metrics = metrics_calc.compute_all_metrics(
        generated_questions,
        gold_questions,
        contexts,
        gold_answers,
        qa_model,
        lang=lang,
        config=eval_config,
    )

    return metrics


def compute_pass_rate(metrics, f1_threshold=0.8):
    if (
        "qa_pass_rate" in metrics
        and "qa_pass_count" in metrics
        and "qa_total" in metrics
    ):
        return metrics["qa_pass_rate"], metrics["qa_pass_count"], metrics["qa_total"]

    if "qa_f1" in metrics:
        qa_f1_scores = metrics["qa_f1"]

        if isinstance(qa_f1_scores, (float, np.floating)):
            pass_rate = 1.0 if qa_f1_scores >= f1_threshold else 0.0
            pass_count = 1 if qa_f1_scores >= f1_threshold else 0
            total = 1
        elif isinstance(qa_f1_scores, (list, np.ndarray)):
            pass_count = sum(1 for f1 in qa_f1_scores if f1 >= f1_threshold)
            total = len(qa_f1_scores)
            pass_rate = pass_count / total if total > 0 else 0.0
        else:
            return 0.0, 0, 0

        return pass_rate, pass_count, total
    return 0.0, 0, 0


def format_metric_with_goal(metric_name, value, goal_info):
    if goal_info is None:
        return f"{value:.4f}"

    if "target" in goal_info:
        target = goal_info["target"]
        if value >= target:
            status = "✅ PASS"
            color = "\033[92m"  # Green
        else:
            status = "❌ FAIL"
            color = "\033[91m"  # Red
        reset = "\033[0m"

        return f"{color}{value:.4f}{reset} (Goal: ≥{target:.4f}) {status}"
    elif goal_info["goal"] == "no_degradation":
        baseline = goal_info["baseline"]
        if value >= baseline:
            status = "✅ OK"
            color = "\033[92m"
        else:
            status = "⚠️  DEGRADED"
            color = "\033[93m"  # Yellow
        reset = "\033[0m"

        return f"{color}{value:.4f}{reset} (Baseline: {baseline:.4f}) {status}"
    else:
        return f"{value:.4f}"


def print_evaluation_report(metrics, config):
    print("\n" + "=" * 80)
    print("📋 EVALUATION REPORT")
    print("=" * 80)

    print(f"\n🔧 Configuration:")
    print(f"   Language: {config.get('language', 'en').upper()}")
    print(f"   Mode: {config.get('mode', 'answer_aware')}")
    print(f"   Model: {config.get('model_name', 'N/A')}")

    print(f"\n📏 LEXICAL SIMILARITY METRICS:")
    print(f"   {'Metric':<15} {'Value':<50} {'Description'}")
    print(f"   {'-'*15} {'-'*50} {'-'*40}")

    for rouge_type in ["rouge-1", "rouge-2", "rouge-l"]:
        if rouge_type in metrics:
            value = metrics[rouge_type]
            goal_info = GOAL_VALUES.get(rouge_type)
            formatted = format_metric_with_goal(rouge_type, value, goal_info)
            desc = goal_info["description"] if goal_info else ""
            print(f"   {rouge_type.upper():<15} {formatted:<50} {desc}")

    if "bleu" in metrics:
        value = metrics["bleu"]
        goal_info = GOAL_VALUES["bleu"]
        formatted = format_metric_with_goal("bleu", value, goal_info)
        desc = goal_info["description"]
        print(f"   {'BLEU':<15} {formatted:<50} {desc}")

    if "bertscore" in metrics:
        value = metrics["bertscore"]
        print(f"   {'BERTScore':<15} {f'{value:.4f}':<50} Semantic similarity")
    elif "bertscore-f1" in metrics:
        value = metrics["bertscore-f1"]
        print(f"   {'BERTScore':<15} {f'{value:.4f}':<50} Semantic similarity")

    print(f"\n🎯 QG→QA FUNCTIONAL METRICS:")
    print(f"   {'Metric':<15} {'Value':<50} {'Description'}")
    print(f"   {'-'*15} {'-'*50} {'-'*40}")

    qa_em_key = "qa_em" if "qa_em" in metrics else "em"
    if qa_em_key in metrics:
        value = metrics[qa_em_key]
        goal_info = GOAL_VALUES["em"]
        formatted = format_metric_with_goal("em", value, goal_info)
        desc = goal_info["description"]
        print(f"   {'EM':<15} {formatted:<50} {desc}")

    qa_f1_key = "qa_f1" if "qa_f1" in metrics else "f1"
    if qa_f1_key in metrics:
        value = metrics[qa_f1_key]
        goal_info = GOAL_VALUES["f1"]
        formatted = format_metric_with_goal("f1", value, goal_info)
        desc = goal_info["description"]
        print(f"   {'F1':<15} {formatted:<50} {desc}")

    f1_threshold = config.get("evaluation.qa_f1_threshold", 0.8)
    pass_rate, pass_count, total = compute_pass_rate(metrics, f1_threshold)
    goal_info = GOAL_VALUES["pass_rate"]
    formatted = format_metric_with_goal("pass_rate", pass_rate, goal_info)
    desc = f"{goal_info['description']} [{pass_count}/{total}]"
    print(f"   {'Pass Rate':<15} {formatted:<50} {desc}")

    print(f"\n👥 HUMAN EVALUATION:")
    print(f"   {'Metric':<15} {'Value':<50} {'Description'}")
    print(f"   {'-'*15} {'-'*50} {'-'*40}")
    goal_info = GOAL_VALUES["mos"]
    print(f"   {'MOS':<15} {'[PLANNED]':<50} {goal_info['description']}")

    print(f"\n🏆 OVERALL ASSESSMENT:")

    goals_met = []
    goals_failed = []

    if "rouge-l" in metrics:
        if metrics["rouge-l"] >= GOAL_VALUES["rouge-l"]["target"]:
            goals_met.append("ROUGE-L")
        else:
            goals_failed.append("ROUGE-L")

    qa_em_key = "qa_em" if "qa_em" in metrics else "em"
    if qa_em_key in metrics:
        if metrics[qa_em_key] >= GOAL_VALUES["em"]["target"]:
            goals_met.append("EM")
        else:
            goals_failed.append("EM")

    qa_f1_key = "qa_f1" if "qa_f1" in metrics else "f1"
    if qa_f1_key in metrics:
        if metrics[qa_f1_key] >= GOAL_VALUES["f1"]["target"]:
            goals_met.append("F1")
        else:
            goals_failed.append("F1")

    if pass_rate >= GOAL_VALUES["pass_rate"]["target"]:
        goals_met.append("Pass Rate")
    else:
        goals_failed.append("Pass Rate")

    print(f"   Goals Met: {len(goals_met)}/{len(goals_met) + len(goals_failed)}")
    if goals_met:
        print(f"   ✅ {', '.join(goals_met)}")
    if goals_failed:
        print(f"   ❌ {', '.join(goals_failed)}")

    print("\n" + "=" * 80)


def save_evaluation_results(metrics, output_path: Path, config, generated_samples=None):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "language": config.get("language", "en"),
            "mode": config.get("mode", "answer_aware"),
            "model_name": config.get("model_name", "N/A"),
        },
        "metrics": {},
        "goals": GOAL_VALUES,
    }

    for key, value in metrics.items():
        if isinstance(value, (list, np.ndarray)):
            results["metrics"][key] = {
                "mean": float(np.mean(value)),
                "std": float(np.std(value)),
                "min": float(np.min(value)),
                "max": float(np.max(value)),
            }
        else:
            results["metrics"][key] = float(value)

    f1_threshold = config.get("evaluation.qa_f1_threshold", 0.8)
    pass_rate, pass_count, total = compute_pass_rate(metrics, f1_threshold)
    results["metrics"]["pass_rate"] = {
        "value": float(pass_rate),
        "pass_count": pass_count,
        "total": total,
        "threshold": f1_threshold,
    }

    results_file = output_path / "evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {results_file}")

    if generated_samples:
        samples_file = output_path / "generated_samples.jsonl"
        with open(samples_file, "w", encoding="utf-8") as f:
            for sample in generated_samples[:100]:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"💾 Sample predictions saved to: {samples_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Question Generation Model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to model config file"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results (default: checkpoint/evaluation)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )

    args = parser.parse_args()

    config = Config(args.config)

    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️  CUDA not available, using CPU (will be slow)")

    model, tokenizer = load_model(args.checkpoint, device)

    print(f"\n📂 Loading dataset...")
    normalizer = TextNormalizer(language=config.get("language", "en"))
    dataset_loader = DatasetLoader(config.config, normalizer)

    if config.get("language", "en") == "en":
        dataset = dataset_loader.load_squad_v2(
            filter_unanswerable=True, deduplicate_by_context=False
        )
    else:
        dataset_path = Path(config.data_dir) / "ukrainian_qa.jsonl"
        raw_dataset = dataset_loader.load_ukrainian_dataset(dataset_path)
        dataset = dataset_loader.stratified_split(
            raw_dataset,
            config["data"]["train_split"],
            config["data"]["val_split"],
            config.get("seed", 42),
        )

    eval_dataset = dataset[args.split]
    print(f"✓ Loaded {len(eval_dataset)} examples from {args.split} split")

    qa_model = QAModel(device=device)
    metrics_calc = MetricsCalculator()

    generated_questions, gold_questions, contexts, gold_answers = generate_questions(
        model, tokenizer, eval_dataset, config, device, args.max_samples, qa_model
    )

    metrics = compute_all_metrics(
        generated_questions,
        gold_questions,
        contexts,
        gold_answers,
        qa_model,
        metrics_calc,
        config,
    )

    print_evaluation_report(metrics, config)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.checkpoint).parent / "evaluation"

    generated_samples = [
        {
            "context": ctx,
            "gold_question": gold_q[0] if gold_q else "",
            "generated_question": gen_q,
            "gold_answer": ans[0] if ans else "",
            "all_gold_questions": gold_q,
            "all_gold_answers": ans,
        }
        for ctx, gold_q, gen_q, ans in zip(
            contexts[:100],
            gold_questions[:100],
            generated_questions[:100],
            gold_answers[:100],
        )
    ]

    save_evaluation_results(metrics, output_path, config, generated_samples)

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
