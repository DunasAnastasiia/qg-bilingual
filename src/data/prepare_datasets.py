#!/usr/bin/env python3
"""
Dataset Preparation Script
Downloads SQuAD 2.0 and prepares Ukrainian Q&A dataset
Supports both demo (limited) and production (full) modes
"""
import sys
import json
import logging
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import random

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / 'data'
DATA_DIR.mkdir(exist_ok=True)


def download_squad(output_dir=None, demo_mode=False, demo_size=1000):
    """
    Download SQuAD 2.0 dataset

    Args:
        output_dir: Directory to save the dataset
        demo_mode: If True, only download a small subset
        demo_size: Number of examples for demo mode
    """
    if output_dir is None:
        output_dir = DATA_DIR / 'squad_v2'

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    mode_str = "DEMO" if demo_mode else "FULL"
    logger.info(f"Downloading SQuAD 2.0 dataset ({mode_str})...")

    try:
        dataset = load_dataset('squad_v2')

        if demo_mode:
            logger.info(f"Demo mode: limiting to {demo_size} examples per split")
            # Take a random sample
            random.seed(42)
            dataset['train'] = dataset['train'].shuffle(seed=42).select(range(min(demo_size, len(dataset['train']))))
            dataset['validation'] = dataset['validation'].shuffle(seed=42).select(range(min(demo_size // 5, len(dataset['validation']))))

        logger.info(f"✓ SQuAD 2.0 downloaded successfully")
        logger.info(f"  - Train examples: {len(dataset['train'])}")
        logger.info(f"  - Validation examples: {len(dataset['validation'])}")

        # Save to local cache
        for split in ['train', 'validation']:
            output_file = output_dir / f'{split}.jsonl'
            with open(output_file, 'w', encoding='utf-8') as f:
                for example in tqdm(dataset[split], desc=f"Saving {split}"):
                    record = {
                        'context': example['context'],
                        'question': example['question'],
                        'answer': example['answers']['text'][0] if example['answers']['text'] else '',
                        'answer_start': example['answers']['answer_start'][0] if example['answers']['answer_start'] else -1,
                        'id': example['id'],
                        'is_impossible': len(example['answers']['text']) == 0
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')

            logger.info(f"  - Saved {split} to {output_file}")

        return True
    except Exception as e:
        logger.error(f"✗ Failed to download SQuAD 2.0: {e}")
        return False


def get_ukrainian_examples():
    """Generate comprehensive Ukrainian Q&A examples"""
    return [
        # History
        {
            'context': 'Київ є столицею та найбільшим містом України, розташованим на річці Дніпро. Це одне з найдавніших міст Східної Європи, засноване понад 1400 років тому. За легендою, місто засновано трьома братами: Києм, Щеком, Хоривом та їхньою сестрою Либіддю.',
            'question': 'Яке місто є столицею України?',
            'answer': 'Київ',
            'language': 'ua'
        },
        {
            'context': 'Київ є столицею та найбільшим містом України, розташованим на річці Дніпро. Це одне з найдавніших міст Східної Європи, засноване понад 1400 років тому. За легендою, місто засновано трьома братами: Києм, Щеком, Хоривом та їхньою сестрою Либіддю.',
            'question': 'На якій річці розташований Київ?',
            'answer': 'Дніпро',
            'language': 'ua'
        },
        {
            'context': 'Україна здобула незалежність 24 серпня 1991 року після розпаду Радянського Союзу. Це була історична подія для українського народу, яка ознаменувала початок нової ери державності.',
            'question': 'Коли Україна здобула незалежність?',
            'answer': '24 серпня 1991 року',
            'language': 'ua'
        },
        {
            'context': 'Україна здобула незалежність 24 серпня 1991 року після розпаду Радянського Союзу. Це була історична подія для українського народу, яка ознаменувала початок нової ери державності.',
            'question': 'Після чого Україна здобула незалежність?',
            'answer': 'після розпаду Радянського Союзу',
            'language': 'ua'
        },
        # Geography
        {
            'context': 'Львів - одне з найкрасивіших міст України, розташоване на заході країни. Місто відоме своєю унікальною архітектурою, кавовою культурою та багатою історією. Населення Львова становить близько 720 тисяч осіб.',
            'question': 'Де розташований Львів?',
            'answer': 'на заході країни',
            'language': 'ua'
        },
        {
            'context': 'Львів - одне з найкрасивіших міст України, розташоване на заході країни. Місто відоме своєю унікальною архітектурою, кавовою культурою та багатою історією. Населення Львова становить близько 720 тисяч осіб.',
            'question': 'Скільки людей проживає у Львові?',
            'answer': 'близько 720 тисяч осіб',
            'language': 'ua'
        },
        {
            'context': 'Карпати - найбільша гірська система на території України, що простягається на заході країни. Найвища точка - гора Говерла, висотою 2061 метр над рівнем моря.',
            'question': 'Яка найвища точка Карпат?',
            'answer': 'гора Говерла',
            'language': 'ua'
        },
        {
            'context': 'Карпати - найбільша гірська система на території України, що простягається на заході країни. Найвища точка - гора Говерла, висотою 2061 метр над рівнем моря.',
            'question': 'Яка висота гори Говерла?',
            'answer': '2061 метр',
            'language': 'ua'
        },
        # Culture
        {
            'context': 'Тарас Шевченко - великий український поет, письменник, художник і громадський діяч. Народився 9 березня 1814 року в селі Моринці на Київщині. Його творчість мала величезний вплив на розвиток української мови та культури.',
            'question': 'Хто такий Тарас Шевченко?',
            'answer': 'великий український поет, письменник, художник і громадський діяч',
            'language': 'ua'
        },
        {
            'context': 'Тарас Шевченко - великий український поет, письменник, художник і громадський діяч. Народився 9 березня 1814 року в селі Моринці на Київщині. Його творчість мала величезний вплив на розвиток української мови та культури.',
            'question': 'Коли народився Тарас Шевченко?',
            'answer': '9 березня 1814 року',
            'language': 'ua'
        },
        {
            'context': 'Леся Українка - видатна українська письменниця, перекладачка, культурна діячка. Справжнє ім\'я - Лариса Петрівна Косач-Квітка. Вона писала в найрізноманітніших жанрах: поезії, драмі, прозі, публіцистиці.',
            'question': 'Яке справжнє ім\'я Лесі Українки?',
            'answer': 'Лариса Петрівна Косач-Квітка',
            'language': 'ua'
        },
        {
            'context': 'Леся Українка - видатна українська письменниця, перекладачка, культурна діячка. Справжнє ім\'я - Лариса Петрівна Косач-Квітка. Вона писала в найрізноманітніших жанрах: поезії, драмі, прозі, публіцистиці.',
            'question': 'В яких жанрах писала Леся Українка?',
            'answer': 'поезії, драмі, прозі, публіцистиці',
            'language': 'ua'
        },
        # Science
        {
            'context': 'Київський університет імені Тараса Шевченка заснований у 1834 році. Це один з найстаріших університетів України та провідний навчальний заклад країни.',
            'question': 'Коли було засновано Київський університет?',
            'answer': 'у 1834 році',
            'language': 'ua'
        },
        {
            'context': 'Харківський національний університет імені В.Н. Каразіна заснований 1804 року. Він є одним з найстаріших університетів Східної Європи.',
            'question': 'Коли засновано Харківський університет?',
            'answer': '1804 року',
            'language': 'ua'
        },
        # More diverse examples
        {
            'context': 'Чорне море омиває південне узбережжя України. Довжина берегової лінії становить близько 1628 км. Основні порти: Одеса, Іллічівськ, Миколаїв.',
            'question': 'Яке море омиває південне узбережжя України?',
            'answer': 'Чорне море',
            'language': 'ua'
        },
        {
            'context': 'Чорне море омиває південне узбережжя України. Довжина берегової лінії становить близько 1628 км. Основні порти: Одеса, Іллічівськ, Миколаїв.',
            'question': 'Які основні порти на Чорному морі?',
            'answer': 'Одеса, Іллічівськ, Миколаїв',
            'language': 'ua'
        },
        {
            'context': 'Українська гривня є офіційною валютою України з 1996 року. Код валюти - UAH. Одна гривня поділяється на 100 копійок.',
            'question': 'Яка офіційна валюта України?',
            'answer': 'Українська гривня',
            'language': 'ua'
        },
        {
            'context': 'Українська гривня є офіційною валютою України з 1996 року. Код валюти - UAH. Одна гривня поділяється на 100 копійок.',
            'question': 'На скільки копійок поділяється гривня?',
            'answer': '100 копійок',
            'language': 'ua'
        },
        {
            'context': 'Софійський собор у Києві - видатна пам\'ятка архітектури XI століття. Заснований 1037 року князем Ярославом Мудрим. Внесений до списку Всесвітньої спадщини ЮНЕСКО.',
            'question': 'Хто заснував Софійський собор?',
            'answer': 'князь Ярослав Мудрий',
            'language': 'ua'
        },
        {
            'context': 'Софійський собор у Києві - видатна пам\'ятка архітектури XI століття. Заснований 1037 року князем Ярославом Мудрим. Внесений до списку Всесвітньої спадщини ЮНЕСКО.',
            'question': 'Коли було засновано Софійський собор?',
            'answer': '1037 року',
            'language': 'ua'
        },
    ]


def prepare_ukrainian_dataset(output_path=None, demo_mode=False, demo_size=1000):
    """
    Prepare Ukrainian Q&A dataset

    Args:
        output_path: Path to save the dataset
        demo_mode: If True, create a small sample
        demo_size: Number of examples for demo (will be multiplied by variations)
    """
    if output_path is None:
        output_path = DATA_DIR / 'ukrainian_qa.jsonl'

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    try:
        # Load from Hugging Face with no checks to avoid NonMatchingSplitsSizesError
        dataset = load_dataset('nogyxo/question-answering-ukrainian', verification_mode="no_checks")

        if demo_mode:
            logger.info(f"Demo mode: limiting to {demo_size} examples per split")
            random.seed(42)
            dataset['train'] = dataset['train'].shuffle(seed=42).select(range(min(demo_size, len(dataset['train']))))
            dataset['test'] = dataset['test'].shuffle(seed=42).select(range(min(demo_size // 5, len(dataset['test']))))

        logger.info(f"✓ Ukrainian QA dataset downloaded successfully")
        logger.info(f"  - Train examples: {len(dataset['train'])}")
        logger.info(f"  - Test examples: {len(dataset['test'])}")

        # Save to local cache as ukrainian_qa.jsonl (merging train and test for later splitting)
        with open(output_path, 'w', encoding='utf-8') as f:
            # First, add manual examples for better variety/quality on common topics
            manual_examples = get_ukrainian_examples()
            for ex in manual_examples:
                record = {
                    'context': ex['context'],
                    'question': ex['question'],
                    'answer': ex['answer'],
                    'all_answers': [ex['answer']],
                    'answer_start': -1, # Will be found by normalizer
                    'is_impossible': False
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

            # Then, add examples from the professional dataset
            for split in ['train', 'test']:
                for example in tqdm(dataset[split], desc=f"Saving Ukrainian {split}"):
                    # nogyxo/question-answering-ukrainian columns: context, question, is_impossible, answer_start, answer_text
                    record = {
                        'context': example['context'],
                        'question': example['question'],
                        'answer': example['answer_text'] if example['answer_text'] else '',
                        'all_answers': [example['answer_text']] if example['answer_text'] else [],
                        'answer_start': example['answer_start'],
                        'is_impossible': example['is_impossible']
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')

        logger.info(f"  - Saved all examples to {output_path}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to prepare Ukrainian dataset: {e}")
        # Fallback to manual examples if HF fails
        logger.info("Falling back to manual examples only...")
        manual_examples = get_ukrainian_examples()
        with open(output_path, 'w', encoding='utf-8') as f:
            for ex in manual_examples:
                record = {
                    'context': ex['context'],
                    'question': ex['question'],
                    'answer': ex['answer'],
                    'all_answers': [ex['answer']],
                    'answer_start': -1,
                    'is_impossible': False
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        return True


def verify_datasets():
    """Verify that all datasets are ready"""
    logger.info("Verifying datasets...")

    required_files = [
        DATA_DIR / 'squad_v2' / 'train.jsonl',
        DATA_DIR / 'squad_v2' / 'validation.jsonl',
        DATA_DIR / 'ukrainian_qa.jsonl'
    ]

    all_present = True
    for file_path in required_files:
        if file_path.exists():
            # Count lines
            with open(file_path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            logger.info(f"  ✓ {file_path.name}: {line_count} examples")
        else:
            logger.error(f"  ✗ Missing: {file_path}")
            all_present = False

    return all_present


def main():
    """Main preparation pipeline"""
    logger.info("="*60)
    logger.info("Starting Dataset Preparation")
    logger.info("="*60)

    # Step 1: Download SQuAD 2.0
    squad_success = download_squad()

    # Step 2: Prepare Ukrainian dataset
    ua_success = prepare_ukrainian_dataset()

    # Step 3: Verify all datasets
    verification_success = verify_datasets()

    logger.info("="*60)
    if squad_success and ua_success and verification_success:
        logger.info("✓ Dataset preparation completed successfully!")
        logger.info(f"  Data directory: {DATA_DIR.absolute()}")
        return 0
    else:
        logger.error("✗ Dataset preparation failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
