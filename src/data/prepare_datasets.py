#!/usr/bin/env python3
"""
Dataset Preparation Script
Downloads SQuAD 2.0 and prepares Ukrainian Q&A dataset
"""
import sys
import json
import logging
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / 'data'
DATA_DIR.mkdir(exist_ok=True)


def download_squad_v2():
    """Download SQuAD 2.0 dataset"""
    logger.info("Downloading SQuAD 2.0 dataset...")

    try:
        dataset = load_dataset('squad_v2')
        logger.info(f"✓ SQuAD 2.0 downloaded successfully")
        logger.info(f"  - Train examples: {len(dataset['train'])}")
        logger.info(f"  - Validation examples: {len(dataset['validation'])}")

        # Save to local cache
        squad_dir = DATA_DIR / 'squad_v2'
        squad_dir.mkdir(exist_ok=True)

        for split in ['train', 'validation']:
            output_file = squad_dir / f'{split}.jsonl'
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


def prepare_ukrainian_dataset():
    """Prepare Ukrainian Q&A dataset"""
    logger.info("Preparing Ukrainian Q&A dataset...")

    ua_file = DATA_DIR / 'ukrainian_qa.jsonl'

    # Check if file exists
    if not ua_file.exists():
        logger.warning(f"Ukrainian dataset not found at {ua_file}")
        logger.info("Creating sample Ukrainian dataset...")

        # Create sample Ukrainian examples
        sample_examples = [
            {
                'context': 'Київ є столицею та найбільшим містом України, розташованим на річці Дніпро. Це одне з найдавніших міст Східної Європи, засноване понад 1400 років тому.',
                'question': 'Яке місто є столицею України?',
                'answer': 'Київ',
                'language': 'ua',
                'is_impossible': False
            },
            {
                'context': 'Львів - одне з найкрасивіших міст України, розташоване на заході країни. Місто відоме своєю унікальною архітектурою та кавовою культурою.',
                'question': 'Де розташований Львів?',
                'answer': 'на заході країни',
                'language': 'ua',
                'is_impossible': False
            },
            {
                'context': 'Україна здобула незалежність 24 серпня 1991 року. Це була історична подія для українського народу.',
                'question': 'Коли Україна здобула незалежність?',
                'answer': '24 серпня 1991 року',
                'language': 'ua',
                'is_impossible': False
            }
        ]

        with open(ua_file, 'w', encoding='utf-8') as f:
            for ex in sample_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')

        logger.info(f"✓ Created sample Ukrainian dataset with {len(sample_examples)} examples")
        logger.warning("⚠ For production, please add more Ukrainian Q&A examples to this file")
    else:
        # Validate existing file
        count = 0
        with open(ua_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    json.loads(line)
                    count += 1
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON line: {e}")

        logger.info(f"✓ Found existing Ukrainian dataset with {count} examples")

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
    squad_success = download_squad_v2()

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
