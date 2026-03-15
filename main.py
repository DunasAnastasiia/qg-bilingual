#!/usr/bin/env python3
"""
Main CLI for Question Generation Project
Supports:
1. Download and train demo version (limited dataset)
2. Download production dataset
3. Train production dataset
4. Run UI with all trained models
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.prepare_datasets import download_squad, prepare_ukrainian_dataset
from src.utils.config import Config


class ProjectCLI:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / 'data'
        self.checkpoints_dir = self.project_root / 'checkpoints'
        self.configs_dir = self.project_root / 'configs'

        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)

    def download_demo(self):
        """Download limited dataset for demo/smoke testing"""
        print("=" * 60)
        print("DOWNLOADING DEMO DATASET (Limited)")
        print("=" * 60)

        # Download English dataset (small subset)
        print("\n[1/2] Downloading English dataset (SQuAD 2.0 demo subset)...")
        demo_size = 1000  # Small subset for demo
        download_squad(
            output_dir=self.data_dir / 'squad_v2',
            demo_mode=True,
            demo_size=demo_size
        )

        # Download/prepare Ukrainian dataset (small subset)
        print("\n[2/2] Preparing Ukrainian dataset (demo subset)...")
        prepare_ukrainian_dataset(
            output_path=self.data_dir / 'ukrainian_qa.jsonl',
            demo_mode=True,
            demo_size=demo_size
        )

        print("\n✓ Demo dataset downloaded successfully!")
        print(f"  - English: {self.data_dir / 'squad_v2'}")
        print(f"  - Ukrainian: {self.data_dir / 'ukrainian_qa.jsonl'}")

    def download_production(self):
        """Download full production dataset"""
        print("=" * 60)
        print("DOWNLOADING PRODUCTION DATASET (Full)")
        print("=" * 60)

        # Download English dataset (full)
        print("\n[1/2] Downloading English dataset (SQuAD 2.0 full)...")
        download_squad(
            output_dir=self.data_dir / 'squad_v2',
            demo_mode=False
        )

        # Download/prepare Ukrainian dataset (full)
        print("\n[2/2] Preparing Ukrainian dataset (full)...")
        prepare_ukrainian_dataset(
            output_path=self.data_dir / 'ukrainian_qa.jsonl',
            demo_mode=False
        )

        print("\n✓ Production dataset downloaded successfully!")
        print(f"  - English: {self.data_dir / 'squad_v2'}")
        print(f"  - Ukrainian: {self.data_dir / 'ukrainian_qa.jsonl'}")

    def train_demo(self):
        """Train demo version with limited dataset"""
        print("=" * 60)
        print("TRAINING DEMO VERSION")
        print("=" * 60)

        # Check if demo data exists
        if not (self.data_dir / 'squad_v2').exists():
            print("⚠ Demo dataset not found. Downloading first...")
            self.download_demo()

        # Train small/fast models for demo
        models_to_train = [
            ('configs/train_t5_smoke_en.yaml', 'T5-small (EN)'),
        ]

        for config_path, name in models_to_train:
            print(f"\n\n[TRAINING] {name}")
            print("-" * 60)
            full_config_path = self.project_root / config_path

            if not full_config_path.exists():
                print(f"⚠ Config not found: {config_path}")
                continue

            try:
                cmd = [sys.executable, str(self.project_root / 'src' / 'train.py'),
                       '--config', str(full_config_path)]
                subprocess.run(cmd, check=True)
                print(f"✓ {name} training completed")
            except subprocess.CalledProcessError as e:
                print(f"✗ {name} training failed: {e}")

        print("\n" + "=" * 60)
        print("✓ Demo training completed!")
        print("=" * 60)

    def train_production(self):
        """Train all production models"""
        print("=" * 60)
        print("TRAINING PRODUCTION MODELS")
        print("=" * 60)

        # Check if production data exists
        if not (self.data_dir / 'squad_v2').exists():
            print("⚠ Production dataset not found. Downloading first...")
            self.download_production()

        # Train all 6 models: T5, BART, mT5 × (aware, agnostic)
        models_to_train = [
            # English - Answer Aware
            ('configs/models/t5_base_en_aware.yaml', 'T5 Base (EN, Answer-Aware)'),
            ('configs/models/bart_base_en_aware.yaml', 'BART Base (EN, Answer-Aware)'),

            # English - Answer Agnostic
            ('configs/models/t5_base_en_agnostic.yaml', 'T5 Base (EN, Answer-Agnostic)'),
            ('configs/models/bart_base_en_agnostic.yaml', 'BART Base (EN, Answer-Agnostic)'),

            # Ukrainian - Answer Aware
            ('configs/models/mt5_base_ua_aware.yaml', 'mT5 Base (UA, Answer-Aware)'),

            # Ukrainian - Answer Agnostic
            ('configs/models/mt5_base_ua_agnostic.yaml', 'mT5 Base (UA, Answer-Agnostic)'),
        ]

        for config_path, name in models_to_train:
            print(f"\n\n[TRAINING] {name}")
            print("-" * 60)
            full_config_path = self.project_root / config_path

            if not full_config_path.exists():
                print(f"⚠ Config not found: {config_path}")
                continue

            try:
                cmd = [sys.executable, str(self.project_root / 'src' / 'train.py'),
                       '--config', str(full_config_path)]
                subprocess.run(cmd, check=True)
                print(f"✓ {name} training completed")
            except subprocess.CalledProcessError as e:
                print(f"✗ {name} training failed: {e}")
                continue

        print("\n" + "=" * 60)
        print("✓ All production models trained successfully!")
        print("=" * 60)

    def run_ui(self):
        """Run UI with all trained models"""
        print("=" * 60)
        print("LAUNCHING UI")
        print("=" * 60)

        # Check if models exist
        required_models = [
            'checkpoints/t5_base_en_aware',
            'checkpoints/bart_base_en_aware',
            'checkpoints/mt5_base_ua_aware'
        ]

        missing_models = []
        for model_path in required_models:
            full_path = self.project_root / model_path
            if not full_path.exists():
                missing_models.append(model_path)

        if missing_models:
            print("\n⚠ WARNING: Some models are missing:")
            for model in missing_models:
                print(f"  - {model}")
            print("\nYou should train models first:")
            print("  python main.py train_demo    (for quick demo)")
            print("  python main.py train_production    (for all models)")
            print("\nContinuing anyway (some models may not be available in UI)...")

        print(f"\nLaunching UI at http://localhost:7860")
        print("Press Ctrl+C to stop\n")

        try:
            cmd = [sys.executable, str(self.project_root / 'src' / 'ui.py')]
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\n\n✓ UI stopped")


    def show_status(self):
        """Show status of datasets and trained models"""
        print("=" * 60)
        print("PROJECT STATUS")
        print("=" * 60)

        print("\n📁 DATASETS:")

        # Check English dataset
        en_data = self.data_dir / 'squad_v2'
        if en_data.exists():
            train_file = en_data / 'train.jsonl'
            val_file = en_data / 'validation.jsonl'
            if train_file.exists() and val_file.exists():
                print(f"  ✓ English (SQuAD 2.0): {en_data}")
            else:
                print(f"  ⚠ English: Incomplete")
        else:
            print(f"  ✗ English: Not downloaded")

        # Check Ukrainian dataset
        ua_data = self.data_dir / 'ukrainian_qa.jsonl'
        if ua_data.exists():
            print(f"  ✓ Ukrainian: {ua_data}")
        else:
            print(f"  ✗ Ukrainian: Not prepared")

        print("\n🤖 TRAINED MODELS:")

        models = [
            ('checkpoints/t5_base_en_aware', 'T5 Base (EN, Answer-Aware)'),
            ('checkpoints/t5_base_en_agnostic', 'T5 Base (EN, Answer-Agnostic)'),
            ('checkpoints/bart_base_en_aware', 'BART Base (EN, Answer-Aware)'),
            ('checkpoints/bart_base_en_agnostic', 'BART Base (EN, Answer-Agnostic)'),
            ('checkpoints/mt5_base_ua_aware', 'mT5 Base (UA, Answer-Aware)'),
            ('checkpoints/mt5_base_ua_agnostic', 'mT5 Base (UA, Answer-Agnostic)'),
        ]

        for path, name in models:
            full_path = self.project_root / path
            if full_path.exists() and (full_path / 'final_model').exists():
                print(f"  ✓ {name}")
            else:
                print(f"  ✗ {name}")

        print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Question Generation Project - Main CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and train demo version (fast, limited dataset)
  python main.py download_demo
  python main.py train_demo

  # Download and train production version (full dataset, all models)
  python main.py download_production
  python main.py train_production

  # Run UI with trained models
  python main.py run_ui

  # Check project status
  python main.py status
        """
    )

    parser.add_argument(
        'command',
        choices=['download_demo', 'download_production', 'train_demo',
                 'train_production', 'run_ui', 'status'],
        help='Command to execute'
    )

    args = parser.parse_args()

    cli = ProjectCLI()

    commands = {
        'download_demo': cli.download_demo,
        'download_production': cli.download_production,
        'train_demo': cli.train_demo,
        'train_production': cli.train_production,
        'run_ui': cli.run_ui,
        'status': cli.show_status,
    }

    commands[args.command]()


if __name__ == '__main__':
    main()
