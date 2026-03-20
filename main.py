#!/usr/bin/env python3
"""
Main CLI for Question Generation Project

NEW Flexible Interface:
  python main.py train --model t5_base_en_aware --dataset 20
  python main.py train --model bart_base_en_agnostic --dataset 50
  python main.py train --model mt5_base_ua_aware --dataset 100

Legacy Commands:
  python main.py download_demo
  python main.py download_production
  python main.py train_production
  python main.py run_ui
  python main.py status
"""

import argparse
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.prepare_datasets import download_squad, prepare_ukrainian_dataset

AVAILABLE_MODELS = {
    "t5_base_en_aware": {
        "config": "configs/models/t5_base_en_aware.yaml",
        "name": "T5 Base (EN, Answer-Aware)",
        "language": "en",
    },
    "t5_base_en_agnostic": {
        "config": "configs/models/t5_base_en_agnostic.yaml",
        "name": "T5 Base (EN, Answer-Agnostic)",
        "language": "en",
    },
    "bart_base_en_aware": {
        "config": "configs/models/bart_base_en_aware.yaml",
        "name": "BART Base (EN, Answer-Aware)",
        "language": "en",
    },
    "bart_base_en_agnostic": {
        "config": "configs/models/bart_base_en_agnostic.yaml",
        "name": "BART Base (EN, Answer-Agnostic)",
        "language": "en",
    },
    "mt5_base_ua_aware": {
        "config": "configs/models/mt5_base_ua_aware.yaml",
        "name": "mT5 Base (UA, Answer-Aware)",
        "language": "ua",
    },
    "mt5_base_ua_agnostic": {
        "config": "configs/models/mt5_base_ua_agnostic.yaml",
        "name": "mT5 Base (UA, Answer-Agnostic)",
        "language": "ua",
    },
}


class ProjectCLI:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data"
        self.checkpoints_dir = self.project_root / "checkpoints"
        self.configs_dir = self.project_root / "configs"

        self.data_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)

    def download_demo(self):
        """Download limited dataset for demo/smoke testing"""
        print("=" * 60)
        print("DOWNLOADING DEMO DATASET (Limited)")
        print("=" * 60)

        print("\n[1/2] Downloading English dataset (SQuAD 2.0 demo subset)...")
        demo_size = 1000
        download_squad(
            output_dir=self.data_dir / "squad_v2", demo_mode=True, demo_size=demo_size
        )

        print("\n[2/2] Preparing Ukrainian dataset (demo subset)...")
        prepare_ukrainian_dataset(
            output_path=self.data_dir / "ukrainian_qa.jsonl",
            demo_mode=True,
            demo_size=demo_size,
        )

        print("\n✓ Demo dataset downloaded successfully!")
        print(f"  - English: {self.data_dir / 'squad_v2'}")
        print(f"  - Ukrainian: {self.data_dir / 'ukrainian_qa.jsonl'}")

    def download_production(self):
        print("=" * 60)
        print("DOWNLOADING PRODUCTION DATASET (Full)")
        print("=" * 60)

        print("\n[1/2] Downloading English dataset (SQuAD 2.0 full)...")
        download_squad(output_dir=self.data_dir / "squad_v2", demo_mode=False)
        print("\n[2/2] Preparing Ukrainian dataset (full)...")
        prepare_ukrainian_dataset(
            output_path=self.data_dir / "ukrainian_qa.jsonl", demo_mode=False
        )

        print("\n✓ Production dataset downloaded successfully!")
        print(f"  - English: {self.data_dir / 'squad_v2'}")
        print(f"  - Ukrainian: {self.data_dir / 'ukrainian_qa.jsonl'}")

    def train_demo(self):
        """Train demo version with limited dataset"""
        print("=" * 60)
        print("TRAINING DEMO VERSION")
        print("=" * 60)

        if not (self.data_dir / "squad_v2").exists():
            print("⚠ Demo dataset not found. Downloading first...")
            self.download_demo()

        models_to_train = [
            ("configs/train_t5_smoke_en.yaml", "T5-small (EN)"),
        ]

        for config_path, name in models_to_train:
            print(f"\n\n[TRAINING] {name}")
            print("-" * 60)
            full_config_path = self.project_root / config_path

            if not full_config_path.exists():
                print(f"⚠ Config not found: {config_path}")
                continue

            try:
                cmd = [
                    sys.executable,
                    str(self.project_root / "src" / "train.py"),
                    "--config",
                    str(full_config_path),
                ]
                subprocess.run(cmd, check=True)
                print(f"✓ {name} training completed")
            except subprocess.CalledProcessError as e:
                print(f"✗ {name} training failed: {e}")

        print("\n" + "=" * 60)
        print("✓ Demo training completed!")
        print("=" * 60)

    def train_production(self):
        print("=" * 60)
        print("TRAINING PRODUCTION MODELS")
        print("=" * 60)

        if not (self.data_dir / "squad_v2").exists():
            print("⚠ Production dataset not found. Downloading first...")
            self.download_production()

        models_to_train = [
            # English - Answer Aware
            ("configs/models/t5_base_en_aware.yaml", "T5 Base (EN, Answer-Aware)"),
            ("configs/models/bart_base_en_aware.yaml", "BART Base (EN, Answer-Aware)"),
            # English - Answer Agnostic
            (
                "configs/models/t5_base_en_agnostic.yaml",
                "T5 Base (EN, Answer-Agnostic)",
            ),
            (
                "configs/models/bart_base_en_agnostic.yaml",
                "BART Base (EN, Answer-Agnostic)",
            ),
            # Ukrainian - Answer Aware
            ("configs/models/mt5_base_ua_aware.yaml", "mT5 Base (UA, Answer-Aware)"),
            # Ukrainian - Answer Agnostic
            (
                "configs/models/mt5_base_ua_agnostic.yaml",
                "mT5 Base (UA, Answer-Agnostic)",
            ),
        ]

        for config_path, name in models_to_train:
            print(f"\n\n[TRAINING] {name}")
            print("-" * 60)
            full_config_path = self.project_root / config_path

            if not full_config_path.exists():
                print(f"⚠ Config not found: {config_path}")
                continue

            try:
                cmd = [
                    sys.executable,
                    str(self.project_root / "src" / "train.py"),
                    "--config",
                    str(full_config_path),
                ]
                subprocess.run(cmd, check=True)
                print(f"✓ {name} training completed")
            except subprocess.CalledProcessError as e:
                print(f"✗ {name} training failed: {e}")
                continue

        print("\n" + "=" * 60)
        print("✓ All production models trained successfully!")
        print("=" * 60)

    def run_ui(self):
        print("=" * 60)
        print("LAUNCHING UI")
        print("=" * 60)

        # Check if models exist
        required_models = [
            "checkpoints/t5_base_en_aware",
            "checkpoints/bart_base_en_aware",
            "checkpoints/mt5_base_ua_aware",
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
            cmd = [sys.executable, str(self.project_root / "src" / "ui.py")]
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\n\n✓ UI stopped")

    def train_model(self, model_name: str, dataset_percent: int = 100):
        print("=" * 80)
        print(f"TRAINING MODEL: {model_name}")
        print(f"Dataset: {dataset_percent}% of full data")
        print("=" * 80)

        if model_name not in AVAILABLE_MODELS:
            print(f"\n❌ Error: Unknown model '{model_name}'")
            print("\nAvailable models:")
            for key, info in AVAILABLE_MODELS.items():
                print(f"  • {key:25} → {info['name']}")
            sys.exit(1)

        model_info = AVAILABLE_MODELS[model_name]
        config_path = self.project_root / model_info["config"]

        if not config_path.exists():
            print(f"\n❌ Error: Config not found at {config_path}")
            sys.exit(1)

        language = model_info["language"]
        if language == "en":
            dataset_path = self.data_dir / "squad_v2"
            if not dataset_path.exists():
                print("\n⚠ English dataset not found. Downloading...")
                download_squad(output_dir=dataset_path, demo_mode=False)
        else:
            dataset_path = self.data_dir / "ukrainian_qa.jsonl"
            if not dataset_path.exists():
                print("\n⚠ Ukrainian dataset not found. Preparing...")
                prepare_ukrainian_dataset(output_path=dataset_path, demo_mode=False)

        print(f"\n✓ Dataset found: {dataset_path}")
        print(f"✓ Config: {config_path}")
        print(f"✓ Model: {model_info['name']}")

        cmd = [
            sys.executable,
            str(self.project_root / "src" / "train.py"),
            "--config",
            str(config_path),
            "--dataset_percent",
            str(dataset_percent),
        ]

        print(f"\n🚀 Starting training...")
        print(f"Command: {' '.join(cmd)}")
        print("-" * 80)

        try:
            subprocess.run(cmd, check=True)
            print("\n" + "=" * 80)
            print(f"✅ Training completed: {model_info['name']}")
            print("=" * 80)
        except subprocess.CalledProcessError as e:
            print("\n" + "=" * 80)
            print(f"❌ Training failed: {model_info['name']}")
            print(f"Error: {e}")
            print("=" * 80)
            sys.exit(1)

    def list_models(self):
        print("=" * 80)
        print("AVAILABLE MODELS")
        print("=" * 80)
        print()

        for key, info in AVAILABLE_MODELS.items():
            checkpoint_path = self.project_root / f"checkpoints/{key}/final_model"
            status = "✅ TRAINED" if checkpoint_path.exists() else "❌ NOT TRAINED"

            print(f"{status:15} | {key:25} | {info['name']}")

        print()
        print("=" * 80)
        print("\nUsage:")
        print("  python main.py train --model <model_name> --dataset <percentage>")
        print("\nExample:")
        print("  python main.py train --model t5_base_en_aware --dataset 20")
        print("=" * 80)

    def evaluate_model(
        self, model_name: str, split: str = "validation", max_samples: int = None
    ):
        print("=" * 80)
        print(f"EVALUATING MODEL: {model_name}")
        print(f"Split: {split}")
        if max_samples:
            print(f"Max samples: {max_samples}")
        print("=" * 80)

        if model_name not in AVAILABLE_MODELS:
            print(f"\n❌ Error: Unknown model '{model_name}'")
            print("\nAvailable models:")
            for key, info in AVAILABLE_MODELS.items():
                print(f"  • {key:25} → {info['name']}")
            sys.exit(1)

        model_info = AVAILABLE_MODELS[model_name]
        config_path = self.project_root / model_info["config"]
        checkpoint_path = self.project_root / f"checkpoints/{model_name}/final_model"

        if not checkpoint_path.exists():
            print(f"\n❌ Error: Model not found at {checkpoint_path}")
            print("\nPlease train the model first:")
            print(f"  python main.py train --model {model_name}")
            sys.exit(1)

        if not config_path.exists():
            print(f"\n❌ Error: Config not found at {config_path}")
            sys.exit(1)

        print(f"\n✓ Model checkpoint: {checkpoint_path}")
        print(f"✓ Config: {config_path}")
        print(f"✓ Model: {model_info['name']}")

        cmd = [
            sys.executable,
            str(self.project_root / "src" / "evaluate_model.py"),
            "--checkpoint",
            str(checkpoint_path),
            "--config",
            str(config_path),
            "--split",
            split,
        ]

        if max_samples:
            cmd.extend(["--max_samples", str(max_samples)])

        print(f"\n🔍 Starting evaluation...")
        print("-" * 80)

        try:
            subprocess.run(cmd, check=True)
            print("\n" + "=" * 80)
            print(f"✅ Evaluation completed: {model_info['name']}")
            print("=" * 80)
        except subprocess.CalledProcessError as e:
            print("\n" + "=" * 80)
            print(f"❌ Evaluation failed: {model_info['name']}")
            print(f"Error: {e}")
            print("=" * 80)
            sys.exit(1)

    def show_status(self):
        print("=" * 60)
        print("PROJECT STATUS")
        print("=" * 60)

        print("\n📁 DATASETS:")

        en_data = self.data_dir / "squad_v2"
        if en_data.exists():
            train_file = en_data / "train.jsonl"
            val_file = en_data / "validation.jsonl"
            if train_file.exists() and val_file.exists():
                print(f"  ✓ English (SQuAD 2.0): {en_data}")
            else:
                print(f"  ⚠ English: Incomplete")
        else:
            print(f"  ✗ English: Not downloaded")

        ua_data = self.data_dir / "ukrainian_qa.jsonl"
        if ua_data.exists():
            print(f"  ✓ Ukrainian: {ua_data}")
        else:
            print(f"  ✗ Ukrainian: Not prepared")

        print("\n🤖 TRAINED MODELS:")

        models = [
            ("checkpoints/t5_base_en_aware", "T5 Base (EN, Answer-Aware)"),
            ("checkpoints/t5_base_en_agnostic", "T5 Base (EN, Answer-Agnostic)"),
            ("checkpoints/bart_base_en_aware", "BART Base (EN, Answer-Aware)"),
            ("checkpoints/bart_base_en_agnostic", "BART Base (EN, Answer-Agnostic)"),
            ("checkpoints/mt5_base_ua_aware", "mT5 Base (UA, Answer-Aware)"),
            ("checkpoints/mt5_base_ua_agnostic", "mT5 Base (UA, Answer-Agnostic)"),
        ]

        for path, name in models:
            full_path = self.project_root / path
            if full_path.exists() and (full_path / "final_model").exists():
                print(f"  ✓ {name}")
            else:
                print(f"  ✗ {name}")

        print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Question Generation Project - Main CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # NEW: Flexible training with dataset percentage
  python main.py train --model t5_base_en_aware --dataset 20
  python main.py train --model bart_base_en_agnostic --dataset 50
  python main.py train --model mt5_base_ua_aware --dataset 100

  # Evaluate a trained model (shows all metrics with goal values)
  python main.py evaluate --model t5_base_en_aware
  python main.py evaluate --model mt5_base_ua_aware --split test
  python main.py evaluate --model t5_base_en_aware --max_samples 100

  # List available models
  python main.py models

  # Legacy commands
  python main.py download_production
  python main.py run_ui
  python main.py status
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    train_parser = subparsers.add_parser("train", help="Train a specific model")
    train_parser.add_argument(
        "--model",
        "-m",
        required=True,
        choices=list(AVAILABLE_MODELS.keys()),
        help="Model to train",
    )
    train_parser.add_argument(
        "--dataset",
        "-d",
        type=int,
        default=100,
        choices=range(1, 101),
        metavar="PERCENT",
        help="Percentage of dataset to use (1-100, default: 100)",
    )

    subparsers.add_parser("models", help="List all available models")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument(
        "--model",
        "-m",
        required=True,
        choices=list(AVAILABLE_MODELS.keys()),
        help="Model to evaluate",
    )
    eval_parser.add_argument(
        "--split",
        "-s",
        default="validation",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate (default: validation)",
    )
    eval_parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples to evaluate (default: all)",
    )

    subparsers.add_parser("download_demo", help="Download demo dataset")
    subparsers.add_parser("download_production", help="Download full dataset")
    subparsers.add_parser("train_demo", help="Train demo model")
    subparsers.add_parser("train_production", help="Train all models")
    subparsers.add_parser("run_ui", help="Launch Gradio UI")
    subparsers.add_parser("status", help="Show project status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    cli = ProjectCLI()

    if args.command == "train":
        cli.train_model(args.model, args.dataset)
    elif args.command == "models":
        cli.list_models()
    elif args.command == "evaluate":
        cli.evaluate_model(args.model, args.split, args.max_samples)
    elif args.command == "download_demo":
        cli.download_demo()
    elif args.command == "download_production":
        cli.download_production()
    elif args.command == "train_demo":
        cli.train_demo()
    elif args.command == "train_production":
        cli.train_production()
    elif args.command == "run_ui":
        cli.run_ui()
    elif args.command == "status":
        cli.show_status()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
