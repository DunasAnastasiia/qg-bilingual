#!/usr/bin/env python3
import sys
import os
import logging
import subprocess
from pathlib import Path
from typing import List, Dict
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent
CONFIGS_DIR = ROOT_DIR / 'configs'
CHECKPOINT_DIR = ROOT_DIR / 'checkpoints'
CHECKPOINT_DIR.mkdir(exist_ok=True)

MODELS_TO_TRAIN: List[Dict[str, str]] = [
    {'name': 't5_base_en_aware', 'config': 'train_t5_base.yaml', 'description': 'T5-base English Answer-Aware'},
    {'name': 't5_base_en_agnostic', 'config': 'train_t5_base.yaml', 'description': 'T5-base English Answer-Agnostic', 'override_mode': 'answer_agnostic'},
    {'name': 'bart_base_en_aware', 'config': 'train_bart_base_aware_en.yaml', 'description': 'BART-base English Answer-Aware'},
    {'name': 'bart_base_en_agnostic', 'config': 'train_bart_base_agnostic_en.yaml', 'description': 'BART-base English Answer-Agnostic'},
    {'name': 'mt5_base_ua_aware', 'config': 'train_mt5_base_aware_ua.yaml', 'description': 'mT5-base Ukrainian Answer-Aware'},
    {'name': 'mt5_base_ua_agnostic', 'config': 'train_mt5_base_aware_ua.yaml', 'description': 'mT5-base Ukrainian Answer-Agnostic', 'override_mode': 'answer_agnostic'}
]

def check_model_trained(model_name: str) -> bool:
    model_dir = CHECKPOINT_DIR / model_name / 'final_model'
    config_file = model_dir / 'config.json'
    if config_file.exists():
        logger.info(f"Model {model_name} already trained, skipping...")
        return True
    return False

def train_model(model_config: Dict[str, str]) -> bool:
    model_name = model_config['name']
    config_path = CONFIGS_DIR / model_config['config']
    logger.info(f"Training: {model_config['description']}")
    if check_model_trained(model_name):
        return True
    cmd = ['python3', 'src/train.py', '--config', str(config_path)]
    if 'override_mode' in model_config:
        cmd.extend(['--mode', model_config['override_mode']])
    
    # Ensure ROOT_DIR and src are in PYTHONPATH so internal imports work
    env = os.environ.copy()
    src_path = str(ROOT_DIR / 'src')
    root_path = str(ROOT_DIR)
    
    # Adding both to be safe, though root_path should be enough if using 'from src.models...'
    # but since current code uses 'from models...', src_path is needed.
    python_path = [src_path, root_path]
    if 'PYTHONPATH' in env:
        python_path.append(env['PYTHONPATH'])
    
    env['PYTHONPATH'] = os.pathsep.join(python_path)

    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=ROOT_DIR, check=True, capture_output=False, text=True, env=env)
        logger.info(f"Successfully trained {model_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to train {model_name}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error training {model_name}: {e}")
        return False

def create_model_registry():
    registry_file = CHECKPOINT_DIR / 'model_registry.json'
    registry = {'models': []}
    for model_config in MODELS_TO_TRAIN:
        model_name = model_config['name']
        model_dir = CHECKPOINT_DIR / model_name / 'final_model'
        if model_dir.exists():
            parts = model_name.split('_')
            architecture = parts[0]
            language = 'en' if 'en' in model_name else 'ua'
            mode = 'aware' if 'aware' in model_name else 'agnostic'
            registry['models'].append({'id': model_name, 'name': model_config['description'], 'architecture': architecture, 'language': language, 'mode': mode, 'checkpoint_path': str(model_dir)})
    with open(registry_file, 'w') as f:
        json.dump(registry, f, indent=2)
    logger.info(f"Created model registry with {len(registry['models'])} models")

def main():
    logger.info("Starting Training Pipeline for All 6 Models")
    trained_count = 0
    failed_count = 0
    for i, model_config in enumerate(MODELS_TO_TRAIN, 1):
        logger.info(f"[{i}/{len(MODELS_TO_TRAIN)}] Processing {model_config['name']}...")
        success = train_model(model_config)
        if success:
            trained_count += 1
        else:
            failed_count += 1
    create_model_registry()
    logger.info(f"Training Summary - Total: {len(MODELS_TO_TRAIN)}, Success: {trained_count}, Failed: {failed_count}")
    return 0 if failed_count == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
