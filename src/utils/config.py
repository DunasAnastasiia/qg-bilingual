import yaml
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._resolve_paths()
        self._merge_base_config()

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _merge_base_config(self):
        if 'base_config' in self.config:
            base_path = self.config_path.parent / self.config['base_config']
            with open(base_path, 'r', encoding='utf-8') as f:
                base_config = yaml.safe_load(f)
            for key, value in base_config.items():
                if key not in self.config:
                    self.config[key] = value
                elif isinstance(value, dict) and isinstance(self.config[key], dict):
                    for sub_key, sub_value in value.items():
                        if sub_key not in self.config[key]:
                            self.config[key][sub_key] = sub_value

    def _resolve_paths(self):
        self.data_dir = Path(os.getenv('DATA_DIR', './data'))
        self.checkpoint_dir = Path(os.getenv('CHECKPOINT_DIR', './checkpoints'))
        self.logs_dir = Path(os.getenv('LOGS_DIR', './logs'))
        self.experiments_dir = Path(os.getenv('EXPERIMENTS_DIR', './experiments'))
        for dir_path in [self.data_dir, self.checkpoint_dir, self.logs_dir, self.experiments_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def __getitem__(self, key):
        return self.config[key]

    def __contains__(self, key):
        return key in self.config
