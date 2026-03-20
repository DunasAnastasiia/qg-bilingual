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
        is_docker = (os.path.exists('/.dockerenv') and os.name != 'nt') or os.getenv('DOCKER_ENV') == 'true'

        project_root = Path(__file__).resolve().parent.parent.parent

        def resolve_env_path(env_key, default_docker, default_local_name):
            val = os.getenv(env_key)
            if is_docker:
                return Path(val or default_docker)
            else:
                if not val or val.startswith('/app'):
                    return project_root / default_local_name
                
                p = Path(val)
                if os.name == 'nt' and not p.is_absolute() and val.startswith(('/', '\\')):
                    return project_root / val.lstrip('/\\')
                
                return p if p.is_absolute() else project_root / p

        self.data_dir = resolve_env_path('DATA_DIR', '/app/data', 'data')
        self.checkpoint_dir = resolve_env_path('CHECKPOINT_DIR', '/app/checkpoints', 'checkpoints')
        self.logs_dir = resolve_env_path('LOGS_DIR', '/app/logs', 'logs')
        self.experiments_dir = resolve_env_path('EXPERIMENTS_DIR', '/app/experiments', 'experiments')

        for dir_path in [self.data_dir, self.checkpoint_dir, self.logs_dir, self.experiments_dir]:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Warning: Could not create directory {dir_path}: {e}")

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
