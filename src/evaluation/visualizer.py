import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List
import json

class MetricsVisualizer:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style('whitegrid')

    def plot_training_curves(self, train_losses: List[float], val_losses: List[float], val_rouge_scores: List[float], save_name: str = 'training_curves.png'):
        if not train_losses or not val_losses or not val_rouge_scores:
            print(f"Warning: Skipping plot - empty data (train: {len(train_losses)}, val: {len(val_losses)}, rouge: {len(val_rouge_scores)})")
            return

        min_len = min(len(train_losses), len(val_losses), len(val_rouge_scores))
        if min_len == 0:
            print("Warning: Skipping plot - no data available")
            return

        train_losses = train_losses[:min_len]
        val_losses = val_losses[:min_len]
        val_rouge_scores = val_rouge_scores[:min_len]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        epochs = range(1, min_len + 1)

        ax1.plot(epochs, train_losses, label='Train Loss', marker='o')
        ax1.plot(epochs, val_losses, label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs, val_rouge_scores, label='Val ROUGE-L', marker='o', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('ROUGE-L')
        ax2.set_title('Validation ROUGE-L Score')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved training curves to {save_path.resolve()}")
