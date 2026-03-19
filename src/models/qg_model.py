import torch
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType

torch.backends.cudnn.benchmark = True

class QGModel:
    def __init__(self, model_name: str, config: dict, device: str = None):
        self.model_name = model_name
        self.config = config

        # Device selection
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if config.get('lora', {}).get('enabled', False):
            self._apply_lora()
        self.model.to(self.device)
        
        # Optimization: torch.compile for faster training (PyTorch 2.0+)
        if hasattr(torch, 'compile') and os.name != 'nt': # torch.compile has issues on Windows sometimes
            try:
                print("Enabling torch.compile for model optimization...")
                self.model = torch.compile(self.model)
            except Exception as e:
                print(f"Could not enable torch.compile: {e}")

    def _apply_lora(self):
        # Determine target modules based on model architecture
        if 't5' in self.model_name.lower() or 'mt5' in self.model_name.lower():
            # Expanded target modules for better T5 fine-tuning
            default_target_modules = ["q", "k", "v", "o", "wi", "wo", "wi_0", "wi_1"]
        elif 'bart' in self.model_name.lower():
            default_target_modules = ['q_proj', 'v_proj', 'k_proj', 'out_proj']
        else:
            default_target_modules = ['q', 'v']

        # Get LoRA configuration from YAML, supporting both prefixed and non-prefixed keys
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=self.config['lora'].get('r', 16),
            lora_alpha=self.config['lora'].get('lora_alpha', self.config['lora'].get('alpha', 32)),
            lora_dropout=self.config['lora'].get('lora_dropout', self.config['lora'].get('dropout', 0.1)),
            target_modules=self.config['lora'].get('target_modules', default_target_modules)
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def get_training_args(self, output_dir: str) -> Seq2SeqTrainingArguments:
        # Disable fp16/bf16 on CPU (only supported on CUDA)
        use_fp16 = self.config['training'].get('fp16', False) and self.device == 'cuda'
        use_bf16 = self.config['training'].get('bf16', False) and self.device == 'cuda'

        if (self.config['training'].get('fp16', False) or self.config['training'].get('bf16', False)) and self.device == 'cpu':
            print("⚠ WARNING: fp16/bf16 disabled (CPU doesn't support mixed precision training)")

        # Performance optimization: adjust dataloader_num_workers on Windows with gradient_checkpointing
        # to avoid pickling errors with PEFT models.
        dataloader_num_workers = self.config['training'].get('dataloader_num_workers', 0)
        if os.name == 'nt' and dataloader_num_workers > 0 and self.config['training'].get('gradient_checkpointing', False):
            print("⚠ WARNING: Setting dataloader_num_workers=0 on Windows because gradient_checkpointing is enabled (avoids pickling errors)")
            dataloader_num_workers = 0

        kwargs = {
            "output_dir": output_dir,
            "num_train_epochs": self.config['training']['num_epochs'],
            "per_device_train_batch_size": self.config['training']['batch_size'],
            "per_device_eval_batch_size": self.config['training']['batch_size'],
            "gradient_accumulation_steps": self.config['training']['gradient_accumulation_steps'],
            "learning_rate": self.config['training']['learning_rate'],
            "warmup_steps": self.config['training'].get('warmup_steps', 0),
            "warmup_ratio": self.config['training'].get('warmup_ratio', 0.0),
            "weight_decay": self.config['training']['weight_decay'],
            "label_smoothing_factor": self.config['training'].get('label_smoothing', 0.0),
            "max_grad_norm": self.config['training'].get('max_grad_norm', 1.0),
            "fp16": use_fp16,
            "bf16": use_bf16,
            "logging_steps": 100,
            "eval_strategy": 'epoch',
            "save_strategy": 'epoch',
            "save_total_limit": self.config['training']['save_total_limit'],
            "load_best_model_at_end": True,
            "metric_for_best_model": 'rouge-l',
            "greater_is_better": True,
            "predict_with_generate": True,
            "generation_max_length": self.config['data']['max_question_len'],
            "generation_num_beams": self.config['training'].get('generation_num_beams', 1), # Default to greedy for speed
            "seed": self.config['seed'],
            "gradient_checkpointing": self.config['training'].get('gradient_checkpointing', False),
            "dataloader_num_workers": dataloader_num_workers,
            "dataloader_pin_memory": self.config['training'].get('dataloader_pin_memory', self.device == 'cuda'),
            "remove_unused_columns": True,
            "group_by_length": True,
            "lr_scheduler_type": "cosine",
            "optim": "adamw_torch_fused" if self.device == 'cuda' else "adamw_torch"
        }
        
        # Prepare kwargs for Seq2SeqTrainingArguments by filtering only supported parameters
        import inspect
        sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
        supported_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        
        # Fallback for save_safetensors
        if "save_safetensors" in sig.parameters:
            supported_kwargs["save_safetensors"] = False
        else:
            # If save_safetensors is not an option, the background thread might still be an issue
            # in older versions. We try to disable it via environment variable as a fallback.
            os.environ["TRANSFORMERS_NO_ADAPTER_CONVERSION"] = "1"
            
        return Seq2SeqTrainingArguments(**supported_kwargs)

    def get_data_collator(self) -> DataCollatorForSeq2Seq:
        return DataCollatorForSeq2Seq(self.tokenizer, model=self.model, label_pad_token_id=-100)

    def save(self, output_dir: str):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def load(self, model_dir: str):
        import os
        from peft import PeftModel, PeftConfig

        # Load tokenizer from checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        adapter_cfg = os.path.join(model_dir, 'adapter_config.json')
        if os.path.exists(adapter_cfg):
            # Fresh base model to avoid double-wrapping and key mismatches
            peft_cfg = PeftConfig.from_pretrained(model_dir)
            base_name = peft_cfg.base_model_name_or_path
            base = AutoModelForSeq2SeqLM.from_pretrained(base_name)
            model = PeftModel.from_pretrained(base, model_dir)
            try:
                # Merge adapters for fast inference and to drop peft_config from live model
                model = model.merge_and_unload()
            except Exception as e:
                print(f"⚠ Unable to merge adapters: {e}. Using PEFT-wrapped model.")
            self.model = model
        else:
            # Load full model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

        self.model.to(self.device)
