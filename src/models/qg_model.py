import torch
import os
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

torch.backends.cudnn.benchmark = True

class QGModel:
    def __init__(self, model_name: str, config: dict, device: str = None):
        self.model_name = model_name
        self.config = config

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, 
            dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32,
            device_map={"": self.device} if self.device == 'cuda' else None
        )
        if config.get('lora', {}).get('enabled', False):
            self._apply_lora()
        self.model.to(self.device)

        self.compiled = False

    def _apply_lora(self):
        if 't5' in self.model_name.lower() or 'mt5' in self.model_name.lower():
            default_target_modules = ["q", "k", "v", "o", "wi", "wo", "wi_0", "wi_1"]
        elif 'bart' in self.model_name.lower():
            default_target_modules = ['q_proj', 'v_proj', 'k_proj', 'out_proj', 'fc1', 'fc2']
        else:
            default_target_modules = ['q', 'v']

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
        use_fp16 = self.config['training'].get('fp16', False) and self.device == 'cuda'
        use_bf16 = self.config['training'].get('bf16', False) and self.device == 'cuda'

        if (self.config['training'].get('fp16', False) or self.config['training'].get('bf16', False)) and self.device == 'cpu':
            print("⚠ WARNING: fp16/bf16 disabled (CPU doesn't support mixed precision training)")

        dataloader_num_workers = self.config['training'].get('dataloader_num_workers', 0)
        if os.name == 'nt' and dataloader_num_workers > 0:
            if self.config['training'].get('gradient_checkpointing', False) or self.compiled:
                reason = "gradient_checkpointing is enabled" if self.config['training'].get('gradient_checkpointing', False) else "torch.compile is enabled"
                print(f"⚠ WARNING: Setting dataloader_num_workers=0 on Windows because {reason} (avoids pickling errors)")
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
            "remove_unused_columns": False,
            "group_by_length": True,
            "lr_scheduler_type": "cosine",
            "optim": "adamw_torch_fused" if self.device == 'cuda' else "adamw_torch"
        }

        import inspect
        sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
        supported_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

        if "save_safetensors" in sig.parameters:
            supported_kwargs["save_safetensors"] = False
        else:
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
            peft_cfg = PeftConfig.from_pretrained(model_dir)
            base_name = peft_cfg.base_model_name_or_path
            base = AutoModelForSeq2SeqLM.from_pretrained(base_name)
            model = PeftModel.from_pretrained(base, model_dir)
            try:
                model = model.merge_and_unload()
            except Exception as e:
                print(f"⚠ Unable to merge adapters: {e}. Using PEFT-wrapped model.")
            self.model = model
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

        self.model.to(self.device)
