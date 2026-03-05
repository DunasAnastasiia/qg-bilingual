import torch
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType

torch.backends.cudnn.benchmark = True

class QGModel:
    def __init__(self, model_name: str, config: dict, device: str = None):
        self.model_name = model_name
        self.config = config
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if config.get('lora', {}).get('enabled', False):
            self._apply_lora()
        self.model.to(self.device)

    def _apply_lora(self):
        if 't5' in self.model_name.lower():
            target_modules = ['q', 'v']
        elif 'bart' in self.model_name.lower():
            target_modules = ['q_proj', 'v_proj']
        else:
            target_modules = ['q', 'v']
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora'].get('alpha', 16),
            lora_dropout=self.config['lora']['dropout'],
            target_modules=target_modules
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def get_training_args(self, output_dir: str) -> Seq2SeqTrainingArguments:
        kwargs = {
            "output_dir": output_dir,
            "num_train_epochs": self.config['training']['num_epochs'],
            "per_device_train_batch_size": self.config['training']['batch_size'],
            "per_device_eval_batch_size": self.config['training']['batch_size'],
            "gradient_accumulation_steps": self.config['training']['gradient_accumulation_steps'],
            "learning_rate": self.config['training']['learning_rate'],
            "warmup_ratio": self.config['training']['warmup_ratio'],
            "weight_decay": self.config['training']['weight_decay'],
            "fp16": self.config['training'].get('fp16', False),
            "bf16": self.config['training'].get('bf16', False),
            "logging_steps": 100,
            "eval_strategy": 'epoch',
            "save_strategy": 'epoch',
            "save_total_limit": self.config['training']['save_total_limit'],
            "load_best_model_at_end": True,
            "metric_for_best_model": 'rouge-l',
            "greater_is_better": True,
            "predict_with_generate": True,
            "generation_max_length": self.config['data']['max_question_len'],
            "seed": self.config['seed'],
            "gradient_checkpointing": True,
            "dataloader_num_workers": 4,
            "dataloader_pin_memory": True,
            "remove_unused_columns": False
        }
        
        # Safely attempt to disable safetensors conversion if supported by the transformers version
        import inspect
        sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
        if "save_safetensors" in sig.parameters:
            kwargs["save_safetensors"] = False
        
        # If save_safetensors is not an option, the background thread might still be an issue
        # in older versions. We try to disable it via environment variable as a fallback.
        if "save_safetensors" not in sig.parameters:
            os.environ["TRANSFORMERS_NO_ADAPTER_CONVERSION"] = "1"
            
        return Seq2SeqTrainingArguments(**kwargs)

    def get_data_collator(self) -> DataCollatorForSeq2Seq:
        return DataCollatorForSeq2Seq(self.tokenizer, model=self.model, label_pad_token_id=-100)

    def save(self, output_dir: str):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def load(self, model_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        self.model.to(self.device)
