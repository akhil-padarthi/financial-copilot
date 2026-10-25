"""Hugging Face / Unsloth backend (GPU). Used in Colab for evaluation and demos.

    backend = HFBackend("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", adapter="outputs/fincopilot-lora")
    with backend.base_model():   # temporarily switch the LoRA off to compare against the base model
        ...
"""
from __future__ import annotations

import contextlib
import importlib.util
from typing import Iterator, List, Optional, Sequence

from ..prompts import format_llama31
from .base import Backend, Message


class HFBackend(Backend):
    name = "hf"

    def __init__(self, model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
                 adapter: Optional[str] = None, load_in_4bit: bool = True,
                 use_unsloth: Optional[bool] = None, max_seq_length: int = 8192, batch_size: int = 8):
        import torch

        self.torch = torch
        self.batch_size = batch_size
        if use_unsloth is None:
            use_unsloth = importlib.util.find_spec("unsloth") is not None and torch.cuda.is_available()

        if use_unsloth:
            from unsloth import FastLanguageModel
            # Unsloth loads a saved LoRA folder directly (it reads the base model from adapter_config.json).
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=adapter or model_name, max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit, dtype=None)
            FastLanguageModel.for_inference(model)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            quant = None
            if load_in_4bit and torch.cuda.is_available():
                from transformers import BitsAndBytesConfig
                quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                           bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
            tokenizer = AutoTokenizer.from_pretrained(adapter or model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant, device_map="auto",
                                                         torch_dtype="auto")
            if adapter:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, adapter)
            model.eval()

        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            vocab = tokenizer.get_vocab()
            tokenizer.pad_token = "<|finetune_right_pad_id|>" if "<|finetune_right_pad_id|>" in vocab else tokenizer.eos_token
        vocab = tokenizer.get_vocab()
        self.stop_ids = [vocab[t] for t in ("<|eot_id|>", "<|eom_id|>", "<|end_of_text|>") if t in vocab]
        self.model, self.tokenizer = model, tokenizer

    @contextlib.contextmanager
    def base_model(self) -> Iterator[None]:
        """Temporarily disable the LoRA adapter (base-vs-fine-tuned comparisons)."""
        if hasattr(self.model, "disable_adapter"):
            with self.model.disable_adapter():
                yield
        else:
            yield

    def generate(self, messages: Sequence[Message], max_new_tokens: int = 384) -> str:
        return self.generate_batch([messages], max_new_tokens)[0]

    def generate_batch(self, batch: Sequence[Sequence[Message]], max_new_tokens: int = 384) -> List[str]:
        outputs: List[str] = []
        for i in range(0, len(batch), self.batch_size):
            chunk = [format_llama31(m) for m in batch[i:i + self.batch_size]]
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
            with self.torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                          temperature=None, top_p=None,
                                          eos_token_id=self.stop_ids, pad_token_id=self.tokenizer.pad_token_id)
            new_tokens = out[:, enc["input_ids"].shape[1]:]
            outputs.extend(self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        return outputs
