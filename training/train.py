"""QLoRA fine-tuning of Llama 3.1 8B Instruct for Financial Copilot.

    python training/train.py                       # full run (GPU)
    python training/train.py --dry-run             # tokenizer only: check lengths and loss masking
    python training/train.py --limit 500 --max-steps 30   # quick smoke test on a GPU

What happens:
1. Load Llama 3.1 8B in 4-bit (QLoRA) with Unsloth and attach LoRA adapters
   (small trainable matrices on every attention and MLP projection).
2. Tokenize each conversation with the exact Llama 3.1 format used at inference,
   masking the loss so only assistant turns are learned (not user text, not tool results).
3. Train with Hugging Face's Trainer, evaluate on the validation split, save the adapter.

Use --backend hf to train with plain transformers + peft + bitsandbytes instead of Unsloth.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fincopilot.prompts import build_training_features  # noqa: E402

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SPECIAL_TOKENS = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]


def load_jsonl(path: str, limit: Optional[int] = None) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return rows[:limit] if limit else rows


def pad_batch(features: Sequence[Dict[str, List[int]]], pad_id: int) -> Dict[str, List[List[int]]]:
    """Right-pad a batch: pad tokens get attention 0 and label -100 (ignored by the loss)."""
    width = max(len(f["input_ids"]) for f in features)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for f in features:
        gap = width - len(f["input_ids"])
        out["input_ids"].append(f["input_ids"] + [pad_id] * gap)
        out["attention_mask"].append(f["attention_mask"] + [0] * gap)
        out["labels"].append(f["labels"] + [-100] * gap)
    return out


def tokenize_all(rows: Sequence[Dict], tokenizer, max_len: int):
    encode = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]  # noqa: E731
    for token in SPECIAL_TOKENS:
        ids = encode(token)
        if len(ids) != 1:
            raise SystemExit(f"Tokenizer splits {token} into {ids}; this is not a Llama 3.x tokenizer.")
    features, dropped = [], 0
    for row in rows:
        feat = build_training_features(row["messages"], encode, max_len)
        if feat is None:
            dropped += 1
        else:
            features.append(feat)
    return features, dropped


def describe(features: Sequence[Dict], tokenizer, name: str) -> Dict[str, float]:
    lengths = sorted(len(f["input_ids"]) for f in features)
    trained = sum(sum(1 for t in f["labels"] if t != -100) for f in features)
    total = sum(lengths)
    stats = {"examples": len(features), "tokens": total, "trained_tokens": trained,
             "mean_len": round(total / len(lengths)), "p95_len": lengths[int(0.95 * (len(lengths) - 1))],
             "max_len": lengths[-1]}
    print(f"[{name}] {stats}")
    return stats


def show_masking(feature: Dict, tokenizer) -> None:
    """Print one example with the learned spans marked, so you can see the loss mask."""
    ids, labels = feature["input_ids"], feature["labels"]
    pieces, current, learned = [], [], labels[0] != -100
    for tok, lab in zip(ids, labels):
        is_learned = lab != -100
        if is_learned != learned:
            pieces.append((learned, tokenizer.decode(current)))
            current, learned = [], is_learned
        current.append(tok)
    pieces.append((learned, tokenizer.decode(current)))
    print("\n----- loss mask preview: text inside [[ ]] is what the model learns -----")
    for is_learned, text in pieces:
        text = text if len(text) < 600 else text[:250] + " ... " + text[-250:]
        print(f"[[{text}]]" if is_learned else text, end="")
    print("\n" + "-" * 72)


def load_model(args):
    import torch
    if args.backend == "unsloth":
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model, max_seq_length=args.max_seq_len, dtype=None, load_in_4bit=True)
        model = FastLanguageModel.get_peft_model(
            model, r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
            target_modules=TARGET_MODULES, use_gradient_checkpointing="unsloth", random_state=args.seed)
    else:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant, device_map="auto")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                                                 target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM"))
    if tokenizer.pad_token is None:
        vocab = tokenizer.get_vocab()
        tokenizer.pad_token = "<|finetune_right_pad_id|>" if "<|finetune_right_pad_id|>" in vocab else tokenizer.eos_token
    return model, tokenizer


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
    p.add_argument("--data", default="data/train.jsonl")
    p.add_argument("--val", default="data/val.jsonl")
    p.add_argument("--out", default="outputs/fincopilot-lora")
    p.add_argument("--backend", choices=["unsloth", "hf"], default="unsloth")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--limit", type=int, default=None, help="use only the first N training examples")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--resume", action="store_true", help="continue from the last checkpoint in --out")
    p.add_argument("--dry-run", action="store_true", help="load only the tokenizer and check the data")
    args = p.parse_args(argv)

    train_rows = load_jsonl(args.data, args.limit)
    val_rows = load_jsonl(args.val) if args.val and os.path.exists(args.val) else []
    print(f"Loaded {len(train_rows)} training and {len(val_rows)} validation conversations.")

    if args.dry_run:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        feats, dropped = tokenize_all(train_rows, tokenizer, args.max_seq_len)
        describe(feats, tokenizer, "train")
        print(f"Dropped {dropped} examples longer than {args.max_seq_len} tokens.")
        repair = next((i for i, r in enumerate(train_rows) if r.get("type") == "repair"), 0)
        show_masking(feats[min(repair, len(feats) - 1)], tokenizer)
        steps = -(-len(feats) // (args.batch_size * args.grad_accum))
        print(f"One epoch = {steps} optimizer steps at an effective batch size of {args.batch_size * args.grad_accum}.")
        return

    if args.backend == "unsloth":
        import unsloth  # noqa: F401  (must be imported before transformers to apply its optimizations)
    import torch
    from transformers import Trainer, TrainingArguments

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found. Run this on Colab (Runtime > Change runtime type > GPU) or use --dry-run.")
    torch.manual_seed(args.seed)
    model, tokenizer = load_model(args)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    train_feats, dropped = tokenize_all(train_rows, tokenizer, args.max_seq_len)
    val_feats, _ = tokenize_all(val_rows, tokenizer, args.max_seq_len) if val_rows else ([], 0)
    stats = describe(train_feats, tokenizer, "train")
    print(f"Dropped {dropped} over-length examples.")
    show_masking(train_feats[0], tokenizer)

    pad_id = tokenizer.pad_token_id

    def collate(batch):
        return {k: torch.tensor(v) for k, v in pad_batch(batch, pad_id).items()}

    bf16 = torch.cuda.is_bf16_supported()
    common = dict(
        output_dir=args.out, per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=args.warmup_ratio, weight_decay=0.01,
        optim="adamw_8bit", bf16=bf16, fp16=not bf16, logging_steps=10, save_strategy="steps",
        save_steps=args.save_steps, save_total_limit=2, eval_steps=args.eval_steps, seed=args.seed,
        report_to="none", remove_unused_columns=False, dataloader_pin_memory=True)
    strategy = "steps" if val_feats else "no"
    try:
        targs = TrainingArguments(eval_strategy=strategy, **common)
    except TypeError:  # transformers < 4.41 used the older argument name
        targs = TrainingArguments(evaluation_strategy=strategy, **common)

    trainer = Trainer(model=model, args=targs, train_dataset=train_feats,
                      eval_dataset=val_feats or None, data_collator=collate)
    start = time.time()
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    minutes = (time.time() - start) / 60

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    final_eval = trainer.evaluate() if val_feats else {}
    summary = {"base_model": args.model, "backend": args.backend, "minutes": round(minutes, 1),
               "train_stats": stats, "dropped": dropped, "args": vars(args), "final_eval": final_eval,
               "log_history": trainer.state.log_history}
    with open(os.path.join(args.out, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Done in {minutes:.1f} min. LoRA adapter saved to {args.out}")
    if final_eval:
        print(f"Validation loss: {final_eval.get('eval_loss')}")


if __name__ == "__main__":
    main()
