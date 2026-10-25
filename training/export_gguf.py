"""Export the fine-tuned model to GGUF so it runs locally in Ollama / llama.cpp.

    python training/export_gguf.py --adapter outputs/fincopilot-lora --out outputs/fincopilot --quant q4_k_m

Merges the LoRA adapter into the base weights, converts to GGUF (q4_k_m is ~4.9 GB
and runs on most laptops with 8 GB+ of RAM), and writes an Ollama Modelfile next to it.
Then, on your own computer:
    ollama create fincopilot -f Modelfile
    python -m fincopilot.cli --backend ollama --model fincopilot
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fincopilot.prompts import build_system_prompt  # noqa: E402

MODELFILE = '''FROM ./{gguf}

# Financial Copilot answers with JSON tool calls; the Python app runs the tools.
# The app sends fully formatted prompts in raw mode, so this template only matters for `ollama run`.
PARAMETER temperature 0
PARAMETER num_ctx 8192
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|eom_id|>"
PARAMETER stop "<|start_header_id|>"

TEMPLATE """{{{{ if .System }}}}<|start_header_id|>system<|end_header_id|>

{{{{ .System }}}}<|eot_id|>{{{{ end }}}}{{{{ if .Prompt }}}}<|start_header_id|>user<|end_header_id|>

{{{{ .Prompt }}}}<|eot_id|>{{{{ end }}}}<|start_header_id|>assistant<|end_header_id|>

{{{{ .Response }}}}<|eot_id|>"""

SYSTEM """{system}"""
'''


def write_modelfile(gguf_path: str) -> str:
    folder = os.path.dirname(gguf_path)
    path = os.path.join(folder, "Modelfile")
    with open(path, "w", encoding="utf-8") as f:
        f.write(MODELFILE.format(gguf=os.path.basename(gguf_path), system=build_system_prompt("compact")))
    return path


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", default="outputs/fincopilot-lora")
    p.add_argument("--out", default="outputs/fincopilot")
    p.add_argument("--quant", default="q4_k_m", help="q4_k_m (recommended), q5_k_m, q8_0 ...")
    args = p.parse_args(argv)

    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=args.adapter, max_seq_length=4096, load_in_4bit=True)
    model.save_pretrained_gguf(args.out, tokenizer, quantization_method=args.quant)

    found = sorted(glob.glob(os.path.join(args.out + "*", "**", "*.gguf"), recursive=True),
                   key=os.path.getmtime, reverse=True)
    wanted = [g for g in found if args.quant.upper() in os.path.basename(g).upper()] or found
    if not wanted:
        raise SystemExit("Export finished but no .gguf file was found; check the Unsloth log above.")
    modelfile = write_modelfile(wanted[0])
    print(f"GGUF:      {wanted[0]}  ({os.path.getsize(wanted[0]) / 1e9:.1f} GB)")
    print(f"Modelfile: {modelfile}")
    print("Copy both files to your computer (same folder), then run:\n  ollama create fincopilot -f Modelfile")


if __name__ == "__main__":
    main()
