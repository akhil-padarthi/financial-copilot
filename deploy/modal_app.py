"""Deploy Financial Copilot on Modal (https://modal.com): one command, two pieces.

  Server  a GPU container running vLLM with Llama 3.1 8B Instruct + your LoRA adapter,
          behind an OpenAI-compatible API protected by an API key. It scales to zero
          after 5 idle minutes, so you only pay while someone is using it.
  web     the Financial Copilot web app (CPU only). This is the public link you share.

Setup (once):
    pip install modal && modal setup
    modal volume create fincopilot-adapters
    modal volume put fincopilot-adapters outputs/fincopilot-lora /fincopilot-lora
    modal secret create fincopilot-secrets FINCOPILOT_API_KEY=<any long random string>

Deploy (from the repository root):
    modal deploy deploy/modal_app.py                            # fine-tuned model
    FINCOPILOT_USE_ADAPTER=0 modal deploy deploy/modal_app.py   # base model only (before you've trained)

Options are read when you deploy and passed into the containers:
    FINCOPILOT_USE_ADAPTER  1 (default) serve your LoRA adapter; 0 serve the untuned base model
    FINCOPILOT_GPU          GPU type (default L4, 24 GB; A10 also works)

Checked against vllm 0.21.0 and the current Modal client (app.server / modal.wsgi_app).
"""
import os
from pathlib import Path

import modal

APP_NAME = "fincopilot"
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"  # full-precision Llama 3.1 8B Instruct, no license gate
ADAPTER_PATH = "/adapters/fincopilot-lora"
VLLM_PORT = 8000
MINUTES = 60

USE_ADAPTER = os.environ.get("FINCOPILOT_USE_ADAPTER", "1") == "1"
GPU = os.environ.get("FINCOPILOT_GPU", "L4")
MODEL_NAME = "fincopilot" if USE_ADAPTER else "base"
PROMPT_STYLE = "compact" if USE_ADAPTER else "full"  # the untuned model needs the full tool descriptions
DEPLOY_ENV = {"FINCOPILOT_USE_ADAPTER": "1" if USE_ADAPTER else "0", "FINCOPILOT_GPU": GPU}

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
adapters = modal.Volume.from_name("fincopilot-adapters", create_if_missing=True)
secrets = [modal.Secret.from_name("fincopilot-secrets")]  # provides FINCOPILOT_API_KEY

app = modal.App(APP_NAME)


@app.server(
    image=vllm_image,
    gpu=GPU,
    env=DEPLOY_ENV,
    secrets=secrets,
    volumes={"/root/.cache/huggingface": hf_cache, "/adapters": adapters},
    port=VLLM_PORT,
    unauthenticated=True,           # reachable by the web app; vLLM's --api-key guards it
    scaledown_window=5 * MINUTES,   # stop the GPU after 5 idle minutes
    startup_timeout=20 * MINUTES,   # the very first start downloads ~16 GB of weights
    max_containers=1,               # never run more than one GPU (caps cost)
    target_concurrency=8,
)
class Server:
    @modal.enter()
    def start(self):
        import subprocess

        cmd = [
            "vllm", "serve", BASE_MODEL,
            "--served-model-name", "base",
            "--host", "0.0.0.0", "--port", str(VLLM_PORT),
            "--max-model-len", "8192",       # room for the long base-model prompt
            "--enforce-eager",               # faster cold starts; plenty fast for a demo
            "--api-key", os.environ["FINCOPILOT_API_KEY"],
        ]
        if USE_ADAPTER:
            if not os.path.exists(os.path.join(ADAPTER_PATH, "adapter_config.json")):
                raise RuntimeError(
                    "No LoRA adapter found. Upload it with:\n"
                    "  modal volume put fincopilot-adapters outputs/fincopilot-lora /fincopilot-lora\n"
                    "or deploy the base model with FINCOPILOT_USE_ADAPTER=0.")
            cmd += ["--enable-lora", "--lora-modules", f"fincopilot={ADAPTER_PATH}", "--max-lora-rank", "16"]
        print(*cmd)
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.process.terminate()


ROOT = Path(__file__).resolve().parent.parent
web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("pydantic>=2.5", "requests>=2.28", "flask>=2.3")
    # add_local_dir (not add_local_python_source) so the web page's HTML is included too
    .add_local_dir(ROOT / "fincopilot", "/root/fincopilot", ignore=["**/__pycache__/**"])
)


def _model_server_url() -> str:
    try:
        url = Server.get_url()
    except Exception:
        url = None
    return url or modal.Server.from_name(APP_NAME, "Server").get_url()


@app.function(
    image=web_image,
    secrets=secrets,
    env={
        **DEPLOY_ENV,
        "FINCOPILOT_BACKEND": "openai",
        "FINCOPILOT_MODEL": MODEL_NAME,
        "FINCOPILOT_PROMPT_STYLE": PROMPT_STYLE,
        "FINCOPILOT_LABEL": ("Fine-tuned Llama 3.1 8B" if USE_ADAPTER else "Llama 3.1 8B (base model, not fine-tuned)"),
        "FINCOPILOT_RATE_LIMIT": "30/600",   # 30 questions per visitor per 10 minutes
        "FINCOPILOT_DAILY_LIMIT": "500",     # 500 questions per day in total
        "FINCOPILOT_TIMEOUT": "540",         # wait through a GPU cold start
    },
    timeout=10 * MINUTES,
    max_containers=1,  # one web container keeps the rate limits consistent
)
@modal.concurrent(max_inputs=20)
@modal.wsgi_app()
def web():
    from fincopilot.web.server import create_app_from_env

    os.environ["FINCOPILOT_HOST"] = _model_server_url().rstrip("/") + "/v1"
    return create_app_from_env()
