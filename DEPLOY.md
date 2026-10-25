# Deploying Financial Copilot

This guide puts Financial Copilot online as a public link: your fine-tuned Llama 3.1 8B on a cloud GPU that sleeps when nobody is using it, plus the web app in front of it.

| Option | What you get | Cost |
|---|---|---|
| **A. Modal (recommended)** | Real model on a GPU + public web app, one command | Free plan includes $30/month of credits; the GPU costs about $0.80/hour *only while awake* |
| B. Docker on any host | Just the web app (demo mode, or pointed at a model server you run) | Depends on the host |
| C. Your laptop | Everything local with Ollama (README Step 6) | Free, but only you can use it |

Prices checked October 2026 on [modal.com/pricing](https://modal.com/pricing): L4 GPU $0.000222/second.

---

## Option A: Modal

### How it works

```
visitor ──> web (Flask app, CPU)  ──API key──>  Server (vLLM + Llama 3.1 8B + your LoRA, L4 GPU)
            public URL                          sleeps after 5 idle minutes
            rate-limited                        at most 1 GPU, ever
```

`deploy/modal_app.py` defines both pieces. When someone opens the page, the app starts waking the GPU immediately, so it's usually warm by the time they've typed a question.

### Step 0. Have something to deploy

- **Fine-tuned model:** run the Colab notebook through training (Step 7). It saves your LoRA adapter to Google Drive at `financial-copilot/fincopilot-lora`.
- **Haven't trained yet?** You can deploy the base model today (see "Deploy the base model first" below) and switch later.

Download the adapter folder into the project as `outputs/fincopilot-lora`. It must contain `adapter_config.json` and `adapter_model.safetensors` at the top level. Delete any `checkpoint-*` subfolders from your local copy first: they're only for resuming training and would make the upload much bigger.

### Step 1. Install Modal and log in

```bash
pip install modal
modal setup          # opens a browser to create an account / log in
```

### Step 2. Upload your adapter

```bash
modal volume create fincopilot-adapters
modal volume put fincopilot-adapters outputs/fincopilot-lora /fincopilot-lora
```

### Step 3. Create an API key for the model server

The model server is reachable on the internet, so it only answers requests carrying this key. The web app gets the same key automatically.

```bash
modal secret create fincopilot-secrets FINCOPILOT_API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

### Step 4. Deploy

From the project folder:

```bash
modal deploy deploy/modal_app.py
```

The output lists two URLs. The one for **web** (it looks like `https://<your-workspace>--fincopilot-web.modal.run`) is your public link.

### Step 5. Test it

```bash
python deploy/smoke_test.py https://<your-workspace>--fincopilot-web.modal.run
```

It asks a known question and checks the answer is $19,671.51. **The very first run takes longest** (roughly 5–15 minutes): Modal builds the container image and downloads about 16 GB of model weights. After that, waking from sleep usually takes a minute or two, and answers take a few seconds once awake.

Then open the link in your browser. The header shows which model is answering, so visitors can see it's the fine-tuned model.

### Deploy the base model first (optional)

Before you've trained, deploy the untuned model to get a live demo and a baseline:

```bash
FINCOPILOT_USE_ADAPTER=0 modal deploy deploy/modal_app.py
```

(On Windows PowerShell: `$env:FINCOPILOT_USE_ADAPTER="0"; modal deploy deploy/modal_app.py`.) This skips Step 2. The app automatically uses the long tool-description prompt the base model needs, and the header says "base model, not fine-tuned". When your adapter is ready, do Step 2 and run the normal deploy command.

### Updating, monitoring, stopping

| Task | Command |
|---|---|
| Upload a retrained adapter | `modal volume put --force fincopilot-adapters outputs/fincopilot-lora /fincopilot-lora`, then `modal deploy deploy/modal_app.py` |
| Watch logs | `modal app logs fincopilot` |
| Take it offline | `modal app stop fincopilot` |
| See spending | the Usage page in your Modal dashboard |

### What it costs

- The GPU only runs while someone is using the demo, plus 5 idle minutes afterwards. One visit (wake-up + a few questions + idle time) is typically under 10 GPU-minutes, around $0.10.
- At about $0.80 per GPU-hour, the $30 monthly credit covers roughly 35+ awake hours.
- Built-in protection: **30 questions per visitor per 10 minutes** and **500 questions per day in total**, and never more than one GPU. Change these in `deploy/modal_app.py` (`FINCOPILOT_RATE_LIMIT`, `FINCOPILOT_DAILY_LIMIT`, `max_containers`).

### Settings you can change at deploy time

| Variable | Default | Meaning |
|---|---|---|
| `FINCOPILOT_USE_ADAPTER` | `1` | `0` serves the base model without your LoRA |
| `FINCOPILOT_GPU` | `L4` | GPU type. L4 and A10 (24 GB) both fit the 8B model |

Example: `FINCOPILOT_GPU=A10 modal deploy deploy/modal_app.py`

### Technical notes

- **Serving:** vLLM 0.21.0 loads the full-precision base model (`unsloth/Meta-Llama-3.1-8B-Instruct`) and applies your LoRA adapter on the fly (`--enable-lora`). Training a LoRA on the 4-bit model and serving it on the full-precision one is the standard way QLoRA adapters are deployed.
- **Same prompt as training:** the app calls vLLM's raw `/v1/completions` endpoint with the exact Llama 3.1 prompt used in training. It strips its own begin-of-text token because vLLM adds one; a doubled token would make deployed prompts differ from training prompts.
- **Security:** the model server requires an API key (`--api-key`), stored as a Modal Secret, never in the code.
- **Cold starts:** `--enforce-eager` skips CUDA graph capture to start faster; the page wakes the GPU as soon as a visitor arrives.

---

## Option B: Docker (web app on any host)

The `Dockerfile` runs only the web app; the model runs elsewhere. Useful for Render, Railway, Fly.io or your own server.

```bash
docker build -t fincopilot .
docker run -p 7860:7860 fincopilot        # demo mode at http://localhost:7860
```

To use a real model, point it at any OpenAI-compatible server, such as the Modal **Server** URL from Option A (add `/v1`), or your own vLLM or llama.cpp server:

```bash
docker run -p 7860:7860 \
  -e FINCOPILOT_BACKEND=openai \
  -e FINCOPILOT_HOST=https://<model-server-url>/v1 \
  -e FINCOPILOT_MODEL=fincopilot \
  -e FINCOPILOT_API_KEY=<your key> \
  -e FINCOPILOT_RATE_LIMIT=30/600 \
  fincopilot
```

All settings are environment variables:

| Variable | Example | Meaning |
|---|---|---|
| `FINCOPILOT_BACKEND` | `openai` | `demo`, `openai`, `ollama` or `hf` |
| `FINCOPILOT_HOST` | `https://…/v1` | model server URL |
| `FINCOPILOT_MODEL` | `fincopilot` | model name on that server (`base` for the untuned model on the Modal server) |
| `FINCOPILOT_API_KEY` | | bearer token for the model server |
| `FINCOPILOT_PROMPT_STYLE` | `compact` | `full` for an untuned base model |
| `FINCOPILOT_LABEL` | `Fine-tuned Llama 3.1 8B` | text in the page header |
| `FINCOPILOT_RATE_LIMIT` | `30/600` | questions per visitor per N seconds (off by default) |
| `FINCOPILOT_DAILY_LIMIT` | `500` | total questions per day (off by default) |
| `FINCOPILOT_TIMEOUT` | `300` | seconds to wait for the model |
| `PORT` | `7860` | port to listen on |

Without Docker, the same app runs with `pip install -r requirements-deploy.txt` and
`gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 16 --timeout 600 "fincopilot.web.server:create_app_from_env()"`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Secret 'fincopilot-secrets' not found` | Do Step 3. |
| Logs say "No LoRA adapter found" | Do Step 2, or deploy with `FINCOPILOT_USE_ADAPTER=0`. Check the folder has `adapter_config.json` at its top level. |
| First question shows "took too long" | The GPU was still starting (especially the very first deploy). Wait a minute and ask again; watch progress with `modal app logs fincopilot`. |
| "The model server rejected the API key" | The web app and model server keys differ. Recreate the secret (Step 3) and redeploy. |
| "has no model named 'fincopilot'" | The server was deployed without the adapter. Redeploy without `FINCOPILOT_USE_ADAPTER=0`. |
| GPU out of memory | Lower `--max-model-len` in `deploy/modal_app.py`, or use a bigger GPU (`FINCOPILOT_GPU=L40S`). |
| "You've asked a lot of questions" | The per-visitor rate limit; raise `FINCOPILOT_RATE_LIMIT` if it's too strict for your use. |

What's verified and what isn't: the web app, the rate limits and the OpenAI-compatible client are covered by automated tests, including a full run against a fake vLLM server over HTTP. The Modal script was checked against the current Modal client source and vLLM 0.21.0's options, but it couldn't be run end to end without a GPU account. `deploy/smoke_test.py` is how you confirm your own deployment.
