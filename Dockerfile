# Financial Copilot web app (CPU only). The model runs elsewhere: on Modal, a vLLM or
# llama.cpp server, or Ollama. Configure it with FINCOPILOT_* environment variables (see DEPLOY.md).
#
#   docker build -t fincopilot .
#   docker run -p 7860:7860 fincopilot                                   # demo mode
#   docker run -p 7860:7860 -e FINCOPILOT_BACKEND=openai \
#     -e FINCOPILOT_HOST=https://<model-server>/v1 -e FINCOPILOT_MODEL=fincopilot \
#     -e FINCOPILOT_API_KEY=<key> fincopilot
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=7860 FINCOPILOT_BACKEND=demo

COPY requirements.txt requirements-deploy.txt ./
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY fincopilot ./fincopilot

RUN useradd --create-home app
USER app

EXPOSE 7860
# One worker keeps the in-memory rate limits consistent; threads handle concurrent visitors.
CMD gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 16 --timeout 600 "fincopilot.web.server:create_app_from_env()"
