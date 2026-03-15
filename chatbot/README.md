---
title: NeuralMemory Docs
emoji: 🧠
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: "5.23.0"
app_file: app.py
pinned: false
license: mit
short_description: NeuralMemory docs — cognitive neural chatbot
---

# NeuralMemory Documentation Assistant v2

Ask questions about [NeuralMemory](https://github.com/nhadaututtheky/neural-memory) — powered by spreading activation, not an LLM.

## What's new in v2

- **Chat interface** — conversational, remembers context within your session
- **Cognitive reasoning** — uncertain answers trigger deeper search automatically
- **Source citations** — see which documentation files were used for each answer
- **Full retrieval pipeline** — RRF score fusion, graph expansion, reflex activation
- **Conversation memory** — follow-up questions understand your previous questions

## How it works

1. Documentation is pre-encoded into a neural memory brain (neurons + synapses + provenance)
2. Your query triggers spreading activation with RRF fusion across the knowledge graph
3. Low-confidence answers automatically escalate to deeper search with cognitive reasoning
4. Source files are cited so you can verify answers directly
5. Each exchange is remembered within your session for contextual follow-ups

## Re-training the brain

```bash
# From the repo root
python chatbot/train_docs_brain.py
```

This trains from `docs/`, `README.md`, `CHANGELOG.md`, and `FAQ.md` with source provenance tracking.

## Local development

```bash
pip install neural-memory gradio
python chatbot/app.py --port 7860
```
