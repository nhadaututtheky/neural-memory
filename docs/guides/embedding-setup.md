# Embedding Setup Guide

NeuralMemory works **without embeddings** — its core retrieval uses spreading activation on a neural graph. Embeddings are an **optional enhancement** that adds cross-language recall and semantic discovery between unrelated memories.

## When Do You Need Embeddings?

| Scenario | Without Embeddings | With Embeddings |
|----------|-------------------|-----------------|
| Recall "auth bug" when you stored "auth bug" | Works perfectly | Works perfectly |
| Recall "lỗi xác thực" when you stored "auth bug" | Won't match | Matches via semantic similarity |
| Discover links between "JWT expired" and "token refresh" | Only if explicitly connected | Auto-discovered via cosine similarity |
| Store/recall in one language only | Full functionality | No benefit |

**Rule of thumb**: If you work in a single language and your recall queries use similar wording to what you stored, you don't need embeddings.

## Quick Start: Auto-Detection

The easiest way to enable embeddings — let NeuralMemory detect what's available:

```toml
# ~/.neuralmemory/config.toml
[embedding]
enabled = true
provider = "auto"
```

Auto-detection checks (in order):

1. **Ollama** running locally → uses `bge-m3` or best available model
2. **sentence-transformers** installed → uses `paraphrase-multilingual-MiniLM-L12-v2`
3. **GEMINI_API_KEY** set → uses Google's free-tier embedding API
4. **OPENAI_API_KEY** set → uses OpenAI's embedding API
5. **OPENROUTER_API_KEY** set → uses OpenRouter's OpenAI-compatible embedding API

If none are available, embedding stays disabled and recall falls back to graph-only (which works great for single-language use).

## Providers

### 1. Sentence Transformer (Free, Local)

Runs entirely on your machine. No API key, no cost, no data leaves your device.

```bash
pip install neural-memory[embeddings]
```

**Config** (`~/.neuralmemory/config.toml`):

```toml
[embedding]
enabled = true
provider = "sentence_transformer"
model = "all-MiniLM-L6-v2"           # English-only, 384D, ~80MB
similarity_threshold = 0.7
```

**Multilingual models** (recommended for non-English or mixed-language use):

| Model | Languages | Dimensions | Size | Speed (CPU) |
|-------|-----------|-----------|------|-------------|
| `all-MiniLM-L6-v2` | English only | 384 | ~80MB | ~15ms |
| `paraphrase-multilingual-MiniLM-L12-v2` | 50+ languages | 384 | ~440MB | ~25ms |
| `multilingual-e5-small` | 100+ languages | 384 | ~500MB | ~30ms |
| `multilingual-e5-large` | 100+ languages | 1024 | ~2.2GB | ~150ms |

For Vietnamese, Chinese, Japanese, or any non-English language, use `paraphrase-multilingual-MiniLM-L12-v2`:

```toml
[embedding]
enabled = true
provider = "sentence_transformer"
model = "paraphrase-multilingual-MiniLM-L12-v2"
similarity_threshold = 0.65
```

> The model downloads automatically on first use (~440MB). Subsequent runs use the cached version.

### 2. Gemini (Google API)

Uses Google's `gemini-embedding-001` (3072D) or `text-embedding-004` (768D).

```bash
pip install neural-memory[embeddings-gemini]
```

Set your API key:

```bash
export GEMINI_API_KEY="your-key-here"
```

**Config**:

```toml
[embedding]
enabled = true
provider = "gemini"
model = "text-embedding-004"         # 768D, lower cost
# model = "gemini-embedding-001"     # 3072D, higher quality
similarity_threshold = 0.7
```

> Get a free API key at [ai.google.dev](https://ai.google.dev/). Free tier includes generous embedding quotas.

### 3. Ollama (Free, Local)

Uses any Ollama model for embeddings. Runs entirely on your machine via the Ollama API.

```bash
pip install neural-memory[embeddings]
# Ensure Ollama is running: ollama serve
# Pull an embedding model: ollama pull nomic-embed-text
```

**Config**:

```toml
[embedding]
enabled = true
provider = "ollama"
model = "nomic-embed-text"              # 768D, fast
# model = "mxbai-embed-large"           # 1024D, higher quality
similarity_threshold = 0.7
# base_url = "http://localhost:11434"    # Default Ollama URL
```

> Requires Ollama running locally. See [ollama.com](https://ollama.com) for installation.

### 4. OpenAI (API)

Uses OpenAI's `text-embedding-3-small` or `text-embedding-3-large`.

```bash
pip install neural-memory[embeddings-openai]
```

Set your API key:

```bash
export OPENAI_API_KEY="your-key-here"
```

**Config**:

```toml
[embedding]
enabled = true
provider = "openai"
model = "text-embedding-3-small"     # 1536D, $0.02/1M tokens
# model = "text-embedding-3-large"   # 3072D, $0.13/1M tokens
similarity_threshold = 0.7
```

### 5. OpenRouter (API)

Uses OpenRouter's OpenAI-compatible embeddings endpoint with provider-routed models like `openai/text-embedding-3-small`.

```bash
pip install neural-memory[embeddings-openrouter]
```

Set your API key:

```bash
export OPENROUTER_API_KEY="your-key-here"
```

**Config**:

```toml
[embedding]
enabled = true
provider = "openrouter"
model = "openai/text-embedding-3-small"   # 1536D
# model = "openai/text-embedding-3-large" # 3072D
similarity_threshold = 0.7
```

> OpenRouter uses the `openai` Python SDK under the hood against `https://openrouter.ai/api/v1`.

## Provider Comparison

| | Sentence Transformer | Ollama | Gemini | OpenAI | OpenRouter |
|---|---|---|---|---|---|
| **Cost** | Free | Free | Free tier / pay-per-use | Pay-per-use | Pay-per-use |
| **Privacy** | 100% local | 100% local | Data sent to Google | Data sent to OpenAI | Data sent to routed provider |
| **Speed** | 15-150ms (CPU) | 10-50ms (GPU) | ~200ms (network) | ~200ms (network) | ~200ms (network) |
| **Quality** | Good | Good-Excellent | Excellent | Excellent | Excellent |
| **Multilingual** | With multilingual model | Model-dependent | Built-in | Built-in | Built-in |
| **Offline** | Yes | Yes | No | No | No |
| **Setup** | `pip install` only | Ollama + model pull | API key required | API key required | API key required |
| **GPU Accel** | Optional | Yes (native) | N/A | N/A | N/A |

**Recommendation**: Start with `sentence_transformer` + `paraphrase-multilingual-MiniLM-L12-v2` for simplicity. Use `ollama` if you have a GPU and want fast local inference. Switch to Gemini, OpenAI, or OpenRouter if you need managed cloud embeddings.

## How It Works

When embeddings are enabled, two things happen:

### 1. Embedding Anchors (during recall)

When you recall a memory, NeuralMemory runs keyword search (FTS5) **and** embedding similarity search **in parallel**. Results are merged — this means you can find memories even when the query uses completely different words than what was stored.

### 2. Semantic Discovery (during consolidation)

When you run `nmem consolidate`, NeuralMemory:
1. Embeds all CONCEPT and ENTITY neurons
2. Computes pairwise cosine similarity
3. Creates `SIMILAR_TO` synapses between semantically related neurons above the threshold

These synapses allow spreading activation to traverse semantic connections during future recalls.

## Tuning

### Similarity Threshold

- **0.7** (default): Conservative — only strong matches. Good for precision.
- **0.6**: Moderate — catches more cross-language matches. Good starting point for multilingual.
- **0.5**: Aggressive — more noise but catches loose associations.

### Changing Provider Mid-Session

You can change providers at any time. However, existing `SIMILAR_TO` synapses created by semantic discovery were computed with the old provider's embeddings. Run `nmem consolidate` after switching to recompute with the new provider.

> **Warning**: Different providers produce different embedding dimensions. Stored `_embedding` metadata from the old provider will be incompatible. This only affects semantic discovery — core recall is unaffected.

## Troubleshooting

**"Embedding provider unavailable — skipping"**: The required package isn't installed. Check `pip list | grep sentence-transformers` (or `google-genai` / `openai`).

**Slow first recall**: Sentence Transformer downloads the model on first use. Subsequent runs use cache at `~/.cache/huggingface/`.

**No cross-language matches**: Check that you're using a multilingual model. `all-MiniLM-L6-v2` is English-only.

**API key errors**: Ensure the environment variable is set in the shell that runs the MCP server. For Claude Code, add it to your MCP config:

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "nmem-mcp",
      "env": {
        "GEMINI_API_KEY": "your-key",
        "OPENROUTER_API_KEY": "your-key"
      }
    }
  }
}
```
