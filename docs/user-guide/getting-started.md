# Getting Started

This guide will help you set up Arandu and run your first pipeline.

## Prerequisites

### Required

- **Python 3.13+**
- **FFmpeg** (for audio/video processing)
- **uv** (recommended) or pip

### Optional

- **Google Drive credentials** (for Drive integration)
- **Ollama** or **OpenAI API key** (for QA/KG pipelines)
- **Docker** (for containerized deployment)

## Installation

### Using uv (Recommended)

```bash
# Clone repository
git clone https://github.com/FredDsR/etno-kgc-preprocessing.git
cd etno-kgc-preprocessing

# Install dependencies
uv sync

# Verify installation
uv run arandu --help
```

### Using pip

```bash
# Clone repository
git clone https://github.com/FredDsR/etno-kgc-preprocessing.git
cd etno-kgc-preprocessing

# Install in editable mode
pip install -e .

# Verify installation
arandu --help
```

### Install FFmpeg

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Verify installation
ffmpeg -version
```

## Quick Start

### 1. Transcribe a Local File

```bash
arandu transcribe audio.mp3
```

### 2. Check System Info

```bash
arandu info
```

This shows your hardware configuration (CPU, GPU, memory).

### 3. Transcribe with Options

```bash
# Use faster turbo model
arandu transcribe audio.mp3 --model-id openai/whisper-large-v3

# Use quantization for reduced VRAM
arandu transcribe audio.mp3 --quantize

# Force CPU execution
arandu transcribe audio.mp3 --cpu
```

## Google Drive Setup (Optional)

For processing files from Google Drive:

1. **Get credentials** from [Google Cloud Console](https://console.cloud.google.com/)
2. Enable the Google Drive API
3. Create OAuth2 credentials and download as `credentials.json`
4. Place in project root

```bash
# Transcribe from Google Drive
arandu drive-transcribe <file-id> --credentials credentials.json
```

## LLM Setup (For QA/KG Pipelines)

### Using Ollama (Recommended for Local)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull qwen3:14b

# Start Ollama server
ollama serve
```

### Using OpenAI

```bash
export OPENAI_API_KEY=sk-...
```

## What's Next?

| Task | Guide |
|------|-------|
| Process multiple files | [Transcription Guide](transcription.md) |
| Validate transcriptions | [Transcription Validation Guide](transcription-validation.md) |
| Generate QA pairs | [QA Generation Guide](qa-generation.md) |
| Build knowledge graphs | [KG Construction Guide](kg-construction.md) |
| Evaluate quality | [Evaluation Guide](evaluation.md) |
| Configure settings | [Configuration Reference](configuration.md) |

## Pipeline Overview

```
Audio/Video Files
       │
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Transcription│ ──▶ │      QA      │ ──▶ │      KG      │
│   Pipeline   │     │  Generation  │     │ Construction │
└──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │
       └────────────────────┴────────────────────┘
                           │
                           ▼
                   ┌──────────────┐
                   │  Evaluation  │
                   └──────────────┘
```

## Troubleshooting

### "No module named 'arandu'"

```bash
pip install -e .
# or
uv sync
```

### "FFmpeg not found"

```bash
sudo apt-get install ffmpeg  # Linux
brew install ffmpeg          # macOS
```

### "CUDA out of memory"

```bash
# Use quantization
arandu transcribe audio.mp3 --quantize

# Or force CPU
arandu transcribe audio.mp3 --cpu
```

---

**See also**: [Transcription](transcription.md) | [Transcription Validation](transcription-validation.md) | [Configuration](configuration.md) | [CLI Reference](cli-reference.md)
