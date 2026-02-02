# Getting Started

This guide will help you set up G-Transcriber and run your first pipeline.

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
uv run gtranscriber --help
```

### Using pip

```bash
# Clone repository
git clone https://github.com/FredDsR/etno-kgc-preprocessing.git
cd etno-kgc-preprocessing

# Install in editable mode
pip install -e .

# Verify installation
gtranscriber --help
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
gtranscriber transcribe audio.mp3
```

### 2. Check System Info

```bash
gtranscriber info
```

This shows your hardware configuration (CPU, GPU, memory).

### 3. Transcribe with Options

```bash
# Use faster turbo model
gtranscriber transcribe audio.mp3 --model-id openai/whisper-large-v3-turbo

# Use quantization for reduced VRAM
gtranscriber transcribe audio.mp3 --quantize

# Force CPU execution
gtranscriber transcribe audio.mp3 --cpu
```

## Google Drive Setup (Optional)

For processing files from Google Drive:

1. **Get credentials** from [Google Cloud Console](https://console.cloud.google.com/)
2. Enable the Google Drive API
3. Create OAuth2 credentials and download as `credentials.json`
4. Place in project root

```bash
# Transcribe from Google Drive
gtranscriber drive-transcribe <file-id> --credentials credentials.json
```

## LLM Setup (For QA/KG Pipelines)

### Using Ollama (Recommended for Local)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.1:8b

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

### "No module named 'gtranscriber'"

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
gtranscriber transcribe audio.mp3 --quantize

# Or force CPU
gtranscriber transcribe audio.mp3 --cpu
```

---

**See also**: [Transcription](transcription.md) | [Configuration](configuration.md) | [CLI Reference](cli-reference.md)
