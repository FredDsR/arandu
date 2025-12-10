# G-Transcriber

Automated transcription system for media files stored in Google Drive using Whisper ASR.

## Overview

G-Transcriber is a robust pipeline for automated transcription of audio and video files. It integrates with Google Drive for seamless file management and uses state-of-the-art speech recognition models from Hugging Face.

## Features

- **Flexible Model Support**: Use any Whisper model from Hugging Face Hub (e.g., `openai/whisper-large-v3`, `distil-whisper/distil-large-v3`)
- **Hardware Agnostic**: Automatic detection and optimization for CPU, CUDA GPU, or Apple Silicon MPS
- **Quantization Support**: 8-bit quantization for reduced VRAM usage on GPUs
- **Google Drive Integration**: Download files, transcribe, and upload results back to Drive
- **Resilient Transfers**: Resumable downloads/uploads with automatic retry logic
- **Rich CLI**: Beautiful command-line interface with progress bars and structured output
- **Structured Output**: JSON output with transcription text, timestamps, and metadata

## Installation

```bash
# Using pip
pip install -e .

# Using uv (recommended)
uv pip install -e .
```

## Usage

### Transcribe a Local File

```bash
gtranscriber transcribe audio.mp3
```

### Transcribe with Custom Model

```bash
gtranscriber transcribe audio.mp3 --model-id openai/whisper-large-v3-turbo
```

### Transcribe with Quantization (Reduced VRAM)

```bash
gtranscriber transcribe audio.mp3 --quantize
```

### Force CPU Execution

```bash
gtranscriber transcribe audio.mp3 --cpu
```

### Transcribe from Google Drive

```bash
gtranscriber drive-transcribe <file-id> --credentials credentials.json
```

### Check System Information

```bash
gtranscriber info
```

## Configuration

The system can be configured via:

1. **Command-line arguments** (highest priority)
2. **Environment variables**:
   - `GTRANSCRIBER_MODEL_ID`: Default model ID
   - `GTRANSCRIBER_FORCE_CPU`: Force CPU execution
   - `GTRANSCRIBER_QUANTIZE`: Enable quantization
   - `GTRANSCRIBER_CREDENTIALS`: Path to Google OAuth credentials
   - `GTRANSCRIBER_TOKEN`: Path to token file

## Output Format

Transcription results are saved as JSON files containing:

```json
{
  "gdrive_id": "...",
  "name": "audio.mp3",
  "mimeType": "audio/mpeg",
  "transcription_text": "...",
  "detected_language": "en",
  "language_probability": 0.95,
  "model_id": "openai/whisper-large-v3",
  "compute_device": "cuda:0",
  "processing_duration_sec": 45.2,
  "transcription_status": "completed",
  "segments": [
    {"text": "...", "start": 0.0, "end": 3.5}
  ]
}
```

## Project Structure

```
gtranscriber/
├── pyproject.toml
├── README.md
└── src/
    └── gtranscriber/
        ├── __init__.py
        ├── main.py         # CLI entrypoint
        ├── config.py       # Configuration
        ├── schemas.py      # Pydantic models
        ├── core/
        │   ├── drive.py    # Google Drive integration
        │   ├── engine.py   # Whisper engine
        │   ├── hardware.py # Hardware detection
        │   └── io.py       # File operations
        └── utils/
            ├── logger.py   # Rich logging
            └── ui.py       # Progress bars
```

## Requirements

- Python >= 3.13
- PyTorch
- Transformers
- Google API Python Client
- Rich
- Typer
- Pydantic

## License

MIT
