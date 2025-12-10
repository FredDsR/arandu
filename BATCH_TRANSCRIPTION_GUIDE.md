# Batch Transcription Guide

This guide explains how to use the parallel batch transcription feature to process multiple audio/video files from Google Drive.

## Overview

The `batch-transcribe` command allows you to:
- Process all audio/video files from a catalog CSV
- Run transcriptions in parallel with multiple workers
- Automatically checkpoint progress for resuming interrupted jobs
- Extract and store media duration in the output
- Save transcriptions with full metadata to a results directory

## Prerequisites

1. **Google Drive Credentials**: You need OAuth2 credentials to access Google Drive
   - Get credentials from [Google Cloud Console](https://console.cloud.google.com/)
   - Save as `credentials.json` in the project root

2. **FFmpeg**: Required for extracting media duration
   ```bash
   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Verify installation
   ffmpeg -version
   ```

3. **Catalog CSV**: A CSV file with Google Drive file metadata containing:
   - `gdrive_id`: Google Drive file ID
   - `name`: File name
   - `mime_type`: MIME type (e.g., audio/mpeg, video/mp4)
   - `size_bytes`: File size in bytes
   - `parents`: Parent folder IDs (JSON array)
   - `web_content_link`: Download link
   - `duration_milliseconds` (optional): Pre-computed duration

## Basic Usage

### Simple Batch Transcription

Process all audio/video files with a single worker:

```bash
gtranscriber batch-transcribe input/catalog.csv --credentials credentials.json
```

This will:
- Read `input/catalog.csv`
- Filter only audio and video files
- Download and transcribe each file
- Save results to `results/` directory
- Create checkpoint at `results/checkpoint.json`

### Parallel Processing

Use multiple workers for faster processing:

```bash
gtranscriber batch-transcribe input/catalog.csv \
  --credentials credentials.json \
  --workers 4
```

**Note**: Each worker loads its own model instance into memory. Consider available RAM/VRAM when choosing worker count.

### Custom Output Directory

Specify a custom output directory:

```bash
gtranscriber batch-transcribe input/catalog.csv \
  --output-dir transcriptions/ \
  --credentials credentials.json \
  --workers 2
```

### Using Different Models

Use a different Whisper model:

```bash
# Faster turbo model
gtranscriber batch-transcribe input/catalog.csv \
  --model-id openai/whisper-large-v3-turbo \
  --credentials credentials.json \
  --workers 4

# Distilled model (smaller, faster)
gtranscriber batch-transcribe input/catalog.csv \
  --model-id distil-whisper/distil-large-v3 \
  --credentials credentials.json \
  --workers 4
```

### Memory Optimization

Use quantization to reduce VRAM usage:

```bash
gtranscriber batch-transcribe input/catalog.csv \
  --credentials credentials.json \
  --quantize \
  --workers 2
```

### Force CPU Processing

Process on CPU instead of GPU:

```bash
gtranscriber batch-transcribe input/catalog.csv \
  --credentials credentials.json \
  --cpu \
  --workers 2
```

**Note**: CPU processing is slower but doesn't require GPU/VRAM.

## Checkpoint and Resume

The batch transcription automatically creates a checkpoint file that tracks:
- Completed files
- Failed files with error messages
- Total progress

### Resuming Interrupted Jobs

If the process is interrupted (Ctrl+C, crash, etc.), simply run the same command again:

```bash
gtranscriber batch-transcribe input/catalog.csv \
  --credentials credentials.json \
  --workers 4
```

The system will:
- Load the checkpoint file
- Skip already completed files
- Resume processing from where it left off

### Custom Checkpoint Location

Specify a custom checkpoint file location:

```bash
gtranscriber batch-transcribe input/catalog.csv \
  --credentials credentials.json \
  --checkpoint my_checkpoint.json \
  --workers 4
```

### Starting Fresh

To start over from scratch, delete the checkpoint file:

```bash
rm results/checkpoint.json
gtranscriber batch-transcribe input/catalog.csv --credentials credentials.json
```

## Output Format

Each transcribed file produces a JSON file with this structure:

```json
{
  "gdrive_id": "1JtKnN2pQGmHEkSPniwES6RmWWp5BtKrU",
  "name": "audio.m4a",
  "mimeType": "audio/mpeg",
  "size_bytes": 143657567,
  "duration_milliseconds": 120000,
  "parents": ["1OusxPzsL5cVb06sMnyTqtCwlalvv2HDd"],
  "webContentLink": "https://drive.google.com/uc?id=...",
  "transcription_text": "Full transcription text here...",
  "detected_language": "pt",
  "language_probability": 0.98,
  "model_id": "openai/whisper-large-v3",
  "compute_device": "cuda:0",
  "processing_duration_sec": 45.2,
  "transcription_status": "completed",
  "created_at_enrichment": "2025-12-10T22:30:00",
  "segments": [
    {
      "text": "Segment text",
      "start": 0.0,
      "end": 3.5
    }
  ]
}
```

Files are named: `{gdrive_id}_transcription.json`

## Performance Considerations

### Choosing Worker Count

**GPU Processing:**
- Each worker loads a full model copy into VRAM
- Typical VRAM usage per model:
  - whisper-large-v3: ~3-6 GB
  - whisper-large-v3-turbo: ~2-4 GB  
  - distil-whisper: ~1-2 GB
- Example: 24 GB GPU → 4 workers with whisper-large-v3

**CPU Processing:**
- Each worker loads model into RAM
- Typical RAM usage per model: 4-8 GB
- CPU processing is 5-10x slower than GPU
- Example: 32 GB RAM → 4 workers

### Optimizing for Speed

1. **Use GPU** if available (much faster)
2. **Use turbo or distilled models** for faster processing
3. **Maximize workers** based on available VRAM/RAM
4. **Use quantization** to fit more workers

### Optimizing for Accuracy

1. **Use large models** (whisper-large-v3)
2. **Use fewer workers** to ensure quality
3. **Avoid quantization** for best accuracy

## Troubleshooting

### "Failed to extract duration"

If FFmpeg is not installed or file format is unsupported:
- Install FFmpeg
- Duration will be `null` in output (processing continues)
- Pre-compute durations in catalog CSV if needed

### "Out of memory" errors

- Reduce number of workers
- Use quantization: `--quantize`
- Use smaller model: `--model-id distil-whisper/distil-large-v3`
- Force CPU: `--cpu`

### "Credentials not found"

- Ensure `credentials.json` exists
- Use `--credentials` to specify path
- Follow Google Cloud Console setup for OAuth2

### Files being skipped

- Check checkpoint file: `results/checkpoint.json`
- Files already in `completed_files` are skipped
- Delete checkpoint to reprocess all files

## Advanced Usage

### Processing Specific File Types

Edit `src/gtranscriber/core/batch.py` to modify `AUDIO_VIDEO_MIME_TYPES` set:

```python
AUDIO_VIDEO_MIME_TYPES = {
    "audio/mpeg",
    "video/mp4",
    # Add or remove types as needed
}
```

### Custom Model Parameters

Modify engine initialization in `transcribe_single_file()` to add custom parameters:

```python
engine = WhisperEngine(
    model_id=config.model_id,
    force_cpu=config.force_cpu,
    quantize=config.quantize,
    chunk_length_s=30,  # Custom chunk length
    stride_length_s=5,  # Custom stride length
)
```

## Examples

### Example 1: Small Dataset (< 10 files)

```bash
gtranscriber batch-transcribe input/catalog.csv \
  --credentials credentials.json \
  --workers 1
```

### Example 2: Medium Dataset (10-100 files)

```bash
gtranscriber batch-transcribe input/catalog.csv \
  --credentials credentials.json \
  --workers 4 \
  --quantize
```

### Example 3: Large Dataset (100+ files)

```bash
# Initial run
gtranscriber batch-transcribe input/catalog.csv \
  --credentials credentials.json \
  --model-id openai/whisper-large-v3-turbo \
  --workers 8 \
  --quantize

# If interrupted, resume:
gtranscriber batch-transcribe input/catalog.csv \
  --credentials credentials.json \
  --model-id openai/whisper-large-v3-turbo \
  --workers 8 \
  --quantize
```

### Example 4: CPU Only (No GPU)

```bash
gtranscriber batch-transcribe input/catalog.csv \
  --credentials credentials.json \
  --cpu \
  --workers 2 \
  --model-id distil-whisper/distil-large-v3
```

## Monitoring Progress

The system logs progress in real-time:

```
INFO - Loaded 817 audio/video files from catalog
INFO - Total files: 817
INFO - Already completed: 0
INFO - Remaining to process: 817
INFO - Using 4 parallel workers
INFO - Processing file: audio1.m4a (1JtK...)
INFO - ✓ Completed: audio1.m4a
INFO - Progress: 1/817 files
INFO - ✓ Completed: audio2.m4a
INFO - Progress: 2/817 files
...
```

At completion:

```
============================================================
Batch transcription completed!
Total files: 817
Successfully transcribed: 810
Failed: 7
Success rate: 99.1%
============================================================
```

## Support

For issues or questions:
- Check logs for error messages
- Review checkpoint file: `results/checkpoint.json`
- Verify Google Drive credentials
- Ensure FFmpeg is installed
- Check available VRAM/RAM vs worker count
