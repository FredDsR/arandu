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
  --model-id openai/whisper-large-v3 \
  --credentials credentials.json \
  --workers 4

# Distilled model (smaller, faster)
gtranscriber batch-transcribe input/catalog.csv \
  --model-id distil-whisper/distil-large-v3 \
  --credentials credentials.json \
  --workers 4
```

### Pipeline ID Tracking

Use a custom pipeline ID to group related processing steps:

```bash
gtranscriber batch-transcribe input/catalog.csv \
  --id etno-project-001 \
  --credentials credentials.json \
  --workers 4
```

The pipeline ID:
- Creates a versioned results directory (e.g., `results/etno-project-001/transcription_YYYYMMDD_HHMMSS/`)
- Enables tracking of the entire pipeline run
- Can be used to link transcription with downstream QA/CEP processing

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

### Language Configuration

Specify the transcription language for better accuracy and downstream processing:

```bash
# Portuguese transcriptions (ETno project)
gtranscriber batch-transcribe input/catalog.csv \
  --credentials credentials.json \
  --language pt \
  --workers 4

# Spanish transcriptions
gtranscriber batch-transcribe input/catalog.csv \
  --credentials credentials.json \
  --language es \
  --workers 4
```

**Important for KG Construction**: The `--language` option sets both `detected_language` and `metadata.lang` in the output JSON. The `metadata.lang` field is **critical** for AutoSchemaKG to route extraction to the correct language-specific prompts during knowledge graph construction.

If language is not specified, Whisper will auto-detect, but it's recommended to set it explicitly for:
- Better transcription accuracy
- Consistent language metadata for downstream processing
- Proper routing in multilingual KG construction

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
  "web_content_link": "https://drive.google.com/uc?id=...",
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
  ],
  "transcription_quality": {
    "script_match_score": 1.0,
    "repetition_score": 0.95,
    "segment_quality_score": 1.0,
    "content_density_score": 0.85,
    "overall_score": 0.94,
    "issues_detected": [],
    "quality_rationale": "High quality transcription"
  },
  "is_valid": true
}
```

**Quality Fields**:
- `transcription_quality`: Quality scores from heuristic-based validation
- `is_valid`: Boolean indicating if transcription passes quality threshold

Files are named: `{gdrive_id}_transcription.json`

## Performance Considerations

### Choosing Worker Count

**GPU Processing:**
- Each worker loads a full model copy into VRAM
- Typical VRAM usage per model:
  - whisper-large-v3: ~3-6 GB
  - whisper-large-v3: ~2-4 GB  
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

### "Incomplete download" errors

If you see errors like:
```
Incomplete download for 'video.mp4': expected 546.43 MB but got 123.45 MB (22.6% complete)
```

This indicates the download was interrupted or truncated. The system will:
- **Automatically retry up to 5 times** with exponential backoff (4s → 60s delays)
- Delete the incomplete file before each retry
- Log retry attempts with wait times

Common causes:
- Network instability or timeouts
- Google Drive rate limiting (for large files or many requests)
- Temporary Google API issues

If retries fail:
- Check your network connection
- Wait a few minutes and run the command again (checkpoint will resume)
- For very large files, try processing during off-peak hours

### "Empty download" errors

If you see:
```
Download resulted in empty file for 'audio.mp3'
```

This typically indicates:
- Google Drive API permission issues
- The file was deleted or moved in Drive
- OAuth token expired

Solutions:
- Delete `token.json` and re-authenticate
- Verify the file exists and is accessible in Google Drive
- Check that your OAuth credentials have Drive read access

### "No audio stream found" errors

If you see:
```
No audio stream found in 'video.mp4'
```

This means the file has no audio track to transcribe. Common causes:
- Video file recorded without audio (e.g., screen recording with mic disabled)
- Corrupted audio track
- Unsupported audio codec

To inspect the file:
```bash
ffprobe -v error -show_streams your_file.mp4
```

The file will be marked as failed and skipped.

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

### "Soundfile is either not in the correct format or is malformed"

This error from the Whisper pipeline usually indicates:
- **Incomplete download** - The file didn't download fully (see "Incomplete download" above)
- **No audio stream** - The file has no audio track (see "No audio stream found" above)
- **Corrupted file** - The source file in Google Drive may be corrupted

The system now validates downloads and audio streams before transcription, so this error should be rare. If it occurs:
1. Check the checkpoint file for the specific file ID
2. Manually download the file from Google Drive and test with `ffprobe`
3. If the source file is valid, report the issue

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
  --model-id openai/whisper-large-v3 \
  --workers 8 \
  --quantize

# If interrupted, resume:
gtranscriber batch-transcribe input/catalog.csv \
  --credentials credentials.json \
  --model-id openai/whisper-large-v3 \
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

### Example 5: Portuguese Corpus (ETno Project)

For the ETno project with Portuguese transcriptions:

```bash
# Transcribe with explicit Portuguese language setting
gtranscriber batch-transcribe input/etno_catalog.csv \
  --credentials credentials.json \
  --language pt \
  --model-id openai/whisper-large-v3 \
  --workers 4 \
  --quantize

# After transcription, the output will have:
# - detected_language: "pt"
# - metadata.lang: "pt"
# This ensures proper routing to Portuguese prompts during KG construction
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
