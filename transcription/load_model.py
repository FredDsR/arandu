from pathlib import Path
import os
import shutil
import subprocess
from typing import Optional
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def get_device_and_dtype(force_cpu: bool = True) -> tuple[str, torch.dtype]:
    # Env override to force CPU, e.g. UV_FORCE_CPU=1
    if os.getenv("UV_FORCE_CPU") == "1":
        force_cpu = True

    if torch.cuda.is_available() and not force_cpu:
        try:
            major, minor = torch.cuda.get_device_capability(0)
        except Exception:
            major, minor = (0, 0)

        # PyTorch 2.x official wheels commonly support >= sm_70
        if major >= 7:
            return "cuda:0", torch.float16
        else:
            print(
                f"[Fallback] Detected CUDA capability {major}.{minor} (<7.0). Using CPU."
            )
            return "cpu", torch.float32
    else:
        return "cpu", torch.float32


def load_model(
    model_id: str, force_cpu: bool = True
) -> tuple[AutoModelForSpeechSeq2Seq, str, torch.dtype]:
    device, torch_dtype = get_device_and_dtype(force_cpu=force_cpu)

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    # Guard: if device is CUDA but moving fails, fallback to CPU
    try:
        model.to(device)
    except Exception as e:
        print(f"[Fallback] Moving model to {device} failed: {e}. Switching to CPU.")
        device, torch_dtype = "cpu", torch.float32
        model.to(device)

    print(f"Model loaded on {device} with dtype {torch_dtype}")
    return model, device, torch_dtype


def get_pipeline(model_id: str, force_cpu: bool = True) -> pipeline:
    model, device, torch_dtype = load_model(model_id, force_cpu=force_cpu)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=torch_dtype,
        device=device,
    )
    print("Pipeline created.")
    return pipe


class TranscriptionModel:
    def __init__(self, model_id: str, force_cpu: bool = True):
        self.pipe = get_pipeline(model_id, force_cpu=force_cpu)

    def transcribe(self, audio: Path) -> dict:
        print(f"Transcribing audio file: {audio}")
        result = self.pipe(
            str(audio), return_timestamps=True, batch_size=8, language="pt"
        )
        return result


def transcribe_audio(model_id: str, audio: Path, force_cpu: bool = True) -> dict:
    model = TranscriptionModel(model_id, force_cpu=force_cpu)
    return model.transcribe(audio)


def _check_ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _extract_audio_with_ffmpeg(
    input_path: Path,
    target_sample_rate: int = 16000,
    channels: int = 1,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    if not _check_ffmpeg_available():
        print("[Warning] ffmpeg not found in PATH; cannot extract audio from video.")
        return None

    # Decide output path: use provided path or cache next to source
    if output_path is None:
        # Save alongside original to enable reuse across runs
        wav_path = input_path.with_suffix("")
        wav_path = wav_path.parent / f"{wav_path.name}_extracted_16k_mono.wav"
    else:
        wav_path = Path(output_path)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(target_sample_rate),
        "-f",
        "wav",
        str(wav_path),
    ]
    print(f"[Info] Extracting audio via ffmpeg -> {wav_path}")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return wav_path
    except subprocess.CalledProcessError as e:
        print(f"[Error] ffmpeg extraction failed: {e}")
        return None


def prepare_audio_input(input_path: Path | str) -> Path:
    """
    Ensure the input is an audio file compatible with the ASR pipeline.
    - If input is a video (e.g., .mp4), attempt to extract the audio track to WAV 16k mono.
    - If input is already audio, pass through.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    video_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    if suffix in video_exts:
        # Preferred cached path alongside original
        cached_wav = input_path.with_suffix("")
        cached_wav = cached_wav.parent / f"{cached_wav.name}_extracted_16k_mono.wav"
        if cached_wav.exists() and cached_wav.stat().st_size > 0:
            print(f"[Info] Using cached audio: {cached_wav}")
            return cached_wav

        extracted = _extract_audio_with_ffmpeg(input_path, output_path=cached_wav)
        if extracted is not None and extracted.exists():
            return extracted
        else:
            print(
                "[Warning] Proceeding with original file; decoding may hang without ffmpeg."
            )
            return input_path
    else:
        return input_path
