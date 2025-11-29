from pathlib import Path
import os
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
    return pipe


class TranscriptionModel:
    def __init__(self, model_id: str, force_cpu: bool = True):
        self.pipe = get_pipeline(model_id, force_cpu=force_cpu)

    def transcribe(self, audio: Path) -> str:
        result = self.pipe(str(audio))
        return result["text"]


def transcribe_audio(model_id: str, audio: Path, force_cpu: bool = True) -> str:
    model = TranscriptionModel(model_id, force_cpu=force_cpu)
    return model.transcribe(audio)
