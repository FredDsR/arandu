import sys
import json
import argparse
from transcription import transcribe_audio
from transcription.load_model import prepare_audio_input


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe an audio/video file using a speech model."
    )
    parser.add_argument(
        "file_path", help="Path to the input audio/video file to transcribe"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution (disables CUDA even if available)",
    )

    parsed = parser.parse_args()

    script_name = sys.argv[0]
    print(f"Script name: {script_name}")
    print(f"Arguments: {[parsed.file_path]}" + (" with --cpu" if parsed.cpu else ""))

    file_path = parsed.file_path
    force_cpu = parsed.cpu

    model_id = "openai/whisper-tiny"
    # model_id = "openai/whisper-large-v3"

    print(f"> Running transcription model {model_id} for file: {file_path}")
    if force_cpu:
        print("> Device override: CPU forced by --cpu flag")

    # Extract/prepare audio before loading the model to avoid delays
    prepared_path = prepare_audio_input(file_path)

    transcription = transcribe_audio(model_id, prepared_path, force_cpu=force_cpu)

    print(f'> Transcription:\n\n"{transcription["text"][:100]}..."')

    output_file = f"{file_path}_transcription.json"
    with open(output_file, "w") as f:
        json.dump(transcription, f)

    print(f"Transcription saved to {output_file}")


if __name__ == "__main__":
    main()
