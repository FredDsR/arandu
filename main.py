import sys
import argparse
from transcription import transcribe_audio


def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio/video file using a speech model.")
    parser.add_argument("file_path", help="Path to the input audio/video file to transcribe")
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

    model_id = "facebook/s2t-small-librispeech-asr"

    print(f"> Running transcription model {model_id} for file: {file_path}")
    if force_cpu:
        print("> Device override: CPU forced by --cpu flag")

    transcription = transcribe_audio(model_id, file_path, force_cpu=force_cpu)

    print(f'> Transcription:\n\n"{transcription[:100]}..."')

    with open(f"{file_path}_transcription.txt", "w") as f:
        f.write(transcription)


if __name__ == "__main__":
    main()
