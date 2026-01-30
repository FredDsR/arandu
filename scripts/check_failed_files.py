#!/usr/bin/env python3
"""
Diagnostic script to check failed audio/video files from the transcription pipeline.

This script:
1. Reads failed file IDs from checkpoint.json
2. Looks up file metadata from catalog.csv
3. Downloads files temporarily and analyzes them with ffprobe
4. Generates a detailed diagnostic report
"""

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import requests

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "results" / "checkpoint.json"
CATALOG_PATH = PROJECT_ROOT / "input" / "catalog.csv"
REPORT_PATH = PROJECT_ROOT / "results" / "failed_files_report.json"


def load_checkpoint() -> dict[str, Any]:
    """Load the checkpoint file with failed files information."""
    with open(CHECKPOINT_PATH) as f:
        return json.load(f)


def load_catalog() -> dict[str, dict[str, Any]]:
    """Load the catalog CSV and index by gdrive_id."""
    catalog = {}
    with open(CATALOG_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            catalog[row["gdrive_id"]] = row
    return catalog


def run_ffprobe(file_path: str) -> dict[str, Any]:
    """Run ffprobe on a file and return the analysis."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-print_format",
                "json",
                file_path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0 and result.stdout:
            return {
                "success": True,
                "data": json.loads(result.stdout),
                "stderr": result.stderr if result.stderr else None,
            }
        else:
            return {
                "success": False,
                "error": result.stderr or "ffprobe returned no output",
                "returncode": result.returncode,
            }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "ffprobe timed out"}
    except FileNotFoundError:
        return {"success": False, "error": "ffprobe not found - please install ffmpeg"}
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Failed to parse ffprobe output: {e}"}


def check_file_header(url: str, timeout: int = 30) -> dict[str, Any]:
    """Download just the file header to check content type."""
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return {
            "status_code": response.status_code,
            "content_type": response.headers.get("Content-Type"),
            "content_length": response.headers.get("Content-Length"),
            "accessible": response.status_code == 200,
        }
    except requests.RequestException as e:
        return {"accessible": False, "error": str(e)}


def download_and_analyze(file_id: str, url: str, filename: str) -> dict[str, Any]:
    """Download a file temporarily and analyze it with ffprobe."""
    result = {"file_id": file_id, "filename": filename, "url": url}

    # First check if file is accessible
    header_check = check_file_header(url)
    result["header_check"] = header_check

    if not header_check.get("accessible"):
        result["diagnosis"] = "FILE_NOT_ACCESSIBLE"
        result["recommendation"] = "Check file permissions or if file still exists in Google Drive"
        return result

    # Download and analyze with ffprobe
    try:
        # Get file extension from filename
        ext = Path(filename).suffix or ".tmp"

        with tempfile.NamedTemporaryFile(suffix=ext, delete=True) as tmp_file:
            print(f"  Downloading {filename}...", end=" ", flush=True)

            response = requests.get(url, timeout=120, stream=True)
            response.raise_for_status()

            # Download file
            total_size = 0
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
                total_size += len(chunk)

            tmp_file.flush()
            print(f"({total_size / 1024 / 1024:.1f} MB)")

            result["downloaded_size"] = total_size

            # Run ffprobe
            print("  Analyzing with ffprobe...", end=" ", flush=True)
            ffprobe_result = run_ffprobe(tmp_file.name)
            result["ffprobe"] = ffprobe_result

            if ffprobe_result["success"]:
                print("OK")
                data = ffprobe_result["data"]
                streams = data.get("streams", [])
                audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
                video_streams = [s for s in streams if s.get("codec_type") == "video"]

                result["has_audio"] = len(audio_streams) > 0
                result["has_video"] = len(video_streams) > 0
                result["audio_codec"] = (
                    audio_streams[0].get("codec_name") if audio_streams else None
                )
                result["video_codec"] = (
                    video_streams[0].get("codec_name") if video_streams else None
                )
                result["format"] = data.get("format", {}).get("format_name")
                result["duration"] = data.get("format", {}).get("duration")

                # Diagnosis
                if not audio_streams:
                    result["diagnosis"] = "NO_AUDIO_STREAM"
                    result["recommendation"] = "File has no audio track - cannot transcribe"
                else:
                    result["diagnosis"] = "VALID_AUDIO"
                    result["recommendation"] = (
                        "File appears valid - error may be transient or codec-specific"
                    )
            else:
                print("FAILED")
                result["diagnosis"] = "FFPROBE_FAILED"
                result["recommendation"] = (
                    f"ffprobe could not analyze: {ffprobe_result.get('error')}"
                )

    except requests.RequestException as e:
        print("DOWNLOAD FAILED")
        result["diagnosis"] = "DOWNLOAD_FAILED"
        result["recommendation"] = f"Could not download file: {e}"
    except Exception as e:
        print(f"ERROR: {e}")
        result["diagnosis"] = "ANALYSIS_ERROR"
        result["recommendation"] = f"Unexpected error during analysis: {e}"

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose failed transcription files")
    parser.add_argument(
        "--limit", "-l", type=int, default=None, help="Limit number of files to check (for testing)"
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true", help="Only show file info without downloading"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=REPORT_PATH, help="Output path for the report JSON"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Failed Files Diagnostic Tool")
    print("=" * 60)

    # Load data
    print("\nLoading checkpoint and catalog...")
    checkpoint = load_checkpoint()
    catalog = load_catalog()

    failed_files = checkpoint.get("failed_files", {})
    print(f"Found {len(failed_files)} failed files")

    # Collect file info
    files_to_check = []
    missing_in_catalog = []

    for file_id, error_msg in failed_files.items():
        if file_id in catalog:
            info = catalog[file_id]
            files_to_check.append(
                {
                    "file_id": file_id,
                    "filename": info["name"],
                    "mime_type": info["mime_type"],
                    "size_bytes": info.get("size_bytes"),
                    "download_url": info["web_content_link"],
                    "original_error": error_msg,
                }
            )
        else:
            missing_in_catalog.append(file_id)

    if missing_in_catalog:
        print(f"\nWarning: {len(missing_in_catalog)} files not found in catalog:")
        for fid in missing_in_catalog[:5]:
            print(f"  - {fid}")
        if len(missing_in_catalog) > 5:
            print(f"  ... and {len(missing_in_catalog) - 5} more")

    # Apply limit if specified
    if args.limit:
        files_to_check = files_to_check[: args.limit]
        print(f"\nLimiting to {args.limit} files")

    # Categorize by error type
    print("\n" + "-" * 60)
    print("Error Summary:")
    print("-" * 60)

    error_categories = {}
    for f in files_to_check:
        error = f["original_error"]
        # Simplify error message for categorization
        if "Soundfile is either not in the correct format" in error:
            category = "MALFORMED_AUDIO"
        elif "No audio stream found" in error:
            category = "NO_AUDIO_STREAM"
        elif "404" in error or "File not found" in error:
            category = "FILE_NOT_FOUND"
        else:
            category = "OTHER"

        error_categories.setdefault(category, []).append(f)

    for category, files in error_categories.items():
        print(f"  {category}: {len(files)} files")

    if args.dry_run:
        print("\n" + "-" * 60)
        print("Dry run - Files that would be checked:")
        print("-" * 60)
        for f in files_to_check:
            print(f"\n  ID: {f['file_id']}")
            print(f"  Name: {f['filename']}")
            print(f"  MIME: {f['mime_type']}")
            print(f"  Size: {int(f['size_bytes'] or 0) / 1024 / 1024:.1f} MB")
            print(f"  Error: {f['original_error'][:80]}...")
        return

    # Analyze files
    print("\n" + "-" * 60)
    print("Analyzing files...")
    print("-" * 60)

    results = []
    for i, f in enumerate(files_to_check, 1):
        print(f"\n[{i}/{len(files_to_check)}] {f['filename']}")
        print(f"  Original error: {f['original_error'][:60]}...")

        result = download_and_analyze(f["file_id"], f["download_url"], f["filename"])
        result["original_error"] = f["original_error"]
        result["mime_type"] = f["mime_type"]
        result["catalog_size"] = f["size_bytes"]
        results.append(result)

    # Generate report
    report: dict = {
        "total_failed": len(failed_files),
        "analyzed": len(results),
        "missing_in_catalog": missing_in_catalog,
        "summary": {"by_diagnosis": {}, "by_mime_type": {}},
        "files": results,
    }

    # Summarize results
    for r in results:
        diag = r.get("diagnosis", "UNKNOWN")
        report["summary"]["by_diagnosis"].setdefault(diag, 0)
        report["summary"]["by_diagnosis"][diag] += 1

        mime = r.get("mime_type", "unknown")
        report["summary"]["by_mime_type"].setdefault(mime, 0)
        report["summary"]["by_mime_type"][mime] += 1

    # Save report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"\nAnalyzed: {len(results)} / {len(failed_files)} failed files")
    print("\nBy Diagnosis:")
    for diag, count in sorted(report["summary"]["by_diagnosis"].items()):
        print(f"  {diag}: {count}")

    print("\nBy MIME Type:")
    for mime, count in sorted(report["summary"]["by_mime_type"].items()):
        print(f"  {mime}: {count}")

    print(f"\nFull report saved to: {args.output}")

    # Print recommendations
    print("\n" + "-" * 60)
    print("RECOMMENDATIONS")
    print("-" * 60)

    recommendations = {}
    for r in results:
        rec = r.get("recommendation")
        if rec:
            recommendations.setdefault(rec, []).append(r["filename"])

    for rec, files in recommendations.items():
        print(f"\n{rec}")
        print(f"  Affected files: {len(files)}")
        for fname in files[:3]:
            print(f"    - {fname}")
        if len(files) > 3:
            print(f"    ... and {len(files) - 3} more")


if __name__ == "__main__":
    main()
