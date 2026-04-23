#!/bin/bash
#SBATCH --job-name=pipeline-cleanup
#SBATCH --partition=draco
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=logs/cleanup_%x_%j.out
#SBATCH --error=logs/cleanup_%x_%j.err

# =============================================================================
# Surgical KG pipeline cleanup (Docker-in-SLURM)
#
# When a KG run fails partway, the container leaves root-owned files under
# results/<pipeline>/kg/ that the user cannot rm from a login shell. This job
# spins up an alpine container that mounts just the pipeline's kg/ dir and:
#   1. deletes kg_extraction/<CLEAN_MODEL>_*.json
#   2. deletes concepts/concept_completed.csv + concepts/concept_shard_*.csv
#   3. restores triples_csv/missing_concepts_*.csv from its .bak (if present)
#   4. resets kg/kg_checkpoint.json to an empty fresh state
#
# Usage:
#   PIPELINE_ID=test-kg-04 sbatch scripts/slurm/kg/pipeline-cleanup.slurm
#   PIPELINE_ID=test-kg-04 CLEAN_MODEL=qwen3:14b sbatch ...
#
# Required env vars:
#   PIPELINE_ID  - pipeline directory under results/ (e.g. test-kg-04)
#
# Optional env vars:
#   CLEAN_MODEL  - prefix for kg_extraction JSONs to delete (default: qwen3.5:9b)
#   PROJECT_DIR  - project root (default: $HOME/etno-kgc-preprocessing)
# =============================================================================

set -euo pipefail

: "${PIPELINE_ID:?PIPELINE_ID env var is required}"
CLEAN_MODEL="${CLEAN_MODEL:-qwen3.5:9b}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/etno-kgc-preprocessing}"

# Reject anything that could escape results/<name>/
case "$PIPELINE_ID" in
    */*|*..*|.|"")
        echo "ERROR: PIPELINE_ID must be a single directory name (got: '$PIPELINE_ID')"
        exit 1
        ;;
esac

PIPELINE_DIR="$PROJECT_DIR/results/$PIPELINE_ID"
if [ ! -d "$PIPELINE_DIR/kg" ]; then
    echo "ERROR: No kg/ directory at $PIPELINE_DIR/kg"
    exit 1
fi

echo "=============================================="
echo "KG Pipeline Cleanup"
echo "=============================================="
echo "Pipeline:    $PIPELINE_ID"
echo "Clean model: $CLEAN_MODEL"
echo "Target dir:  $PIPELINE_DIR/kg"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        $(hostname)"
echo "=============================================="

# Narrow blast radius: mount ONLY the pipeline's kg/ dir into the container.
docker run --rm \
    -v "$PIPELINE_DIR/kg":/target \
    -e CLEAN_MODEL="$CLEAN_MODEL" \
    alpine sh -c '
        set -eu
        TARGET=/target
        ATLAS="$TARGET/outputs/atlas_output"

        echo ""
        echo "--- kg_extraction: remove ${CLEAN_MODEL}_*.json ---"
        if [ -d "$ATLAS/kg_extraction" ]; then
            find "$ATLAS/kg_extraction" -maxdepth 1 -type f \
                -name "${CLEAN_MODEL}_*.json" -print -delete || true
        fi

        echo ""
        echo "--- concepts: remove concept_completed.csv + concept_shard_*.csv ---"
        rm -fv "$ATLAS/concepts/concept_completed.csv" 2>/dev/null || true
        for f in "$ATLAS/concepts/concept_shard_"*.csv; do
            [ -e "$f" ] && rm -fv "$f"
        done

        echo ""
        echo "--- triples_csv: restore missing_concepts from .bak ---"
        BAK="$ATLAS/triples_csv/missing_concepts_transcriptions.json_from_json.csv.bak"
        CSV="$ATLAS/triples_csv/missing_concepts_transcriptions.json_from_json.csv"
        if [ -f "$BAK" ]; then
            mv -v "$BAK" "$CSV"
        else
            echo "No .bak to restore (skipping)"
        fi

        echo ""
        echo "--- kg_checkpoint.json: reset ---"
        cat > "$TARGET/kg_checkpoint.json" <<EOF
{
  "completed_files": [],
  "failed_files": {},
  "total_files": 0,
  "started_at": null,
  "last_updated": null
}
EOF
        echo "Wrote fresh kg_checkpoint.json"

        echo ""
        echo "--- final state ---"
        echo "kg_extraction/:"
        ls -la "$ATLAS/kg_extraction" 2>/dev/null || echo "(none)"
        echo ""
        echo "concepts/:"
        ls -la "$ATLAS/concepts" 2>/dev/null || echo "(none)"
        echo ""
        echo "triples_csv/:"
        ls -la "$ATLAS/triples_csv" 2>/dev/null || echo "(none)"
    '

echo ""
echo "=============================================="
echo "Cleanup complete"
echo "=============================================="
