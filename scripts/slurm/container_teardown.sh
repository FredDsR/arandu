#!/bin/bash
# =============================================================================
# Shared container-teardown trap for SLURM job scripts.
#
# Now delegates to scripts/slurm/container_lib.sh (Podman pods).
# Maintained for backward compatibility.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/slurm/container_lib.sh
source "${SCRIPT_DIR}/container_lib.sh"
