#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROFILE_ROOT="${OUT:-${PROFILE_ROOT:-${ROOT}/outputs/joint_native_parallel_$(date +%Y%m%d_%H%M%S)}}"
export SKIP_PROFILE=1
exec bash "${ROOT}/scripts/profile_alpha_joint.sh" all --preset native-parallel --require-exclusive "$@"
