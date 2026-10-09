#!/usr/bin/env bash
set -euo pipefail
SIX_ARM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${UICA_PYTHON:-python}" "$SIX_ARM_ROOT/tools/check_code.py" "$@"
