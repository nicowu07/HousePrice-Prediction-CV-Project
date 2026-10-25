#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
if [[ -d /Library/TeX/texbin ]]; then
    export PATH="/Library/TeX/texbin:$PATH"
fi
if ! command -v latexmk >/dev/null 2>&1; then
    printf '%s\n' 'latexmk is missing. Install TeX Live / MacTeX before watching.' >&2
    exit 1
fi
# Watch only LaTeX dependencies. No viewer is launched and no login service is installed.
exec latexmk -pdf -pvc -view=none main.tex
