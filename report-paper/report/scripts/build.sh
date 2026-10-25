#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
if [[ -d /Library/TeX/texbin ]]; then
    export PATH="/Library/TeX/texbin:$PATH"
fi
if ! command -v latexmk >/dev/null 2>&1; then
    printf '%s\n' 'latexmk is missing. Install TeX Live / MacTeX before building.' >&2
    exit 1
fi
exec latexmk -pdf main.tex
