#!/usr/bin/env bash
set -euo pipefail

# Install uv if not available
if ! command -v uv &>/dev/null; then
    echo "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Source the env so uv is available in the current shell
    export PATH="$HOME/.local/bin:$PATH"

    if ! command -v uv &>/dev/null; then
        echo "ERROR: uv installation failed" >&2
        exit 1
    fi
    echo "uv installed successfully: $(uv --version)"
fi

# Run the training script with uv (PyPy for speed, 5GB virtual memory limit for safety)
ulimit -v 5242880
uv run --python pypy gpt.py
