#!/bin/bash

# archive_run.sh
# Usage: ./archive_run.sh <RUN_DIR> <RESULT_DIR> [LOG_FILE]

set -e  # Exit on error

RUN_DIR="$1"
RESULT_DIR="$2"
LOG_FILE="$3"

if [ -z "$RUN_DIR" ] || [ -z "$RESULT_DIR" ]; then
    echo "Usage: $0 <RUN_DIR> <RESULT_DIR> [LOG_FILE]"
    exit 1
fi

echo "=== Archiving run to: $RUN_DIR ==="

# 1. Create directory structure from template (or mkdir if template missing)
TEMPLATE_DIR="$(dirname "$0")/../run_template"
if [ -d "$TEMPLATE_DIR" ]; then
    echo "Creating structure from template..."
    # Copy structure without overwriting existing files if RUN_DIR exists
    mkdir -p "$RUN_DIR"
    cp -r -n "$TEMPLATE_DIR"/* "$RUN_DIR/" 2>/dev/null || true
else
    echo "Template not found, creating directories manually..."
    mkdir -p "$RUN_DIR"/{outputs,logs,env,meta,git}
    touch "$RUN_DIR/run_log.md"
fi

# 2. Archive Outputs
echo "Archiving outputs from $RESULT_DIR..."
if [ -d "$RESULT_DIR" ]; then
    cp -r "$RESULT_DIR"/* "$RUN_DIR/outputs/"
else
    echo "Warning: RESULT_DIR $RESULT_DIR does not exist."
fi

# 3. Archive Logs (if provided)
if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
    echo "Archiving log file: $LOG_FILE"
    cp "$LOG_FILE" "$RUN_DIR/logs/"
fi

# 4. Freeze Environment
echo "Freezing environment..."
pip freeze > "$RUN_DIR/env/pip_freeze.txt"

# 5. Record Meta Info
echo "Recording meta info..."
python3 --version > "$RUN_DIR/meta/python_version.txt" 2>&1 || true
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi > "$RUN_DIR/meta/nvidia_smi.txt"
else
    echo "nvidia-smi not found" > "$RUN_DIR/meta/nvidia_smi.txt"
fi

# 6. Record Git Info
echo "Recording git info..."
GIT_ROOT="$(dirname "$0")/.."
if [ -d "$GIT_ROOT/.git" ]; then
    git -C "$GIT_ROOT" rev-parse HEAD > "$RUN_DIR/git/commit_id.txt"
    git -C "$GIT_ROOT" status --porcelain > "$RUN_DIR/git/status_porcelain.txt"
    git -C "$GIT_ROOT" status > "$RUN_DIR/git/status_full.txt"
    git -C "$GIT_ROOT" diff > "$RUN_DIR/git/diff.patch"
else
    echo "Warning: Not a git repository or .git missing." > "$RUN_DIR/git/no_git.txt"
fi

# 7. Update Run Log
echo "Updating run_log.md..."
DATE=$(date +%F_%T)
cat >> "$RUN_DIR/run_log.md" <<EOF

## Run Archived: $DATE
- Result Source: $RESULT_DIR
- Log Source: ${LOG_FILE:-"N/A"}
- Git Commit: $(cat "$RUN_DIR/git/commit_id.txt" 2>/dev/null || echo "Unknown")
EOF

echo "=== Archive Complete: $RUN_DIR ==="
