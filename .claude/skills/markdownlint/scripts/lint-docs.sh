#!/bin/bash
# Lint all documentation markdown files in the open-responses-server repo.
# Uses the global markdownlint config at ~/.markdownlint-cli2.yaml unless
# a project-level .markdownlint-cli2.yaml exists.
#
# Usage:
#   bash lint-docs.sh          # check all docs
#   bash lint-docs.sh --fix    # auto-fix fixable issues
#   bash lint-docs.sh FILE...  # check specific files

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

# Determine config: project-level takes precedence over global
if [ -f "$REPO_ROOT/.markdownlint-cli2.yaml" ]; then
    CONFIG_FLAG=""
elif [ -f "$HOME/.markdownlint-cli2.yaml" ]; then
    CONFIG_FLAG="--config $HOME/.markdownlint-cli2.yaml"
else
    CONFIG_FLAG=""
fi

# Check for --fix flag
FIX_FLAG=""
FILES=()
for arg in "$@"; do
    if [ "$arg" = "--fix" ]; then
        FIX_FLAG="--fix"
    else
        FILES+=("$arg")
    fi
done

# Default file set: docs/ (excluding plan/ and prompts/), index.md, CLAUDE.md, skills
if [ ${#FILES[@]} -eq 0 ]; then
    FILES=(
        "$REPO_ROOT/docs/*.md"
        "$REPO_ROOT/index.md"
        "$REPO_ROOT/CLAUDE.md"
        "$REPO_ROOT/.claude/skills/**/*.md"
    )
fi

# Ignore patterns for historical/archive docs
IGNORE_FLAGS=(
    "!$REPO_ROOT/docs/plan/**"
    "!$REPO_ROOT/docs/prompts/**"
    "!$REPO_ROOT/docs/pip-publish-instructions.md"
    "!$REPO_ROOT/docs/using-uv.md"
)

echo "Running markdownlint-cli2 on ${#FILES[@]} pattern(s)..."
# shellcheck disable=SC2086
set +e
markdownlint-cli2 $FIX_FLAG $CONFIG_FLAG "${FILES[@]}" "${IGNORE_FLAGS[@]}"
EXIT_CODE=$?
set -e

if [ $EXIT_CODE -eq 0 ]; then
    echo "All files pass."
else
    echo "Found lint issues (exit code $EXIT_CODE)."
fi

exit $EXIT_CODE
