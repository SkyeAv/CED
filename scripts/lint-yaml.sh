#!/usr/bin/env bash
set -euo pipefail

if (($# == 0)); then
  exit 0
fi

npx --yes prettier@3.6.2 --write --config .prettierrc.yaml -- "$@"
uv run --with yamllint==1.38.0 yamllint --strict --config-file .yamllint.yaml -- "$@"

# prek stashes unstaged hunks during commits, so this stages only the previously
# indexed YAML plus successful fixes, never unrelated paths or unstaged hunks.
git add -- "$@"
