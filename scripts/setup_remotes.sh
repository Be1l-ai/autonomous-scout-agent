#!/usr/bin/env bash
# Usage: ./scripts/setup_remotes.sh <github-user>/<repo> <hf-user>/<space>
set -euo pipefail

GH="${1:-}"
HF="${2:-}"

if [[ -z "$GH" || -z "$HF" ]]; then
  echo "usage: $0 <github-user>/<repo> <hf-user>/<space-name>" >&2
  exit 1
fi

[[ -d .git ]] || { git init -b main; echo "initialised git repo"; }

git remote remove origin 2>/dev/null || true
git remote remove space  2>/dev/null || true
git remote add origin "https://github.com/${GH}.git"
git remote add space   "https://huggingface.co/spaces/${HF}"

echo "remotes configured:"
git remote -v
echo
echo "next:  git add -A && git commit -m 'initial commit' && git push -u origin main && git push space main"
