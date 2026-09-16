"""Fetch the scout's GGUF weights.

Filenames inside HF GGUF repos aren't stable across quants/uploads, so we list
the repo and pick the best match rather than hard-coding a filename (which is
the usual cause of a Docker build failing at the download step).
"""

from __future__ import annotations

import sys

from huggingface_hub import HfApi, hf_hub_download

from config import settings


def main() -> int:
    api = HfApi()
    files = [f for f in api.list_repo_files(settings.scout_repo_id) if f.endswith(".gguf")]
    if not files:
        print(f"No .gguf files in {settings.scout_repo_id}", file=sys.stderr)
        return 1

    matches = sorted(f for f in files if settings.scout_quant.lower() in f.lower())
    chosen = matches or sorted(files)
    # Split models ship as -00001-of-0000N; grab every shard of the chosen set.
    if matches and any("-of-" in f for f in matches):
        targets = matches
    else:
        targets = [chosen[0]]

    for filename in targets:
        print(f"Downloading {settings.scout_repo_id}/{filename} -> {settings.scout_model_dir}")
        hf_hub_download(
            repo_id=settings.scout_repo_id,
            filename=filename,
            local_dir=settings.scout_model_dir,
        )

    print("Resolved model path:", settings.resolve_model_path())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
