"""Utilities for stripping notebook outputs before they enter git history."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


_CELL_METADATA_KEYS = {"collapsed", "execution", "ExecuteTime", "scrolled"}
_JUPYTER_METADATA_KEYS = {"outputs_hidden", "source_hidden"}


def clean_notebook_content(notebook: dict) -> tuple[dict, bool]:
    """Return a cleaned notebook object and whether any content changed."""
    cleaned = notebook
    changed = False

    metadata = cleaned.setdefault("metadata", {})
    if "widgets" in metadata:
        metadata.pop("widgets", None)
        changed = True

    for cell in cleaned.get("cells", []):
        cell_metadata = cell.setdefault("metadata", {})
        for key in list(cell_metadata.keys()):
            if key in _CELL_METADATA_KEYS:
                cell_metadata.pop(key, None)
                changed = True

        jupyter_metadata = cell_metadata.get("jupyter")
        if isinstance(jupyter_metadata, dict):
            for key in list(jupyter_metadata.keys()):
                if key in _JUPYTER_METADATA_KEYS:
                    jupyter_metadata.pop(key, None)
                    changed = True
            if not jupyter_metadata:
                cell_metadata.pop("jupyter", None)
                changed = True

        if cell.get("cell_type") == "code":
            if cell.get("execution_count") is not None:
                cell["execution_count"] = None
                changed = True
            if cell.get("outputs"):
                cell["outputs"] = []
                changed = True

    return cleaned, changed


def clean_notebook_file(path: str | Path) -> bool:
    """Clean a notebook file in place and return whether it changed."""
    notebook_path = Path(path)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cleaned, changed = clean_notebook_content(notebook)
    if changed:
        notebook_path.write_text(
            json.dumps(cleaned, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return changed


def list_staged_notebooks(repo_root: str | Path | None = None) -> list[Path]:
    """Return staged notebook paths relative to *repo_root*."""
    root = Path("." if repo_root is None else repo_root)
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    notebooks = []
    for line in result.stdout.splitlines():
        if line.lower().endswith(".ipynb"):
            notebook_path = root / line
            if notebook_path.exists():
                notebooks.append(notebook_path)
    return notebooks


def clean_staged_notebooks(repo_root: str | Path | None = None) -> list[Path]:
    """Clean all staged notebooks and re-stage the ones that changed."""
    root = Path("." if repo_root is None else repo_root)
    changed_paths = []
    for notebook_path in list_staged_notebooks(root):
        if clean_notebook_file(notebook_path):
            subprocess.run(["git", "add", str(notebook_path)], cwd=root, check=True)
            changed_paths.append(notebook_path)
    return changed_paths


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for stripping notebook outputs."""
    parser = argparse.ArgumentParser(description="Strip outputs and execution metadata from Jupyter notebooks.")
    parser.add_argument("paths", nargs="*", help="Notebook paths to clean")
    parser.add_argument("--staged", action="store_true", help="Clean staged notebooks and re-stage them")
    args = parser.parse_args(argv)

    if args.staged:
        changed_paths = clean_staged_notebooks()
        if changed_paths:
            print("Cleaned staged notebooks:")
            for path in changed_paths:
                print(f"  {path}")
        return 0

    for path in args.paths:
        changed = clean_notebook_file(path)
        status = "cleaned" if changed else "unchanged"
        print(f"{status}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI helper
    raise SystemExit(main())
