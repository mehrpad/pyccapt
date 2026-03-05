# Building the Documentation

PyCCAPT documentation is built with Sphinx.

## Prerequisites

From the repository root, install documentation dependencies:

```bash
pip install -r docs/requirements.txt
```

## Build HTML (Recommended)

From the repository root:

```bash
sphinx-build -b html docs docs/_build/html
```

Open the generated entry point:

```text
docs/_build/html/index.html
```

## Build via `make` Helpers

If you prefer `make` scripts, run from the `docs/` directory:

```bash
cd docs
make clean
make html
```

On Windows, use:

```powershell
cd docs
.\make.bat clean
.\make.bat html
```

## Regenerate API Stubs (When Needed)

Regenerate API `.rst` files only when package/module structure changes:

```bash
cd docs
sphinx-apidoc -o . ../pyccapt
```
