# Contributing to PyCCAPT

Thank you for contributing to PyCCAPT. This guide describes the expected workflow for issues, documentation updates, and code contributions.

## Ways to Contribute

- Report bugs or regressions
- Propose and implement new features
- Improve tutorials and documentation
- Add tests and reliability improvements

## Report an Issue

Open an issue at [GitHub Issues](https://github.com/mmonajem/pyccapt/issues) and include:

- a clear problem description
- steps to reproduce
- expected behavior and observed behavior
- environment details (OS, Python version, install method)
- relevant logs, stack traces, or screenshots

## Development Setup

1. Fork and clone the repository:

```bash
git clone https://github.com/<your-username>/pyccapt.git
cd pyccapt
```

2. Create and activate an environment:

```bash
conda create -n apt_env python=3.11
conda activate apt_env
python -m pip install --upgrade pip
```

3. Install an editable development build:

```bash
pip install -e ".[full]"
```

If you only need one module:

```bash
pip install -e ".[control]"
pip install -e ".[calibration]"
```

## Branching and Commits

1. Create a feature branch:

```bash
git checkout -b <short-descriptive-branch-name>
```

2. Make focused changes with clear commit messages.

3. Push your branch:

```bash
git push origin <short-descriptive-branch-name>
```

## Testing

Run tests from the repository root before opening a pull request:

```bash
pytest -q --run-control
pytest -q --run-calibration
```

You can also run:

```bash
pytest -q
```

This runs whichever test groups have their optional dependencies installed.

## Documentation

If your change affects behavior, configuration, public APIs, or user workflows, update the relevant documentation in:

- `README.md`
- `docs/`
- module docs under `pyccapt/control` or `pyccapt/calibration`

To build documentation locally:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

## Pull Request Checklist

Before submitting a pull request, confirm that:

- the change is scoped and well described
- tests pass locally for affected areas
- documentation is updated where needed
- backward-incompatible behavior is called out explicitly
- related issues are linked in the pull request description

## Style and Pre-commit

Linting and formatting are handled by [ruff](https://docs.astral.sh/ruff/). Run
the same checks the CI uses with:

```bash
ruff check .
ruff format --check .
```

To wire that into every commit automatically, install the pre-commit hooks
once:

```bash
pip install pre-commit
pre-commit install
```

The hook config lives in `.pre-commit-config.yaml` and runs ruff plus a few
hygiene checks (trailing whitespace, EOF newline, large files, merge markers).

## Continuous Integration

Every push and pull request triggers five workflows under `.github/workflows/`:

| Workflow      | Trigger                          | Purpose                                                  |
|---------------|----------------------------------|----------------------------------------------------------|
| `tests.yml`   | push / PR                        | pytest matrix on Linux + Windows × Python 3.10/3.11/3.12 |
| `lint.yml`    | push / PR                        | ruff lint + format check, smoke import test, pip check   |
| `docs.yml`    | push / PR / tag                  | Sphinx build (deploys to gh-pages on push to `main`)     |
| `docker.yml`  | push to `main` / tag             | Builds and pushes image to `ghcr.io/<owner>/pyccapt`     |
| `release.yml` | tag `v*`                         | Builds sdist+wheel, publishes to PyPI, opens GH Release  |

## Releasing a new version

1. Bump `__version__` in `pyccapt/__init__.py`.
2. Update `CHANGELOG.md` (or commit history) with the user-facing changes.
3. Tag the commit and push:

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

4. The `release.yml` workflow then builds the sdist+wheel, publishes to PyPI
   via [Trusted Publishing](https://docs.pypi.org/trusted-publishers/), and
   opens a GitHub Release with auto-generated notes. The `docker.yml`
   workflow pushes `:vX.Y.Z`, `:X.Y`, `:X`, and `:latest` tags to GHCR.

   *One-time setup on PyPI:* register the publisher under
   <https://pypi.org/manage/account/publishing/> with workflow `release.yml`
   and environment `pypi`. No PyPI token is stored in this repo.

## Running the Docker image locally

Build and run JupyterLab with the calibration tutorials:

```bash
docker build -t pyccapt .
docker run --rm -p 8888:8888 -v "$PWD":/work pyccapt
```

Once the official tags exist, you can also pull them directly:

```bash
docker pull ghcr.io/mmonajem/pyccapt:edge   # latest main
docker pull ghcr.io/mmonajem/pyccapt:latest # latest release
```

## Code of Conduct

Be respectful and constructive in all project interactions.
