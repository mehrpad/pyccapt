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

Releases are **tag-driven**, not commit-driven. Pushing or merging to `main` runs tests, lint, docs, and a Docker `:edge` build, but it does **not** publish a new PyPI version or create a GitHub Release. A release happens only when a tag matching `v*` is pushed.

### One-time setup (first release only)

PyPI publishing uses [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC) — no API tokens are stored in this repo. Before the first release, the project owner must register the publisher once on PyPI:

1. Go to <https://pypi.org/manage/account/publishing/>.
2. Add a new pending publisher with:
   - PyPI project name: `pyccapt`
   - Owner: your GitHub user/org (e.g. `mmonajem`)
   - Repository: `pyccapt`
   - Workflow filename: `release.yml`
   - Environment name: `pypi`
3. In the GitHub repo settings, create an environment called `pypi` (Settings → Environments → New environment). Add reviewers if you want a manual approval gate.

You also need GitHub Pages enabled (Settings → Pages → Source: GitHub Actions) for the docs deploy, and the repo must allow GHCR package writes (Settings → Actions → General → Workflow permissions → Read and write).

### Release runbook

1. **Make sure `main` is green.** Check the latest [tests](https://github.com/mmonajem/pyccapt/actions/workflows/tests.yml), [lint](https://github.com/mmonajem/pyccapt/actions/workflows/lint.yml), and [docs](https://github.com/mmonajem/pyccapt/actions/workflows/docs.yml) runs. Don't tag a red commit.

2. **Bump the version.** The single source of truth is `__version__` in [`pyccapt/__init__.py`](pyccapt/__init__.py); `setup.py` reads it from there. Use [SemVer](https://semver.org/): bump `MAJOR` for breaking changes, `MINOR` for new features, `PATCH` for bug fixes.

3. **Update `CHANGELOG.txt`** with the user-facing changes for this version.

4. **Commit and push the bump to `main`** through the normal PR flow:

   ```bash
   git checkout -b release/vX.Y.Z
   # edit pyccapt/__init__.py and CHANGELOG.txt
   git commit -am "Release vX.Y.Z"
   git push origin release/vX.Y.Z
   # open and merge the PR, then:
   git checkout main
   git pull
   ```

5. **Tag the merge commit and push the tag.** The tag must start with `v` and match the version exactly:

   ```bash
   git tag -a vX.Y.Z -m "PyCCAPT vX.Y.Z"
   git push origin vX.Y.Z
   ```

6. **Watch the workflows.** The tag push triggers:

   - [`release.yml`](.github/workflows/release.yml) — builds sdist + wheel, publishes to PyPI via Trusted Publishing, creates a GitHub Release with auto-generated notes and the dist files attached.
   - [`docker.yml`](.github/workflows/docker.yml) — pushes `ghcr.io/mmonajem/pyccapt:X.Y.Z`, `:X.Y`, `:X`, and `:latest`.
   - [`docs.yml`](.github/workflows/docs.yml) — rebuilds and deploys the Sphinx site.

7. **Verify.** Confirm the new version is live:

   ```bash
   pip install --upgrade pyccapt
   python -c "import pyccapt; print(pyccapt.__version__)"
   docker pull ghcr.io/mmonajem/pyccapt:latest
   ```

   Check the [PyPI project page](https://pypi.org/project/pyccapt/), the [GitHub Releases page](https://github.com/mmonajem/pyccapt/releases), and the [GHCR package page](https://github.com/mmonajem/pyccapt/pkgs/container/pyccapt).

### Recovering from a failed release

PyPI versions are **immutable** — once `X.Y.Z` is uploaded you cannot reupload the same version, even after deleting it. If the release workflow fails:

- **Failed before PyPI upload** (build/test step): fix the issue on `main`, then delete and recreate the tag at the new commit.

  ```bash
  git tag -d vX.Y.Z
  git push origin :refs/tags/vX.Y.Z
  # fix, merge to main, then re-tag:
  git tag -a vX.Y.Z -m "PyCCAPT vX.Y.Z"
  git push origin vX.Y.Z
  ```

- **Failed after PyPI upload** (e.g. GitHub Release step): yank the broken release on PyPI (Manage project → Releases → Options → Yank), bump to `X.Y.Z+1`, and run the runbook again. Don't try to reuse `X.Y.Z`.

- **Manual re-trigger:** `release.yml` also accepts `workflow_dispatch`, so you can re-run it from the Actions tab without retagging — useful if only the GitHub Release or Docker step failed transiently.

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
