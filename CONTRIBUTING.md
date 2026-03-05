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

## Code of Conduct

Be respectful and constructive in all project interactions.
