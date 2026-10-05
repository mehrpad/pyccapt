# Building the Documentation

PyCCAPT documentation is built with Sphinx.

The user guide for TOML experiment queues is in
[Experiment Plans](experiment_plans.rst), sourced from
`pyccapt/control/EXPERIMENT_PLANS.md`. Its downloadable example is
[`experiment_plan.example.toml`](../pyccapt/files/experiment_plan.example.toml).

The shared control-state contract, resource owners and diagnosis instructions
are in [Control State Mechanisms](state_mechanisms.rst), sourced from
`pyccapt/control/STATE_MECHANISMS.md`.

## Prerequisites

From the repository root, install documentation dependencies:

```bash
pip install -r docs/requirements.txt
```
cd docs
pip install -r requirements.txt
```
## Create rst files

If there is no conf.py file, create one with `sphinx-quickstart`.

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
