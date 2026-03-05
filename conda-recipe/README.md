# Conda Packaging

This recipe builds a conda package named `pyccapt`.

## Build locally

```bash
conda install -c conda-forge conda-build
conda build conda-recipe
```

## Install locally built package

```bash
conda install --use-local pyccapt
```
