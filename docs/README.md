# How to build the documentation

The documentation is built with `Sphinx` and a set of extensions.

## Install Requirements

```bash
cd docs
pip install -r requirements.txt
```

## Create RST Files

If there is no conf.py file, create one with `sphinx-quickstart`.

Then generate the API `.rst` files with `sphinx-apidoc`:

```bash
sphinx-apidoc -o .  ../pyccapt
```

## Build

```bash
make clean html
make html
```

The generated documentation entrypoint is:

```text
./_build/html/index.html
```

