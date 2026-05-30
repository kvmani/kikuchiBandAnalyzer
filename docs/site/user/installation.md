# Installation

Use Python 3.10 or newer. A virtual environment is strongly recommended because
the scientific stack includes compiled packages.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev,docs]
```

For a minimal runtime installation, omit the optional extras:

```powershell
python -m pip install -e .
```

Validate the environment with:

```powershell
python -m pytest -q
python -m sphinx -b html docs/site docs/site/_build/html
```

The package installs the following console entry points:

`kikuchi-band-width`
: Config-driven scan processing and export.

`kikuchi-band-detector`
: Single-pattern analysis, including interactive debug visualization.

`ebsd-compare`
: EBSD comparison GUI entry point.

`oh5-to-ang-exporter`
: Export selected `.oh5` fields into `.ang`-style text data.
