"""Sphinx configuration for the Kikuchi Band Analyzer documentation."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

project = "Kikuchi Band Analyzer"
author = "Kikuchi Band Analyzer contributors"
copyright = "2026, Kikuchi Band Analyzer contributors"

version_path = REPO_ROOT / "VERSION"
release = version_path.read_text(encoding="utf-8").strip() if version_path.exists() else "0.0.0"
version = release

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
}
master_doc = "index"
exclude_patterns = ["_build", "README.md", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
toc_object_entries = True
toc_object_entries_show_parents = "hide"
suppress_warnings = ["ref.python"]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
]
nb_execution_mode = "off"
nb_execution_timeout = 120

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

html_theme = "furo"
html_title = "Kikuchi Band Analyzer"
html_static_path = ["_static"]
html_css_files = ["architecture.css"]
html_theme_options = {}

nitpicky = False
os.environ.setdefault("MPLBACKEND", "Agg")
