# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import datetime
import os
import sys
import warnings
from importlib.metadata import version as get_version

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.deprecation import RemovedInSphinx10Warning
from sphinx.util.docutils import SphinxRole

sys.path.insert(0, os.path.abspath("."))
from config_reference import setup as config_reference_setup
from jaqmc_sphinx_utils import TypehintsFormatter

warnings.filterwarnings("ignore", category=RemovedInSphinx10Warning)

project = "JaQMC"
copyright = f"2025–{datetime.datetime.now().year}, ByteDance Seed"
author = "ByteDance Seed"
release = get_version("jaqmc")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "protocol_autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.linkcode",
    "sphinx.ext.graphviz",
    "myst_nb",
    "sphinx_design",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_llm.txt",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    ".jupyter_cache",
    "jupyter_execute",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_copy_source = False
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_favicon = "_static/jaqmc-light.svg"
html_css_files = ["custom.css"]
html_logo = "_static/jaqmc-light-large.svg"
html_baseurl = "https://bytedance.github.io/jaqmc/"
html_theme_options = {
    "repository_url": "https://github.com/bytedance/jaqmc",
    "use_source_button": True,
    "use_repository_button": True,
    "use_download_button": False,
    "use_edit_page_button": True,
    "use_issues_button": True,
}

# -- Options for LaTeX output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-latex-output

latex_engine = "xelatex"
latex_logo = "_static/jaqmc-light-large.pdf"

# -- Extensions configurations ------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
    "upath": ("https://universal-pathlib.readthedocs.io/en/latest/", None),
    "pyscf": ("https://pyscf.org/", None),
    "optax": ("https://optax.readthedocs.io/en/latest/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "flax": ("https://flax-linen.readthedocs.io/en/latest/", None),
}

myst_enable_extensions = ["colon_fence", "dollarmath", "fieldlist"]
myst_heading_anchors = 4
myst_dmath_allow_space = False
myst_dmath_double_inline = True
napoleon_custom_sections = [("Type Parameters", "params_style")]
nitpick_ignore = [
    ("py:class", "DataT"),
    ("py:class", "OutputT"),
]


def linkcode_resolve(domain: str, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None
    filename = info["module"].replace(".", "/")
    return f"https://github.com/bytedance/jaqmc/blob/main/src/{filename}.py"


typehints_defaults = "comma"
always_use_bars_union = True
typehints_formatter = TypehintsFormatter()

nb_execution_mode = "cache"
nb_execution_in_temp = True
nb_execution_timeout = 360

graphviz_output_format = "svg"

llms_txt_suffix_mode = "replace"


class GitHubSourceRole(SphinxRole):
    """Inline role that links a source path to its GitHub location.

    Usage in MyST: ``{ghsrc}`src/jaqmc/app/hydrogen_atom.py` ``

    Renders as ``<a href="https://github.com/..."><code>path</code></a>``.
    """

    base_url = "https://github.com/bytedance/jaqmc/blob/main/"

    def run(self):
        url = self.base_url + self.text
        code_node = nodes.literal("", self.text)
        ref_node = nodes.reference("", "", code_node, refuri=url)
        return [ref_node], []


def setup(app: Sphinx):
    app.add_role("ghsrc", GitHubSourceRole())
    config_reference_setup(app)

    def configure_for_builder(app):
        if app.builder.name == "markdown":
            app.config.nb_execution_mode = "off"

    app.connect("builder-inited", configure_for_builder)
