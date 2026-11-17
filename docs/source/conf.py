# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

# --- sphinx 9 compatibility shim -------------------------------------------
# enum_tools.autoenum imports `logger` from sphinx.ext.autodoc, which sphinx 9
# removed. Inject the equivalent logger until enum-tools ships sphinx 9 support;
# the hasattr guard makes this inert once it does.
import sphinx.ext.autodoc
import sphinx.util.logging
if not hasattr(sphinx.ext.autodoc, 'logger'):
    sphinx.ext.autodoc.logger = sphinx.util.logging.getLogger('sphinx.ext.autodoc')
# --- end shim ---------------------------------------------------------------


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pysndlib'
copyright = '2023-2024, Todd Ingalls'
author = 'Todd Ingalls'
# version is derived from the installed package (git tag via setuptools-scm);
# there is no hardcoded release string to keep in sync
import pysndlib
release = pysndlib.__version__
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'enum_tools.autoenum',
    'sphinx.ext.autosectionlabel'
]

templates_path = ['_templates']
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']
autodoc_typehints = "description"
add_module_names = False
autodoc_type_aliases = {'cclm.mus_long_t' : 'int', 'cython.double' : 'float', 'mus_long_t' : 'int' }


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
