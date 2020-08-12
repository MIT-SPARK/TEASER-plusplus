# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import sphinx_rtd_theme
import subprocess, os

read_the_docs_build = os.environ.get('READTHEDOCS', None) == 'True'

# If building for readthedocs, generate doxygen xml files
if read_the_docs_build:
    subprocess.call('TEASERPP_SOURCE_DIR=../teaser doxygen', shell=True)

# -- Project information -----------------------------------------------------

project = 'TEASER++'
copyright = '2020, Heng Yang, Jingnan Shi, Luca Carlone'
author = 'Heng Yang, Jingnan Shi, Luca Carlone'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'breathe',
    'sphinx_rtd_theme',
              ]
source_suffix = ['.rst']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Breathe configuration --------------------------------------------------
breathe_projects = { "teaserpp": "./doxygen/xml" }
breathe_default_project = "teaserpp"
