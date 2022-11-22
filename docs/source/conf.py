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
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))


# -- Project information -----------------------------------------------------

project = 'HiveNAS'
copyright = '2022, Mohamed Shahawy'
author = 'Mohamed Shahawy'

# The full version, including alpha/beta/rc tags
release = '0.1.3'


# -- General configuration ---------------------------------------------------


add_module_names = False


html_title = "HiveNAS - Neural Architecture Search using Artificial Bee Colony Optimization"
html_short_title = "HiveNAS"
html_favicon = '_static/favicon.png'

# autoclass_content = 'both'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
	'sphinx.ext.autodoc',
	#'sphinx.ext.autosummary',
	'sphinx.ext.coverage',
	'sphinx.ext.napoleon',
	'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    'myst_parser',
	# 'sphinx.ext.graphviz',
	# 'sphinx.ext.inheritance_diagram'
]

source_suffix = ['.rst', '.md']

pygments_style = 'sphinx'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
apidoc_template_dir = 'source/_templates/autoapi'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'api/src.rst', 'api/modules.rst']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo' # 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom CSS
html_css_files = ['../../../source/_static/css/custom.css']

# Framework logo
# html_logo = "_static/hivenas_logo.png"

# Bypass rtd theme preset navigation limit
# html_theme_options = {'navigation_depth': 6}
html_theme_options = {
    'sidebar_hide_name': True,
    'light_logo': 'hivenas_logo.svg',
    'dark_logo': 'hivenas_logo_light.svg'
}

html_show_sphinx = False
html_show_furo = False

# -- Options for AutoAPI -----------------------------------------------------

autoapi_options = [
	'members', 
	'undoc-members', 
	'private-members', 
	'show-inheritance', 
	'show-module-summary', 
	'special-members', 
	'imported-members',
	'titlesonly'
]

toc_object_entries_show_parents = 'hide'

