# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'QSPRpred'
# copyright = '2022, Helle van den Maagdenberg, Linde Schoenmaker, Martin Sicho'
author = 'Helle van den Maagdenberg, Linde Schoenmaker, Martin Sicho'

# The full version, including alpha/beta/rc tags
from importlib.metadata import version
release = version('qsprpred')
version = f'v{release}'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
    "sphinx_design",
    "sphinx_design_elements",
]

intersphinx_mapping = {'python': ('https://docs.python.org/3.10', None)}
intersphinx_mapping = {'python': ('https://docs.python.org/3.9', None)}
autoclass_content = "both"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "special-members": False,
    "inherited-members": True,
    "private-members": False,
    "show-inheritance": True
}

# napoleon settings
# https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_ivar = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, maps document names to template names.
html_sidebars = { '**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'], }

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If false, no module index is generated.
html_domain_indices = True

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = 'any'

autoclass_content = 'both'
