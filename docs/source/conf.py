# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import inspect
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../HierarchBayesParcel'))
import HierarchBayesParcel

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'HierarchBayesParcel'
copyright = '2024, Da Zhi, Joern Diedrichsen'
author = 'Da Zhi, Joern Diedrichsen'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.napoleon',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosectionlabel',
              'sphinx.ext.mathjax',
              'sphinx.ext.intersphinx',
              'sphinx.ext.doctest',
              'nbsphinx',
              'sphinx.ext.viewcode']

napoleon_custom_sections = [('Returns', 'params_style')]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Source Code available via sphinx
extensions.append('sphinx.ext.linkcode')

def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None

    obj = info['module']
    submodule = info['module']
    fullname = info['fullname']
    try:
        # Import the module and get the object
        mod = __import__(submodule, fromlist=[''])
        obj = mod
        for part in fullname.split('.'):
            obj = getattr(obj, part)
        
        # Get the source file and line number
        file = inspect.getsourcefile(obj)
        source, lineno = inspect.getsourcelines(obj)
        file = os.path.relpath(file, start=os.path.dirname(HierarchBayesParcel.__file__))
        return f"https://github.com/DiedrichsenLab/HierarchBayesParcel/blob/main/HierarchBayesParcel/{file}#L{lineno}"
    except Exception as e:
        print(f"Error in linkcode_resolve: {e}")
        return None
