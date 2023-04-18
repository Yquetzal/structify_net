# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'structify_net'
copyright = '2023, Rémy Cazabet'
author = 'Rémy Cazabet'
release = '2'

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../structify_net'))

from mock import Mock as MagicMock

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()
    
MOCK_MODULES = ['numpy', 'scipy', 'matplotlib', 'matplotlib.pyplot', 'scipy.interpolate',"seaborn","pandas","scipy.special","scipy.spatial","tqdm","tqdm.auto","networkx","math","numbers","random"]

sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc','sphinx.ext.autosummary',"sphinx_rtd_theme"]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

autosummary_generate = True
