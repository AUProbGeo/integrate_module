# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# Add the parent directory (where the integrate package is located)
sys.path.insert(0, os.path.abspath('..'))

# Try to import integrate to ensure it's available
try:
    import integrate
    print(f"Successfully imported integrate from {integrate.__file__}")
except ImportError as e:
    print(f"Failed to import integrate: {e}")

# Set up the environment for Sphinx autodoc
autodoc_mock_imports = []

# Mock problematic imports if needed
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

project = 'INTEGRATE'
copyright = '2023,2024,2025 INTEGRATE WORKING GROUP'
author = 'INTEGRATE WORKING GROUP'

# Dynamically get version from pyproject.toml
try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Fallback for older Python
    except ImportError:
        tomllib = None

if tomllib:
    pyproject_path = os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml')
    with open(pyproject_path, 'rb') as f:
        pyproject_data = tomllib.load(f)
    version = pyproject_data['project']['version']
    release = version
else:
    # Fallback to importlib.metadata if tomllib not available
    try:
        from importlib.metadata import version as get_version
        version = get_version('integrate_module')
        release = version
    except Exception:
        version = '0.31'  # Fallback version
        release = version
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',

]
autosummary_generate = True

# -- Gallery -----------------------------------------------------------------
# Which examples are *executed* is chosen per run; everything in the gallery is
# always rendered, executed or not. See doc/Makefile and DOC_REFACTOR.md.
#
#   make html                   no execution (GALLERY_TIER=none)
#   make gallery                the cheap, self-contained examples
#   make gallery TIER=all       adds the slow but still self-contained ones
#   make gallery FILE=x.py      one specific example
import re

from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey

_GALLERY_TIER = os.environ.get('GALLERY_TIER', 'cheap')
_GALLERY_FILE = os.environ.get('GALLERY_FILE', '')

# Self-contained (data fetched via ig.get_case_data) and small enough to run
# routinely.
_TIER_CHEAP = (r'integrate_(getting_started.*|synthetic_case|linear_logspace'
               r'|dual_data|priors|merge_prior|query)\.py')
# Also self-contained, but slow: N up to 2e6, or a full timing sweep.
_TIER_ALL = (r'integrate_(getting_started.*|synthetic_case|linear_logspace'
             r'|dual_data|priors|merge_prior|query|gaussian_noise|esbjerg'
             r'|merge_data|timing_example)\.py')

if _GALLERY_FILE:
    _filename_pattern = re.escape(_GALLERY_FILE) + r'$'
elif _GALLERY_TIER == 'none':
    _filename_pattern = r'(?!)'          # matches nothing
elif _GALLERY_TIER == 'all':
    _filename_pattern = _TIER_ALL
else:
    _filename_pattern = _TIER_CHEAP

sphinx_gallery_conf = {
    'examples_dirs': '../examples/gallery',
    'gallery_dirs': 'auto_examples',
    'filename_pattern': _filename_pattern,
    # A helper imported by the rawmaterial examples, not an example itself.
    'ignore_pattern': r'integrate_rawmaterial_utils\.py',
    'subsection_order': ExplicitOrder([
        '../examples/gallery/10_getting_started',
        '../examples/gallery/20_workflow',
        '../examples/gallery/30_data',
        '../examples/gallery/40_noise',
        '../examples/gallery/50_hypothesis',
        '../examples/gallery/60_query',
        '../examples/gallery/70_synthetic',
        '../examples/gallery/80_plotting',
        '../examples/gallery/85_rawmaterial',
        '../examples/gallery/90_other',
    ]),
    'within_subsection_order': FileNameSortKey,
    # Gives each section its own index page, so the sidebar nests as
    # Example Gallery > section > example. This requires every subsection
    # GALLERY_HEADER.rst to underline its title with '-', one level below the
    # root gallery's '='; using '=' there makes the sections siblings of the
    # gallery title instead, and the sidebar then lists them twice.
    'nested_sections': True,
    'remove_config_comments': True,
    # An example that fails must not take the whole doc build down with it.
    'abort_on_example_error': False,
    'download_all_examples': False,
}

# Napoleon settings (for allowing using Google and NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# sphinx_gallery_conf holds class and function objects (the sort keys), which
# Sphinx cannot pickle into its environment cache. Harmless, but noisy.
suppress_warnings = ['config.cache']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'README.md', 'tools']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'sphinx_rtd_theme'
html_theme = 'furo'
#html_static_path = ['_static']

# GitHub Pages compatibility
html_baseurl = 'https://cultpenguin.github.io/integrate_module/'
html_copy_source = False
html_show_sourcelink = False

def setup(app):
    """Custom setup function to add .nojekyll file to output."""
    import os
    
    def add_nojekyll_file(app, exception):
        if exception is None:  # Build was successful
            nojekyll_path = os.path.join(app.outdir, '.nojekyll')
            with open(nojekyll_path, 'w') as f:
                f.write('')
    
    app.connect('build-finished', add_nojekyll_file)
