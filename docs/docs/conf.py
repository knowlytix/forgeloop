import os
import sys

# This conf.py is the Sphinx *config directory*; the *source directory* is the
# parent ``forgeloop-docs/`` folder, so the authored documentation lives in the
# self-describing top-level folders:
#   ../2-user-guide      user guide (.rst)
#   ../3-api-reference   API reference (.rst, autodoc)
#   ../auto_examples     sphinx-gallery output (generated)
# Build with:  make html   (from this directory)
# or directly: sphinx-build -b html -c . .. ../docs/_build/html

_HERE = os.path.dirname(os.path.abspath(__file__))              # forgeloop-docs/docs
_SOURCE = os.path.abspath(os.path.join(_HERE, ".."))            # forgeloop-docs (Sphinx source dir)
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))   # agent-tutorial-private (holds the forgeloop package)

sys.path.insert(0, _REPO_ROOT)

import forgeloop  # noqa: E402  (must follow the sys.path setup above)

extensions = [
    'sphinx.ext.napoleon',   # forgeloop docstrings mix Google and NumPy styles
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'myst_nb',               # render the book chapter notebooks (.ipynb) in-theme
]

# The Examples section renders the actual chapter notebooks from the three books
# (copied under 4-notebook-examples/). Execution is OFF: myst-nb renders each
# notebook's markdown, code and whatever outputs are already saved, without
# running it, so no GPU or licensed backend is needed at build time. To populate
# outputs, execute the notebooks offline and rebuild.
nb_execution_mode = 'off'
nb_execution_allow_errors = True
# Notebooks repeat short headings across chapters; do not fail on duplicate
# auto-generated section anchors.
suppress_warnings = ['myst.header', 'myst.xref_missing', 'mystnb.unknown_mime_type']
myst_heading_anchors = 2

# The API reference documents both forgeloop and the knowlytix GMS backend it
# wraps. Documenting knowlytix requires importing it for real, so we mock a
# heavy dependency only when it is genuinely absent from the build environment.
# In the licensed spark-venv all of these are present, so nothing is mocked and
# every signature renders from the real objects. In a plain environment the
# missing ones are mocked so the forgeloop pages still build (the knowlytix
# pages then require the licensed package). numpy, pydantic and pyyaml are light
# real dependencies and are never mocked.
import importlib.util as _ilu  # noqa: E402  (must follow the sys.path setup above)

_CANDIDATE_MOCKS = [
    'torch', 'knowlytix', 'transformers', 'peft',
    'anthropic', 'huggingface_hub', 'pandas',
]
autodoc_mock_imports = [m for m in _CANDIDATE_MOCKS if _ilu.find_spec(m) is None]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

source_suffix = '.rst'
# Master doc lives at the source-directory root so the landing page renders to
# the site root index.html.
master_doc = 'index'
project = u'forgeloop'
author = u'Knowlytix'
copyright = u'Knowlytix'
version = forgeloop.__version__
release = forgeloop.__version__

# Source dir holds only authored docs plus the generated gallery; exclude build
# scratch and environments.
exclude_patterns = [
    'docs/_build',
    '**/_build/**',
    # Gallery SOURCE scripts and their headers live under the config dir; they
    # are read directly by sphinx-gallery via examples_dirs, so they must not
    # also be picked up as stray documents in the Sphinx source tree.
    'docs/auto_galleries',
    '.venv',
    '**/.venv/**',
    '*.egg-info',
    '**/.ipynb_checkpoints',
    '**/__pycache__',
]

pygments_style = 'sphinx'
autoclass_content = 'both'
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
# Render a docstring "Attributes:" section as inline :ivar: fields rather than
# separate attribute objects, so it does not collide with the attributes autodoc
# already documents via :members: (which would duplicate every dataclass field).
napoleon_use_ivar = True
autodoc_inherit_docstrings = True
autodoc_member_order = 'groupwise'
# Member documentation is requested per directive (each autoclass carries
# :members:), so no global members default is set here — a global default would
# override the explicit per-page symbol lists and double-document re-exported
# classes. Undocumented members (e.g. dataclass fields) are still shown because
# each autoclass sets :undoc-members: via the generator.
autodoc_default_options = {
    'show-inheritance': True,
}

html_theme = 'sphinx_rtd_theme'
html_show_sourcelink = True
# Show the full nested navigation tree in the sidebar (book section -> subpackage
# -> module) rather than collapsing to a flat list of the current branch.
html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 4,
    'titles_only': False,
}
