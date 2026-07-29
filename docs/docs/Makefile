# Minimal makefile for Sphinx documentation
#
# The config directory is this folder; the source directory is the parent
# forgeloop-docs/ folder (one level up) so documentation content can live in the
# top-level 2-user-guide / 3-api-reference folders and the generated gallery.

SPHINXOPTS    = -c .
SPHINXBUILD   = sphinx-build
SOURCEDIR     = ..
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# The notebook gallery under ../4-notebook-examples/ is assembled from the
# books rather than committed. 47 user-guide pages :doc:-link into it, so it
# must exist before sphinx-build runs or those links fail as unknown documents.
gallery:
	@python3 ../../scripts/build_forgeloop.py --docs

.PHONY: help gallery Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile gallery
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
