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

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
