# Makefile for the INTEGRATE Sphinx documentation.
#
# The gallery is built from the plain .py examples in ../examples/gallery by
# sphinx-gallery, which EXECUTES them and embeds the resulting figures. Which
# examples run is chosen per invocation, so the normal edit/preview cycle never
# pays for the slow ones:
#
#   make html                      no execution at all -- fast structural check
#   make gallery                   the cheap, self-contained examples
#   make gallery TIER=all          adds the slow but self-contained ones
#   make gallery FILE=x.py         one specific example
#   make deploy                    publish _build/html to the gh-pages branch
#
# sphinx-gallery caches on an md5 of each example, so re-running `make gallery`
# only re-executes what actually changed. That cache lives in auto_examples/,
# which is why `make clean` does NOT touch it -- use `make clean-gallery`.

SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = INTEGRATE
SOURCEDIR     = .
BUILDDIR      = _build
GALLERYDIR    = auto_examples

TIER ?= cheap
FILE ?=

.PHONY: help html gallery clean clean-gallery deploy

help:
	@echo "html           build without executing any example"
	@echo "gallery        build, executing the cheap example tier"
	@echo "gallery TIER=all    build, executing every self-contained example"
	@echo "gallery FILE=x.py   build, executing only x.py"
	@echo "clean          remove $(BUILDDIR) (keeps the gallery cache)"
	@echo "clean-gallery  remove $(GALLERYDIR), forcing every example to re-run"
	@echo "deploy         publish $(BUILDDIR)/html to the gh-pages branch"

html:
	GALLERY_TIER=none $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

gallery:
	GALLERY_TIER=$(TIER) GALLERY_FILE=$(FILE) \
		$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Deliberately does not remove $(GALLERYDIR): that would discard the md5 cache
# and force every example to be executed again from scratch.
clean:
	rm -rf "$(BUILDDIR)"

clean-gallery:
	rm -rf "$(GALLERYDIR)" sg_execution_times.rst

deploy:
	@test -d "$(BUILDDIR)/html" || { echo "nothing built -- run 'make gallery' first"; exit 1; }
	ghp-import -n -p -f -o "$(BUILDDIR)/html"
