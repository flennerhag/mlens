# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = mlens
SOURCEDIR     = 
BUILDDIR      = build


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Install necessary modules
# TODO: should we load these as mockups?
install:
			pip install -U sphinx-gallery numpydoc matplotlib seaborn

# Update API docs
api:
	rm -rf module
	sphinx-apidoc -ef -o module ../../mlens/   ../../mlens/externals

# Render landing page
boostrap:
	cp -rf landing/ ../
	cd .. | gulp

# Populate parent dir
arrange:
	cp -r build/html ../docs
	rm -r build
	-rm -rf tutorials
	cp CNAME ../

clean-build:
	rm -rf tutorials

# Make docs
docs:
	make api
	make html
	make arrange
	make boostrap
	make clean-build

html-noplot:
	$(SPHINXBUILD) -D plot_gallery=0 -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

# Clear all in docs except source
clean-cache:
	make clean
	rm -rf tutorials
	bash clean-html.sh
