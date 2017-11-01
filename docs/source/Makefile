# Minimal makefile for Sphinx documentation

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
install:
			pip install -U sphinx-gallery numpydoc matplotlib seaborn

# Populate parent dir
arange:
	cp -r build/html/ ../
	rm -r build

# Make docs
docs:
	make html
	make arange

# Clear all in docs except source
clean-cache:
	make clean
	bash clean-html.sh
