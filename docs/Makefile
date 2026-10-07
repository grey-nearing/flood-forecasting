# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAPIDOC  ?= sphinx-apidoc
SOURCEDIR     = source
BUILDDIR      = build
PACKAGEDIR    = ../model
PACKAGEDIR2   = ../return_periods

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile apidoc clean-api

# Target to auto-generate API docs
# -f: Force overwriting of any existing generated files.
# -o: Directory to place the output files.
# -e: Put documentation for each module on its own page.
# -T: Do not generate a table of contents file (managed manually).
apidoc:
	@echo "Generating API documentation..."
	@$(SPHINXAPIDOC) -f -e -T -o "$(SOURCEDIR)/api" "$(PACKAGEDIR)" "$(PACKAGEDIR)/tests" "$(PACKAGEDIR)/tutorial"
	@$(SPHINXAPIDOC) -f -e -T -o "$(SOURCEDIR)/api" "$(PACKAGEDIR2)" "$(PACKAGEDIR2)/tests" "$(PACKAGEDIR2)/tools"
	@printf "Modules\n=======\n\n.. toctree::\n   :maxdepth: 4\n\n   model\n   catchment_delineation\n   static_extractor\n   gridded_archive_builders\n   timeseries_extractors\n   return_periods\n   benchmarks\n" > "$(SOURCEDIR)/api/modules.rst"

# Target to clean API docs
clean-api:
	@echo "Cleaning API documentation..."
	@rm -f "$(SOURCEDIR)/api/model"*.rst "$(SOURCEDIR)/api/return_periods"*.rst "$(SOURCEDIR)/api/modules.rst"

# Explicitly define html target to run apidoc first
html: apidoc
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Clean build directory and API docs
clean: clean-api
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)