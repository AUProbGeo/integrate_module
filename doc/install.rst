============
Installation
============

INTEGRATE Python Module
========================

This repository contains the INTEGRATE Python module for localized probabilistic data integration in geophysics.

Assuming you already have Python 3.10+ installed:

::
    
    pip install integrate_module

On Windows, this will also install the Python wrapper for GA-AEM (1D EM forward modeling - GPL v2 code): `ga-aem-forward-win <https://pypi.org/project/ga-aem-forward-win/>`_.

On Linux/macOS, you will need to install GA-AEM manually.

Using uv (recommended, from PyPI)
==================================

`uv <https://github.com/astral-sh/uv>`_ is a fast Python package manager. Install it first if needed:

::

    # Install uv (Linux/macOS)
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # or: pip install uv

    # Create virtual environment in .venv/ inside the module root
    cd path/to/integrate_module
    uv venv .venv --python 3.11

    # Activate
    source .venv/bin/activate      # Linux/macOS
    .venv\Scripts\activate         # Windows

    # Install integrate module
    uv pip install integrate_module

Using uv (from source)
=======================

::

    # Install uv (Linux/macOS)
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # or: pip install uv

    # Create .venv and install all dependencies in one step (recommended for development)
    cd path/to/integrate_module
    uv sync

    # Activate
    source .venv/bin/activate      # Linux/macOS
    .venv\Scripts\activate         # Windows

Using pip + venv (from PyPI, on Ubuntu)
========================================

::

    # Install python3-venv
    sudo apt install python3-venv

    # Create virtual environment in .venv/ inside the module root
    cd path/to/integrate_module
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip

    # Install integrate module
    pip install integrate_module

Using pip + venv (from source, on Ubuntu)
==========================================

::

    # Install python3-venv
    sudo apt install python3-venv

    # Create virtual environment in .venv/ inside the module root
    cd path/to/integrate_module
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip

    # Install integrate module from source
    pip install -e .

Installing documentation dependencies
======================================

To also install the packages needed to build the Sphinx documentation, use the ``docs`` extra:

::

    # With uv (from source)
    uv sync --extra docs

    # With uv pip (from source)
    uv pip install -e ".[docs]"

    # With pip (from source)
    pip install -e ".[docs]"

Using Conda + pip (from PyPI)
==============================

Create a Conda environment (called integrate) and install the required modules:

::

    conda create --name integrate python=3.11
    conda activate integrate 
    conda install -c conda-forge pip
    pip install integrate_module
    
Using Conda + pip (from source)
================================

Create a Conda environment (called integrate) and install integrate_module from source using pip:

::

    # Download source code, and unzip (if you use a zipped archive)
    cd path/to/integrate_module
    conda create --name integrate python=3.11
    conda activate integrate
    conda install -c conda-forge pip
    pip install -e .

    
GA-AEM
======

In order to use GA-AEM for forward EM modeling, the 'gatdaem1d' Python module must be installed. Follow instructions at `https://github.com/GeoscienceAustralia/ga-aem <https://github.com/GeoscienceAustralia/ga-aem>`_ or use the information below.

PyPI package for Windows
-------------------------

On Windows, the `ga-aem-forward-win <https://pypi.org/project/ga-aem-forward-win/>`_ package will be automatically installed, providing access to the GA-AEM forward code. It can be installed manually using:

::

    pip install ga-aem-forward-win

Pre-compiled Python module for Windows
---------------------------------------

1. Download the pre-compiled version of GA-AEM for Windows from the latest release: https://github.com/GeoscienceAustralia/ga-aem/releases (GA-AEM.zip)

2. Download precompiled FFTW3 Windows DLLs from https://www.fftw.org/install/windows.html (fftw-3.3.5-dll64.zip)

3. Extract both archives:
   - ``unzip GA-AEM.zip`` to get GA-AEM
   - ``unzip fftw-3.3.5-dll64.zip`` to get fftw-3.3.5-dll64

4. Copy FFTW3 DLLs to GA-AEM Python directory:

::

    cp fftw-3.3.5-dll64/*.dll GA-AEM/python/gatdaem1d/

5. Install the Python gatdaem1d module:

::

    cd GA-AEM/python/
    pip install -e .

    # Test the installation
    cd examples
    python skytem_example.py

Compile GA-AEM Python module on Debian/Ubuntu/Linux
---------------------------------------------------

A script that downloads and installs GA-AEM is located in ``scripts/cmake_build_script_DebianUbuntu_gatdaem1d.sh``. This script has been tested and confirmed to work on both Debian and Ubuntu distributions. Be sure to use the appropriate Python environment and then run:

::

    sh scripts/cmake_build_script_DebianUbuntu_gatdaem1d.sh
    
Compile GA-AEM Python module on macOS/Homebrew
-----------------------------------------------

First install Homebrew, then run:

::

    sh ./scripts/cmake_build_script_homebrew_gatdaem1d.sh
    cd ga-aem/install-homebrew/python
    pip install .


Development
===========

The ``main`` branch is the most stable, with less frequent updates but larger changes.

The ``develop`` branch contains the current development code and may be updated frequently. Some functions and examples may be broken.

An extra set of tests and examples are located in the ``experimental`` sub-branch `https://github.com/cultpenguin/integrate_module_experimental/ <https://github.com/cultpenguin/integrate_module_experimental/>`_.
Please ask the developers for access to this branch if needed. To clone the main repository with the experimental branch, use:

::

    git clone --recurse-submodules git@github.com:AUProbGeo/integrate_module.git

You may need to run the following command to update the submodules:

::

    cd experimental
    git submodule update --init --recursive
