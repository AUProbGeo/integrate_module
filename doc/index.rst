.. INTEGRATE documentation master file, created by
   sphinx-quickstart on Thu Jan 24 09:18:42 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


INTEGRATE: Fast Probabilistic inversion of EM data using informed prior models
------------------------------------------------------------------------------
Last updated: |today| (version |version|).

INTEGRATE provides a python module and methods for fast probabilistic inversion of local information (e.g. electromagnetic data (EM), well log data, ...) using informed prior models. 

The aim is to provide methods for the following tasks, that together represent a probabilistic workflow: 

Prior modeling
   Tools will be developed to quantify (through forward simulation) as much information as possible about the subsurface, such as the expected distribution of lithological layers and a model that links resistivity to lithology.
   See, for example, [MADSEN2023]_ and [GEOPRIOR1D]_.
  
Forward modeling
   For each type of data considered a forward model must be available. 

   For EM type data use GA-AEM (https://github.com/GeoscienceAustralia/ga-aem), based on [FALK2025]_.

  
Probabilistic Inversion
   An implementation of the 1D probabilistic localized inversion using the **Localized Rejection Sampler** [HANSEN2021]_ and **Machine Learning** [HANSENFINLAY2022]_.

   Features
      - Fast probabilistic inversion with informed prior models
      - Multiple Data Types
      - Multiple Forward Models
      - Joint inversion 


Analysis
   Tools for visual illustrations of the results will be developed, such as 1D, 2D cross-sections, 3D rendering, as well as uncertainty quantification.
   

Getting started
===============
Refer to the documentation in :doc:`install` for installation instructions.

Examples of using the module can be found in the :doc:`notebooks`.



The INTEGRATE project
=====================
The project is developed as part of the INTEGRATE project, where the goal is to develop probabilistic support tools that allow quantifying the potential for finding raw material resources close to where it is to be utilized.

For more information, please visit the INTEGRATE website (https://integrate.nu/).

Source Code
===========

   The latest stable code is available on GitHub at
   https://github.com/AUProbGeo/integrate_module


License (MIT)
=============

MIT License

Copyright (c) 2023-2025 Thomas Mejer Hansen and INTEGRATE Working Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The manual
----------

.. toctree::
   :maxdepth: 3

   install
   gettingstarted
   format
   format_wells
   format_query
   workflow
   notebooks
   contributions
   references
   modules


   
