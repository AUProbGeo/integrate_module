

==============================================================================
INTEGRATE: Fast Probabilistic inversion of EM data using informed prior models
==============================================================================
Last updated: |today| (version |version|).

INTEGRATE provides a python module and methods for fast probabilistic inversion of local information (e.g. electromagnetic data (EM), well log data, ...) using informed prior models. 

The aim is to provide methods for the following tasks, that together represent a probabilistic workflow: 

Prior modeling
   Tools will be developed to quantify (through forward simulation) as much information as possible about the subsurface, such as the expected distribution of lithological layers and a model that links resistivity to lithology.
   See, for example, [MADSEN2023]_, [NORGAARD2026]_ and [GEOPRIOR1D]_.
  
Forward modeling
   For each type of data considered a forward model must be available. 

   For EM type data, forward modelling uses GA-AEM (https://github.com/GeoscienceAustralia/ga-aem), based on [FALK2025]_, or the anemone (PyTorch) and SimPEG backends. See :doc:`prior_data`.

  
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

Examples of using the module can be found in the :doc:`example gallery <auto_examples/index>`.

Start with :doc:`the getting started examples <auto_examples/10_getting_started/index>`.

The gallery is organised in sections:

* :doc:`Getting started <auto_examples/10_getting_started/index>`
* :doc:`The complete workflow <auto_examples/20_workflow/index>`
* :doc:`Data <auto_examples/30_data/index>`
* :doc:`Noise <auto_examples/40_noise/index>`
* :doc:`Forward models <auto_examples/45_forward/index>`
* :doc:`Hypothesis testing <auto_examples/50_hypothesis/index>`
* :doc:`Querying the posterior <auto_examples/60_query/index>`
* :doc:`Synthetic case <auto_examples/70_synthetic/index>`
* :doc:`Plotting <auto_examples/80_plotting/index>`
* :doc:`Raw material assessment <auto_examples/85_rawmaterial/index>`
* :doc:`Other examples <auto_examples/90_other/index>`




The INTEGRATE project
=====================
The project is developed as part of the INTEGRATE project, where the goal is to develop probabilistic support tools that allow quantifying the potential for finding raw material resources close to where it is to be utilized.

For more information, please visit the INTEGRATE website (https://integrate.nu/).

Source Code
===========

   The latest stable code is available on GitHub at
   https://github.com/AUProbGeo/integrate_module


License
=======

INTEGRATE is released under the MIT License. See the
`LICENSE <https://github.com/AUProbGeo/integrate_module/blob/main/LICENSE>`_ file
in the repository for the full text.

Contents
========

.. toctree::
   :maxdepth: 3

   install
   gettingstarted
   format
   format_wells
   format_query
   prior_data
   rejection
   workflow
   plotting
   cli
   auto_examples/index
   contributions
   references
   modules


   
