===============
Getting started
===============


:: 

    import integrate as ig
    import matplotlib.pyplot as plt


0. Get some TTEM data
=====================

A number of test cases are available in the INTEGRATE package.
To see which cases are available, check the `get_case_data` function.

The code below will download the file DAUGAARD_AVG.h5 that contains 
a number of TTEM soundings from DAUGAARD, Denmark.
It will also download the corresponding GEX file, TX07_20231016_2x4_RC20-33.gex, 
that contains information about the TTEM system used.

::

    case = 'DAUGAARD'
    files = ig.get_case_data(case=case, showInfo=2)
    f_data_h5 = files[0]
    file_gex = ig.get_gex_file_from_data(f_data_h5)

    print("Using data file: %s" % f_data_h5)
    print("Using GEX file: %s" % file_gex)


Plot the geometry and the data
-------------------------------

ig.plot_geometry plots the geometry of the data, i.e. the locations of the soundings.
ig.plot_data plots the data, i.e. the measured data for each sounding.

::

    # The next line plots LINE, ELEVATION and data id, as three scatter plots
    # ig.plot_geometry(f_data_h5)
    # Each of these plots can be plotted separately by specifying the pl argument
    ig.plot_geometry(f_data_h5, pl='LINE')
    ig.plot_geometry(f_data_h5, pl='ELEVATION')
    ig.plot_geometry(f_data_h5, pl='id')

    # The data, d_obs and d_std, can be plotted using ig.plot_data
    ig.plot_data(f_data_h5, hardcopy=True)


1. Setup the prior model
=========================

In this example a simple layered prior model will be considered.

1a. First, a sample of the prior model parameters will be generated
-------------------------------------------------------------------

As an example, we choose a simple layered model. 
The number of layers follow a chi-squared distribution with 4 degrees of freedom, and the resistivity in each layer is log-uniform between [1,3000].
This will create N realizations of 3 types of model parameters: 

    PRIOR:/M1: 1D resistivity values in layers of 1m thickness down to 90m depth
    PRIOR:/M2: 1D resistivity values in discrete sets of [Nlayer,Nlayer-1] parameters where the first Nlayer parameters are resistivities, and the last Nlayer-1 parameters are depths to the base of each layer.
    PRIOR:/M3: The number of layers in each model

::

    # Select how many, N, prior realizations should be generated
    N = 100000

    f_prior_h5 = ig.prior_model_layered(N=N, lay_dist='chi2', NLAY_deg=4, RHO_min=1, RHO_max=3000, f_prior_h5='PRIOR.h5')
    print('%s is used to hold prior realizations' % (f_prior_h5))
    
    # Plot some summary statistics of the prior model, to QC the prior choice
    ig.plot_prior_stats(f_prior_h5, hardcopy=True)


1b. Compute prior data (the forward response of the prior realizations)
------------------------------------------------------------------------

Then the prior data, corresponding to the prior model parameters, are computed, using the GA-AEM code and the GEX file (from the DATA).

:: 

    # To update the PRIOR.h5
    f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, doMakePriorCopy=False)
    # To create a COPY of PRIOR.h5 and update that
    # f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex)

    print('Updated %s to hold prior DATA' % (f_prior_data_h5))

It can be useful to compare the prior data to the observed data before inversion. If there is little to no overlap of the observed data with the prior data, there is little chance that the inversion will go well. This would be an indication of inconsistency.
In the figure below, one can see that the observed data (red) is clearly within the space of the prior data.

::

    ig.plot_data_prior(f_prior_data_h5, f_data_h5, nr=1000, hardcopy=True)


2. Sample the posterior distribution
====================================

The posterior distribution is sampled using the extended rejection sampler.

::

    # Rejection sampling of the posterior can be done using
    # f_post_h5 = ig.integrate_rejection(f_prior_h5, f_data_h5)

    # One can also control a number of options.
    # One can choose to make use of only a subset of the prior data. Decreasing the sample size used makes the inversion faster, but increasingly approximate
    N_use = N
    T_base = 1  # The base annealing temperature. 
    autoT = 1   # Automatically set the annealing temperature
    f_post_h5 = ig.integrate_rejection(f_prior_h5,
                                       f_data_h5,
                                       f_post_h5='POST.h5',
                                       N_use=N_use,
                                       autoT=autoT,
                                       T_base=T_base,
                                       showInfo=1)


3. Plot some statistics from the posterior distribution
=======================================================

Prior and posterior data
------------------------

First, compare prior (beige) to posterior (black) data, as well as observed data (red), for two specific data IDs.

::

    ig.plot_data_prior_post(f_post_h5, i_plot=100, hardcopy=True)
    ig.plot_data_prior_post(f_post_h5, i_plot=0, hardcopy=True)


Evidence and Temperature
------------------------

::

    # Plot the Temperature used for inversion
    ig.plot_T_EV(f_post_h5, pl='T', hardcopy=True)
    # Plot the evidence (prior likelihood) estimated as part of inversion
    ig.plot_T_EV(f_post_h5, pl='EV', hardcopy=True)


Profile
-------

Plot a profile of posterior statistics of model parameters 1 (resistivity)

::

    ig.plot_profile(f_post_h5, i1=1, i2=2000, im=1, hardcopy=True)


Plot 2D Features
-----------------

First plot the median resistivity in layers 5, 30, and 50

:: 

    # Plot a 2D feature: Resistivity in different layers
    try:
        ig.plot_feature_2d(f_post_h5, im=1, iz=5, key='Median', uselog=1, cmap='jet', s=10, hardcopy=True)
        plt.show()
    except:
        pass

    try:
        ig.plot_feature_2d(f_post_h5, im=1, iz=30, key='Median', uselog=1, cmap='jet', s=10, hardcopy=True)
        plt.show()
    except:
        pass

    try:
        ig.plot_feature_2d(f_post_h5, im=1, iz=50, key='Median', uselog=1, cmap='jet', s=10, hardcopy=True)
        plt.show()
    except:
        pass

    try:
        # Plot a 2D feature: The number of layers
        ig.plot_feature_2d(f_post_h5, im=2, iz=0, key='Median', title_text='Number of layers', uselog=0, clim=[1,6], cmap='jet', s=12, hardcopy=True)
        plt.show()
    except:
        pass


Export to CSV format
====================

::

    f_csv, f_point_csv = ig.post_to_csv(f_post_h5)

    # Read the CSV file
    import pandas as pd
    df = pd.read_csv(f_point_csv)
    df.head()