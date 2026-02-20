#!/usr/bin/env python
# %% [markdown]
# # Muliple surveys
# 8 individual tTEM survey wer conducted, using 4 distinct GEX files .
# Below these data are inverted in 4 separate inversions on for each specific GEX file.
#


# %%
try:
    # Check if the code is running in an IPython kernel (which includes Jupyter notebooks)
    get_ipython()
    # If the above line doesn't raise an error, it means we are in a Jupyter environment
    # Execute the magic commands using IPython's run_line_magic function
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except:
    # If get_ipython() raises an error, we are not in a Jupyter environment
    # # # # # # # #%load_ext autoreload
    # # # # # # # #%autoreload 2
    pass
# %%
import integrate as ig
import numpy as np
import matplotlib.pyplot as plt
hardcopy = True

# %%
# Load ESBJERG as 4 subsets (4 different observed data sets, ewach with their own calibrations, and hence unique GEX file)
f_files = ig.get_case_data(case='ESBJERG',loadType='gex' , filelist=[])
f_gex = f_files[:4] 
f_data_h5_files = f_files[4:] 

# The last dataset from 2024, used different number of gates, which cannot be handles my merging, so we consider the first three datasets from 2023.

nsubsets = 3
f_gex = f_gex[:nsubsets]
f_data_h5_files = f_data_h5_files[:nsubsets]


# %%
N=100000
# All surveys are inverted with the same prior model
#f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='chi2', NLAY_deg=3, RHO_min=1, RHO_max=500)
f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='uniform', NLAY_min=1, NLAY_max=8, RHO_min=1, RHO_max=500)


# %%
plFigs = False

f_post_h5_files = []
f_prior_data_h5_files = []
showInfo=0
for i in range(nsubsets):
    f_data_h5 = f_data_h5_files[i]

    print('------------------------')
    print('- f_data_h5 = "%s"' % (f_data_h5))
    
    #X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

    file_gex= ig.get_gex_file_from_data(f_data_h5)
    # Extract filename without extension from f_data_h5
    filename = f_data_h5.split('.')[0]
    
    if plFigs:
        ig.plot_geometry(f_data_h5, pl='LINE', hardcopy=hardcopy)
    
    # Compute prior DATA - 
    # Even though the prior model parameters are the same, the prior data diffeer, due to using a different GEX file.
    f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, showInfo=0)
    if plFigs:
        ig.plot_data_prior(f_prior_data_h5,f_data_h5,nr=1000,alpha=1, ylim=[1e-13,1e-5], hardcopy=hardcopy, showInfo=showInfo)

    # Perform inversion
    N_use = N
    f_post_h5 = ig.integrate_rejection(f_prior_data_h5,
                                   f_data_h5,
                                   N_use = N_use,
                                   showInfo=showInfo,
                                   Ncpu = 10,
                                   updatePostStat=False
                                      )

    f_post_h5_files.append(f_post_h5)
    f_prior_data_h5_files.append(f_prior_data_h5)

    # Plot posterior stats
    if plFigs:
        # Update posterior stats
        ig.integrate_posterior_stats(f_post_h5)
        # compare prior an posterior data
        ig.plot_data_prior_post(f_post_h5, i_plot=100, hardcopy=hardcopy);
        ig.plot_data_prior_post(f_post_h5, i_plot=0, hardcopy=hardcopy);        

        # Plot the Temperature used for inversion
        ig.plot_T_EV(f_post_h5, pl='T', hardcopy=hardcopy)
        # Plot the evidence (prior likelihood) estimated as part of inversion
        ig.plot_T_EV(f_post_h5, pl='EV', hardcopy=hardcopy)

        # Plot Profiles
        ig.plot_profile(f_post_h5, im=1, hardcopy=hardcopy)

# %% [markdown]
# ## Merge posterior and, update posterior statistics in ther poster hdf5 file.

# %%
f_post_merged_h5, f_data_merged_h5 = ig.merge_posterior(f_post_h5_files, f_data_h5_files, showInfo=4)
ig.integrate_posterior_stats(f_post_merged_h5)

# %%
# Plot some figures 
ig.plot_geometry(f_data_merged_h5, pl='LINE')

# %%
ig.plot_T_EV(f_post_merged_h5, pl='T', hardcopy=hardcopy)

# %%
ig.plot_profile(f_post_merged_h5, im=1, i1=0, i2=1000, hardcopy=hardcopy)

# %%
ig.plot_feature_2d(f_post_merged_h5,im=1,iz=20, key='Median', uselog=1, cmap='jet', s=1, hardcopy=hardcopy);

# %%

# %%
