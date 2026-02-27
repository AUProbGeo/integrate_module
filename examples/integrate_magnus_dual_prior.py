#!/usr/bin/env python
# %% [markdown]
# # The complete INTEGRATE workflow: from data to posterior
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
    # # # # # #%load_ext autoreload
    # # # # # #%autoreload 2
    pass

# %%
import os
import integrate as ig
from geoprior1d import geoprior1d

hardcopy = True
import matplotlib.pyplot as plt
import numpy as np
import copy

# %%
N = 2000 # Number of prior model realizations to generate (this is just for testing, use a larger number for better results)

# %% [markdown]
# ## GETTING THE DATA AND GEX FILE for gthe chosen area

# %%
case = 'DAUGAARD'
f_xlsx_files = ['daugaard_standard.xlsx','daugaard_valley.xlsx']
ig.get_case_data(case=case, filelist=['daugaard_standard.xlsx','daugaard_valley.xlsx'])
files = ig.get_case_data(case=case)
f_data_h5 = files[0]
file_gex= ig.get_gex_file_from_data(f_data_h5)

print("Using data file: %s" % f_data_h5)
print("Using GEX file: %s" % file_gex)

f_data_old_h5 = f_data_h5
f_data_h5 = f_data_h5.replace('.h5', '_WF.h5')
ig.copy_hdf5_file(f_data_old_h5, f_data_h5)



# %% [markdown]
# ## the PRIOR model(s) :
# simulate prior model realizations using geoprior1d # https://github.com/GEUSjesper/geoprior1d
# We are using two prior models representing expected variability outstide and inside a buried valley system.

# %%
f_prior_h5_list = []
for file_xlsx in f_xlsx_files:
    # get filename without extension for naming the output hdf5 file
    fname = file_xlsx.split('.')[0]
    f_prior_h5, flags  = geoprior1d(file_xlsx, Nreals=N, dz=1, dmax =90, output_file='%s_prior_N%d.h5' % (fname, N))
    ig.plot_prior_stats(f_prior_h5, hardcopy=hardcopy)
    f_prior_h5_list.append(f_prior_h5)

# Optionally Merge the two prior hdf5 files into one, which will be used for the rest of the workflow. The merged file will be named 'daugaard_merged_prior_N10000.h5' (or with the appropriate N value).
#f_prior_h5_merged = ig.merge_prior(f_prior_h5_list, f_prior_merged_h5='daugaard_merged_prior_N%d.h5' % N)
#ig.plot_prior_stats(f_prior_h5_merged, hardcopy=hardcopy)


# %% [markdown]
# ## the tTEM DATA

# The electromagnetic data (d_obs and d_std) can be plotted using ig.plot_data:
ig.plot_data(f_data_h5, hardcopy=hardcopy)
# Plot data channel 15 in an XY grid
ig.plot_data_xy(f_data_h5, data_channel=15, cmap='jet');


# %% [markdown]
# ## COMPUTE PRIOR DATA !!

# %%
for f_prior_h5 in f_prior_h5_list:
    f_prior_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, doMakePriorCopy=False)


# %% [markdown]
# ### Select a profile

# %% [markdown]
# ### INVERSION
# The data is now ready for inversion with the rejection sampler.

# %%
# This prt of the can be rerun using different selection of data types without rerunning the abobe parts
nr=1000

N_use = N
f_post_h5_list = []
for f_prior_h5 in f_prior_h5_list:
    # get string from id_use
    f_post_h5 = ig.integrate_rejection(f_prior_h5, 
                                    f_data_h5, 
                                    showInfo=1, 
                                    N_use = N_use,
                                    nr=nr,
                                    updatePostStat=True)
    f_post_h5_list.append(f_post_h5)

# %% [markdown]
# ### POSTERIOR ANALYSIS

# %%
for f_post_h5 in f_post_h5_list:
    ig.plot_profile(f_post_h5, im=1, i1=0, i2=500, gap_threshold=100, xaxis='x', hardcopy=hardcopy, alpha = 1,std_min = 0.5, std_max = 0.6)
    ig.plot_profile(f_post_h5, im=2, i1=0, i2=500, gap_threshold=100, xaxis='x', hardcopy=hardcopy, alpha=1, entropy_min =0.7, entropy_max=0.8)


# %%
