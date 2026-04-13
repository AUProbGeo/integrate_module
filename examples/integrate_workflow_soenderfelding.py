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
    # # # # # # #%load_ext autoreload
    # # # # # # #%autoreload 2
    pass

# %%
import os
import integrate as ig
from geoprior1d import geoprior1d

hardcopy = True
import matplotlib.pyplot as plt
import numpy as np
import copy

# %% [markdown]
# ## GETTING THE DATA AND GEX FILE for gthe chosen area

# %%
case = 'SOENDER_FELDING'
files = ig.get_case_data(case=case)
files_xyz = [f for f in files if f.endswith('.xyz')]
files_gex = [f for f in files if f.endswith('.gex')]


# %% Just for 'fun' convert all xyz files to hdf5 data format, 
# using the first gex file.
# This is probably NOT usefull at all, and leading to errors.
# Update: FDoes not work as some GEX files define 29 and som 40 data
f_data_all_h5 = ig.xyz_to_h5(files_xyz, files_gex[0], f_data_h5='SOENDER_FELDING_data.h5', showInfo=1)

# The electromagnetic data (d_obs and d_std) can be plotted using ig.plot_data:
ig.plot_data(f_data_all_h5, hardcopy=hardcopy)
# Plot data channel 15 in an XY grid
ig.plot_data_xy(f_data_all_h5, data_channel=15, cmap='jet')



# %% Create a data file for each GEX file . This is boring handwork, but it is just an example, and it is not expected that users will do this manually in the future. In the future we will probably have a more automated way to link the XYZ files to the GEX files, e.g. by using a naming convention or by reading the GEX file to see which XYZ files are associated with it.
# 
# Read @README_SOENDER_FELDING, and create a an f_data_h5 for each 
# gex filem, using the XYZ files as indicate by the readme file
f_data_sub=[]

# Gex file 1: TX07_20240802_2x4_RC20-39.gex
# Dates: 20240819, 20240820, 20240821, 20240911
file_gex = 'TX07_20240802_2x4_RC20-39.gex'
file_xyz = ['20240819_AVG_export.xyz', '20240820_AVG_export.xyz', '20240821_AVG_export.xyz', '20240911_AVG_export.xyz']
fname = file_gex.split('.')[0]
f_data_sub.append(ig.xyz_to_h5(file_xyz, file_gex, f_data_h5='%s_data.h5' % fname, showInfo=1))

# Gex file 2: TX07_20240802_2x4_RC20-39_eksternGPS.gex
# Dates: 20240911_eksternGPS
file_gex = 'TX07_20240802_2x4_RC20-39_eksternGPS.gex'
file_xyz = ['20240911_eksterngps_AVG_export.xyz']
fname = file_gex.split('.')[0]
f_data_sub.append(ig.xyz_to_h5(file_xyz, file_gex, f_data_h5='%s_data.h5' % fname, showInfo=1))

# Gex file 3: TX07_20240912_2x4_RC20-39_eksterngps.gex
# Dates: 20240924, 20240924_test, 20241007, 20241008
file_gex = 'TX07_20240912_2x4_RC20-39_eksterngps.gex'
file_xyz = ['20240924_AVG_export.xyz', '20240924_test_AVG_export.xyz', '20241007_AVG_export.xyz', '20241008_AVG_export.xyz']
fname = file_gex.split('.')[0]
f_data_sub.append(ig.xyz_to_h5(file_xyz, file_gex, f_data_h5='%s_data.h5' % fname, showInfo=1))

# # Gex file 4: TX07_20241014_2x4_RC20_33_and_57_EksternGPS.gex
# Dates: 20241029
file_gex = 'TX07_20241014_2x4_RC20_33_and_57_EksternGPS.gex'
file_xyz = ['20241029_AVG_export.xyz']
fname = file_gex.split('.')[0]
f_data_sub.append(ig.xyz_to_h5(file_xyz, file_gex, f_data_h5='%s_data.h5' % fname, showInfo=1))

# # Gex file 5: TX07_20241202_2x4_RC20_57_EksternGPS.gex
# Dates: 20241210
file_gex = 'TX07_20241202_2x4_RC20_57_EksternGPS.gex'
file_xyz = ['20241210_AVG_export.xyz']
fname = file_gex.split('.')[0]
f_data_sub.append(ig.xyz_to_h5(file_xyz, file_gex, f_data_h5='%s_data.h5' % fname, showInfo=1))

# Gex file 6: TX07_20241202_2x4_RC20_57.gex
# Dates: 20241210_InternGPS
file_gex = 'TX07_20241202_2x4_RC20_57.gex'
file_xyz = ['20241210_InternGPS_AVG_export.xyz']
fname = file_gex.split('.')[0]
f_data_sub.append(ig.xyz_to_h5(file_xyz, file_gex, f_data_h5='%s_data.h5' % fname, showInfo=1))



# %% [markdown]
N = 100_000
# ## the PRIOR model(s) :
f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='chi2', NLAY_deg=3, RHO_min=1, RHO_max=3000, f_prior_h5='PRIOR_N%d.h5' % N, 
                                    showInfo=1)

# ## the PRIOR data :
f_prior_sub = []
for file_gex in files_gex:
    f_prior_sub_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, doMakePriorCopy=True)
    f_prior_sub.append(f_prior_sub_h5)


# %% [markdown]
# ### Select a profile

# %%
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_all_h5)

# Find points within buffer distance
X1 = 483272.0
Y1 = 6201000.0
X2 = 488850.0
Y2 = 6203000.0
Xl = np.array([X1-100,X1, X2, X2+1500])
Yl = np.array([Y1, Y1, Y2, Y2-150])
buffer = 15.0
indices, distances, segment_ids = ig.find_points_along_line_segments(
    X, Y, Xl, Yl, tolerance=buffer
)
id_line = indices

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, c=ELEVATION, s=1,label='X')
plt.plot(X[id_line],Y[id_line], 'r.', markersize=8, label='Profile', zorder=2, linewidth=5)
plt.grid()
plt.colorbar(label='Number of non-Nan data points')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Survey Points Colored by Number of Non-NaN Data Points')
plt.axis('equal')
plt.legend()
if hardcopy:
    plt.savefig('DAUGAARD_survey_points_nonnan.png', dpi=300)
plt.show()

i1=np.min(id_line)
i2=np.max(id_line)+1


# %% [markdown]
# ### INVERSION

# %% INVERT ALL
# This prt of the can be rerun using different selection of data types without rerunning the abobe parts
N_use = N
nr=10
T_N_above=50
T_P_acc_level=0.2 
autoT = 1 # We need minium of T_N_above realizations with an acceptance probability above T_P_acc_level

f_post_sub = []

for i in range(len(f_data_sub)):
    f_post_h5_single = ig.integrate_rejection(f_prior_sub[i], 
                                    f_data_sub[i], 
                                    showInfo=1, 
                                    N_use = N_use,
                                    nr=nr,
                                    T_N_above = T_N_above,
                                    T_P_acc_level = T_P_acc_level,
                                    updatePostStat=True)
    f_post_sub.append(f_post_h5_single)

#%%  Merge posterior models from different data sets (this is just for testing, and it is not expected that users will do this manually in the future, but it is just to show how it can be done)
f_post_merged_h5, f_data_merged_h5 = ig.merge_posterior(f_post_sub, f_data_sub, showInfo=4)

# %% [markdown]
# ### POSTERIOR ANALYSIS

# %%
ig.plot_profile(f_post_merged_h5, im=1, ii=id_line, gap_threshold=100, xaxis='x', hardcopy=hardcopy, alpha = 1,std_min = 0.5, std_max = 0.6)
#ig.plot_profile(f_post_merged_h5, im=2, ii=id_line, gap_threshold=100, xaxis='x', hardcopy=hardcopy, alpha=1, entropy_min =0.7, entropy_max=0.8)

# %%
ig.plot_feature_2d(f_post_merged_h5, key='Median', im=1, elevation=-10)
plt.legend()    
plt.show()

# %%
#ig.plot_feature_2d(f_post_merged_h5, key='Mode', im=2, elevation=45)
#plt.legend()    
#plt.show()

