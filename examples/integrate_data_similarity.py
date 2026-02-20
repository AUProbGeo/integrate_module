#!/usr/bin/env python
# %% [markdown]
# # Distance based wights
# Weights based on distance and data similarity
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
    # # # # #%load_ext autoreload
    # # # # #%autoreload 2
    pass
# %%
import numpy as np
import integrate as ig
import matplotlib.pyplot as plt

# %%
case = 'DAUGAARD'
files = ig.get_case_data(case=case)
f_data_h5 = files[0]
f_data_h5 = 'DAUGAARD_AVG.h5'
file_gex= ig.get_gex_file_from_data(f_data_h5)

print("Using data file: %s" % f_data_h5)
print("Using GEX file: %s" % file_gex)



# %%
# Load data and ploti
DATA = ig.load_data(f_data_h5)
d_obs = DATA['d_obs'][0]
d_std = DATA['d_std'][0]

# get geometry
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

# %%
r_dis = 400
r_data = 2

i_ref = 6000-1


# select gates to use 
# find the number of data points for each gate that has non-nan values
n_not_nan = np.sum(~np.isnan(d_obs), axis=0)
n_not_nan_freq = n_not_nan/d_obs.shape[0]
# use the data for which n_not_nan_freq>0.8
i_use = np.where(n_not_nan_freq>0.8)[0]

# select gates to use, manually
i_use = [5,17,18,19,20]

d_ref = np.log10(d_obs[i_ref,i_use])
d_test = np.log10(d_obs[:,i_use])
dd = np.abs(d_test - d_ref)
sum_dd = np.sum(dd, axis=1)

w_data = np.exp(-1*sum_dd**2/r_data**2)

# COmpute the distance from d_ref to all other points
dis = np.sqrt((X-X[i_ref])**2 + (Y-Y[i_ref])**2)
w_dis = np.exp(-1*dis**2/r_dis**2)

w_combined = w_data * w_dis

fig, axs = plt.subplots(3, 1, figsize=(5, 10))

# Plot w_data
axs[0].plot(X, Y, '.', ms=1, zorder=1, color='lightgray')
axs[0].plot(X[i_ref], Y[i_ref], 'k.', ms=6, zorder=4)
sc0 = axs[0].scatter(X, Y, c=w_data, cmap='hot_r', s=.4, zorder=3)
fig.colorbar(sc0, ax=axs[0])
axs[0].set_title('w_data')
axs[0].axis('equal')

# Plot w_dis
axs[1].plot(X, Y, '.', ms=1, zorder=1, color='lightgray')
axs[1].plot(X[i_ref], Y[i_ref], 'k.', ms=6, zorder=4)
sc1 = axs[1].scatter(X, Y, c=w_dis, cmap='hot_r', s=.4, zorder=3)
fig.colorbar(sc1, ax=axs[1])
axs[1].set_title('w_dis')
axs[1].axis('equal')

# Plot w_combined
axs[2].plot(X, Y, '.', ms=1, zorder=1, color='lightgray')
axs[2].plot(X[i_ref], Y[i_ref], 'k.', ms=6, zorder=4)
sc2 = axs[2].scatter(X, Y, c=w_combined, cmap='hot_r', s=.4, zorder=3)
fig.colorbar(sc2, ax=axs[2])
axs[2].set_title('w_combined')
axs[2].axis('equal')

plt.tight_layout()

plt.savefig('Daugard_weights_%04d.png' % (i_ref))
plt.show()





