#!/usr/bin/env python


# %% Setup
try:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except Exception:
    pass

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import libaarhusxyz
import integrate as ig
from scipy.ndimage import maximum_filter1d

plt.ion()
hardcopy = True

# %% [markdown]
# # Get some data

# %% Load DAUGAARD data
ig.get_case_data(case='DAUGAARD', loadType='post')
ig.get_case_data(case='DAUGAARD', loadType='WB_sharp')
ig.get_case_data(case='DAUGAARD', loadType='WB_smooth')

cmap, clim = ig.get_colormap_and_limits('resistivity')
fontsize = 16

f_post_h5 = 'POST_DAUGAARD_AVG_prior_detailed_general_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu2000000_aT1.h5'
f_post_h5 = 'post_PRIOR_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nuse1000000_inflateNoise2_main.h5'
#f_post_h5 = 'post_daugaard_merged_N2000000_Nuse1000000_inflateNoise2_main.h5'

with h5py.File(f_post_h5, 'r') as f:
    f_data_h5  = f.attrs['f5_data']
    f_prior_h5 = f.attrs['f5_prior']

X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)


# %% [markdown]
# # Comparison to WorkBench inversion
#
# Reprojects a WorkBench LSQ inversion result onto the INTEGRATE posterior z-grid
# and writes Mean, LogMean, and Std into the posterior HDF5 file for profile comparison.

# Posterior HDF5 (INTEGRATE result)
# get id of profile
# Find points within buffer distance
X, Y, LINE, ELEVATION = ig.get_geometry(f_post_h5)
Xl = np.array([544000, 543550])
Yl = np.array([6174500, 6176500])
buffer = 10.0
indices, distances, segment_ids = ig.find_points_along_line_segments(
    X, Y, Xl, Yl, tolerance=buffer
)
id_line = indices
i_plot_1 = indices[5]
i_plot_2 = 1000

'''
plt.figure(figsize=(10, 6))
plt.plot(X, Y,'k.', markersize=11,label='Survey Points')
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
'''



# WorkBench LSQ XYZ file to compare against (choose smooth or sharp)
f_xyz_list = [
    'SCI7_40_ml_Daugaard_I01_MOD_inv.xyz',
    'SCI7_40_ml_sharp2_I02_MOD_inv.xyz'
    ]

for f_xyz in f_xyz_list:
    #% Read posterior geometry and z-grid
    # Copy posterior HDF5 and use it as the output file for the LSQ comparison
    f_lsq_h5 = os.path.splitext(f_xyz)[0] + '.h5'
    ig.copy_hdf5_file(f_post_h5, f_lsq_h5)

    X, Y, LINE, ELEVATION = ig.get_geometry(f_lsq_h5)

    with h5py.File(f_lsq_h5, 'r') as f:
        f_data_h5  = f.attrs['f5_data']
        f_prior_h5 = f.attrs['f5_prior']
        M1_median  = f['/M1/Median'][:]

    nd, nz = M1_median.shape

    with h5py.File(f_prior_h5, 'r') as f:
        z = f['/M1'].attrs['x'][:].flatten()

    #% Read WorkBench LSQ model
    model = libaarhusxyz.XYZ(f_xyz)

    rho     = model.layer_data['rho'].values      # (n_lsq, n_layers_lsq)
    rho_std = model.layer_data['rho_std'].values
    dep_top = model.layer_data['dep_top'].values  # top depth of each LSQ layer

    n_lsq, n_layers_lsq = rho.shape

    utmx = model.flightlines['utmx'].values
    utmy = model.flightlines['utmy'].values

    #% Reproject LSQ model onto posterior z-grid
    # For each LSQ data point, find the nearest posterior grid location and
    # interpolate resistivity onto the posterior z-grid via layer lookup.
    M1_mean = np.full_like(M1_median, np.nan)
    M1_std  = np.full_like(M1_median, np.nan)

    for i in range(len(utmx)):
        idx = np.argmin((X - utmx[i])**2 + (Y - utmy[i])**2)
        #print(f"i={i:4d}/{len(utmx)}, idx={idx:4d}  "
        #    f"[{utmx[i]:.1f}, {utmy[i]:.1f}] -> [{X[idx]:.1f}, {Y[idx]:.1f}]")

        # For each posterior z cell, find which LSQ layer contains it
        j_idx = np.searchsorted(dep_top[i], z, side='right') - 1
        valid = j_idx >= 0
        M1_mean[idx, valid] = rho[i, j_idx[valid]]
        M1_std[idx, valid]  = rho_std[i, j_idx[valid]]

    #% Write LSQ results into output HDF5
    with h5py.File(f_lsq_h5, 'a') as f:
        for key in ['/M1/LogMean', '/M1/Mean', '/M1/Std']:
            if key in f:
                del f[key]
        f['/M1/Mean']    = M1_mean
        f['/M1/LogMean'] = np.log10(M1_mean)
        f['/M1/Std']     = np.log10(M1_std)

    #% Plot profiles
    ig.plot_profile(f_lsq_h5,  ii=id_line, gap_threshold=50, xaxis='y', fontsize = fontsize, panels=['mean'],             im=1, hardcopy=hardcopy)
    ig.plot_profile(f_lsq_h5,  ii=id_line, gap_threshold=50, xaxis='y',fontsize = fontsize, panels=['mean', 'std'],        im=1, hardcopy=hardcopy)
ig.plot_profile(f_post_h5, ii=id_line, gap_threshold=50, xaxis='y',fontsize = fontsize, panels=['median'],               im=1, hardcopy=hardcopy)
ig.plot_profile(f_post_h5, ii=id_line, gap_threshold=50, xaxis='y',fontsize = fontsize, panels=['harmonicmean'],               im=1, hardcopy=hardcopy)
ig.plot_profile(f_post_h5, ii=id_line, gap_threshold=50, xaxis='y',fontsize = fontsize, panels=['mean'],               im=1, hardcopy=hardcopy)
ig.plot_profile(f_post_h5, ii=id_line, gap_threshold=50, xaxis='y',fontsize = fontsize, panels=['mean', 'std', 'stats'], im=1, hardcopy=hardcopy)



# %% [markdown]
# # Analyze the generic prior with variable number of layers

f_post_h5 = 'post_PRIOR_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nuse100000_inflateNoise1_main.h5'
# f_base is f_post_h5 withtout extension and 'post_' prefix
f_base = os.path.splitext(os.path.basename(f_post_h5))[0].removeprefix('post_')
with h5py.File(f_post_h5, 'r') as f:
    f_data_h5  = f.attrs['f5_data']
    f_prior_h5 = f.attrs['f5_prior']
    M1_median  = f['/M1/Median'][:]
    M3_median  = f['/M3/Median'][:]
    M3_mean  = f['/M3/Mean'][:]
    i_use = f['/i_use'][:]


i_plot = i_plot_1
#i_plot = id_line[80]
#i_plot = id_line[20]
nd, nr = i_use.shape

D, M, idx = ig.load_prior(f_prior_h5)

bins = np.arange(1 - 0.5, 17 + 1.5)
plt.figure()
plt.hist(M[2], bins=bins, density=True, color='black', alpha=1, label='Prior')
plt.hist(M[2][i_use[i_plot]], bins=bins, density=True, color='darkgray', alpha=0.8, label='Posterior')
plt.axvline(np.mean(M[2][i_use[i_plot]]),   color='darkgray', linestyle='--', label='Posterior mean')
plt.axvline(np.median(M[2][i_use[i_plot]]), color='darkgray', linestyle=':',  label='Posterior median')
plt.xlabel('Number of layers')
plt.ylabel('Density')
plt.grid()
plt.legend()
plt.show()
plt.gcf().savefig('%s_prior_M3_median.png' % f_base, dpi=300)

#%%
# Plot the mean
#ig.plot_feature_2d(f_post_h5, key='Mean', im=3, uselog=False, cmap='jet', s=2)
#plt.gcf().savefig('%s_prior_M3_mean.png' % f_base, dpi=300)
#plt.plot(X[i_plot], Y[i_plot], 'ko', markersize=30, label='Profile Point')
#plt.show()
#% Plot the median 
cmap_disc = plt.cm.get_cmap('seismic_r', 8)
norm_disc = plt.matplotlib.colors.BoundaryNorm(np.arange(0.5, 9.5), 8)
ig.plot_feature_2d(f_post_h5, key='Median', im=3, uselog=False, cmap=cmap_disc, norm=norm_disc, s=2)
#plt.plot(X[id_line],Y[id_line],'k.', markersize=1.1, label='Survey Points')
plt.plot(X[i_plot], Y[i_plot], 'ko', markersize=10, markerfacecolor='none', label='Profile Point')
plt.gcf().axes[-1].set_yticks(range(1, 9))
plt.gcf().savefig('%s_prior_M3_median_2d.png' % f_base, dpi=300)

#%% 
# Now compute the probability of a layer boundary at each depth index, as the
M1_boundary = M1_median.copy()*0
for i in range(nd):
    i_use_single = i_use[i]
    rho_post = M[0][i_use_single].T
    # 1 if any relative change >= rel_change occurs within ±delta_z depth cells
    rel_change = 0.001
    delta_z = 3
    rho_change_raw = np.zeros_like(rho_post)
    rho_change_raw[1:, :] = (np.abs(rho_post[1:, :] - rho_post[:-1, :]) / rho_post[:-1, :] >= rel_change).astype(float)
    rho_change = maximum_filter1d(rho_change_raw, size=2*delta_z+1, axis=0)
    M1_boundary[i, :] = rho_change.mean(axis=1)

plt.imshow(M1_boundary[id_line].T, cmap='gray_r', vmin=0.1, vmax=1)
plt.colorbar()
plt.gcf().savefig('%s_prior_M1_boundary_2d.png' % f_base, dpi=300)

# %%
plt.figure()
nshow = 11
for i in range(nshow):
    data = M[1][i]
    
    n = int((len(data) + 1) / 2)
    z = np.insert(data[0:n-1], 0, 0)  # n depth edges, starting at surface
    rho = data[n-1:]                   # n resistivity values
    n_valid = int(np.sum(~np.isnan(rho)))
    if n_valid == 0:
        continue
    rho = rho[:n_valid]
    z = z[:n_valid]
    dz = (z[-1] - z[-2]) if n_valid > 1 else 10
    z_edges = np.append(z, z[-1] + dz)  # n_valid+1 edges for n_valid values
    plt.stairs(rho, z_edges, orientation='horizontal')

plt.gca().invert_yaxis()
plt.xlabel('Resistivity (Ohm·m)')
plt.ylabel('Depth (m)')
plt.title('Staircase plot of resistivity vs depth')
plt.show()


# %%
