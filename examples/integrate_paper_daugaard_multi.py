#!/usr/bin/env python
# %%
# # Daugaard Case Study with three lithology-resistivity prior models.
#
# This notebook contains an example of inverison of the DAUGAARD tTEM data using three different lithology-resistivity prior models

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
    # # # # # # # # #%load_ext autoreload
    # # # # # # # # #%autoreload 2
    pass

import integrate as ig
# Check if parallel computations can be performed
parallel = ig.use_parallel(showInfo=1)
import numpy as np
import os
import matplotlib.pyplot as plt
plt.ion()
import h5py

from integrate.integrate_io import copy_prior
hardcopy=True

# %% [markdown]
# ## Download the data DAUGAARD data including non-trivial prior data realizations

# %%
cmap, clim = ig.get_colormap_and_limits('resistivity')
useMergedPrior=True
useGenericPrior=False
inflateNoise = 4   # 1,2, 4
useLogData = False
N_use = 100000
N_use_org= N_use
#N_use = 100000
#N_use = 100000

doEffectSize = True
doTbase = True
doPlotAll=True
doTestInversion = False
doSamplePrior = False # Load prior model and data? (True) or load precomputed prior data realizations (False)

# Load the data from DAUGAARD
ig.get_case_data(case='DAUGAARD')

# Load the Excel files for the two different geological scenarios (for use with geoprior1d)
f_standard_xls = 'daugaard_standard.xlsx'
f_valley_xls = 'daugaard_valley.xlsx'
ig.get_case_data(case='DAUGAARD', filelist=[f_standard_xls])
ig.get_case_data(case='DAUGAARD', filelist=[f_valley_xls])


if doSamplePrior is False:
    # Load prior data, allready incuding prior model and data realizations
    files = ig.get_case_data(case='DAUGAARD', loadType='prior_data') # Load data and prior+data realizations

f_data_h5 = 'DAUGAARD_AVG.h5'
file_gex= ig.get_gex_file_from_data(f_data_h5)

# check that file_gex exists
if not os.path.isfile(file_gex):
    print("file_gex=%s does not exist in the current folder." % file_gex)

print('Using hdf5 data file %s with gex file %s' % (f_data_h5,file_gex))

ig.plot_geometry(f_data_h5, pl='NDATA', hardcopy= hardcopy, cmap='viridis')
plt.show()

# %%
if doSamplePrior:
    N=N_use
    dmax=90
    dz=1
    from geoprior1d import geoprior1d

    f_prior_standard_h5='prior_daugaard_standard_N%d.h5' % N
    f_prior_valley_h5='prior_daugaard_valley_N%d.h5' % N
    # Generate priors
    filename, flags = geoprior1d(
        input_data=f_standard_xls,
        Nreals=N, dmax=dmax, dz=dz,        
        output_file = f_prior_standard_h5  # Optional: specify custom output filename
    )

    # Generate priors
    filename, flags = geoprior1d(
        input_data=f_valley_xls,
        Nreals=N, dmax=dmax, dz=dz,
        output_file=f_prior_valley_h5      # Optional: specify custom output filename
    )

    # Compute prior data
    f_prior_standard_h5 = ig.prior_data_gaaem(f_prior_standard_h5, file_gex, doMakePriorCopy=False)
    f_prior_valley_h5 = ig.prior_data_gaaem(f_prior_valley_h5, file_gex, doMakePriorCopy=False)
else:
    # Load precomputed prior data realizations
    f_prior_standard_h5='daugaard_standard_new_N1000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5'
    f_prior_valley_h5='daugaard_valley_new_N1000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5'
    



# %%

if useLogData:
    f_data_h5_org = f_data_h5
    f_data_h5 = 'DATA_LOGSPACE.h5'
    ig.copy_hdf5_file(f_data_h5_org, f_data_h5)
    DATA = ig.load_data(f_data_h5_org)
    D_obs = DATA['d_obs'][0]
    D_std = DATA['d_std'][0]
    lD_obs = np.log10(D_obs)
    lD_std_up = np.abs(np.log10(D_obs+D_std)-lD_obs)
    lD_std_down = np.abs(np.log10(D_obs-D_std)-lD_obs)
    corr_std = 0.02
    lD_std = np.abs((lD_std_up+lD_std_down)/2) + corr_std
    ig.save_data_gaussian(lD_obs, D_std = lD_std, f_data_h5 = f_data_h5, id=1, showInfo=0, is_log=1)



# %%
if inflateNoise != 1:
    gf=inflateNoise
    print("="*60)
    print("Increasing noise level (std) by a factor of %d" % gf)
    print("="*60)
    D = ig.load_data(f_data_h5)
    D_obs = D['d_obs'][0]
    D_std = D['d_std'][0]*gf
    f_data_old_h5 = f_data_h5
    f_data_h5 = 'DAUGAARD_AVG_gf%g.h5' % (gf) 
    ig.copy_hdf5_file(f_data_old_h5, f_data_h5)
    ig.save_data_gaussian(D_obs, D_std=D_std, f_data_h5=f_data_h5, file_gex=file_gex)

ig.plot_data(f_data_h5, useLog = 0, hardcopy= hardcopy)
plt.show()

# %%
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

with h5py.File(f_data_h5,'r') as f_data:
    # find number of nan values on d_obs
    NON_NAN = np.sum(~np.isnan(f_data['/%s' % 'D1']['d_obs']), axis=1)



# Find a unique list of line number, then find the LINEnumber that occurs most frequently
unique_lines, counts = np.unique(LINE, return_counts=True)
most_frequent_line = unique_lines[np.argmax(counts)]
print("Most frequent line number:", most_frequent_line)

# find the indexes of the most frequent line
id_line = np.where(LINE == most_frequent_line)[0]

# Only use the first entries of id_line until the index change more than 2
id_line_diff = np.diff(id_line)
id_line_cut = np.where(id_line_diff > 2)[0]
if len(id_line_cut) > 0:
    id_line = id_line[:id_line_cut[0]+1]

# set id_line to 100,101...,1001
id_line_1 = np.arange(755, 815)
id_line_2 = np.arange(7720, 7730)
id_line_3 = np.flip(np.arange(2705, 2731))
id_line = np.concatenate((id_line_1, id_line_2, id_line_3))
#id_liine = id_line_3

# Find points within buffer distance
Xl = np.array([X[id_line][0], X[id_line][-1]])
Yl = np.array([Y[id_line][0], Y[id_line][-1]])
buffer = 10.0
indices, distances, segment_ids = ig.find_points_along_line_segments(
    X, Y, Xl, Yl, tolerance=buffer
)
id_line = indices

# Find points within buffer distance
Xl = np.array([np.min(X), np.max(X)])
Yl = np.array([np.max(Y), np.min(Y)])
buffer = 10.0
indices, distances, segment_ids = ig.find_points_along_line_segments(
    X, Y, Xl, Yl, tolerance=buffer
)
id_line = indices


# Find points within buffer distance
Xl = np.array([544000, 543550])
Yl = np.array([6174500, 6176500])
buffer = 10.0
indices, distances, segment_ids = ig.find_points_along_line_segments(
    X, Y, Xl, Yl, tolerance=buffer
)
id_line = indices

i_plot_1 = indices[5]
i_plot_2 = 1000


plt.figure(figsize=(10, 6))
plt.scatter(X, Y, c=NON_NAN, s=1,label='Survey Points')
#plt.plot(X[id_line],Y[id_line], 'k-', markersize=8, label='Profile', zorder=2, linewidth=5)
plt.plot(X[id_line],Y[id_line], 'r.', markersize=8, label='Profile', zorder=2, linewidth=5)
plt.plot(X[i_plot_1],Y[i_plot_1], 'k*', markersize=10, label='P1')
plt.plot(X[i_plot_2],Y[i_plot_2], 'k*', markersize=10, label='P2')
plt.grid()
#plt.colorbar(label='Elevation (m)')
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
# ## Compute prior data from prior model if they do not already exist

# %%
f_prior_data_h5_list = []
f_prior_data_h5_list.append(f_prior_valley_h5)
f_prior_data_h5_list.append(f_prior_standard_h5)

if useLogData == True:
    f_prior_log_data_h5  = 'd_valley_log.h5'
    ig.copy_hdf5_file(f_prior_data_h5_list[0], f_prior_log_data_h5, showInfo=2)
    D,id =  ig.load_prior_data(f_prior_data_h5_list[0])
    Dlog = np.log10(D[0])
    ig.save_prior_data(f_prior_log_data_h5, Dlog, id=1, showInfo=2, force_delete=True)
    f_prior_data_h5_list[0] = f_prior_log_data_h5

    f_prior_log_data_h5  = 'd_out_log.h5'
    ig.copy_hdf5_file(f_prior_data_h5_list[1], f_prior_log_data_h5, showInfo=2)
    D,id =  ig.load_prior_data(f_prior_data_h5_list[1])
    Dlog = np.log10(D[0])
    ig.save_prior_data(f_prior_log_data_h5, Dlog, id=1, showInfo=2, force_delete=True)
    f_prior_data_h5_list[1] = f_prior_log_data_h5

# %%

# %%
if useMergedPrior:
    f_prior_data_merged_h5 = ig.merge_prior(f_prior_data_h5_list, 
                                            f_prior_merged_h5='daugaard_merged_N%d.h5' % (2*N_use), 
                                            showInfo=2,
                                            shuffle=True)
    ig.hdf5_info(f_prior_data_merged_h5)
    f_prior_data_h5_list.append(f_prior_data_merged_h5)

# %%
if useGenericPrior:
    f_prior_h5 = ig.prior_model_layered(N=N_use,lay_dist='chi2', NLAY_deg=4, RHO_min=1, RHO_max=3000, f_prior_h5='PRIOR.h5')
    f_prior_data_generic_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, NdoMakePriorCopy=True)
    f_prior_data_h5_list.append(f_prior_data_generic_h5)

# %%
for i_prior in range(len(f_prior_data_h5_list)):

    f_prior_data_h5= f_prior_data_h5_list[i_prior]
    ig.integrate_update_prior_attributes(f_prior_data_h5)
    ig.plot_data_prior(f_prior_data_h5, f_data_h5, i_plot=100, hardcopy=hardcopy)


# %%
for i_prior in range(len(f_prior_data_h5_list)):
    ig.plot_prior_stats(f_prior_data_h5_list[i_prior])

# %%
# Select how many prior model realizations (N) should be generated
autoT=True
nr=1000
f_post_h5_list = []
for i_prior in range(len(f_prior_data_h5_list)):

    f_prior_data_h5= f_prior_data_h5_list[i_prior]
  
    # Get filename without extension
    fileparts = os.path.splitext(f_prior_data_h5)
    f_post_h5 = 'post_%s_Nuse%d_inflateNoise%d_main.h5' % (fileparts[0], N_use,inflateNoise)

    f_post_h5 = ig.integrate_rejection(f_prior_data_h5, 
                                    f_data_h5, 
                                    f_post_h5, 
                                    N_use = 2*N_use, 
                                    showInfo=1, 
                                    nr=nr,
                                    parallel=True, 
                                    autoT=autoT,
                                    T_base=1,
                                    updatePostStat=False)
   

    f_post_h5_list.append(f_post_h5)    

# %%
for i_post in range(len(f_post_h5_list)):
    ig.integrate_posterior_stats(f_post_h5_list[i_post], showInfo=1)

# %%
useShapeFiles = True
try:
    files = ig.get_case_data(case='DAUGAARD', loadType='shapefiles')
    import geopandas as gpd
    gdf = gpd.read_file('Begravet dal.shp')
    line_coords = gdf[gdf.geometry.type == 'LineString'].geometry.apply(lambda geom: list(geom.coords))
    line1=np.array(line_coords[0])
    line2=np.array(line_coords[1])
    gdf = gpd.read_file('Erosion øvre.shp')
    line_coords = gdf[gdf.geometry.type == 'LineString'].geometry.apply(lambda geom: list(geom.coords))
    line1_erosion=np.array(line_coords[0])
    line2_erosion=np.array(line_coords[1])
except:
    useShapeFiles = False
    print("Could not load shapefiles for buried valleys.")


 # %%
 #if useMergedPrior:
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

# Compute hypothesis probabilities using the new function
# This computes P(valley|data) and P(standard|data) from the evidence values in the posterior files
P_from_EV, Mode_from_EV, ENT_from_EV = ig.compute_hypothesis_probability([f_post_h5_list[0], f_post_h5_list[1]], showInfo=1)
P_valley = P_from_EV[:, 0]  # Probability of valley hypothesis
P_standard = P_from_EV[:, 1]  # Probability of standard hypothesis

# Load the probabilities from the merged prior approach (for comparison)
with h5py.File(f_post_h5_list[2],'r') as f:
    P = f['/M3/P'][:]
P_valley_check = P[:,:,0]

# Alternative manual calculation (should give identical results):
# with h5py.File(f_post_h5_list[0],'r') as f:
#     EV1 = f['/EV'][:]
# with h5py.File(f_post_h5_list[1],'r') as f:
#     EV2 = f['/EV'][:]
# log_sum = np.logaddexp(EV1, EV2)
# P_valley = np.exp(EV1 - log_sum)
#P_valley = EV1/(EV1+EV2)
# use cmap red white blue
cmap_valley = plt.get_cmap('RdBu_r')
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, c=P_valley, s=1, cmap=cmap_valley, vmin=0, vmax=1);plt.colorbar(label='P(Valley)');plt.axis('equal')
plt.grid()
plt.savefig('DAUGAARD_Pvalley_EV_N%d_No%d_aT%d_l%d.png' % (N_use,inflateNoise,autoT,useLogData), dpi=300)
if useShapeFiles:
    plt.plot(line1[:,0],line1[:,1],'y-',linewidth=6, alpha=0.4)
    plt.plot(line1[:,0],line1[:,1],'k--',linewidth=2)
    plt.plot(line2[:,0],line2[:,1],'y-',linewidth=6, alpha=0.4)
    plt.plot(line2[:,0],line2[:,1],'k--',linewidth=2)
    #plt.plot(line1_erosion[:,0],line1_erosion[:,1],'c-',linewidth=6)
    #plt.plot(line2_erosion[:,0],line2_erosion[:,1],'r-',linewidth=6)
    plt.savefig('DAUGAARD_Pvalley_EV_N%d_No%d_aT%d_l%d_shape.png' % (N_use,inflateNoise,autoT,useLogData), dpi=300)

plt.figure(figsize=(8, 6))
#plt.scatter(X, Y, c=P_valley_check[:,0], s=1, cmap=cmap_valley, vmin=.45, vmax=.55);plt.colorbar(label='P(Valley)');plt.axis('equal')
plt.scatter(X, Y, c=P_valley_check[:,0], s=1, cmap=cmap_valley, vmin=0, vmax=1);plt.colorbar(label='P(Valley)');plt.axis('equal')
plt.grid()
plt.savefig('DAUGAARD_Pvalley_MIXTURE_N%d_No%d_aT%d_l%d.png' % (N_use,inflateNoise,autoT,useLogData), dpi=300)
if useShapeFiles:
    plt.plot(line1[:,0],line1[:,1],'y-',linewidth=6, alpha=0.4)
    plt.plot(line1[:,0],line1[:,1],'k--',linewidth=2)
    plt.plot(line2[:,0],line2[:,1],'y-',linewidth=6, alpha=0.4)
    plt.plot(line2[:,0],line2[:,1],'k--',linewidth=2)
    #plt.plot(line1_erosion[:,0],line1_erosion[:,1],'c-',linewidth=6)
    #plt.plot(line2_erosion[:,0],line2_erosion[:,1],'r-',linewidth=6)
plt.savefig('DAUGAARD_Pvalley_MIXTURE_N%d_No%d_aT%d_l%d_shape.png' % (N_use,inflateNoise,autoT,useLogData), dpi=300)




plt.figure(figsize=(4, 4))
plt.plot(P_valley.flatten(),P_valley_check[:,0].flatten(),'k.', markersize=.1);plt.xlabel('P(Valley) from EV');plt.ylabel('P(Valley) from M3');plt.axis('equal' )
plt.grid(True, which='both', alpha=0.3)
plt.gca().set_xticks(np.arange(0, 1.1, 0.1))
plt.gca().set_yticks(np.arange(0, 1.1, 0.1))
plt.grid(True, which='major', alpha=0.7)
plt.savefig('DAUGAARD_Pvalley_compare_N%d_No%d_aT%d_l%d.png' % (N_use,inflateNoise,autoT,useLogData), dpi=300)


# %%
plLevel = 2    
for i_post in range(len(f_post_h5_list)):
    f_post_h5 = f_post_h5_list[i_post]

    if plLevel>0:
        #ig.plot_profile(f_post_h5, ii=id_line, gap_threshold=50, xaxis='y', cmap=cmap, clim=clim,hardcopy=hardcopy)
        ig.plot_profile(f_post_h5, ii=id_line, gap_threshold=50, xaxis='y', cmap=cmap, clim=clim,hardcopy=hardcopy, im=1, alpha=.9, std_min = 0.6, std_max=0.63)
        ig.plot_profile(f_post_h5, ii=id_line, gap_threshold=50, xaxis='y', cmap=cmap, clim=clim,hardcopy=hardcopy, im=2, alpha=.8, entropy_min = 0.5)
        
    if plLevel>1:

        ig.plot_data_prior_post(f_post_h5, i_plot=i_plot_1, hardcopy=hardcopy)
        ig.plot_data_prior_post(f_post_h5, i_plot=i_plot_2, hardcopy=hardcopy)

        ig.plot_data_prior_post(f_post_h5, i_plot=100, hardcopy=hardcopy)

        ig.plot_T_EV(f_post_h5, pl='CHI2', hardcopy=hardcopy)
        ig.plot_T_EV(f_post_h5, pl='T', hardcopy=hardcopy)


        ig.plot_feature_2d(f_post_h5,im=1,iz=15, key='Mean', uselog=1, s=10,hardcopy=hardcopy)
        plt.show()
        ig.plot_feature_2d(f_post_h5,im=2,iz=15, key='Mode', uselog=0, s=10,hardcopy=hardcopy)
        plt.show()

        # Plot prior and post model parameter stats
        ig.plot_post_stats(f_post_h5, i_plot = i_plot_1)
        ig.plot_post_stats(f_post_h5, i_plot = i_plot_2)

    try:
        ig.plot_feature_2d(f_post_h5,im=3, key='Mode', uselog=1, s=10, cmap='jet', hardcopy=hardcopy)
    except:
        pass


# %% [markdown]
# ## Effect of size of prior data set

# %%
f_post_h5_N_list = []

if doEffectSize:
    N_use_arr = [1000,10000,100000,1000000,2000000]
    #N_use_arr = [1000,10000,10000]

    for N_use in N_use_arr:
        for i_prior in range(len(f_prior_data_h5_list)):

            print('# ------')
            print('TESTING N_use=%d, f_prior_data_h5=%s' % (N_use, f_prior_data_h5_list[i_prior]))
            print('# ------')

            f_prior_data_h5= f_prior_data_h5_list[i_prior]
            # Get filename without extension
            fileparts = os.path.splitext(f_prior_data_h5)
            f_post_h5 = 'post_%s_Nuse%d_inflateNoise%d.h5' % (fileparts[0], N_use,inflateNoise)

            f_post_h5 = ig.integrate_rejection(f_prior_data_h5, 
                                            f_data_h5, 
                                            f_post_h5, 
                                            N_use = N_use, 
                                            showInfo=1, 
                                            parallel=True, 
                                            updatePostStat=True)
            
            f_post_h5_N_list.append(f_post_h5)
        
            plLevel = 1
            if plLevel>0:
                ig.plot_profile(f_post_h5, ii=id_line, gap_threshold=50, xaxis='y', hardcopy=hardcopy)




# %% [markdown]
# ## Effect of size of prior data set

# %%
f_post_h5_T_list = [] 
if doTbase:
    T_base_arr = [1,2,10,20,100]
    N_use = N_use_org
    for T_base in T_base_arr:
        for i_prior in range(len(f_prior_data_h5_list)):

            print('# ------')
            print('TESTING T_base=%d, f_prior_data_h5=%s' % (T_base, f_prior_data_h5_list[i_prior]))
            print('# ------')
                            
            f_prior_data_h5= f_prior_data_h5_list[i_prior]
            # Get filename without extension
            fileparts = os.path.splitext(f_prior_data_h5)
            f_post_h5 = 'post_%s_Nuse%d_T%d_inflateNoise%d.h5' % (fileparts[0], N_use,T_base,inflateNoise)

            f_post_h5 = ig.integrate_rejection(f_prior_data_h5, 
                                            f_data_h5, 
                                            f_post_h5, 
                                            T_base = T_base,
                                            N_use = 2*N_use,
                                            autoT=False,
                                            showInfo=1, 
                                            parallel=True, 
                                            updatePostStat=True)
            
            f_post_h5_T_list.append(f_post_h5)

            plLevel = 1
            if plLevel>0:
                ig.plot_profile(f_post_h5, ii=id_line, gap_threshold=50, xaxis='y', cmap=cmap, clim=clim, hardcopy=hardcopy)



# %% 
f_post_h5_all_list = f_post_h5_list + f_post_h5_N_list + f_post_h5_T_list
    

# %%
# concatenate f_post_h5_list, f_post_h5_N_list, f_post_h5_T_list

if doPlotAll:
    plLevel=2
    
    #f_post_h5_all_list = f_post_h5_T_list

    for i_post in range(len(f_post_h5_all_list)):
        f_post_h5 = f_post_h5_all_list[i_post]
        
        with h5py.File(f_post_h5,'r') as f:
            f_prior_h5 = f.attrs['f5_prior']

        ig.plot_profile(f_post_h5, ii=id_line, gap_threshold=50, xaxis='y', cmap=cmap, clim=clim,hardcopy=hardcopy)

        ig.plot_profile(f_post_h5, ii=id_line, gap_threshold=50, xaxis='y', 
                    cmap=cmap, 
                    clim=clim,
                    hardcopy=hardcopy,
                    panels = ['Median','Mode'],
                    alpha=0)
        
        ig.plot_profile(f_post_h5, ii=id_line, gap_threshold=50, xaxis='y', 
                    cmap=cmap, 
                    clim=clim,
                    hardcopy=hardcopy,
                    panels = ['std','entropy'],
                    alpha=0)

        ig.plot_T_EV(f_post_h5, pl='T', hardcopy=hardcopy)
            
        if plLevel>0:
            ig.plot_data_prior_post(f_post_h5, i_plot=i_plot_1, hardcopy=hardcopy, title='a) P1')
            ig.plot_data_prior_post(f_post_h5, i_plot=i_plot_2, hardcopy=hardcopy, title='b) P2')
    
        if plLevel>1:
            ig.plot_T_EV(f_post_h5, pl='CHI2', hardcopy=hardcopy)
            ig.plot_T_EV(f_post_h5, pl='T', hardcopy=hardcopy)
            ig.plot_T_EV(f_post_h5, pl='EV', hardcopy=hardcopy)
            ig.plot_T_EV(f_post_h5, pl='ND', hardcopy=hardcopy)
            ig.plot_T_EV(f_post_h5, pl='N_UNIQUE', hardcopy=hardcopy, N_UNIQUE_min=0, N_UNIQUE_max=nr)

            ig.plot_feature_2d(f_post_h5,im=1,iz=15, key='LogMean', uselog=1, hardcopy=hardcopy, clim=clim, cmap=cmap, title = 'log(Mean)' )
            plt.show()
            ig.plot_feature_2d(f_post_h5,im=1,iz=15, key='Median', uselog=1, hardcopy=hardcopy, clim=clim, cmap=cmap, title= 'Median' )
            plt.show()
            ig.plot_feature_2d(f_post_h5,im=2,iz=15, key='Mode', uselog=0, hardcopy=hardcopy, title='Mode' )
            plt.show()
            ig.plot_feature_2d(f_post_h5,im=1,iz=5, key='Median', uselog=1, hardcopy=hardcopy, clim=clim, cmap=cmap, title= 'Median' )
            plt.show()
            ig.plot_feature_2d(f_post_h5,im=2,iz=5, key='Mode', uselog=0, hardcopy=hardcopy, title='Mode' )
            plt.show()

            try:
                ig.plot_feature_2d(f_post_h5,im=3, key='Mode', uselog=1, cmap='jet', hardcopy=hardcopy)
            except:
                pass

            # Plot prior and post model parameter stats
            ig.plot_prior_stats(f_prior_h5)
            ig.plot_post_stats(f_post_h5, i_plot = i_plot_1)
            ig.plot_post_stats(f_post_h5, i_plot = i_plot_2)
            

# %% TEST 
doTest=True
if doTest:
    for i_post in range(len(f_post_h5_all_list)):
    #for i_post in range(1):
        f_post_h5 = f_post_h5_all_list[i_post]

        # fname is ifle name wihtout extension
        fname = os.path.splitext(os.path.basename(f_post_h5))[0]
        #ig.plot_T_EV(f_post_h5, pl='N_UNIQUE', hardcopy=hardcopy, N_UNIQUE_min=1, N_UNIQUE_max=1000)

        # MAKE A SCATTER PLOT OF f_post:/T versus f_post:/CHI2 with KDE background
        with h5py.File(f_post_h5,'r') as f:
            T = f['/T'][:]
            CHI2 = f['/CHI2'][:]
            N_UNIQUE = f['/N_UNIQUE'][:]

        # Handle case where CHI2 might be 2D (multiple data types)
        if len(CHI2.shape) == 2:
            # If multiple data types, sum across them for visualization
            CHI2 = np.nansum(CHI2, axis=1)

        # Flatten arrays to ensure 1D
        T = T.flatten()
        CHI2 = CHI2.flatten()
        N_UNIQUE = N_UNIQUE.flatten()

        # Remove NaN values for KDE computation
        mask = ~(np.isnan(T) | np.isnan(CHI2))
        T_clean = T[mask]
        CHI2_clean = CHI2[mask]
        N_UNIQUE_clean = N_UNIQUE[mask]

        # Compute 2D KDE for background
        from scipy.stats import gaussian_kde

        # Check if data has sufficient variance for KDE
        CHI2_std = np.std(CHI2_clean) if len(CHI2_clean) > 0 else 0
        log_T_std = np.std(np.log10(T_clean)) if len(T_clean) > 0 else 0
        has_variance = (CHI2_std > 1e-10) and (log_T_std > 1e-10)

        # Only compute KDE if we have enough data points and variance
        if len(T_clean) > 100 and has_variance:
            # Use log(T) for KDE computation since we're plotting with log scale
            log_T_clean = np.log10(T_clean)

            # Create KDE
            values = np.vstack([CHI2_clean, log_T_clean])
            try:
                kernel = gaussian_kde(values)

                # Create grid for evaluation
                CHI2_grid = np.linspace(np.nanmin(CHI2_clean), np.nanmax(CHI2_clean), 100)
                log_T_grid = np.linspace(np.nanmin(log_T_clean), np.nanmax(log_T_clean), 100)
                CHI2_mesh, log_T_mesh = np.meshgrid(CHI2_grid, log_T_grid)
                positions = np.vstack([CHI2_mesh.ravel(), log_T_mesh.ravel()])
                density = kernel(positions).reshape(CHI2_mesh.shape)

                # Convert log_T back to linear scale for plotting
                T_mesh = 10**log_T_mesh

                # Plot
                plt.figure(figsize=(4, 4))

                # Plot KDE as filled contours in background
                contour = plt.contourf(CHI2_mesh, T_mesh, density, levels=15, cmap='hot_r')

                # Overlay scatter plot
                plt.plot(CHI2[::10], T[::10], 'k.', alpha=1, markersize=1)

                # Use log on the yaxis
                plt.yscale('log')
                plt.xscale('log')
                plt.xlabel('CHI2')
                plt.ylabel('T')
                plt.colorbar(contour, label='Density')
                plt.grid(alpha=0.3)
                # Plot a black line at x=1 along the y axis
                plt.axvline(x=1, color='k', linestyle='--', linewidth=1)
                # set axis to 0,5, 1, 20
                plt.xlim(0, 15)
                plt.ylim(1, 101)
                plt.savefig('scatter_T_CHI2_%s.png' % fname, dpi=300)
                plt.show()
            except np.linalg.LinAlgError:
                print(f"Warning: KDE failed for {fname} (singular covariance matrix - insufficient variance in data)")
                # Fallback to simple scatter
                plt.figure(figsize=(4, 4))
                plt.plot(CHI2[::10], T[::10], 'k.', alpha=1, markersize=1)
                plt.yscale('log')
                plt.xscale('log')
                plt.xlabel('CHI2')
                plt.ylabel('T')
                plt.grid(alpha=0.3)
                plt.axvline(x=1, color='k', linestyle='--', linewidth=1)
                plt.xlim(0, 15)
                plt.ylim(1, 101)
                plt.title('Scatter (KDE unavailable)')
                plt.savefig('scatter_T_CHI2_%s.png' % fname, dpi=300)
                plt.show()
        else:
            # Fallback to regular scatter plot if insufficient data or variance
            if len(T_clean) <= 100:
                reason = f"insufficient samples (n={len(T_clean)})"
            else:
                reason = "insufficient variance in data"

            print(f"Warning: Skipping KDE for {fname} ({reason})")

            plt.figure(figsize=(4, 4))
            plt.plot(CHI2[::10], T[::10], 'k.', alpha=1, markersize=1)
            plt.yscale('log')
            plt.xscale('log')
            plt.xlabel('CHI2')
            plt.ylabel('T')
            plt.grid(alpha=0.3)
            plt.axvline(x=1, color='k', linestyle='--', linewidth=1)
            plt.xlim(0, 15)
            plt.ylim(1, 101)
            plt.title(f'Scatter (KDE: {reason})')
            plt.savefig('scatter_T_CHI2_%s.png' % fname, dpi=300)
            plt.show()




# %%
if doPlotAll:
    f_txt = 'f_post_h5_all_list_%s_Nuse%d_inflateNoise%d.txt' % (fileparts[0], N_use,inflateNoise)

    with open(f_txt, 'w') as f:
        for item in f_post_h5_all_list:
            f.write("%s\n" % item)

# %%
doReadList = True
if doReadList:
    #f_txt = 'f_post_h5_all_list_%s_Nuse%d_inflateNoise%d.txt' % (fileparts[0], N_use,inflateNoise)

    f_txt_list=[]
    f_txt_list.append('f_post_h5_all_list_PRIOR_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nuse1000000_inflateNoise1.txt')
    f_txt_list.append('f_post_h5_all_list_PRIOR_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nuse1000000_inflateNoise2.txt')
    f_txt_list.append('f_post_h5_all_list_PRIOR_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nuse1000000_inflateNoise4.txt')


    f_post_h5_all_list = []
    for f_txt in f_txt_list:
        with open(f_txt, 'r') as f:
            for line in f:
                f_post_h5_all_list.append(line.strip())



# %%
# TEST INVERSION
# #################################################################################################

# %%

#f_prior_data_h5_list = ['daugaard_valley_new_N1000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5',
# 'daugaard_standard_new_N1000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5',
# 'daugaard_merged.h5']

if doTestInversion:
    D_D = []
    D_idx = []
    D_M = []

    for i in np.arange(len(f_prior_data_h5_list)):
        D_t, idx = ig.load_prior_data(f_prior_data_h5_list[i], Randomize=True, showInfo=1)
        D_D.append(D_t)
        D_idx.append(idx)
        M_t, idx=ig.load_prior_model(f_prior_data_h5_list[i])
        D_M.append(M_t)

    DATA = ig.load_data(f_data_h5, id_arr = [1])
        
    # The data point to invert
    ip = 1000 # Phyp=30,60
    ip = 100 # Phyp=3,97

    N=D_D[0][0].shape[0]
    ipd=0    
    print('data ipd=%d in first [%g,%g]' %  (ipd,D_D[0][0][ipd][0],D_D[2][0][ipd][0]))
    print('data ipd=%d in 2nd dset [%g,%g]' % (ipd,D_D[1][0][ipd][0],D_D[2][0][ipd+N][0]))

    d_obs = DATA['d_obs'][0][ip]
    d_std = DATA['d_std'][0][ip]

    # now invert data sat id
    autoT=False
    T_base = 1
    nr=4000
    EV_est=[]
    EV_rej=[]
    i_use_man=[]
    i_use_rej=[]
    for i in np.arange(len(D_D)):
    #for i in np.arange(1):
        OUT = ig.integrate_rejection_range(D=D_D[i], 
                                        DATA = DATA,
                                        idx = idx,                                                                   
                                        autoT=autoT,
                                        T_base = T_base,
                                        ip_range = [ip],
                                        useRandomData=True,
                                        nr=nr,
                                        showInfo=1)
        i_use, T, EV, EV_post, EV_post_mean, LOGL_mean, N_UNIQUE, ip_range = OUT
        i_use_rej.append(i_use.flatten())
        
        # compute logL manually
        Ns = len(D_D[i][0])
        logL_manual = np.zeros(Ns)
        #print("Computing logL manually for %d samples" % Ns)
        for j in range(Ns):
            dd = D_D[i][0][j]-d_obs
            logL_manual[j]   = -0.5 * np.nansum((dd / d_std)**2)

        # compute logL using likelihood function
        logL = ig.likelihood_gaussian_diagonal(D_D[i][0],d_obs,d_std)
        P_acc = np.exp(logL-logL.max())
        r = np.random.rand(len(logL))
        i_use_temp = np.where(r < P_acc)[0]
        i_use_man.append(i_use_temp)
        #p=P_acc/np.sum(P_acc)
        #i_use_temp2 = np.random.choice(len(P_acc), nr, p=p)
        #i_use_man.append(i_use_temp2)

        #print('logL manual = ')
        #print(logL[0:3])

        EV_est_single=np.log(np.mean(np.exp(logL)))
        EV_est.append(EV_est_single.flatten())
        EV_rej.append(EV.flatten())

        print("logL (likelihood)=%g, logL(manual)=%g" % (logL[0],logL_manual[0]))
        for k in np.arange(i):
            print("EV(rej)=%g, EV(mix)=%g" % (EV_rej[k],EV_est[k]))

        doPlot=True
        if doPlot:
            plt.figure(figsize=(4,3))
            plt.semilogy(D_D[i][0][i_use_rej[i],:].T,'g-',linewidth=1,alpha=0.3)
            plt.semilogy(D_D[i][0][i_use_man[i],:].T,'k-',linewidth=.5,alpha=0.3)
            plt.semilogy(d_obs,'r:')
            plt.title(f_prior_data_h5_list[i])
            plt.show()  
    

# %%%
#a ** a
# %%


if doTestInversion:

    iHYP = D_M[2][2]
    i_post_type_man = iHYP[i_use_man[2]]
    i_post_type_rej = iHYP[i_use_rej[2]]

    N_cat_rej = np.array([np.sum(i_post_type_rej==i) for i in [1,2]])
    P_rej = N_cat_rej/np.sum(N_cat_rej)
    N_cat_man = np.array([np.sum(i_post_type_man==i) for i in [1,2]])
    P_man = N_cat_man/np.sum(N_cat_man)
    print("P from MIXING (rejection): ", P_rej.flatten())
    print("P from MIXING (manual): ", P_man.flatten())
    print("Difference  = ", P_rej - P_man)

    P_from_EV_man = np.exp(EV_est)/(np.exp(EV_est[0])+np.exp(EV_est[1]))
    P_from_EV_rej = np.exp(EV_rej)/(np.exp(EV_rej[0])+np.exp(EV_rej[1]))
    print("P from EV (manual): ", P_from_EV_man.flatten())
    print("P from EV (rejection): ", P_from_EV_rej.flatten())

# %%
# DO we get the same if we call integrate_rejection_range() and integrate_rejection()

if doTestInversion:
    Nd=len(DATA['d_obs'][0])
    ip_range = np.arange(Nd)
    #ip_range = np.arange(0, Nd, 100)
    i_use_compare=[]
    EV_compare=[]

    for i in np.arange(len(D_D)):
    #for i in np.arange(1):
        OUT = ig.integrate_rejection_range(D=D_D[i], 
                                        DATA = DATA,
                                        idx = idx,                                                                   
                                        autoT=autoT,
                                        T_base = T_base,
                                        ip_range = ip_range,
                                        # useRandomData=False,
                                        nr=nr,
                                        showInfo=1)
        i_use, T, EV, EV_post, EV_post_mean, LOGL_mean, N_UNIQUE, ip_range_alt = OUT
        i_use_compare.append(i_use)
        EV_cplt.plot(P_compare_ev[:,0],P_valley_check[ip_range,0].flatten(),'g.', markersize=.1);plt.xlabel('P(Valley) from EV (_range)');plt.ylabel('P(Valley) from M3');plt.axis('equal' )
    ompare.append(EV.flatten())

# %%
if doTestInversion:
    Np = len(ip_range)
    P_compare_mix = np.zeros((Np,2))
    P_compare_ev = np.zeros((Np,2))

    for i in np.arange(Np):
        i_post_type_compare = iHYP[i_use_compare[2][i]]
        Nc = np.array([np.sum(i_post_type_compare==i) for i in [1,2]])
        P_compare_mix[i,:] = Nc/np.sum(Nc)

        EV=np.array([EV_compare[0][i],EV_compare[1][i]])
        P_compare_ev[i,:] = np.exp(EV)/(np.exp(EV[0])+np.exp(EV[1]))
        
    plt.figure(figsize=(14,4))
    plt.plot(P_compare_mix[:,0],'k-', label='P(Valley) from Mixing')
    plt.plot(P_compare_ev[:,0],'r-', label='P(Valley) from EV')
    plt.plot(P_valley_check[ip_range,0],'b--', label='P(Valley) from M3')
    plt.plot(P_valley[ip_range],'g--', label='P(Valley) from EV calc')
    plt.grid(True, which='both', alpha=0.3)
    plt.xlabel('P(Valley) from Mixing')
    plt.ylabel('P(Valley) from EV')
    plt.ylim(0,1)
    plt.legend()

    plt.figure(figsize=(8,8))
    plt.subplot(2,2,1)
    plt.scatter(X[ip_range], Y[ip_range], c=P_compare_mix[:,0], s=1, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar()
    plt.title('P(Valley) from Mixing')
    plt.axis('equal')
    plt.subplot(2,2,2)
    plt.scatter(X[ip_range], Y[ip_range], c=P_compare_ev[:,0], s=1, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar()
    plt.title('P(Valley) from EV')  
    plt.axis('equal')
    try:
        plt.subplot(2,2,3)
        plt.scatter(X, Y, c=P_valley, s=1, cmap=cmap, vmin=0, vmax=1);plt.colorbar(label='P(Valley)');plt.axis('equal')
        plt.subplot(2,2,4)
        plt.scatter(X, Y, c=P_valley_check[:,0], s=1, cmap=cmap, vmin=0, vmax=1);plt.colorbar(label='P(Valley)');plt.axis('equal')
    except:
        pass


    plt.figure(figsize=(4, 4))
    plt.plot(P_valley.flatten(),P_valley_check[:,0].flatten(),'k.', markersize=.1);plt.xlabel('P(Valley) from EV');plt.ylabel('P(Valley) from M3');plt.axis('equal' )
    plt.plot(P_compare_ev[:,0],P_compare_mix[:,0],'r.', markersize=.5);plt.xlabel('P(Valley) from EV');plt.ylabel('P(Valley) from Mixing');plt.axis('equal' )
    plt.grid(True, which='both', alpha=0.3)

    plt.figure(figsize=(4, 4))
    #plt.plot(P_compare_ev[:,0],P_valley_check[ip_range,0].flatten(),'g.', markersize=.1);plt.xlabel('P(Valley) from EV (_range)');plt.ylabel('P(Valley) from M3 - integrate_rejection()');plt.axis('equal' )
    plt.plot(P_compare_ev[:,0],P_valley.flatten()[ip_range].flatten(),'g.', markersize=.1);plt.xlabel('P(Valley) from EV (_range)');plt.ylabel('P(Valley) from EV - integrate_rejection()');plt.axis('equal' )
    plt.plot(P_compare_ev[:,0],P_compare_mix[:,0],'r.', markersize=.5);#plt.xlabel('P(Valley) from EV');plt.ylabel('P(Valley) from Mixing');plt.axis('equal - integrate_rejection_range()' )
    plt.grid(True, which='both', alpha=0.3)


    plt.figure(figsize=(4, 4))
    plt.plot(EV_compare[0],EV1,'.')
    plt.xlabel('EV [] from integrate_rejection_range()');
    plt.ylabel('EV [] from integrate_rejection()');plt.axis('equal' )
    plt.title('EV1 (Valley) - %s' % f_prior_data_h5_list[0][0:19])
    plt.grid(True, which='both', alpha=0.3)


#%%  TEST
f_post_h5 = 'post_daugaard_merged_N2000000_Nuse1000000_T1_inflateNoise2.h5'
ig.plot_feature_2d(f_post_h5, im=1, iz=10)
plt.show()
ig.plot_feature_2d(f_post_h5, im=2, iz=10)
plt.show()

# %% 
ig.plot_profile(f_post_h5, ii=id_line, gap_threshold=50, xaxis='y')


# %% 
cmap, clim = ig.get_colormap_and_limits('evidence', custom_clim=[0, 1])
f_post_h5='post_daugaard_merged_N2000000_Nuse1000000_T1_inflateNoise2.h5'
ele=-20
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)
nd = len(X)

class_id, class_name = ig.get_discrete_classes(f_post_h5, im=2)
nclass = len(class_id)

#E = ig.extract_feature_at_elevation(f_post_h5, elevation=ele)
POST_MODE = ig.extract_feature_at_elevation(f_post_h5, im=2, elevation=ele, key='Mode')
POST_ENTROPY = ig.extract_feature_at_elevation(f_post_h5, im=2, elevation=ele, key='Entropy')


#%% Make a movie of class probabilities at different elevations

POST_P_CLASS = np.zeros( (nd, nclass) )
i=0
for ele in np.arange(80,-51,-1):
    i=i+1 
    for ic in range(nclass):
        POST_P_CLASS[:, ic] = ig.extract_feature_at_elevation(f_post_h5, im=2, elevation=ele, key='P', ic=ic)

    plt.figure()
    for ic in range(nclass):
        plt.subplot(3,3,ic+1)
        plt.plot(X, Y, 'k.', markersize=.01)
        plt.scatter(X, Y, c=POST_P_CLASS[:, ic], s=.2, cmap='magma_r', vmin=0, vmax=1)
        #plt.colorbar(label='P(Class %d: %s) at %d m elevation' % (class_id[ic], class_name[ic], ele));plt.axis('equal')
        #plt.colorbar()
        # Remove axis labels for clarity
        plt.xticks([])
        plt.yticks([])
        plt.title('P(Class %d: %s)' % (class_id[ic], class_name[ic]), fontsize=5)
        
    plt.suptitle('Class probabilities at %3.1f m elevation' % ele)
    plt.axis('equal')
    plt.savefig('PROB_CLASS_%04d.png' % i, dpi=300)

# %% 
# ffmpeg -framerate 4 -pattern_type glob -i "PROB_CLASS_*.png" -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p output.mp4

# ffmpeg -f concat -safe 0 -i files.txt -c:v libx264 -pix_fmt yuv420p output.mp4



#%% Example use og  ig.plot_feature_2d() 

# %%
# Plot Median resistivity ad 10m Depth.
ig.plot_feature_2d(f_post_h5, im=1, iz=10)
plt.show()

# Plot Median resistivity at elevation -10 Depth.
ig.plot_feature_2d(f_post_h5, im=1, elevation=-10)
plt.show()

# Plot Mean, Mode, Std
# Plot Media resistivity at elevation -10 Depth.
ig.plot_feature_2d(f_post_h5, im=1, elevation=-10, key='Mean')
plt.show()
# Plot Median resistivity at elevation -10 Depth.
ig.plot_feature_2d(f_post_h5, im=1, elevation=-10, key='Median')
plt.show()
# Plot Median resistivity at elevation -10 Depth.
ig.plot_feature_2d(f_post_h5, im=1, elevation=-10, key='Std', uselog=0)
plt.show()



# %%
# Plot Mode lithology at 10m Depth.
ig.plot_feature_2d(f_post_h5, im=2, iz=10)
plt.show()

# Plot Mode lithology at elevation -10 Depth.
ig.plot_feature_2d(f_post_h5, im=2, elevation=-10)
plt.show()

# Plot Mode lithology at elevation -10 Depth.
ig.plot_feature_2d(f_post_h5, im=2, elevation=-10, key='Mode')
plt.show()

# Plot Mode lithology at elevation -10 Depth.
ig.plot_feature_2d(f_post_h5, im=2, elevation=-10, key='Entropy', clim=[0,1])
plt.show()

# plot class probability of ic=7
ig.plot_feature_2d(f_post_h5, im=2, iz=80, key='P', ic=7, uselog=0, clim=[0,1])

# Plot class probability
ig.plot_feature_2d(f_post_h5, im=2, iz=80, key='P', ic=7, uselog=0, clim=[0,1])
plt.show()

ig.plot_feature_2d(f_post_h5, im=2, elevation=-30, key='P', ic=6, uselog=0, clim=[0,1], cmap='magma_r')
plt.show()

# same as above, but the position of data points are plotted as well
ig.plot_feature_2d(f_post_h5, im=2, elevation=-30, key='P', ic=5, uselog=0, clim=[0,1], cmap='magma_r', plotPoints=True)
plt.show()







# %% 

#POST_P0 = ig.extract_feature_at_elevation(f_post_h5, im=2, elevation=ele, key='P', ic=0)
#POST_P1 = ig.extract_feature_at_elevation(f_post_h5, im=2, elevation=ele, key='P', ic=1)#
#POST_P2 = ig.extract_feature_at_elevation(f_post_h5, im=2, elevation=ele, key='P', ic=2)

#P1 = ig.extract_feature_at_elevation(f_post_h5, im=2, ic=1, elevation=ele)
#P1 = ig.extract_feature_at_elevation(f_post_h5, im=2, ic=1, elevation=ele)

plt.subplot(2,2,1)
plt.plot(X, Y, 'k.', markersize=.1)
plt.scatter(X, Y, c=POST_MODE, s=.2, cmap='jet')
plt.colorbar(label='Feature at 10 m elevation');plt.axis('equal')

plt.subplot(2,2,2)
plt.plot(X, Y, 'k.', markersize=.1)
plt.scatter(X, Y, c=POST_ENTROPY, s=.2, cmap='jet', vmin=0, vmax=1)
plt.colorbar(label='Feature at 10 m elevation');plt.axis('equal')


#%% 
ig.plot_feature_2d(f_post_h5, key='Mode', im=2)
plt.show()
ig.plot_feature_2d(f_post_h5, key='Mode', im=2, elevation=0)
plt.show()
ig.plot_feature_2d(f_post_h5, elevation = 0, key='P', ic=1, im=2)
plt.show()
ig.plot_feature_2d(f_post_h5, elevation = 0, key='P', ic=2, im=2)
plt.show()
