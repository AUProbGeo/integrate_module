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

from integrate import integrate_query

hardcopy = True
import matplotlib.pyplot as plt
import numpy as np
import copy

# %%
N = 1_000_000 # Number of prior model realizations to generate (this is just for testing, use a larger number for better results)
N = 1_000 # Smaller N for testing

showInfo = -1 # Determines how much nfo to print to screen

# %%
case = 'SOENDER_FELDING'
files = ig.get_case_data(case=case, showInfo = 1)


# %% [markdown]
# ## A : the PRIOR model(s) :
# First a prior model needs to be defined, create, or loaded

# %%

useGeneric = False 
if useGeneric:  
    f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='chi2', NLAY_deg=3, RHO_min=1, RHO_max=3000, f_prior_h5='PRIOR_N%d.h5' % N, 
                                    showInfo=1)
else:
    usePriorGenerator = False
    if usePriorGenerator:
        # use geoprior1        
        file_xlsx = 'Sddr_Felding_prior_standard.xlsx'
        f_prior_h5, flags  = geoprior1d(file_xlsx, Nreals=N, dz=1, dmax =90, output_file='%s_prior_N%d.h5' % (fname, N))
    else:
        f_prior_h5 = 'Sdr_Felding_prior_210526_N1000000_dmax90_20260521_1616.h5'
        #f_prior_h5 = files[-1]


ig.plot_prior_stats(f_prior_h5, hardcopy=hardcopy)


# %% [markdown]
# ## B: Setup the data 


# %% [markdown]
# ###  B1: setup tTEM fdata
# We need to convert XYZ files and GEX files to HDF5 files in the format expected by INTEGRATE. 
# The GEX files contain metadata about the survey geometry and which data channels are available, 
# while the XYZ files contain the actual observed data. 
# The function `ig.xyz_to_h5` can be used to convert the XYZ and GEX files into an HDF5 file that can be used as input for INTEGRATE. 
# This function will read the XYZ file, extract the relevant data channels, and save them in an HDF5 file along with the metadata from the GEX file.


# %%
files_xyz = [f for f in files if f.endswith('.xyz')]
files_gex = [f for f in files if f.endswith('.gex')]

# %%
print(files_xyz)
print(files_gex)

# %%
# 
# Read @README_SOENDER_FELDING, and create a an f_data_h5 for each 
# gex filem, using the XYZ files as indicate by the readme file
f_data_sub=[]

# Gex file 1: TX07_20240802_2x4_RC20-39.gex
# Dates: 20240819, 20240820, 20240821, 20240911
file_gex = 'TX07_20240802_2x4_RC20-39.gex'
file_xyz = ['20240819_AVG_export.xyz', '20240820_AVG_export.xyz', '20240821_AVG_export.xyz', '20240911_AVG_export.xyz']
fname = file_gex.split('.')[0]
f_data_sub.append(ig.xyz_to_h5(file_xyz, file_gex, f_data_h5='%s_data.h5' % fname, showInfo=showInfo))

# Gex file 2: TX07_20240802_2x4_RC20-39_eksternGPS.gex
# Dates: 20240911_eksternGPS
file_gex = 'TX07_20240802_2x4_RC20-39_eksternGPS.gex'
file_xyz = ['20240911_eksterngps_AVG_export.xyz']
fname = file_gex.split('.')[0]
f_data_sub.append(ig.xyz_to_h5(file_xyz, file_gex, f_data_h5='%s_data.h5' % fname, showInfo=showInfo))

# Gex file 3: TX07_20240912_2x4_RC20-39_eksterngps.gex
# Dates: 20240924, 20240924_test, 20241007, 20241008
file_gex = 'TX07_20240912_2x4_RC20-39_eksterngps.gex'
file_xyz = ['20240924_AVG_export.xyz', '20240924_test_AVG_export.xyz', '20241007_AVG_export.xyz', '20241008_AVG_export.xyz']
fname = file_gex.split('.')[0]
f_data_sub.append(ig.xyz_to_h5(file_xyz, file_gex, f_data_h5='%s_data.h5' % fname, showInfo=showInfo))

# # Gex file 4: TX07_20241014_2x4_RC20_33_and_57_EksternGPS.gex
# Dates: 20241029
file_gex = 'TX07_20241014_2x4_RC20_33_and_57_EksternGPS.gex'
file_xyz = ['20241029_AVG_export.xyz']
fname = file_gex.split('.')[0]
f_data_sub.append(ig.xyz_to_h5(file_xyz, file_gex, f_data_h5='%s_data.h5' % fname, showInfo=showInfo))

# # Gex file 5: TX07_20241202_2x4_RC20_57_EksternGPS.gex
# Dates: 20241210
file_gex = 'TX07_20241202_2x4_RC20_57_EksternGPS.gex'
file_xyz = ['20241210_AVG_export.xyz']
fname = file_gex.split('.')[0]
f_data_sub.append(ig.xyz_to_h5(file_xyz, file_gex, f_data_h5='%s_data.h5' % fname, showInfo=showInfo))

# Gex file 6: TX07_20241202_2x4_RC20_57.gex
# Dates: 20241210_InternGPS
file_gex = 'TX07_20241202_2x4_RC20_57.gex'
file_xyz = ['20241210_InternGPS_AVG_export.xyz']
fname = file_gex.split('.')[0]
f_data_sub.append(ig.xyz_to_h5(file_xyz, file_gex, f_data_h5='%s_data.h5' % fname, showInfo=showInfo))


# %%
for id in range(len(f_data_sub)):
    f_data_h5 = f_data_sub[id]
    ig.plot_data(f_data_h5, hardcopy=hardcopy, showInfo = -1)
    ig.plot_data_xy(f_data_h5, data_channel=20, cmap='jet')

# %%
# Read the number of d_obs data in f_data_h5 and print to screen
for id in range(len(f_data_sub)):
    f_data_h5 = f_data_sub[id]

    DATA = ig.load_data(f_data_h5, showInfo = -1)
    n_data = DATA['d_obs'][0].shape[0]
    n_gates = DATA['d_obs'][0].shape[1]
    print("id=%d %10d  data points [%2d gates] in %s"   % (id, n_data, n_gates, f_data_h5))

# %%
# Use subset of available data only?
usePartOfData = True
if usePartOfData:
    i_sub_use = [0,2] # only use first subsets
    # use all subsets
    # i_sub_use = list(range(len(f_data_sub)))

    f_data_sub_new = []
    print("Using %d data subsets:" % len(i_sub_use))
    for i in i_sub_use:
        print("-- using data file %s" % f_data_sub[i])
        f_data_sub_new.append(f_data_sub[i])

        ig.plot_data(f_data_sub[i], hardcopy=hardcopy, showInfo = -1)
        ig.plot_data_xy(f_data_sub[i], data_channel=20, cmap='jet')

    f_data_sub = f_data_sub_new


# %%
# Optionally merge all data
mergeData = True
if mergeData:
    f_gex = 'TX07_20240802_2x4_RC20-39.gex'
    f_data_all_h5 = ig.merge_data(f_data_sub, f_gex, f_data_merged_h5='SDR_FEDL_ALL.h5')

    ig.plot_data(f_data_all_h5, hardcopy=hardcopy, showInfo = -1)
    ig.plot_data_xy(f_data_all_h5, data_channel=20, cmap='jet')
    ig.plot_geometry(f_data_all_h5)

    # Now consider only 1 data subset, which is the merged data
    f_data_sub=[] 
    f_data_sub.append(f_data_all_h5)


# %%

X, Y, LINE, ELEVATION = ig.get_geometry(f_data_sub[0])

BHOLES = ig.read_borehole('SdrFelding_boreholes.json', showInfo=1)

# Go trhoug the boreholes. and if elevation is set at -9999 replace it with the elevation from the geometry of the tTEM data at the 
# borehole location. This is just to have a more correct elevation for plotting, and it does not affect the inversion since the elevation is not used in the inversion (the depth intervals are defined relative to the borehole top, which is at depth 0).
for ibh in range(len(BHOLES)):
    d = np.sqrt((X-BHOLES[ibh]['X'])**2 + (Y-BHOLES[ibh]['Y'])**2)
    i_closest = np.argmin(d)
    ELEVATION_close = ELEVATION[i_closest]
    if BHOLES[ibh]['elevation'] == -9999:
        print('** ibh=%2d, elevation from borehole file = %5g, elevation from geometry = %g' % (ibh, BHOLES[ibh]['elevation'], ELEVATION_close))    
        BHOLES[ibh]['elevation'] = ELEVATION_close
    else:
        print('   ibh=%2d, elevation from borehole file = %5g, elevation from geometry = %g' % (ibh, BHOLES[ibh]['elevation'], ELEVATION_close))
    BH = BHOLES[ibh]
    
    #print(f"  {BH['name']:30s}  {len(BH['depth_top'])} intervals  X={BH['X']:.1f}  Y={BH['Y']:.1f}   ELEVATION={BH['elevation']:.1f} m") 


# Plot without prior info – classes labelled by numeric ID, default colours
# plt borholes in sets of 10, 0:10, 10:20, etc. if there are more than 10 boreholes
for i in range(0, len(BHOLES), 10):
    #ig.plot_boreholes(BHOLES[i:i+10], f_prior_h5, hardcopy=hardcopy)
    ig.plot_boreholes(BHOLES[i:i+10], hardcopy=hardcopy)


# %%

# ## Compute the PRIOR data for all data subsets (with unique gex files):
f_prior_sub = []
#N=100
for i in range(len(f_data_sub)):
    file_gex = ig.get_gex_file_from_data(f_data_sub[i])
    f_prior_sub_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, N=N, doMakePriorCopy=True, )
    f_prior_sub.append(f_prior_sub_h5)




# %%
# COPY FROM FREF, FOR WORKSHOP
f_prior_sub = []
f_prior_sub.append('Sdr_Felding_prior_210526_N1000000_dmax90_20260521_1616_TX07_20240802_2x4_RC20-39_Nh280_Nf12.h5')
f_prior_h5 = f_prior_sub[0]

# %%
#
# For each borehole BH in BHOLES:
#   1. Compute prior borehole data (mode class per interval per realization)
#      and save to f_prior_h5  →  returns P_obs and id_prior
#   2. Extrapolate point observations across the survey grid with
#      distance-based weighting  →  d_obs, i_use
#   3. Save the observed borehole data to f_data_h5  →  id_out
import tqdm

im_prior = 2       # lithology model index (M2)
r_data   = 4       # tTEMN data based radius (db/dT) — If this is very high it will have no effect
r_dis    = 300     # fade-out radius (m) — weight approaches zero at this distance

# Compute and save prior + observed data for all boreholes in one step per borehole
#for i in range(len(f_data_sub)):
for i in [0]:
    f_data_h5 = f_data_sub[i]
    file_gex = ig.get_gex_file_from_data(f_data_h5)
    f_prior_h5 = f_prior_sub[i]

    id_borehole_list = []
    for BH in tqdm.tqdm(BHOLES, desc="Processing boreholes"):
#    for BH in [BHOLES[0]]:
        id_prior, id_out = ig.save_borehole_data(
            f_prior_h5, f_data_h5, BH,
            im_prior=im_prior, r_data=r_data, r_dis=r_dis,
            doPlot=True,
            showInfo=11)
        id_borehole_list.append(id_out)




# %% [markdown]
# ### INVERSION
# The data is now ready for inversion with the rejection sampler.
#
# On total we have 3 data types (one tTEM and two WellLog). They can be all jointly inverted (the default) or one can select which data types to ínver using `id_use`
#
#     id_use = [1] # tTEM 
#     id_use = [2] # Well 1
#     id_use = [3] # Well 2
#     id_use = [2,3] # Wells 1,2
#     id_use = [1,2,3] # tTEM, Wells 1,2 (the default if id_use is not set)
#     id_use = [1,2,3,4,5,6,7,8,9,10,11,12,13] # tTEM, and all 12 borehole data channels (the default if id_use is not set)
#

# %%
# This prt of the can be rerun using different selection of data types without rerunning the abobe parts
nr=1000
T_N_above=50
T_P_acc_level=0.2 
autoT = 1 # We need minium of T_N_above realizations with an acceptance probability above T_P_acc_level
id_use_arr = []
id_use_arr.append([1]) # tTEM 
id_use_arr.append(list(range(2, len(BHOLES)+2))) # ONLY BORREHOLES
id_use_arr.append(list(range(1, len(BHOLES)+2))) # ALL data

#id_use = id_use_arr[0]
#id_use = id_use_arr[1]
#id_use = id_use_arr[-1]

f_post_sub = []
    
for id_use in id_use_arr:
    print("Using data types with id in %s" % id_use)

    for i in range(len(f_prior_sub)):

        f_prior_h5 = f_prior_sub[i]
        f_data_h5 = f_data_sub[i]

        N_use = N
        
        # get string from id_use
        fileparts = os.path.splitext(f_data_h5)
        id_use_str = '_'.join(map(str, id_use))
        f_post_h5 = 'post_%s_id%s.h5' % (fileparts[0], id_use_str)
        f_post_h5 = ig.integrate_rejection(f_prior_h5, 
                                        f_data_h5, 
                                        f_post_h5, 
                                        showInfo=1, 
                                        N_use = N_use,
                                        id_use = id_use,
                                        nr=nr,
                                        T_N_above = T_N_above,
                                        T_P_acc_level = T_P_acc_level,
                                        updatePostStat=True)
        f_post_sub.append(f_post_h5)


# %%
mergePost = False
if len(f_post_sub) ==1:
    mergePost = False

if not mergePost:
    f_post_merged_h5 = f_post_sub[0]
    f_data_merged_h5 = f_data_sub[0]
else:
    f_post_merged_h5, f_data_merged_h5 = ig.merge_posterior(f_post_sub, f_data_sub, showInfo=4)



# %%
# use merged
f_post_h5 = f_post_merged_h5
f_data_h5 = f_data_merged_h5

# %%
f_post_h5 = 'post_SDR_FEDL_ALL_id2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32_33_34_35_36_37.h5'
#f_post_h5 = 'post_SDR_FEDL_ALL_id1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32_33_34_35_36_37.h5'
#f_post_h5 = 'post_SDR_FEDL_ALL_id1.h5'

import h5py
with h5py.File(f_post_h5, 'r') as f:
    f_prior_h5 = str(f.attrs.get('f5_prior', ''))
    f_data_h5 = str(f.attrs.get('f5_data', ''))

# %%
# PLot prior and observed data 
ig.plot_data_prior(f_prior_h5, f_data_h5, hardcopy=hardcopy, showInfo=-1)

# %% [markdown]
# ### Select a profile

# %%

X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

# find the index of the [X,Y] points closest to each borehole, keep only those within dist_max
dist_max = 1000
i_bh = []
BHOLES_filtered = []
for BH in BHOLES:
    d = np.sqrt((X-BH['X'])**2 + (Y-BH['Y'])**2)
    i_closest = np.argmin(d)
    print("Closest point to %s is at index %d with distance %.1f m" % (BH['name'], i_closest, d[i_closest]))
    if d[i_closest] <= dist_max:
        i_bh.append(i_closest)
        BHOLES_filtered.append(BH)
    else:
        print("  -> skipping %s (distance %.1f m > %.1f m)" % (BH['name'], d[i_closest], dist_max))

# The following does not work unless the DATA.h5 and PRIOR.h5 is updated consistently!!
#BHOLES = BHOLES_filtered
#print("Using %d boreholes within %.1f m of a data point" % (len(BHOLES), dist_max))

ibh_use = [23, 7, 4, 11]
Xl = np.array([BHOLES[i]['X'] for i in ibh_use])
Yl = np.array([BHOLES[i]['Y'] for i in ibh_use])

buffer = 15.0
indices, distances, segment_ids = ig.find_points_along_line_segments(
    X, Y, Xl, Yl, tolerance=buffer
)
id_line = indices

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, c=ELEVATION, s=1,label='X')
plt.plot(X[id_line],Y[id_line], 'r.', markersize=8, label='Profile', zorder=2, linewidth=5)
for i in range(len(i_bh)):
    #plt.plot(X[i_bh[i]], Y[i_bh[i]], 'k*', markersize=10, label='BH%d' % (i+1), zorder=3)
    plt.plot(BHOLES[i]['X'], BHOLES[i]['Y'], 'k*', markersize=10, label='BH%d' % (i), zorder=3)
    plt.text(BHOLES[i]['X']+10, BHOLES[i]['Y']+10, '%s [%d]' % (BHOLES[i]['name'], i), fontsize=9, zorder=4, rotation=20)

plt.grid()
plt.colorbar(label='Number of non-Nan data points')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Survey Points Colored by Number of Non-NaN Data Points')
plt.axis('equal')
#plt.legend()
if hardcopy:
    plt.savefig('SDR_FELDING_survey_points_nonnan.png', dpi=300)
plt.show()
# %%
#ig.plot_feature_2d(f_post_h5, key='HarmonicMean', im=1, elevation=elevation, plotPoints=True)
ig.plot_feature_2d(f_post_h5, key='HarmonicMean', im=1, iz=10, plotPoints=True, clim=[.1,3000])
plt.plot(X[i_bh], Y[i_bh], 'k*', markersize=10, label='Boreholes')
#plt.legend()    
plt.show()


# %% [markdown]
# ### POSTERIOR ANALYSIS
# ig.plot_profile(f_post_h5, im=1, ii=id_line, gap_threshold=100, xaxis='x', hardcopy=hardcopy)
# ig.plot_profile(f_post_h5, im=2, ii=id_line, gap_threshold=100, xaxis='x', hardcopy=hardcopy, alpha=.5, entropy_min =0.7, entropy_max=0.8)

# %%
for elevation in [40, 20, 0, -20]:
    #ig.plot_feature_2d(f_post_h5, key='HarmonicMean', im=1, elevation=elevation, plotPoints=True)
    ig.plot_feature_2d(f_post_h5, key='HarmonicMean', im=1, elevation=elevation, plotPoints=True)
    plt.plot(X[i_bh], Y[i_bh], 'k*', markersize=10, label='Boreholes')
    #plt.legend()    
    plt.show()

# %%
#ig.plot_feature_2d(f_post_h5, key='HarmonicMean', im=1, elevation=elevation, plotPoints=True)
ig.plot_feature_2d(f_post_h5, key='HarmonicMean', im=1, elevation=elevation, plotPoints=True)
plt.plot(X[i_bh], Y[i_bh], 'k*', markersize=10, label='Boreholes')
#plt.legend()    
plt.show()

# %%
for elevation in range(40, -41, -5):
    ig.plot_feature_2d(f_post_h5, key='Mode', im=2, elevation=elevation, plotPoints=True, hardcopy=hardcopy)
    #plt.plot(X[i_bh], Y[i_bh], 'k*', markersize=10, label='Boreholes')
    #plt.legend()    
    plt.show()


# %%
ig.plot_profile(f_post_h5, im=1, ii=id_line, gap_threshold=100, xaxis='x', hardcopy=hardcopy) 
ig.plot_profile(f_post_h5, im=2, ii=id_line, gap_threshold=100, xaxis='x', hardcopy=hardcopy, alpha=.5, entropy_min =0.7, entropy_max=0.8)

# %%
ig.plot_feature_2d(f_post_h5, key='Mean', im=3, plotPoints=True, uselog=False, hardcopy=hardcopy)


# %% [markdown]
# ### QUERY POSTERIOR MODEL REALIZATIONS
# QUERY : Probability that the cumulative thickness of lithology class 2
# within 0–30 m depth is greater than 10 m, and with an additional constraint: any top layer that is NOT sand/gravel
# cannot be thicker than 3m.
#
# ig.plot_data_prior_post(f_post_h5, i_plot=3000)


# %%
doLoadQuery = False
if doLoadQuery:
    query = ig.load_query('query_ex1.json')
else:
    query = {
        "constraints": [
            {
                "im": 2,
                "classes": [1,2, 4],
                "thickness_mode": "cumulative",
                "thickness_comparison": ">",
                "thickness_threshold": 5.0,
                "depth_min": 0.0,
                "depth_max": 30.0,
                "negate": False
            },
            {
                "im": 2,
                "classes": [3,5],  # All classes except sand (2) and gravel (5)
                "thickness_mode": "first_occurrence",
                "thickness_comparison": "<",
                "thickness_threshold": 3.0,
                "depth_min": 0.0,
                "depth_max": 30.0,
                "negate": False
            }
        ]
    }

    query = {
        "constraints": [
            {
                "im": 2,
                "classes": [1],
                "thickness_mode": "cumulative",
                "thickness_comparison": ">",
                "thickness_threshold": 10.0,
                "depth_min": 0.0,
                "depth_max": 80.0,
                "negate": False
            },
            {
                "im": 2,
                "classes": [1],
                "thickness_mode": "cumulative",
                "thickness_comparison": "<",
                "thickness_threshold": 20.0,
                "depth_min": 0.0,
                "depth_max": 80.0,
                "negate": False
            }
        ]
    }


    ig.save_query(query, 'query_daugaard.json')

for f_post_h5 in f_post_sub:
    # Compute the probability of the query being satisfied for each model realization in the posterior, and get metadata about the query results (e.g. which realizations satisfy the query, etc.)
    P, meta = ig.query(f_post_h5, query)
    #  Plot the predicted probability map, and the the outcome at the borehole locations
    ig.query_plot(P, meta, ip=i_bh[0], query_dict=query, f_post_h5=f_post_h5)
    ig.query_plot(P, meta, ip=i_bh[1], query_dict=query, f_post_h5=f_post_h5)

# %%
