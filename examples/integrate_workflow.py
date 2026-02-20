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
import integrate as ig
from geoprior1d import geoprior1d

hardcopy = True
import matplotlib.pyplot as plt
import numpy as np
import copy


#%% SOME BASIC SETTINGS
N = 10000 # Number of prior model realizations to generate (this is just for testing, use a larger number for better results)
useLogData = False # Whether to transform data to log10 space (recommended for resistivity data)
inflateNoise = 1 # Factor to increase noise level (std) in the data, to



# %% [markdown]
# ## GETTING THE DATA AND GEX FILE for gthe chosen area

# %%
case = 'DAUGAARD'
f_xlsx_files = ['daugaard_standard.xlsx','daugaard_valley.xlsx']
ig.get_case_data(case=case, showInfo=2, filelist=['daugaard_standard.xlsx','daugaard_valley.xlsx'])
files = ig.get_case_data(case=case, showInfo=2)
f_data_h5 = files[0]
file_gex= ig.get_gex_file_from_data(f_data_h5)

print("Using data file: %s" % f_data_h5)
print("Using GEX file: %s" % file_gex)



# %% [markdown]
# ## the PRIOR model(s) : 
# simulate prior model realizations using geoprior1d # https://github.com/GEUSjesper/geoprior1d

# Generate N realizations of prirpo as defined  by the XLS files
f_prior_h5_list = []
for file_xlsx in f_xlsx_files:
    # get filename without extension for naming the output hdf5 file
    fname = file_xlsx.split('.')[0]
    f_prior_h5, flags  = geoprior1d(file_xlsx, Nreals=N, dz=1, dmax =90, output_file='%s_prior_N%d.h5' % (fname, N))
    f_prior_h5_list.append(f_prior_h5)

# Merge the two prior hdf5 files into one, which will be used for the rest of the workflow. The merged file will be named 'daugaard_merged_prior_N10000.h5' (or with the appropriate N value).
f_prior_h5 = ig.merge_prior(f_prior_h5_list, f_prior_merged_h5='daugaard_merged_prior_N%d.h5' % N)
ig.plot_prior_stats(f_prior_h5, hardcopy=hardcopy)


# %% [markdown]
# ## the tTEM DATA


# %% Determine if the data should be hanlded in log10 space, which is often recommended for resistivity data. If True, a new hdf5 file will be created with the log-transformed data and updated standard deviations.

if useLogData:
    f_data_h5_org = f_data_h5
    fname_data = f_data_h5.split('.')[0]
    f_data_h5 = '%s_LOGSPACE.h5' % fname_data
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



# %% Determined if the noise level in the data should be increased by a factor (inflateNoise). This can be useful for testing the sensitivity of the inversion results to noise. If inflateNoise is not equal to 1, a new hdf5 file will be created with the inflated noise level (standard deviation) and updated data values if necessary.
if inflateNoise != 1:
    gf=inflateNoise
    print("="*60)
    print("Increasing noise level (std) by a factor of %d" % gf)
    print("="*60)
    D = ig.load_data(f_data_h5)
    D_obs = D['d_obs'][0]
    D_std = D['d_std'][0]*gf
    f_data_old_h5 = f_data_h5
    f_data_h5 = '%s_gf%g.h5' % (fname_data, gf)
    ig.copy_hdf5_file(f_data_old_h5, f_data_h5)
    ig.save_data_gaussian(D_obs, D_std=D_std, f_data_h5=f_data_h5, file_gex=file_gex)

# The electromagnetic data (d_obs and d_std) can be plotted using ig.plot_data:
ig.plot_data(f_data_h5, hardcopy=hardcopy)
# Plot data channel 15 in an XY grid
ig.plot_data_xy(f_data_h5, data_channel=15, cmap='jet');


# %% [markdown]
# ## the tTEM DATA


# %% Define the well data for the Daugaard case, which can be used for validation of inversion results
BHOLES=[]
P_single=0.9 # Probability assigned to the observed class in the prior model (for discrete parameters)

use_well_type = 'compressed' # Options: 'full', 'compressed', 'single_layer'
if use_well_type == 'full':
    BH = {}
    BH['depth_top'] =    [0  , 0.3, 0.5, 1, 1.5, 2, 10, 10.5, 13.2, 16.6]
    BH['depth_bottom'] = [0.3, 0.5, 1, 1.5, 2, 10, 10.5, 13.2, 16.6, 20]
    BH['class_obs'] = [2, 2, 2, 2, 2, 2, 5, 2, 5, 3]
    BH['class_prob'] = [P_single, P_single, P_single, P_single, P_single, P_single, P_single, P_single, P_single, P_single]
    BH['X'] = 542983.01
    BH['Y'] = 6175822.76
    BH['name'] = 'DAU02 - Full'
    BHOLES.append(BH)

    # WELL 2: 116.1602
    BH = {}
    BH['depth_top'] =     [0,  8, 15, 17, 20, 23.5, 45]
    BH['depth_bottom'] =  [8, 15, 17, 20, 23.5, 45, 46]
    BH['class_obs'] = [3,  5,  3,  5,  5,  6,  7]
    BH['class_prob'] = [P_single, P_single, P_single, P_single, P_single, P_single, P_single]
    BH['X'] = 543584.098
    BH['Y'] = 6175788.478
    BH['name'] = '116.1602 - Full'
    BHOLES.append(BH)

elif use_well_type == 'compressed':

    BH = {}
    BH['depth_top'] =    [0   , 13.2, 16.6]
    BH['depth_bottom'] = [12.2, 15.6, 20]
    BH['class_obs'] = [2, 5, 3]
    BH['class_prob'] = [P_single, P_single, P_single]
    BH['X'] = 542983.01
    BH['Y'] = 6175822.76
    BH['name'] = 'DAU02 - Compressed'
    BHOLES.append(BH.copy())

    BH = {}
    BH = BH.copy()
    BH['depth_top'] =     [0,  8, 15,   17, 23.5, 45]
    BH['depth_bottom'] =  [7, 14, 16, 22.5,   44, 46]
    BH['class_obs'] = [3,  5,  3,    5,    6, 7]
    BH['class_prob'] = [P_single, P_single, P_single, .99, P_single, P_single]
    BH['X'] = 543584.098
    BH['Y'] = 6175788.478
    BH['name'] = '116.1602 - Compressed'
    BHOLES.append(BH.copy())

else :
    # SINGLE LAYER: lithoilogy 5 from 20-24 m
    BH = {}
    BH['depth_top'] =     [20]
    BH['depth_bottom'] = [24]
    BH['class_obs'] = [5]
    BH['class_prob'] = [P_single]
    BH['X'] = 542983.01
    BH['Y'] = 6175822.76
    BH['name'] = 'DAU02'
    BHOLES.append(BH.copy())

    # SINGLE LAYER: lithoilogy 5 from 20-24 m
    BH = {}
    BH['depth_top'] =     [20]
    BH['depth_bottom'] = [24]
    BH['class_obs'] = [5]
    BH['class_prob'] = [P_single]
    BH['X'] = 543584.098
    BH['Y'] = 6175788.478
    BH['name'] = '116.1602 - Single Layer'
    BHOLES.append(BH.copy())

ig.plot_boreholes(BHOLES)

# %%
# Write each borehole to its own JSON file
for BH in BHOLES:
    fname = 'daugaard_borehole_%s.json' % BH['name'].replace(' ', '_').replace('/', '-')
    ig.write_borehole(BH, fname, showInfo=1)

# Write all boreholes together in one JSON file
ig.write_borehole(BHOLES, 'daugaard_boreholes.json', showInfo=1)

# Read back a single borehole and inspect it
BH_loaded = ig.read_borehole('daugaard_borehole_%s.json' % BHOLES[0]['name'].replace(' ', '_').replace('/', '-'), showInfo=1)
print('name    :', BH_loaded['name'])
print('X, Y    :', BH_loaded['X'], BH_loaded['Y'])
print('intervals:', list(zip(BH_loaded['depth_top'], BH_loaded['depth_bottom'])))
print('class_obs:', BH_loaded['class_obs'])

# Read back all boreholes at once
BHOLES_loaded = ig.read_borehole('daugaard_boreholes.json', showInfo=1)
for BH in BHOLES_loaded:
    print(f"  {BH['name']:30s}  {len(BH['depth_top'])} intervals  X={BH['X']:.1f}  Y={BH['Y']:.1f}")

# %% Plot boreholes as lithology sticks
# Plot without prior info – classes labelled by numeric ID, default colours
ig.plot_boreholes(BHOLES)

# Plot without prior info 
ig.plot_boreholes(BHOLES, f_prior_h5)








# %% [markdown]
# ## COMPUTE PRIOR DATA !!


# %% COMPUTE prior tTEM data
f_prior_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, doMakePriorCopy=False)


# %% COMPUTE PRIOR BOREHOEL DATA and OBSERVED DATA
#
# Select how to extrapolate the borehoole observations from the point of obsevation to all posiutions in the grid


# %% [markdown]
# ### INVERSION


# %% [markdown]
# ### POSTERIOR ANALYSIS



# %% [markdown]
# ### QUERY POSTERIOR MODEL REALIZATIONS

