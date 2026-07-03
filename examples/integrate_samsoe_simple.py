# %% [markdown]
# # Ex for Daniel
#
# Needed packages:
# pip install scikit-learn, tensorflow, keras_tuner, mlmapping, libaarhusxyz
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
from integrate import mlmapping as mm
import time
import libaarhusxyz
# Check if parallel computations can be performed
parallel = ig.use_parallel(showInfo=1)
hardcopy = True 
import matplotlib.pyplot as plt
doPlot = 0


# %% [markdown]
# # Load data
#
# Load the SkyTEM XYZ survey data and convert it to the `f_data_h5` format used
# by the integrate module, either automatically via `ig.xyz_to_h5()` or manually.


# %% 


doLoadFull = False

if not doLoadFull:
    # We load only data from Samsø Nord
    file_xyz = 'SamsoeNord_v1_20260227_AVG_export.xyz'
    file_gex = '20251114_DEN_Diamond_Samsoe_PROTECT_SR.gex'
    f_data_h5 = ig.xyz_to_h5(file_xyz=file_xyz, file_gex = file_gex, 
                            rx_altitude='tx_altitude', rx_altitude_std='tx_altitude_std',
                            tx_altitude='rx_altitude', tx_altitude_std='rx_altitude_std')

else:
    # We load ALL Samsø data, but right now there is a problem with the scaling of the input data.
    doLoadAutomatic = True
    if doLoadAutomatic: 
            
        # AUTOMATIC Load XYZ data and convert to f_data_h5 format for integrate module
        import integrate as ig
        loadXYZ = True

        file_xyz  = '20260501_PROTECT_Samsoe_SkyTEM_2Hz.XYZ'
        file_gex = '20251114_DEN_Diamond_Samsoe_PROTECT_SR.gex'
        # altitude std = 2
        #f_data_h5 = ig.xyz_to_h5(file_xyz=file_xyz, file_gex = file_gex, altitude='Alt', altitude_std=2.0)
        # altitude std = 5%
        f_data_h5 = ig.xyz_to_h5(file_xyz=file_xyz, file_gex = file_gex, altitude='Alt', altitude_std=0.05)

    else: 
        #% MANUAL Alternate method to load XYZ data and convert to f_data_h5 format for integrate module
        model = libaarhusxyz.XYZ(file_xyz)
        gex = libaarhusxyz.GEX(file_gex)

        fl = model.flightlines
        ld = model.layer_data

        # --- gate selection (single-channel SkyTEM system) ---
        i_start = int(gex.remove_initial_gates(1))
        i_end = int(gex.no_gates(1))

        # --- replace the XYZ dummy/missing value with NaN ---
        nan_value = model.model_info.get('dummy', 9999)
        dbdt = ld['z_dbdt'].replace(nan_value, np.nan).values[:, i_start:i_end].astype(float)
        dbdt_relstd = ld['relunc_z_dbdt'].replace(nan_value, np.nan).values[:, i_start:i_end].astype(float)
        dbdt_std = ld['STD_Z_dBdt'.lower()].replace(nan_value, np.nan).values[:, i_start:i_end].astype(float)

        # --- construct DATA and DATA_std (relative std in the XYZ file -> absolute) ---
        DATA = dbdt
        #DATA_std = dbdt_relstd * np.abs(DATA)
        DATA_std = dbdt_std

        # --- geometry and flight altitude (SkyTEM column names: e/n/line/dem/alt) ---
        UTMX = fl['e'].values.astype(float)
        UTMY = fl['n'].values.astype(float)
        LINE = fl['line'].values.astype(float)
        ELEVATION = fl['dem'].values.astype(float)
        ALT = fl['alt'].values.reshape(-1, 1).astype(float)
        ALT_std = 0.05 * np.abs(ALT)

        # --- drop soundings that are all-NaN ---
        keep = ~np.all(np.isnan(DATA), axis=1)
        DATA, DATA_std = DATA[keep], DATA_std[keep]
        UTMX, UTMY, LINE, ELEVATION = UTMX[keep], UTMY[keep], LINE[keep], ELEVATION[keep]
        ALT, ALT_std = ALT[keep], ALT_std[keep]

        # --- save to HDF5 (mirrors what ig.xyz_to_h5() does internally) ---
        f_data_h5 = 'DATA_samsoe_manual.h5'
        ig.save_data_gaussian(
            DATA, D_std=DATA_std, f_data_h5=f_data_h5,
            UTMX=UTMX, UTMY=UTMY, LINE=LINE, ELEVATION=ELEVATION,
            delete_if_exist=True, f_gex=file_gex, showInfo=1,
        )
        ig.save_data_gaussian(
            ALT, D_std=ALT_std, f_data_h5=f_data_h5,
            id=2, name='Tx Altitude', delete_if_exist=False, showInfo=1,
        )
        ig.save_data_gaussian(
            ALT, D_std=ALT_std, f_data_h5=f_data_h5,
            id=3, name='Rx Altitude', delete_if_exist=False, showInfo=1,
        )

# %% Plot the geometry of the data in f_data_h5
ig.plot_geometry(f_data_h5, pl='ELEVATION')
ig.plot_geometry(f_data_h5, pl='id')
ig.plot_geometry(f_data_h5, pl='NDATA')

# %% Plot the actual data in f_data_h5
#ig.plot_data(f_data_h5, id=1)
ig.plot_data(f_data_h5)

# %% Load data!
D_obs = ig.load_data(f_data_h5)

# %% [markdown]
# # Setup the prior
#


# %% 

N=2_000_000
f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='chi2', NLAY_deg=3, RHO_min=1, RHO_max=3000, 
                                    f_prior_h5='PRIOR_N%d.h5' % N, 
                                    dz = 1, 
                                    z_max = 149,
                                    showInfo=1)

# simulate Tx/Rx altitude and save to f_prior_h5:/M2 and /M3
alt_min = np.percentile(D_obs['d_obs'][1].flatten(),2.5)
alt_max = np.percentile(D_obs['d_obs'][1].flatten(),100-2.5)
tx_rx = np.mean(D_obs['d_obs'][1]-D_obs['d_obs'][2])

Tx_sim = np.random.uniform(alt_min, alt_max, N)
Rx_sim = Tx_sim + tx_rx 
# write Tx_sim to f_prior_h5:/M2
ig.save_prior_model(f_prior_h5, Tx_sim, im=2, name='Tx_altitude', showInfo=1, force_replace=True)
# write Rx_sim to f_prior_h5:/M3
ig.save_prior_model(f_prior_h5, Rx_sim, im=3, name='Rx_altitude', showInfo=1, force_replace=True)

ig.plot_prior_stats(f_prior_h5)

# %% FORWARD MODELING

# %% Load MM model and predict forward response
nn_model_file = 'resistivity_forward_model_HL4_UN160_chi2_log-uniform_N1000000_zmax150_abslog_gates5-41_NoNegatives_Samsoe251114'
mm.get_model_info(nn_model_file)

# Load prior data
M, ind = ig.load_prior_model(f_prior_h5)


t0 = time.time()    
D1_pred_ml = mm.predict([np.log10(M[0]), np.log10(M[1]), np.log10(M[2])], nn_model_file, batch_size=10000)
#D1_ml = 10 **(D1_pred_ml[0][:,use_gates])
D1_ml = 10 **(D1_pred_ml[0])
t1= time.time()
print(f"Time taken for ML prediction: {t1-t0:.2f} seconds")


# Now update the prior file with the forward-modeled data
ig.save_prior_data(f_prior_h5, D1_ml, id=1, name='dB/dT', showInfo=1, force_delete=True)
# D2 = M2 (identity mapping): M[1] is already loaded in memory, so just re-save it as data
ig.save_prior_data(f_prior_h5, M[1], id=2, name='Tx_altitude', showInfo=1, force_delete=True)
# D3 = M3 (identity mapping): M[2] is already loaded in memory, so just re-save it as data
ig.save_prior_data(f_prior_h5, M[2], id=3, name='Rx_altitude', showInfo=1, force_delete=True)


# %%
ig.plot_data_prior(f_prior_h5, f_data_h5)

# %% 
backend = 'jax'
#backend = 'numpy'
autoT = True
N_use = N
f_post_h5 = 'POST_N%d.h5' % N_use

f_post_h5 = ig.integrate_rejection(f_prior_h5,
                                f_data_h5,
                                f_post_h5,
                                N_use = N_use,
                                autoT=autoT,
                                showInfo=1,
                                parallel=True,
                                updatePostStat=True,
                                backend=backend)

# %% 
ig.plot_data_prior_post(f_post_h5, id=1)
ig.plot_data_prior_post(f_post_h5, i_plot=100)

# %%
cmap, clim = ig.get_colormap_and_limits('resistivity')
hardcopy = True

ig.plot_profile(f_post_h5, i1=0, i2=1000, clim=clim, cmap=cmap, hardcopy=hardcopy)

ig.plot_T_EV(f_post_h5, pl='N_UNIQUE', hardcopy=hardcopy, N_UNIQUE_min=1,plot_data_locations=True)

ig.plot_feature_2d(f_post_h5,im=1,iz=15, key='LogMean', uselog=1, hardcopy=hardcopy, clim=clim, cmap=cmap, title = 'log(Mean) iz = 15')
ig.plot_feature_2d(f_post_h5,im=1,iz=45, key='LogMean', uselog=1, hardcopy=hardcopy, clim=clim, cmap=cmap, title = 'log(Mean), iz = 45')

# Harmonic mean resistivity at elevation -10 m depth.
ig.plot_feature_2d(f_post_h5, im=1, elevation=10, key='HarmonicMean')

ig.plot_feature_2d(f_post_h5,im=2,iz=0, key='HarmonicMean', uselog=0, hardcopy=hardcopy, 
                   title = 'Posterior Tx',  cmap='jet_r', clim=[0,50]) 


# %% Plot profile using coordinates
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)
# Find points within buffer distance
Xl = np.array([595000, 598500])
Yl = np.array([6204000, 6203500])
buffer = 40.0
indices, distances, segment_ids = ig.find_points_along_line_segments(
    X, Y, Xl, Yl, tolerance=buffer
)
id_line = indices

plt.figure()
plt.plot(X,Y,'k.',markersize=1)
plt.plot(Xl,Yl,'r-')
plt.plot(X[id_line],Y[id_line],'r*',markersize=11)
plt.axis('equal')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.grid()

ig.plot_profile(f_post_h5, ii=id_line ,  gap_threshold=150, xaxis='x', clim=clim, cmap=cmap, hardcopy=hardcopy)



# %%
