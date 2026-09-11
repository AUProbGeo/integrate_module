#!/usr/bin/env python
# %% [markdown]
# # Effect of using different forward models INTEGRATE
#

# %%
import integrate as ig
hardcopy = True
import matplotlib.pyplot as plt
import numpy as np
import time

# %% [markdown]
# ## 0. Get TTEM data

# %%
case = 'DAUGAARD'
files = ig.get_case_data(case=case, showInfo=2)
f_data_h5 = files[0]
file_gex= ig.get_gex_file_from_data(f_data_h5)

print("Using data file: %s" % f_data_h5)
print("Using GEX file: %s" % file_gex)

forward_models = ['gaaem','anemone']

# %% [markdown]
# Select a profile

# %%
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)
# Find points within buffer distance
Xl = np.array([544000, 543550])
Yl = np.array([6174500, 6176500])
#Xl = np.array([544000, 543550, 543000])
#Yl = np.array([6174500, 6176500, 6176400])
buffer = 10.0
indices, distances, segment_ids = ig.find_points_along_line_segments(
    X, Y, Xl, Yl, tolerance=buffer
)
i_line = indices

plt.figure()
ig.plot_geometry(f_data_h5, pl='ELEVATION')
plt.plot(Xl, Yl,'ko', markersize=15)
plt.plot(X[i_line], Y[i_line], 'ko', markersize=5, zorder=3)
plt.title('Profile line')
plt.show()

i_use = np.arange(len(X))

# %%
# The electromagnetic data (d_obs and d_std) can be plotted using ig.plot_data:
ig.plot_data(f_data_h5, hardcopy=hardcopy)
# Plot data channel 15 in an XY grid
ig.plot_data_xy(f_data_h5, data_channel=15, cmap='jet');

# %% [markdown]
# ## 1. Set up the prior model ($\rho(\mathbf{m},\mathbf{d})$)

# %%
# Select how many prior model realizations (N) should be generated
N=100_000
f_prior_h5 = ig.prior_model_layered(N=N,
                                    lay_dist='uniform', 
                                    NLAY_min=5,
                                    NLAY_max=5,                 # Minimum 3 layer
                                    #lay_dist='chi2',                                                                         
                                    #NLAY_max=6, 
                                    RHO_deg=3, 
                                    RHO_max=3000, 
                                    f_prior_h5='PRIOR_N%d.h5' % N, 
                                    showInfo=1)
print('%s is used to hold prior realizations' % (f_prior_h5))


# %%
# Plot summary statistics of the prior model for quality control of the prior choice
ig.plot_prior_stats(f_prior_h5, im=1, panels=['hist'],hardcopy=hardcopy)


# %% [markdown]
# ### 1b. Generate corresponding prior data
# Next, we generate a corresponding sample of $\rho(\mathbf{d})$ (prior data distribution).
# Here we test using both ga-aem and anemone

# %% GA-AEM

# set the default hdf file to be the f_prior_data_h5 (without extension) + file_gex (without extension) + .h5
# Make a copy of the å
f_prior_data_h5 = '%s_%s_data.h5' % (f_prior_h5[:-3], file_gex[:-4])
t0=time.time()
f_prior_data_h5 = ig.prior_data_em(f_prior_h5, file_gex, 
                                doMakePriorCopy=True, 
                                f_prior_data_h5=f_prior_data_h5, 
                                id=1, # The id '/D1' of the prior data! 
                                im=1, # the id '/M1' the im of the PRIOR resistivity
                                method=forward_models[0] # ga-aem is forward model type
                                )
t_gaaem=time.time()-t0

# %% Now use AnEMone
# anemone CPU
t0=time.time()
f_prior_data_h5 = ig.prior_data_em(f_prior_data_h5, file_gex, 
                                doMakePriorCopy=False,   # We KEEP also the ga-aem data                                                                    
                                id=2, # The id '/D2' of the prior data. Incremented from above
                                im=1, # The resistivity is still '/M1'
                                method=forward_models[1], # 'anemone' # anemone is the forward type
                                device = 'cpu'
                                )
t_anemone_cpu=time.time()-t0
id_anemone  = 2
# anemone GPU
t0=time.time()
try:
    f_prior_data_h5 = ig.prior_data_em(f_prior_data_h5, file_gex, 
                                    doMakePriorCopy=False,   # We KEEP also the ga-aem data                                                                    
                                    id=3, # The id '/D2' of the prior data. Incremented from above
                                    im=1, # The resistivity is still '/M1'
                                    method=forward_models[1], # 'anemone' # anemone is the forward type
                                    device = 'cuda'
                                    )
    id_anemone  = 3

except:
    print('GPU not available for anemone')
t_anemone_gpu=time.time()-t0
    

#%% 
print('t_gaaem = %3.1fs' % (t_gaaem))
print('t_anemone_cpu = %3.1fs' % (t_anemone_cpu))
print('t_anemone vs ga-aem speedup = %3.1f' % (t_gaaem/t_anemone_cpu))
try:
    print('t_anemone_gpu = %3.1fs' % (t_anemone_gpu))
    print('t_anemone_gpu vs ga-aem speedup = %3.1f' % (t_gaaem/t_anemone_gpu))  
except:
    pass
print('%s is used to hold prior model and data realizations' % (f_prior_data_h5))

# %%
D , id= ig.load_prior_data(f_prior_data_h5)
D1 = D[0]
D2 = D[1]
DD = D2-D1
DD_mean = np.mean(np.abs(DD), axis=0)
DD_std = np.std(DD, axis=0)

#plt.semilogy(np.abs(DD[0:10].T));
plt.figure()
plt.semilogy(D1[0:10].T, 'k-', linewidth=3, label=forward_models[0]);
plt.semilogy(D2[:10].T, 'r-', linewidth=1, label=forward_models[1]);
plt.semilogy(DD_std.T, 'b:', label='std')
plt.semilogy(DD_mean.T, 'b-', label='mean')
plt.xlabel('Gate id')
plt.ylabel('dB/dT')
#plt.legend()
plt.show()

# %% [markdown]
# Prior data

# %%
ig.plot_data_prior(f_prior_data_h5,f_data_h5,nr=1000, id=1, id_data=1, hardcopy=hardcopy)
ig.plot_data_prior(f_prior_data_h5,f_data_h5,nr=1000, id=2, id_data=1, hardcopy=hardcopy)

# %% [markdown]
# ## 2. Sample the posterior distribution $\sigma(\mathbf{m})$

# %%
N_use = N   # Number of prior samples to use (use all available)
T_base = 1  # Base annealing temperature for rejection sampling
autoT = 1   # Automatically estimate optimal annealing temperature

f_post_h5_arr = []
for id in [1,id_anemone]: # Loop over the two forward models
    f_post_h5 = ig.integrate_rejection(f_prior_data_h5,
                                    f_data_h5,
                                    f_post_h5 = 'POST_D%d.h5' % (id),
                                    ip_range = i_line,
                                    N_use = N_use,
                                    autoT = autoT,
                                    T_base = T_base,
                                    showInfo=0,
                                    id_use=1,
                                    id_prior = id, # Important to ONLY use one data set a time for this test
                                    updatePostStat = True)
    f_post_h5_arr.append(f_post_h5)

# %% [markdown]
# ## 3. Plot statistics from the posterior $\sigma(\mathbf{m})$
#
# ### Compare prior and posterior data
for f_post_h5 in f_post_h5_arr:
    ig.plot_data_prior_post(f_post_h5, i_plot=i_line[0],hardcopy=hardcopy)
for f_post_h5 in f_post_h5_arr:
    ig.plot_data_prior_post(f_post_h5, i_plot=i_line[-1],hardcopy=hardcopy)

# %% [markdown]
# ### Evidence and annealing temperature
# The evidence quantifies how well the data fits the model,
# while temperature controls the acceptance rate in rejection sampling.

# %% 
import h5py
CHI2 = []
EV = []
T = []
i=-1
for f_post_h5 in f_post_h5_arr:
    i=i+1
    with h5py.File(f_post_h5, 'r') as f:
        CHI2.append(f['CHI2'][:])
        EV.append(f['EV'][:])
        T.append(f['T'][:])
        print('Mean CHI2 for %s: %.3f - %s' % (f_post_h5, np.nanmean(CHI2[-1]),forward_models[i]))
        print('Mean logEV for %s: %.3f - %s' % (f_post_h5, np.nanmean(EV[-1]),forward_models[i]))
        print('Mean T for %s: %.3f - %s' % (f_post_h5, np.nanmean(T[-1]),forward_models[i]))
        
plt.figure(figsize=(15,5))
for i in range(3):
    plt.subplot(1,3,i+1)
    if i==0:
        plt.plot(CHI2[0].flatten(),CHI2[1].flatten(), 'r.', markersize=1)
        rel_err = np.nanmean((CHI2[0].flatten()-CHI2[1].flatten())/CHI2[0].flatten())
        plt.title('Comparison of CHI2 values, RelErr = %5.1f %%' % (100*rel_err))
    elif i==1:
        plt.plot(EV[0].flatten(),EV[1].flatten(), 'r.', markersize=1)
        rel_err = np.nanmean((EV[0].flatten()-EV[1].flatten())/T[0].flatten())
        plt.title('Comparison of EV values, RelErr = %5.1f %%' % (100*rel_err))
    elif i==2:
        plt.plot(T[0].flatten(),T[1].flatten(), 'r.', markersize=1)
        rel_err = np.nanmean((T[0].flatten()-T[1].flatten())/T[0].flatten())
        plt.title('Comparison of T values, RelErr = %5.1f %%' % (100*rel_err))
    xlim = plt.xlim();ylim = plt.ylim()
    lim = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
    plt.plot(lim, lim, 'k--')
    plt.gca().set_aspect('equal')
    plt.xlabel(forward_models[0])
    plt.ylabel(forward_models[1])
    plt.grid()
plt.show()

# %% [markdown]
# ### Resistivity profiles

# %%
# Plot resistivity profile for model M1
for f_post_h5 in f_post_h5_arr:
    ig.plot_profile(f_post_h5, ii=i_line, im=1, key='HarmonicMean', hardcopy=hardcopy)
