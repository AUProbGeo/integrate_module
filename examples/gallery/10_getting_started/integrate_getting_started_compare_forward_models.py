# %% 
import integrate as ig
import time

# %%
case = 'DAUGAARD'
files = ig.get_case_data(case=case, showInfo=2)
f_data_h5 = files[0]
file_gex= files[1]

print("Using data file: %s" % f_data_h5)
print("Using GEX file: %s" % file_gex)


# %%
# Select how many prior model realizations (N) should be generated
N=2_000_000
#N=1_000_000
N=10_000

f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='uniform', NLAY_min=8, NLAY_max=8, RHO_min=1, RHO_max=3000, f_prior_h5='PRIOR_N%d.h5' % N, 
                                    showInfo=0)
#print('%s is used to hold prior realizations' % (f_prior_h5))

# %% FORWARD - PRIOR DATA
# prior data

method = ['ga-aem','simpeg','anemone','anemone']
device = ['cpu','cpu','cpu','cuda']
#method = ['anemone']
#device = ['cuda']

t_run = []
D_all = []

for i in range(len(method)):
    t0=time.time()
    f_prior_data_h5 = ig.prior_data_em(f_prior_h5, file_gex, doMakePriorCopy=False, id=1,
                                        method=method[i], device=device[i], showInfo=1)
#    f_prior_data_h5 = ig.prior_data_em(f_prior_h5, file_gex, doMakePriorCopy=True, 
#                                       f_prior_data_h5='PRIOR_DATA_N%d.h5_%s_%s' % (N, method[i],device[i]), 
#                                       method=method[i], device=device[i], showInfo=0)
    D = ig.load_prior_data(f_prior_data_h5)[0][0]
    D_all.append(D)
    t_run.append(time.time()-t0)

# %% TIMING
print('\nForward timing, N=%d' % N)
print('%-10s %-6s %10s %14s' % ('method', 'device', 'time [s]', 'ms/sounding'))
for i in range(len(method)):
    print('%-10s %-6s %10.1f %14.3f' % (method[i], device[i], t_run[i], 1000*t_run[i]/N))

# %% COMPARE FORWARD RESPONSES (first 9 soundings)
import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(3, 3, figsize=(12, 10), sharex=True)
for k, ax in enumerate(axs.flat):
    for i in range(len(method)):
        ax.semilogy(np.abs(D_all[i][k]), '.-', label='%s (%s)' % (method[i], device[i]))
    ax.set_title('Prior data #%d' % k)
    ax.grid(True, which='both', alpha=0.3)
for ax in axs[-1, :]:
    ax.set_xlabel('Gate index')
for ax in axs[:, 0]:
    ax.set_ylabel('|d|')
axs.flat[0].legend(fontsize=8)
fig.suptitle('Forward response comparison, N=%d' % N)
fig.tight_layout()
plt.savefig('tiny_forward_compare_N%d.png' % N)
plt.show()
