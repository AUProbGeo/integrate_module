"""
INTEGRATE -Generic Prior Model Generation Examples
==================================================

This example demonstrates the different prior model generation functions 
available in the INTEGRATE module. The script creates multiple prior models
using different parameterizations and distributions:

1. **prior_model_layered()** - Variable number of layers with uniform or chi2 distribution
2. **prior_model_smooth()** - Workbench Smooth (L2): vertically correlated resistivity
3. **prior_model_blocky()** - Workbench Blocky (L1): Laplace random walk in log-resistivity
4. **prior_model_sharp()** - Workbench Sharp (MGS): few sharp interfaces at random depths

Each function demonstrates different resistivity distributions and layer configurations
commonly used in electromagnetic geophysical inversions.
"""
# %%
try:
    # Check if the code is running in an IPython kernel (which includes Jupyter notebooks)
    get_ipython()
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except:
    pass

# %%
import integrate as ig
import numpy as np
import matplotlib.pyplot as plt
import os

hardcopy = True

print("="*60)
print("INTEGRATE - Prior Model Generation Examples")
print("="*60)

# %%
# Configuration Parameters
# ------------------------
#
# Set the number of realizations and common parameters for all prior models

# %%
# Number of realizations to generate for each prior model
N = 500000  # Adjust this value as needed
N = 10000

# Common depth parameters  
z_max = 90      # Maximum depth (m)
dz = 1          # Depth discretization (m)

# Common resistivity range
RHO_min = 1     # Minimum resistivity (Ohm-m) 
RHO_max = 500   # Maximum resistivity (Ohm-m)

print(f"Configuration:")
print(f"- Number of realizations per model: {N:,}")
print(f"- Maximum depth: {z_max} m")
print(f"- Resistivity range: {RHO_min} - {RHO_max} Ohm-m")
print(f"- Depth discretization: {dz} m")

# %%
# 1. prior_model_layered() Examples
# ---------------------------------
#
# Generate layered resistivity models with different layer number distributions and resistivity distributions

# %%
print(f"\n{'='*60}")
print("1. PRIOR_MODEL_LAYERED() EXAMPLES")
print(f"{'='*60}")

# ### 1a. Layered model with uniform layer distribution and log-uniform resistivity
print(f"\n1a. Layered model: uniform layers (3-6), log-uniform resistivity")
f_prior_layered_uniform = ig.prior_model_layered(
    N=N,
    lay_dist='uniform',         # Uniform distribution of layer numbers
    z_max=z_max,
    dz=dz,
    NLAY_min=3,                 # Minimum 3 layers
    NLAY_max=6,                 # Maximum 6 layers
    RHO_dist='log-uniform',     # Log-uniform resistivity distribution
    RHO_min=RHO_min,
    RHO_max=RHO_max,
    f_prior_h5='PRIOR_layered_uniform_log-uniform_N%d.h5' % N,
    save_sparse=False,
    showInfo=1
)

# ### 1b. Layered model with chi2 layer distribution and uniform resistivity  
print(f"\n1b. Layered model: chi2 layers (3-8), uniform resistivity")
f_prior_layered_chi2 = ig.prior_model_layered(
    N=N,
    lay_dist='chi2',            # Chi-square distribution of layer numbers
    z_max=z_max,
    dz=dz,
    NLAY_min=3,                 # Minimum 3 layers
    NLAY_max=8,                 # Maximum 8 layers  
    NLAY_deg=5,                 # Degrees of freedom for chi2 distribution
    RHO_dist='uniform',         # Uniform resistivity distribution
    RHO_min=RHO_min,
    RHO_max=RHO_max,
    f_prior_h5='PRIOR_layered_chi2_uniform_N%d.h5' % N,
    showInfo=1
)

# ### 1c. Layered model with normal resistivity distribution
print(f"\n1c. Layered model: uniform layers (4-7), normal resistivity")
f_prior_layered_normal = ig.prior_model_layered(
    N=N,
    lay_dist='uniform',
    z_max=z_max,
    dz=dz,
    NLAY_min=4,                 # Minimum 4 layers
    NLAY_max=7,                 # Maximum 7 layers
    RHO_dist='normal',          # Normal resistivity distribution
    RHO_mean=100,               # Mean resistivity
    RHO_std=50,                 # Standard deviation
    f_prior_h5='PRIOR_layered_uniform_normal_N%d.h5' % N,
    showInfo=1
)

# %%
# 2. prior_model_smooth / blocky / sharp() Examples
# -------------------------------------------------
#
# Aarhus Workbench Smooth (L2), Blocky (L1) and Sharp (MGS) 1D model types,
# generated as sample ensembles. Each writes
# /M1 (realizations on a regular dz grid, used for forward modelling),
# /M2 (realizations in the native formulation) and /M3 (native layer count).
# Smooth/Blocky share a fixed geometric native stack and differ only in the
# vertical prior on log-resistivity; Sharp uses a sparse Poisson jump model.

# %%
print(f"\n{'='*60}")
print("2. WORKBENCH SMOOTH / BLOCKY / SHARP PRIORS")
print(f"{'='*60}")

# ### 2a. Smooth (L2): correlated Gaussian process in depth
print(f"\n2a. Smooth: L2 vertical constraint, corr_length=15 m")
f_prior_smooth = ig.prior_model_smooth(
    N=N,
    nlayers=30,                 # native geometric layers
    z1=0,
    z_max=z_max,
    dz=dz,                      # regular /M1 output grid
    corr_length=15.0,           # vertical correlation length (m)
    sigma_logrho=0.25,          # prior std of log-resistivity (BetaV)
    RHO_ref=100.0,
    RHO_min=RHO_min,
    RHO_max=RHO_max,
    f_prior_h5='PRIOR_smooth_N%d.h5' % N,
    showInfo=1
)

# ### 2b. Blocky (L1): cumulative Laplace increments -> sparse steps
print(f"\n2b. Blocky: L1 vertical constraint, blocky_scale=0.3")
f_prior_blocky = ig.prior_model_blocky(
    N=N,
    nlayers=30,
    z1=0,
    z_max=z_max,
    dz=dz,
    blocky_scale=0.3,
    RHO_min=RHO_min,
    RHO_max=RHO_max,
    f_prior_h5='PRIOR_blocky_N%d.h5' % N,
    showInfo=1
)

# ### 2c. Sharp (MGS): sparse sharp interfaces on the fixed grid
print(f"\n2c. Sharp: ~3 sharp jumps, log-uniform inter-jump resistivity")
f_prior_sharp = ig.prior_model_sharp(
    N=N,
    z1=0,
    z_max=z_max,
    dz=dz,
    n_jumps_mean=3.0,
    RHO_dist='log-uniform',
    RHO_min=RHO_min,
    RHO_max=RHO_max,
    f_prior_h5='PRIOR_sharp_N%d.h5' % N,
    showInfo=1
)

# %%
# 3. Summary and Visualization
# ----------------------------
#
# Display summary information and create comparison plots

# %%
print(f"\n{'='*60}")
print("3. SUMMARY OF GENERATED PRIOR MODELS")
print(f"{'='*60}")

# List all generated prior files
prior_files = [
    f_prior_layered_uniform,
    f_prior_layered_chi2,
    f_prior_layered_normal,
    f_prior_smooth,
    f_prior_blocky,
    f_prior_sharp
]

print(f"\nGenerated {len(prior_files)} prior model files:")
for i, fname in enumerate(prior_files, 1):
    if os.path.exists(fname):
        file_size_mb = os.path.getsize(fname) / (1024*1024)
        print(f"  {i:2d}. {fname} ({file_size_mb:.1f} MB)")
    else:
        print(f"  {i:2d}. {fname} (FILE NOT FOUND)")

# %%
# 4. Plot Prior Statistics
# ------------------------
#
# Generate comparison plots for selected prior models

# %%
print(f"\n{'='*60}")
print("4. PLOTTING PRIOR STATISTICS")
print(f"{'='*60}")

# Plot statistics for a few representative examples
example_priors = [
    (f_prior_layered_uniform, "Layered: Uniform layers, Log-uniform ρ"),
    (f_prior_smooth, "Smooth (L2)"),
    (f_prior_blocky, "Blocky (L1)"),
    (f_prior_sharp, "Sharp (MGS)")
]

for fname, title in example_priors:
    if os.path.exists(fname):
        print(f"\nPlotting statistics for: {title}")
        try:
            fig = ig.plot_prior_stats(fname)
            if fig is not None and hardcopy:
                plot_name = fname.replace('.h5', '_stats.png')
                plt.savefig(plot_name, dpi=150, bbox_inches='tight')
                print(f"  Saved plot: {plot_name}")
        except Exception as e:
            print(f"  Error plotting {fname}: {e}")

# %%
print(f"\n{'='*60}")
print("SCRIPT COMPLETED SUCCESSFULLY")
print(f"{'='*60}")
print(f"\nGenerated {len(prior_files)} INTEGRATE prior model files:")
print("- Different layer parameterizations (layered, smooth, blocky, sharp)")  
print("- Various resistivity distributions (log-uniform, uniform, normal, lognormal)")
print("- Different layer number distributions (uniform, chi2)")
print("- Range of model complexities (2-10 layers)")
print(f"\nAll files contain {N:,} realizations and are ready for:")
print("- Forward modeling with prior_data_em()")
print("- Probabilistic inversion with integrate_rejection()")
print("- Model analysis and visualization")

# %%
