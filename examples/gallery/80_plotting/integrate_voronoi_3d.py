"""
3D posterior volumes on Voronoi prisms
======================================

``ig.post_to_voronoi_grid()`` puts a posterior statistic (e.g. ``/M1/Median``)
on a 3D grid for PyVista and ParaView:

* each sounding gets its 2D Voronoi cell (the area closest to it), clipped to
  a square of half-width ``r_max`` around the sounding;
* the cell is extruded below the surface elevation of the sounding along the
  depth grid of the prior, so the model follows the terrain;
* cells with NaN (not inverted, below the depth of investigation) are left out.

The result is a ``pyvista.UnstructuredGrid`` of wedge cells with cell data
``value``, ``log10_value``, ``sounding`` and ``iz``. It can be saved as
``.vtu`` and shown with

* ``ig.plot_voronoi_3d()``: values inside a range opaque, the rest transparent;
* ``ig.plot_voronoi_slices()``: x / y / z slices, or one interactive slice.

The DAUGAARD case is used.
"""
# %%
import numpy as np
import integrate as ig

# Set to True to open the interactive PyVista windows
showPyVista = False

# %%
# Data files
# ----------

# %%
f_prior_h5 = 'daugaard_merged_N2000000.h5'
f_post_h5 = 'post_daugaard_merged_N2000000_Nuse1000000_inflateNoise2.h5'
f_data_h5 = 'DAUGAARD_AVG_gf2.h5'
ig.get_case_data(case='DAUGAARD', filelist=[f_prior_h5, f_post_h5, f_data_h5])

# %%
# Build the Voronoi grid
# ----------------------
#
# Several million cells render slowly: use every 5th sounding (``ii``) and
# every 2nd depth cell (``iz_step``).

# %%
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)
ii = np.arange(len(X))[::5]
grid = ig.post_to_voronoi_grid(f_post_h5, ii=ii, im=1, key='Median', r_max=100, iz_step=2)
print(f'{grid.n_cells} cells, clim={list(grid.field_data["clim"])}')
grid.save('DAUGAARD_voronoi3d.vtu')

# %%
# Thresholded volume with transparency
# ------------------------------------
#
# Resistivities between 10 and 60 Ohm.m are opaque, the rest at opacity 0.05.

# %%
if showPyVista:
    ig.plot_voronoi_3d(grid, threshold=[10, 60], opacity_out=0.05, z_scale=10,
                       title='DAUGAARD, Median')

# %%
# Slices
# ------

# %%
if showPyVista:
    ig.plot_voronoi_slices(grid, x=np.median(X), y=np.median(Y), z=np.median(ELEVATION) - 30,
                           title='DAUGAARD, Median')
    ig.plot_voronoi_slices(grid, interactive=True, normal='x', title='DAUGAARD, Median')
