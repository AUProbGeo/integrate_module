import multiprocessing
multiprocessing.freeze_support()

# Import rejection sampling functions from new module
from integrate.integrate_rejection import integrate_rejection
from integrate.integrate_rejection import integrate_rejection_range
from integrate.integrate_rejection import integrate_posterior_chunk
from integrate.integrate_rejection import integrate_posterior_main
from integrate.integrate_rejection import likelihood_gaussian_diagonal
from integrate.integrate_rejection import likelihood_gaussian_diagonal_old
from integrate.integrate_rejection import likelihood_gaussian_full
from integrate.integrate_rejection import likelihood_multinomial
from integrate.integrate_rejection import likelihood_multinomial_old
from integrate.integrate_rejection import select_subset_for_inversion
from integrate.integrate_rejection import cleanup_shared_memory
from integrate.integrate_rejection import reconstruct_shared_arrays
from integrate.integrate_rejection import create_shared_memory
from integrate.integrate_rejection import compute_hypothesis_probability

# Import other functions from main module
from integrate.integrate import integrate_update_prior_attributes   
from integrate.integrate import integrate_posterior_stats
from integrate.integrate import logl_T_est
from integrate.integrate import lu_post_sample_logl
from integrate.integrate import prior_data
from integrate.integrate import prior_data_gaaem
from integrate.integrate import prior_data_identity
from integrate.integrate import forward_gaaem
from integrate.integrate import synthetic_case
from integrate.integrate import prior_model_layered
from integrate.integrate import prior_model_workbench
from integrate.integrate import prior_model_workbench_direct
from integrate.integrate import posterior_cumulative_thickness
#from integrate.integrate import integrate_rejection_multi  
from integrate.integrate import use_parallel
from integrate.integrate import kl_divergence
from integrate.integrate import entropy
from integrate.integrate import class_id_to_idx
from integrate.integrate import is_notebook
from integrate.integrate import get_hypothesis_probability
from integrate.integrate import sample_posterior_multiple_hypotheses
from integrate.integrate import timing_compute
from integrate.integrate import timing_plot
#from integrate.integrate import compute_P_obs_from_log
#from integrate.integrate import rescale_P_obs_temperature
#from integrate.integrate import Pobs_to_datagrid

# Import matplotlib backend setup
from integrate.integrate_plot import setup_matplotlib_backend

from integrate.integrate_io import copy_prior
from integrate.integrate_io import load_prior
from integrate.integrate_io import load_prior_data
from integrate.integrate_io import save_prior_data
from integrate.integrate_io import load_prior_model
from integrate.integrate_io import save_prior_model
from integrate.integrate_io import load_data
from integrate.integrate_io import get_geometry
from integrate.integrate_io import extract_feature_at_elevation
from integrate.integrate_io import get_discrete_classes
from integrate.integrate_io import get_number_of_datasets
from integrate.integrate_io import get_number_of_data
from integrate.integrate_io import read_gex
from integrate.integrate_io import read_gex_workbench
from integrate.integrate_io import gex_to_stm
from integrate.integrate_io import get_gex_file_from_data
from integrate.integrate_io import write_stm_files
from integrate.integrate_io import post_to_csv
from integrate.integrate_io import copy_hdf5_file
from integrate.integrate_io import hdf5_scan
from integrate.integrate_io import get_case_data
from integrate.integrate_io import save_data_gaussian
from integrate.integrate_io import xyz_to_h5
from integrate.integrate_io import save_data_multinomial
from integrate.integrate_io import write_data_gaussian  # Deprecated - use save_data_gaussian
from integrate.integrate_io import write_data_multinomial  # Deprecated - use save_data_multinomial
from integrate.integrate_io import check_data
from integrate.integrate_io import merge_prior
from integrate.integrate_io import merge_data
from integrate.integrate_io import merge_posterior
from integrate.integrate_io import filter_prior
from integrate.integrate_io import read_usf
from integrate.integrate_io import read_usf_mul
from integrate.integrate_io import test_read_usf
from integrate.integrate_io import hdf5_info
from integrate.integrate_io import read_borehole
from integrate.integrate_io import write_borehole

from integrate.integrate_plot import plot_geometry
from integrate.integrate_plot import plot_profile
from integrate.integrate_plot import plot_profile_continuous
from integrate.integrate_plot import plot_profile_discrete
from integrate.integrate_plot import plot_cumulative_probability_profile
from integrate.integrate_plot import plot_T_EV
from integrate.integrate_plot import plot_data_xy
from integrate.integrate_plot import plot_data
from integrate.integrate_plot import plot_data_prior_post
from integrate.integrate_plot import plot_data_prior
from integrate.integrate_plot import plot_prior_stats
from integrate.integrate_plot import plot_post_stats
from integrate.integrate_plot import plot_feature_2d
from integrate.integrate_plot import plot_posterior_cumulative_thickness
from integrate.integrate_plot import h5_get_clim_cmap
from integrate.integrate_plot import get_colormap_and_limits
from integrate.integrate_plot import find_points_along_line_segments
from integrate.integrate_plot import plot_boreholes

# Import from borehole module
from integrate.integrate_borehole import compute_P_obs_discrete
from integrate.integrate_borehole import compute_P_obs_discrete as Pobs_discrete_compute
from integrate.integrate_borehole import welllog_compute_P_obs_class_mode
from integrate.integrate_borehole import rescale_P_obs_temperature
from integrate.integrate_borehole import rescale_P_obs_temperature as Pobs_rescale_temperature
from integrate.integrate_borehole import Pobs_to_datagrid
from integrate.integrate_borehole import get_weight_from_position
from integrate.integrate_borehole import prior_data_borehole
from integrate.integrate_borehole import prior_data_borehole_class_mode
from integrate.integrate_borehole import prior_data_borehole_class_layer
from integrate.integrate_borehole import save_borehole_data

# Import query functions
from integrate.integrate_query import query
from integrate.integrate_query import query_plot
from integrate.integrate_query import save_query
from integrate.integrate_query import load_query
from integrate.integrate_query import get_prior_model_info
from integrate.integrate_query import query_from_text
from integrate.integrate_query import query_test_llm

# Import gex module functions
from integrate.gex import read_gex as read_gex2
from integrate.gex import describe_gex as describe_gex2


# REMOVE CLI IMPORTS - These cause circular dependencies
# from . import integrate_cli
# from . import integrate_timing

