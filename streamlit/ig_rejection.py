"""
INTEGRATE Rejection Sampling Interface

Streamlit interface for Bayesian inversion using rejection sampling.
Provides access to integrate_rejection() function with progress monitoring.

Author: Generated for the INTEGRATE module
"""

import streamlit as st
import os
import sys
import h5py
import numpy as np
import time
import threading

# Add the parent directory to Python path to import integrate module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ig_style import apply_style, page_header
from ig_progress import make_progress_callback

try:
    import integrate as ig
except ImportError:
    st.error("Could not import integrate module. Please ensure it is properly installed.")
    st.stop()

def display_h5_info(file_path):
    """Display basic information about an H5 file"""
    try:
        with h5py.File(file_path, 'r') as f:
            st.subheader(f"File Info: {os.path.basename(file_path)}")
            
            # Show datasets
            datasets = []
            def visit_func(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets.append((name, obj.shape, obj.dtype))
            
            f.visititems(visit_func)
            
            if datasets:
                st.write("**Datasets:**")
                for name, shape, dtype in datasets[:10]:  # Show first 10
                    st.text(f"  {name}: {shape} {dtype}")
                if len(datasets) > 10:
                    st.text(f"  ... and {len(datasets) - 10} more datasets")
            
            # Show attributes
            if f.attrs:
                st.write("**File Attributes:**")
                for key, value in list(f.attrs.items())[:5]:  # Show first 5
                    st.text(f"  {key}: {value}")
                if len(f.attrs) > 5:
                    st.text(f"  ... and {len(f.attrs) - 5} more attributes")
                    
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

def get_h5_files():
    """Get list of H5 files in current directory"""
    return [f for f in os.listdir('.') if f.endswith('.h5')]

def get_prior_files():
    """Get H5 files that look like prior files"""
    h5_files = get_h5_files()
    # Filter for files that likely contain prior data (have forward model data)
    prior_files = []
    for f in h5_files:
        if 'PRIOR' in f.upper() or any(x in f.upper() for x in ['_N', 'LAYERED', 'WORKBENCH']):
            prior_files.append(f)
    return prior_files if prior_files else h5_files

def get_data_files():
    """Get H5 files that look like data files"""
    h5_files = get_h5_files()
    # Filter for files that likely contain observational data
    data_files = []
    for f in h5_files:
        if any(x in f.upper() for x in ['DATA', 'OBS', 'TEM', 'DAUGAARD', 'AVG']):
            data_files.append(f)
    return data_files if data_files else h5_files

def run_rejection_app():
    page_header("Rejection Sampling Inversion",
                "Perform Bayesian inversion using rejection sampling with temperature "
                "annealing, finding model parameters that best fit the observed data.")
    
    # File selection
    st.subheader("File Selection")
    
    prior_files = get_prior_files()
    data_files = get_data_files()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Prior/Forward Model Files**")
        if prior_files:
            f_prior_h5 = st.selectbox("Select prior/forward model file:", prior_files)
            if st.button("Show Prior Info"):
                display_h5_info(f_prior_h5)
        else:
            st.warning("No suitable prior files found")
            f_prior_h5 = None
    
    with col2:
        st.write("**Observational Data Files**")
        if data_files:
            f_data_h5 = st.selectbox("Select data file:", data_files)
            if st.button("Show Data Info"):
                display_h5_info(f_data_h5)
        else:
            st.warning("No suitable data files found")
            f_data_h5 = None
    
    if not (f_prior_h5 and f_data_h5):
        st.error("Please ensure both prior and data files are selected")
        return
    
    st.markdown("---")
    
    # Parameters
    st.subheader("Inversion Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Sampling Parameters**")
        N_use = st.number_input("Max prior samples to use", value=100000, min_value=1000)
        nr = st.number_input("Posterior samples per data point", value=400, min_value=10)
        autoT = st.selectbox("Temperature control", [1, 0], index=0, 
                           format_func=lambda x: "Automatic" if x == 1 else "Manual")
        if autoT == 0:
            T_base = st.number_input("Base temperature", value=1.0, min_value=0.1)
        else:
            T_base = 1.0
    
    with col2:
        st.write("**Processing Options**")
        parallel = st.checkbox("Use parallel processing", value=True)
        Ncpu = st.number_input("Number of CPUs (0 = auto)", value=0, min_value=0)
        Nchunks = st.number_input("Number of chunks (0 = auto)", value=0, min_value=0)
        use_N_best = st.number_input("Use N best samples (0 = disabled)", value=0, min_value=0)
        
        st.write("**Advanced Options**")
        updatePostStat = st.checkbox("Compute posterior statistics", value=True)
        showInfo = st.selectbox("Verbosity level", [0, 1, 2], index=0)
    
    with col3:
        st.write("**Data Selection**")
        # Simple interface - could be expanded for more complex selections
        st.info("Using all available data points and data types")
        id_use_text = st.text_input("Data types to use (comma-separated, empty = all)", "")
        ip_range_text = st.text_input("Data point range (start:end, empty = all)", "")
        
        # Parse selections
        id_use = []
        if id_use_text:
            try:
                id_use = [int(x.strip()) for x in id_use_text.split(',')]
            except:
                st.error("Invalid data type format")
        
        ip_range = []
        if ip_range_text:
            try:
                if ':' in ip_range_text:
                    start, end = ip_range_text.split(':')
                    ip_range = list(range(int(start), int(end)))
                else:
                    ip_range = [int(ip_range_text)]
            except:
                st.error("Invalid data point range format")
    
    # Output options
    st.write("**Output Options**")
    f_post_h5 = st.text_input("Output filename (optional)", 
                             placeholder="Leave empty for auto-generated name")
    
    # Run inversion
    if st.button("Run Rejection Sampling Inversion", type="primary"):
        if not os.path.exists(f_prior_h5):
            st.error(f"Prior file {f_prior_h5} not found")
            return
        if not os.path.exists(f_data_h5):
            st.error(f"Data file {f_data_h5} not found")
            return
        
        # Create progress tracking
        callback, finish = make_progress_callback()
        time_text = st.empty()

        start_time = time.time()

        try:
            # Prepare arguments
            kwargs = {
                'showInfo': showInfo,
                'updatePostStat': updatePostStat,
                'progress_callback': callback,
                'console_progress': False  # Disable console progress for cleaner Streamlit output
            }

            # Run inversion
            f_post_result = ig.integrate_rejection(
                f_prior_h5=f_prior_h5,
                f_data_h5=f_data_h5,
                f_post_h5=f_post_h5,
                N_use=N_use,
                id_use=id_use,
                ip_range=ip_range,
                nr=nr,
                autoT=autoT,
                T_base=T_base,
                Nchunks=Nchunks,
                Ncpu=Ncpu,
                parallel=parallel,
                use_N_best=use_N_best,
                **kwargs
            )

            elapsed_time = time.time() - start_time

            finish("Inversion completed!")
            time_text.text(f"Total time: {elapsed_time:.1f} seconds")

            st.success(f"✅ Rejection sampling completed successfully!")
            st.info(f"Results saved as: {f_post_result}")
            
            # Show basic statistics
            if os.path.exists(f_post_result):
                st.markdown("---")
                display_h5_info(f_post_result)
                
                # Try to show some basic inversion statistics
                try:
                    with h5py.File(f_post_result, 'r') as f:
                        if 'T' in f:
                            T_values = f['T'][:]
                            T_mean = np.nanmean(T_values)
                            st.metric("Average Temperature", f"{T_mean:.2f}")
                        
                        if 'EV' in f:
                            EV_values = f['EV'][:]
                            EV_mean = np.nanmean(EV_values)
                            st.metric("Average Log Evidence", f"{EV_mean:.2f}")
                            
                        if 'N_UNIQUE' in f:
                            N_unique = f['N_UNIQUE'][:]
                            N_unique_mean = np.nanmean(N_unique)
                            st.metric("Average Unique Samples", f"{N_unique_mean:.1f}")
                except:
                    pass
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            time_text.text(f"Time elapsed: {elapsed_time:.1f} seconds")
            
            st.error(f"Error during inversion: {str(e)}")
            st.error("Common issues:")
            st.error("1. Incompatible file formats between prior and data")
            st.error("2. Insufficient memory for large datasets")
            st.error("3. Data/model dimension mismatches")
            st.error("4. Corrupted H5 files")

    # Information section
    st.markdown("---")
    st.subheader("About Rejection Sampling")
    
    st.markdown("""
    **Rejection Sampling** is a Bayesian inference method that:
    
    1. **Evaluates likelihood** of observed data given each prior model
    2. **Applies temperature annealing** to control acceptance rates  
    3. **Accepts/rejects** prior samples based on data fit quality
    4. **Generates posterior** ensemble of models consistent with data
    
    **Key Parameters:**
    - **N_use**: Number of prior samples to evaluate (more = better coverage, slower)
    - **nr**: Posterior samples per data point (more = better statistics) 
    - **autoT**: Automatic temperature finds optimal acceptance rates
    - **parallel**: Use multiple cores for faster computation
    
    **Output:**
    The output file contains posterior model statistics (mean, std, mode) and
    temperature/evidence fields showing inversion quality.
    """)

if __name__ == "__main__":
    st.set_page_config(page_title="INTEGRATE - Rejection Sampling",
                       page_icon=":material/target:", layout="wide")
    apply_style()
    run_rejection_app()