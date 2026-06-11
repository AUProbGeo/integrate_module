"""
INTEGRATE Forward Modeling Interface

Streamlit interface for electromagnetic forward modeling using GA-AEM.
Provides access to forward_gaaem() and prior_data_gaaem() functions.

Author: Generated for the INTEGRATE module
"""

import streamlit as st
import os
import sys
import h5py
import numpy as np

# Add the parent directory to Python path to import integrate module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ig_style import apply_style, page_header

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

def get_gex_files():
    """Get list of GEX files in current directory"""
    return [f for f in os.listdir('.') if f.endswith('.gex')]

def get_stm_files():
    """Get list of STM files in current directory"""  
    return [f for f in os.listdir('.') if f.endswith('.stm')]

def run_forward_app():
    page_header("Forward Modeling with GA-AEM",
                "Compute electromagnetic forward modeling using the GA-AEM method. "
                "This generates synthetic data from prior model realizations.")
    
    # File selection
    st.subheader("File Selection")
    
    # Show existing files
    h5_files = get_h5_files()
    gex_files = get_gex_files()
    stm_files = get_stm_files()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Prior Model Files (.h5)**")
        if h5_files:
            f_prior_h5 = st.selectbox("Select prior model file:", h5_files)
            if st.button("Show Prior Info"):
                display_h5_info(f_prior_h5)
        else:
            st.warning("No H5 files found")
            f_prior_h5 = None
    
    with col2:
        st.write("**System Files**")
        if gex_files:
            file_gex = st.selectbox("Select GEX file:", gex_files)
            st.write(f"Selected: {file_gex}")
        else:
            st.warning("No GEX files found")
            file_gex = None
        
        if stm_files:
            st.write("**Available STM files:**")
            for stm in stm_files:
                st.text(f"  {stm}")
    
    if not (f_prior_h5 and file_gex):
        st.error("Please ensure both prior model (.h5) and GEX file are available")
        return
    
    st.markdown("---")
    
    # Parameters
    st.subheader("Forward Modeling Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Parameters**")
        N = st.number_input("Number of soundings (0 = use all)", value=0, min_value=0)
        im = st.number_input("Model index", value=1, min_value=1)
        id = st.number_input("Data index", value=1, min_value=1)
        im_height = st.number_input("Height model index (0 = none)", value=0, min_value=0)
        
        st.write("**Processing Options**")
        doMakePriorCopy = st.checkbox("Make prior copy", value=True)
        parallel = st.checkbox("Use parallel processing", value=True)
        is_log = st.checkbox("Apply log scaling to data", value=False)
    
    with col2:
        st.write("**GA-AEM Parameters**")
        Nhank = st.number_input("Hankel filter points", value=280, min_value=10)
        Nfreq = st.number_input("Number of frequencies", value=12, min_value=1)
        
        st.write("**Processing Control**")
        Ncpu = st.number_input("Number of CPUs (0 = auto)", value=0, min_value=0)
        showInfo = st.selectbox("Verbosity level", [0, 1, 2], index=0)
        
    # Output filename
    st.write("**Output Options**")
    output_filename = st.text_input("Output filename (optional)", 
                                  placeholder="Leave empty for auto-generated name")
    
    # Run forward modeling
    if st.button("Run Forward Modeling", type="primary"):
        if not os.path.exists(f_prior_h5):
            st.error(f"Prior file {f_prior_h5} not found")
            return
        if not os.path.exists(file_gex):
            st.error(f"GEX file {file_gex} not found")
            return
        
        with st.spinner("Running forward modeling... This may take several minutes."):
            try:
                # Prepare progress display
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Starting forward modeling...")
                progress_bar.progress(10)
                
                # Run forward modeling
                f_prior_data_h5 = ig.prior_data_gaaem(
                    f_prior_h5=f_prior_h5,
                    file_gex=file_gex,
                    N=N,
                    doMakePriorCopy=doMakePriorCopy,
                    im=im,
                    id=id,
                    im_height=im_height,
                    Nhank=Nhank,
                    Nfreq=Nfreq,
                    is_log=is_log,
                    parallel=parallel,
                    Ncpu=Ncpu,
                    showInfo=showInfo
                )
                
                progress_bar.progress(100)
                status_text.text("Forward modeling completed!")
                
                st.success(f"✅ Forward modeling completed successfully!")
                st.info(f"Output saved as: {f_prior_data_h5}")
                
                # Show info about generated file
                if os.path.exists(f_prior_data_h5):
                    st.markdown("---")
                    display_h5_info(f_prior_data_h5)
                    
            except Exception as e:
                st.error(f"Error during forward modeling: {str(e)}")
                st.error("This could be due to:")
                st.error("1. GA-AEM not properly installed")
                st.error("2. Invalid GEX file format")
                st.error("3. Incompatible prior model structure")
                st.error("4. System configuration issues")
    
    # Additional information
    st.markdown("---")
    st.subheader("About GA-AEM Forward Modeling")
    
    st.markdown("""
    **GA-AEM (Geophysical Analysis - Airborne Electromagnetic)** is used for computing 
    electromagnetic responses from layered earth models.
    
    **Requirements:**
    - Prior model file with resistivity/conductivity values
    - GEX file containing system configuration
    - STM files for system transfer functions (optional, can be auto-generated)
    
    **Process:**
    1. Load prior model parameters (resistivity, thickness)
    2. Set up electromagnetic system from GEX configuration
    3. Compute forward responses for each model realization
    4. Save synthetic data to output H5 file
    
    **Output:**
    The output file contains forward modeled data (D1, D2, etc.) corresponding 
    to the input model parameters, ready for use in rejection sampling inversion.
    """)

if __name__ == "__main__":
    st.set_page_config(page_title="INTEGRATE - Forward Modeling",
                       page_icon=":material/bolt:", layout="wide")
    apply_style()
    run_forward_app()