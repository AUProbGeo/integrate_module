"""
INTEGRATE Prior Model Generation Interface

Streamlit interface for creating prior model ensembles using the INTEGRATE module.
Provides access to prior_model_layered(), prior_model_workbench(), and 
prior_model_workbench_direct() functions.

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

def run_prior_app():
    page_header("Prior Model Generation",
                "Generate prior model ensembles for Bayesian inversion, based on "
                "your geological assumptions.")
    
    # Model selection
    model_type = st.selectbox(
        "Select Prior Model Type",
        [
            "prior_model_layered",
            "prior_model_workbench", 
            "prior_model_workbench_direct"
        ]
    )
    
    # Show existing H5 files
    h5_files = get_h5_files()
    if h5_files:
        st.subheader("Existing H5 Files")
        selected_file = st.selectbox("Select file to inspect (optional):", ["None"] + h5_files)
        if selected_file != "None":
            display_h5_info(selected_file)
    
    st.markdown("---")
    
    if model_type == "prior_model_layered":
        st.subheader("Layered Earth Model Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Layer Parameters**")
            lay_dist = st.selectbox("Layer distribution", ["uniform", "chi2"])
            dz = st.number_input("Depth step (dz)", value=1.0, min_value=0.1)
            z_max = st.number_input("Maximum depth", value=90.0, min_value=1.0)
            NLAY_min = st.number_input("Min layers", value=3, min_value=1)
            NLAY_max = st.number_input("Max layers", value=6, min_value=1)
            
            if lay_dist == "chi2":
                NLAY_deg = st.number_input("Chi2 degrees of freedom", value=6, min_value=1)
            else:
                NLAY_deg = 6
        
        with col2:
            st.write("**Resistivity Parameters**")
            RHO_dist = st.selectbox("Resistivity distribution", 
                                  ["log-uniform", "uniform", "normal", "lognormal"])
            RHO_min = st.number_input("Min resistivity", value=0.1, min_value=0.001)
            RHO_max = st.number_input("Max resistivity", value=100.0, min_value=0.001)
            
            if RHO_dist in ["normal", "lognormal"]:
                RHO_MEAN = st.number_input("Mean resistivity", value=100.0)
                RHO_std = st.number_input("Std resistivity", value=80.0)
            else:
                RHO_MEAN = 100.0
                RHO_std = 80.0
        
        N = st.number_input("Number of realizations", value=100000, min_value=100)
        
        output_filename = st.text_input("Output filename (optional)", 
                                      placeholder="Leave empty for auto-generated name")
        
        if st.button("Generate Layered Model"):
            with st.spinner("Generating prior model..."):
                try:
                    kwargs = {}
                    if output_filename:
                        kwargs['f_prior_h5'] = output_filename
                    
                    f_prior_h5 = ig.prior_model_layered(
                        lay_dist=lay_dist,
                        dz=dz,
                        z_max=z_max,
                        NLAY_min=NLAY_min,
                        NLAY_max=NLAY_max,
                        NLAY_deg=NLAY_deg,
                        RHO_dist=RHO_dist,
                        RHO_min=RHO_min,
                        RHO_max=RHO_max,
                        RHO_MEAN=RHO_MEAN,
                        RHO_std=RHO_std,
                        N=N,
                        **kwargs
                    )
                    
                    st.success(f"✅ Prior model generated successfully!")
                    st.info(f"Saved as: {f_prior_h5}")
                    
                    # Show info about generated file
                    if os.path.exists(f_prior_h5):
                        display_h5_info(f_prior_h5)
                        
                except Exception as e:
                    st.error(f"Error generating model: {str(e)}")
    
    elif model_type == "prior_model_workbench":
        st.subheader("Workbench Model Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Geometry Parameters**")
            p = st.number_input("Power parameter", value=2, min_value=1)
            z1 = st.number_input("Min depth", value=0.0, min_value=0.0)
            z_max = st.number_input("Maximum depth", value=100.0, min_value=1.0)
            dz = st.number_input("Depth step", value=1.0, min_value=0.1)
            
            st.write("**Layer Parameters**")
            lay_dist = st.selectbox("Layer distribution", ["uniform", "chi2"])
            nlayers = st.number_input("Fixed number of layers (0 for variable)", value=0, min_value=0)
            
            if nlayers == 0:
                NLAY_min = st.number_input("Min layers", value=3, min_value=1)
                NLAY_max = st.number_input("Max layers", value=6, min_value=1)
            else:
                NLAY_min = nlayers
                NLAY_max = nlayers
                
            if lay_dist == "chi2":
                NLAY_deg = st.number_input("Chi2 degrees of freedom", value=5, min_value=1)
            else:
                NLAY_deg = 5
        
        with col2:
            st.write("**Resistivity Parameters**")
            RHO_dist = st.selectbox("Resistivity distribution", 
                                  ["log-uniform", "uniform", "normal", "lognormal", "chi2"])
            RHO_min = st.number_input("Min resistivity", value=1.0, min_value=0.001)
            RHO_max = st.number_input("Max resistivity", value=300.0, min_value=0.001)
            RHO_mean = st.number_input("Mean resistivity", value=180.0)
            RHO_std = st.number_input("Std resistivity", value=80.0)
            
            if RHO_dist == "chi2":
                chi2_deg = st.number_input("Chi2 degrees of freedom", value=100, min_value=1)
            else:
                chi2_deg = 100
        
        N = st.number_input("Number of realizations", value=100000, min_value=100)
        
        output_filename = st.text_input("Output filename (optional)", 
                                      placeholder="Leave empty for auto-generated name")
        
        if st.button("Generate Workbench Model"):
            with st.spinner("Generating prior model..."):
                try:
                    kwargs = {}
                    if output_filename:
                        kwargs['f_prior_h5'] = output_filename
                    
                    f_prior_h5 = ig.prior_model_workbench(
                        N=N,
                        p=p,
                        z1=z1,
                        z_max=z_max,
                        dz=dz,
                        lay_dist=lay_dist,
                        nlayers=nlayers,
                        NLAY_min=NLAY_min,
                        NLAY_max=NLAY_max,
                        NLAY_deg=NLAY_deg,
                        RHO_dist=RHO_dist,
                        RHO_min=RHO_min,
                        RHO_max=RHO_max,
                        RHO_mean=RHO_mean,
                        RHO_std=RHO_std,
                        chi2_deg=chi2_deg,
                        **kwargs
                    )
                    
                    st.success(f"✅ Prior model generated successfully!")
                    st.info(f"Saved as: {f_prior_h5}")
                    
                    # Show info about generated file
                    if os.path.exists(f_prior_h5):
                        display_h5_info(f_prior_h5)
                        
                except Exception as e:
                    st.error(f"Error generating model: {str(e)}")
    
    elif model_type == "prior_model_workbench_direct":
        st.subheader("Workbench Direct Model Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Geometry Parameters**")
            z1 = st.number_input("Min depth", value=0.0, min_value=0.0)
            z_max = st.number_input("Maximum depth", value=100.0, min_value=1.0)
            nlayers = st.number_input("Number of layers", value=30, min_value=1)
            p = st.number_input("Power parameter", value=2, min_value=1)
        
        with col2:
            st.write("**Resistivity Parameters**")
            RHO_dist = st.selectbox("Resistivity distribution", 
                                  ["log-uniform", "uniform", "normal", "lognormal", "chi2"])
            RHO_min = st.number_input("Min resistivity", value=1.0, min_value=0.001)
            RHO_max = st.number_input("Max resistivity", value=300.0, min_value=0.001)
            RHO_mean = st.number_input("Mean resistivity", value=180.0)
            RHO_std = st.number_input("Std resistivity", value=80.0)
            
            if RHO_dist == "chi2":
                chi2_deg = st.number_input("Chi2 degrees of freedom", value=100, min_value=1)
            else:
                chi2_deg = 100
        
        N = st.number_input("Number of realizations", value=100000, min_value=100)
        
        output_filename = st.text_input("Output filename (optional)", 
                                      placeholder="Leave empty for auto-generated name")
        
        if st.button("Generate Workbench Direct Model"):
            with st.spinner("Generating prior model..."):
                try:
                    kwargs = {}
                    if output_filename:
                        kwargs['f_prior_h5'] = output_filename
                    
                    f_prior_h5 = ig.prior_model_workbench_direct(
                        N=N,
                        RHO_dist=RHO_dist,
                        z1=z1,
                        z_max=z_max,
                        nlayers=nlayers,
                        p=p,
                        RHO_min=RHO_min,
                        RHO_max=RHO_max,
                        RHO_mean=RHO_mean,
                        RHO_std=RHO_std,
                        chi2_deg=chi2_deg,
                        **kwargs
                    )
                    
                    st.success(f"✅ Prior model generated successfully!")
                    st.info(f"Saved as: {f_prior_h5}")
                    
                    # Show info about generated file
                    if os.path.exists(f_prior_h5):
                        display_h5_info(f_prior_h5)
                        
                except Exception as e:
                    st.error(f"Error generating model: {str(e)}")

if __name__ == "__main__":
    st.set_page_config(page_title="INTEGRATE - Prior Models",
                       page_icon=":material/casino:", layout="wide")
    apply_style()
    run_prior_app()