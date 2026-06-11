"""
INTEGRATE Data Analysis Interface

This module provides a Streamlit interface for analyzing HDF5 files used in the INTEGRATE workflow.
It allows users to inspect HDF5 files, identify their type (DATA, PRIOR, POSTERIOR), and visualize
basic information about the datasets contained within.

Features:
- File selection and filtering
- HDF5 file type detection
- Dataset information display
- Geometry plotting using ig.plot_geometry()
"""

import streamlit as st
import h5py
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from ig_style import apply_style, page_header, file_type_badge

# Add the parent directory to Python path to import integrate module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import integrate as ig
except ImportError:
    st.error("Could not import integrate module. Please ensure it's properly installed.")
    st.stop()

def identify_h5_type(filepath):
    """
    Identify the type of HDF5 file based on its contents according to the official format.
    
    Based on format.rst specification:
    - DATA.h5: Contains observed data with d_obs in /D1/, /D2/, etc.
    - PRIOR.h5: Contains prior model realizations (/M1, /M2, etc.) and data (/D1, /D2, etc.)
    - POST.h5: Contains posterior indices (/i_use), temperature (/T), and evidence (/EV)
    - FORWARD.h5: Contains forward model information with /method attribute
    
    Parameters
    ----------
    filepath : str
        Path to the HDF5 file
        
    Returns
    -------
    str
        File type: 'DATA', 'PRIOR', 'POSTERIOR', 'FORWARD', or 'UNKNOWN'
    """
    try:
        with h5py.File(filepath, 'r') as f:
            keys = list(f.keys())
            
            # Check for POSTERIOR file indicators (POST.h5)
            # Must have /i_use as mandatory field
            if 'i_use' in keys:
                return 'POSTERIOR'
            
            # Check for FORWARD file indicators (FORWARD.h5)
            # Must have /method attribute
            if 'method' in f.attrs:
                return 'FORWARD'
            
            # Check for DATA file indicators (DATA.h5)
            # Look for d_obs in data groups (D1, D2, etc.)
            data_groups = [k for k in keys if k.startswith('D') and k[1:].isdigit()]
            for dgroup in data_groups:
                try:
                    if 'd_obs' in f[dgroup]:
                        return 'DATA'
                except:
                    continue
            
            # Check for geometry indicators typical in DATA files
            geometry_keys = ['UTMX', 'UTMY', 'ELEVATION', 'LINE']
            if any(geo_key in keys for geo_key in geometry_keys):
                return 'DATA'
            
            # Check for PRIOR file indicators (PRIOR.h5)
            # Look for model parameters (M1, M2, etc.) or data arrays (D1, D2, etc.)
            model_groups = [k for k in keys if k.startswith('M') and k[1:].isdigit()]
            if model_groups:
                return 'PRIOR'
            
            # Additional check for PRIOR: data groups without d_obs (prior data)
            if data_groups:
                # If we have data groups but no d_obs, likely PRIOR
                has_d_obs = False
                for dgroup in data_groups:
                    try:
                        if 'd_obs' in f[dgroup]:
                            has_d_obs = True
                            break
                    except:
                        continue
                if not has_d_obs:
                    return 'PRIOR'
            
            return 'UNKNOWN'
    except Exception as e:
        return 'ERROR'

def get_h5_info(filepath):
    """
    Extract basic information from an HDF5 file with format-specific details.
    
    Parameters
    ----------
    filepath : str
        Path to the HDF5 file
        
    Returns
    -------
    dict
        Dictionary containing file information
    """
    info = {
        'filename': os.path.basename(filepath),
        'size_mb': os.path.getsize(filepath) / (1024 * 1024),
        'type': identify_h5_type(filepath),
        'datasets': {},
        'attributes': {},
        'format_details': {}
    }
    
    try:
        with h5py.File(filepath, 'r') as f:
            keys = list(f.keys())
            
            # Get dataset information
            def visit_func(name, obj):
                if isinstance(obj, h5py.Dataset):
                    info['datasets'][name] = {
                        'shape': obj.shape,
                        'dtype': str(obj.dtype),
                        'size_mb': obj.size * obj.dtype.itemsize / (1024 * 1024)
                    }
                    # Add attributes for this dataset
                    attrs = {}
                    for attr_key, attr_value in obj.attrs.items():
                        try:
                            if isinstance(attr_value, bytes):
                                attrs[attr_key] = attr_value.decode('utf-8')
                            else:
                                attrs[attr_key] = str(attr_value)
                        except:
                            attrs[attr_key] = str(type(attr_value))
                    if attrs:
                        info['datasets'][name]['attributes'] = attrs
            
            f.visititems(visit_func)
            
            # Get root attributes
            for key, value in f.attrs.items():
                try:
                    if isinstance(value, bytes):
                        info['attributes'][key] = value.decode('utf-8')
                    else:
                        info['attributes'][key] = str(value)
                except:
                    info['attributes'][key] = str(type(value))
            
            # Add format-specific details based on file type
            if info['type'] == 'DATA':
                info['format_details'] = analyze_data_file(f, keys)
            elif info['type'] == 'PRIOR':
                info['format_details'] = analyze_prior_file(f, keys)
            elif info['type'] == 'POSTERIOR':
                info['format_details'] = analyze_posterior_file(f, keys)
            elif info['type'] == 'FORWARD':
                info['format_details'] = analyze_forward_file(f, keys)
                    
    except Exception as e:
        info['error'] = str(e)
    
    return info

def analyze_data_file(f, keys):
    """Analyze DATA.h5 file structure."""
    details = {'data_groups': [], 'geometry': [], 'noise_models': []}
    
    # Check for geometry data
    geometry_keys = ['UTMX', 'UTMY', 'ELEVATION', 'LINE']
    for geo_key in geometry_keys:
        if geo_key in keys:
            details['geometry'].append(geo_key)
    
    # Check data groups and their noise models
    data_groups = [k for k in keys if k.startswith('D') and k[1:].isdigit()]
    for dgroup in data_groups:
        try:
            group_info = {'name': dgroup}
            if 'd_obs' in f[dgroup]:
                group_info['has_d_obs'] = True
                group_info['d_obs_shape'] = f[dgroup]['d_obs'].shape
            if 'd_std' in f[dgroup]:
                group_info['has_d_std'] = True
            if 'noise_model' in f[dgroup].attrs:
                group_info['noise_model'] = f[dgroup].attrs['noise_model'].decode('utf-8') if isinstance(f[dgroup].attrs['noise_model'], bytes) else str(f[dgroup].attrs['noise_model'])
                details['noise_models'].append(group_info['noise_model'])
            details['data_groups'].append(group_info)
        except:
            continue
    
    return details

def analyze_prior_file(f, keys):
    """Analyze PRIOR.h5 file structure."""
    details = {'model_groups': [], 'data_groups': [], 'n_realizations': 0}
    
    # Check model groups (M1, M2, etc.)
    model_groups = [k for k in keys if k.startswith('M') and k[1:].isdigit()]
    for mgroup in model_groups:
        try:
            group_info = {'name': mgroup, 'shape': f[mgroup].shape}
            if 'name' in f[mgroup].attrs:
                group_info['description'] = f[mgroup].attrs['name'].decode('utf-8') if isinstance(f[mgroup].attrs['name'], bytes) else str(f[mgroup].attrs['name'])
            if 'is_discrete' in f[mgroup].attrs:
                group_info['is_discrete'] = bool(f[mgroup].attrs['is_discrete'])
            details['model_groups'].append(group_info)
            # Track number of realizations from first model group
            if details['n_realizations'] == 0:
                details['n_realizations'] = f[mgroup].shape[0]
        except:
            continue
    
    # Check data groups (D1, D2, etc.)
    data_groups = [k for k in keys if k.startswith('D') and k[1:].isdigit()]
    for dgroup in data_groups:
        try:
            group_info = {'name': dgroup, 'shape': f[dgroup].shape}
            if 'with_noise' in f[dgroup].attrs:
                group_info['with_noise'] = bool(f[dgroup].attrs['with_noise'])
            if 'f5_forward' in f[dgroup].attrs:
                group_info['forward_file'] = f[dgroup].attrs['f5_forward'].decode('utf-8') if isinstance(f[dgroup].attrs['f5_forward'], bytes) else str(f[dgroup].attrs['f5_forward'])
            details['data_groups'].append(group_info)
        except:
            continue
    
    return details

def analyze_posterior_file(f, keys):
    """Analyze POST.h5 file structure."""
    details = {'has_mandatory': [], 'statistics': [], 'n_data': 0, 'n_samples': 0}
    
    # Check mandatory fields
    mandatory_fields = ['i_use', 'T', 'EV']
    for field in mandatory_fields:
        if field in keys:
            details['has_mandatory'].append(field)
            if field == 'i_use' and details['n_data'] == 0:
                try:
                    details['n_data'] = f[field].shape[0]
                    details['n_samples'] = f[field].shape[1] if len(f[field].shape) > 1 else 0
                except:
                    pass
    
    # Check for statistics in model groups
    model_groups = [k for k in keys if k.startswith('M') and k[1:].isdigit()]
    for mgroup in model_groups:
        try:
            stats_in_group = []
            for stat in ['Mean', 'Median', 'Std', 'Mode', 'Entropy', 'P']:
                if stat in f[mgroup]:
                    stats_in_group.append(stat)
            if stats_in_group:
                details['statistics'].append({'group': mgroup, 'stats': stats_in_group})
        except:
            continue
    
    # Check for file references
    if 'f5_data' in f.attrs:
        details['data_file'] = f.attrs['f5_data'].decode('utf-8') if isinstance(f.attrs['f5_data'], bytes) else str(f.attrs['f5_data'])
    if 'f5_prior' in f.attrs:
        details['prior_file'] = f.attrs['f5_prior'].decode('utf-8') if isinstance(f.attrs['f5_prior'], bytes) else str(f.attrs['f5_prior'])
    
    return details

def analyze_forward_file(f, keys):
    """Analyze FORWARD.h5 file structure."""
    details = {}
    
    # Check mandatory attributes
    if 'method' in f.attrs:
        details['method'] = f.attrs['method'].decode('utf-8') if isinstance(f.attrs['method'], bytes) else str(f.attrs['method'])
    if 'type' in f.attrs:
        details['forward_type'] = f.attrs['type'].decode('utf-8') if isinstance(f.attrs['type'], bytes) else str(f.attrs['type'])
    
    return details

def display_h5_info(info):
    """
    Display HDF5 file information in Streamlit with format-specific details.
    
    Parameters
    ----------
    info : dict
        File information dictionary from get_h5_info()
    """
    st.subheader(f"📁 {info['filename']}")
    
    # File metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("File Type")
        st.markdown(file_type_badge(info['type']), unsafe_allow_html=True)
    with col2:
        st.metric("File Size", f"{info['size_mb']:.2f} MB")
    with col3:
        st.metric("Datasets", len(info['datasets']))
    
    # Error handling
    if 'error' in info:
        st.error(f"Error reading file: {info['error']}")
        return
    
    # Datasets information - show first
    if info['datasets']:
        st.subheader("📊 Datasets")
        
        dataset_data = []
        for name, data in info['datasets'].items():
            row = {
                'Name': name,
                'Shape': str(data['shape']),
                'Type': data['dtype'],
                'Size (MB)': f"{data['size_mb']:.3f}"
            }
            # Add attributes column if any dataset has attributes
            if 'attributes' in data and data['attributes']:
                attr_summary = ', '.join([f"{k}={v}" for k, v in list(data['attributes'].items())[:2]])
                if len(data['attributes']) > 2:
                    attr_summary += f", +{len(data['attributes'])-2} more"
                row['Key Attributes'] = attr_summary
            dataset_data.append(row)
        
        st.dataframe(dataset_data, use_container_width=True)
    
    # Format-specific details - show after datasets
    if info.get('format_details'):
        st.subheader(f"📋 {info['type']} File Details")
        display_format_details(info['type'], info['format_details'])
    
    # Root attributes
    if info['attributes']:
        st.subheader("🏷️ File Attributes")
        
        # Display in expandable sections for better organization
        with st.expander("Root Attributes"):
            for key, value in info['attributes'].items():
                st.text(f"{key}: {value}")

def display_format_details(file_type, details):
    """Display format-specific information for different file types."""
    
    if file_type == 'DATA':
        # DATA.h5 specific information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🗺️ Geometry Data:**")
            if details['geometry']:
                for geo in details['geometry']:
                    st.text(f"✓ {geo}")
            else:
                st.text("❌ No geometry data found")
        
        with col2:
            st.markdown("**📡 Noise Models:**")
            if details['noise_models']:
                for model in set(details['noise_models']):
                    st.text(f"• {model}")
            else:
                st.text("No noise models detected")
        
        if details['data_groups']:
            st.markdown("**📊 Data Groups:**")
            for group in details['data_groups']:
                with st.expander(f"Group {group['name']}"):
                    if group.get('has_d_obs'):
                        st.text(f"✓ Observed data (d_obs): {group.get('d_obs_shape', 'N/A')}")
                    if group.get('has_d_std'):
                        st.text("✓ Data uncertainty (d_std)")
                    if 'noise_model' in group:
                        st.text(f"Noise model: {group['noise_model']}")
    
    elif file_type == 'PRIOR':
        # PRIOR.h5 specific information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎲 Model Parameters:**")
            st.metric("Realizations", details['n_realizations'])
            for group in details['model_groups']:
                with st.expander(f"Model {group['name']} - Shape: {group['shape']}"):
                    if 'description' in group:
                        st.text(f"Description: {group['description']}")
                    if 'is_discrete' in group:
                        st.text(f"Type: {'Discrete' if group['is_discrete'] else 'Continuous'}")
        
        with col2:
            st.markdown("**📊 Prior Data:**")
            for group in details['data_groups']:
                with st.expander(f"Data {group['name']} - Shape: {group['shape']}"):
                    if 'with_noise' in group:
                        st.text(f"Noise added: {'Yes' if group['with_noise'] else 'No'}")
                    if 'forward_file' in group:
                        st.text(f"Forward file: {group['forward_file']}")
    
    elif file_type == 'POSTERIOR':
        # POST.h5 specific information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Sampling Results:**")
            st.metric("Data Points", details['n_data'])
            st.metric("Samples per Point", details['n_samples'])
            
            st.markdown("**✅ Mandatory Fields:**")
            for field in details['has_mandatory']:
                st.text(f"✓ {field}")
        
        with col2:
            st.markdown("**📊 Available Statistics:**")
            for stat_group in details['statistics']:
                with st.expander(f"Model {stat_group['group']}"):
                    for stat in stat_group['stats']:
                        st.text(f"• {stat}")
        
        # File references
        if 'data_file' in details or 'prior_file' in details:
            st.markdown("**🔗 File References:**")
            if 'data_file' in details:
                st.text(f"Data file: {details['data_file']}")
            if 'prior_file' in details:
                st.text(f"Prior file: {details['prior_file']}")
    
    elif file_type == 'FORWARD':
        # FORWARD.h5 specific information
        st.markdown("**⚡ Forward Model Configuration:**")
        if 'method' in details:
            st.text(f"Method: {details['method']}")
        if 'forward_type' in details:
            st.text(f"Implementation: {details['forward_type']}")
    
    else:
        st.info("No format-specific details available for this file type.")

def plot_geometry_if_possible(filepath, file_type):
    """
    Plot geometry if the file contains appropriate data.
    
    Parameters
    ----------
    filepath : str
        Path to the HDF5 file
    file_type : str
        Type of the file (DATA, PRIOR, POSTERIOR, etc.)
    """
    try:
        if file_type in ['DATA', 'POSTERIOR']:
            # Try to plot geometry
            fig, ax = plt.subplots(figsize=(10, 6))
            
            try:
                ig.plot_geometry(filepath, ax=ax)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.warning(f"Could not plot geometry: {str(e)}")
                plt.close(fig)
        else:
            st.info("Geometry plotting is only available for DATA and POSTERIOR files.")
            
    except Exception as e:
        st.error(f"Error in geometry plotting: {str(e)}")

def run_data_app():
    """
    Main function to run the data analysis Streamlit app.
    """
    page_header("HDF5 Data Analysis",
                "Analyze and inspect HDF5 files used in the INTEGRATE workflow.")
    
    # Get list of H5 files in current directory
    current_dir = os.getcwd()
    all_h5_files = [f for f in os.listdir(current_dir) if f.endswith('.h5')]
    
    if not all_h5_files:
        st.warning("No H5 files found in the current directory.")
        st.info(f"Current directory: {current_dir}")
        return
    
    # Filter toggle
    st.subheader("🔍 File Selection")
    filter_prior_post = st.checkbox(
        "Filter out PRIOR* and POST* files", 
        value=False,
        help="Hide files that start with 'PRIOR' or 'POST' prefixes"
    )
    
    # Apply filter
    if filter_prior_post:
        h5_files = [f for f in all_h5_files if not (f.startswith('PRIOR') or f.startswith('POST'))]
    else:
        h5_files = all_h5_files
    
    if not h5_files:
        st.warning("No files remaining after filtering.")
        return
    
    # File selection
    selected_file = st.selectbox(
        "Select HDF5 file to analyze:",
        h5_files,
        help="Choose a file to inspect its contents and metadata"
    )
    
    if selected_file:
        filepath = os.path.join(current_dir, selected_file)
        
        # Get file information first to determine type
        with st.spinner("Analyzing file..."):
            info = get_h5_info(filepath)
        
        # Geometry plotting section - show right after file selection for DATA files
        if info['type'] == 'DATA':
            st.subheader("🗺️ Geometry Visualization")
            
            plot_geometry = st.checkbox(
                "Plot geometry",
                help="Visualize the spatial distribution of data points"
            )
            
            if plot_geometry:
                with st.spinner("Generating geometry plot..."):
                    plot_geometry_if_possible(filepath, info['type'])
        
        # Display detailed file information
        display_h5_info(info)
        
        # Raw dataset preview
        st.subheader("👁️ Dataset Preview")
        
        if info['datasets']:
            dataset_names = list(info['datasets'].keys())
            preview_dataset = st.selectbox(
                "Select dataset to preview:",
                ['None'] + dataset_names,
                help="View first few values of selected dataset"
            )
            
            if preview_dataset != 'None':
                try:
                    with h5py.File(filepath, 'r') as f:
                        dataset = f[preview_dataset]
                        
                        # Show dataset info
                        st.text(f"Shape: {dataset.shape}")
                        st.text(f"Type: {dataset.dtype}")
                        
                        # Preview data (limit to reasonable size)
                        if dataset.size > 0:
                            if len(dataset.shape) == 1:
                                # 1D array
                                preview_size = min(20, dataset.shape[0])
                                data_preview = dataset[:preview_size]
                                st.text(f"First {preview_size} values:")
                                st.text(data_preview)
                            elif len(dataset.shape) == 2:
                                # 2D array
                                preview_rows = min(10, dataset.shape[0])
                                preview_cols = min(10, dataset.shape[1])
                                data_preview = dataset[:preview_rows, :preview_cols]
                                st.text(f"First {preview_rows}x{preview_cols} values:")
                                st.dataframe(data_preview)
                            else:
                                st.text("Multi-dimensional array - showing shape and type only")
                        else:
                            st.text("Empty dataset")
                            
                except Exception as e:
                    st.error(f"Error previewing dataset: {str(e)}")

if __name__ == "__main__":
    st.set_page_config(page_title="INTEGRATE - Data Analysis",
                       page_icon=":material/database:", layout="wide")
    apply_style()
    run_data_app()