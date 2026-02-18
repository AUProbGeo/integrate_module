"""
INTEGRATE Web Interface - Main Application

This is the main Streamlit application for the INTEGRATE module providing
a web-based interface for probabilistic geophysical data integration.

The application provides access to:
- Prior model generation
- Forward modeling with GA-AEM
- Rejection sampling inversion
- Visualization and plotting tools

Author: Generated for the INTEGRATE module
"""

import streamlit as st
import os
import sys

# Add the parent directory to Python path to import integrate module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def main():
    st.set_page_config(
        page_title="INTEGRATE - Probabilistic Geophysical Data Integration",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🌍 INTEGRATE - Probabilistic Geophysical Data Integration")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("Select a module to get started:")
    
    page = st.sidebar.selectbox(
        "Choose Module",
        [
            "Home",
            "Data Analysis",
            "Prior Model Generation",
            "Forward Modeling",
            "Rejection Sampling",
            "Visualization & Plotting",
            "Query Tool",
        ]
    )
    
    # Home page
    if page == "Home":
        st.header("Welcome to INTEGRATE")
        
        st.markdown("""
        **INTEGRATE** is a Python module for localized probabilistic data integration in geophysics, 
        with a focus on electromagnetic (EM) data analysis. The module implements rejection sampling 
        algorithms for Bayesian inversion and probabilistic data integration.
        
        ### Key Features:
        - 📊 **Data Analysis**: Analyze and inspect HDF5 files with automatic type detection
        - 🎲 **Prior Model Generation**: Create layered earth models with various distributions
        - ⚡ **Forward Modeling**: Electromagnetic forward modeling with GA-AEM integration
        - 🎯 **Rejection Sampling**: Bayesian inversion using temperature-controlled rejection sampling  
        - 📈 **Visualization**: Comprehensive plotting tools for analysis and results interpretation
        - 🔍 **Query Tool**: Compute per-data-point probabilities for user-defined geological features

        ### Getting Started:
        1. Use the sidebar to navigate between modules
        2. Start with **Data Analysis** to inspect existing HDF5 files
        3. Use **Prior Model Generation** to create model ensembles
        4. Apply **Forward Modeling** to compute synthetic data
        5. Run **Rejection Sampling** for probabilistic inversion
        6. Visualize results with **Plotting** tools
        """)
        
        # Display current working directory and available files
        st.subheader("Current Directory")
        cwd = os.getcwd()
        st.text(f"Working directory: {cwd}")
        
        # Show available H5 files
        h5_files = [f for f in os.listdir('.') if f.endswith('.h5')]
        if h5_files:
            st.subheader("Available H5 Files")
            st.write(h5_files)
        else:
            st.info("No H5 files found in current directory")
            
    # Import and run individual modules
    elif page == "Data Analysis":
        try:
            from ig_data import run_data_app
            run_data_app()
        except ImportError:
            st.error("ig_data.py module not found. Please ensure all modules are properly installed.")
            
    elif page == "Prior Model Generation":
        try:
            from ig_prior import run_prior_app
            run_prior_app()
        except ImportError:
            st.error("ig_prior.py module not found. Please ensure all modules are properly installed.")
            
    elif page == "Forward Modeling":
        try:
            from ig_forward import run_forward_app
            run_forward_app()
        except ImportError:
            st.error("ig_forward.py module not found. Please ensure all modules are properly installed.")
            
    elif page == "Rejection Sampling":
        try:
            from ig_rejection import run_rejection_app
            run_rejection_app()
        except ImportError:
            st.error("ig_rejection.py module not found. Please ensure all modules are properly installed.")
            
    elif page == "Visualization & Plotting":
        try:
            from ig_plot import run_plot_app
            run_plot_app()
        except ImportError:
            st.error("ig_plot.py module not found. Please ensure all modules are properly installed.")

    elif page == "Query Tool":
        try:
            from ig_query import run_query_app
            run_query_app()
        except ImportError:
            st.error("ig_query.py module not found. Please ensure all modules are properly installed.")

if __name__ == "__main__":
    main()