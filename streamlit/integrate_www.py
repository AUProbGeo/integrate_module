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
import datetime

# Add the parent directory to Python path to import integrate module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ig_style import apply_style


def _run_module(module_name, func_name):
    """Import and run a module's app function, with a friendly error if missing."""
    try:
        module = __import__(module_name)
        getattr(module, func_name)()
    except ImportError:
        st.error(f"{module_name}.py module not found. Please ensure all modules are properly installed.")


def data_analysis():
    _run_module("ig_data", "run_data_app")


def prior_models():
    _run_module("ig_prior", "run_prior_app")


def forward_modeling():
    _run_module("ig_forward", "run_forward_app")


def rejection_sampling():
    _run_module("ig_rejection", "run_rejection_app")


def visualization():
    _run_module("ig_plot", "run_plot_app")


def query_tool():
    _run_module("ig_query", "run_query_app")


# Module cards shown on the home page: (page key, icon, title, description)
_MODULES = [
    ("data", ":material/database:", "Data Analysis",
     "Inspect HDF5 files with automatic detection of DATA, PRIOR, and POSTERIOR types."),
    ("prior", ":material/casino:", "Prior Models",
     "Generate layered earth model ensembles from a range of prior distributions."),
    ("forward", ":material/bolt:", "Forward Modeling",
     "Compute synthetic electromagnetic data with the GA-AEM forward engine."),
    ("rejection", ":material/target:", "Rejection Sampling",
     "Bayesian inversion using temperature-controlled rejection sampling."),
    ("plot", ":material/monitoring:", "Visualization",
     "Publication-quality profiles, maps, and statistical plots of your results."),
    ("query", ":material/search:", "Query Tool",
     "Per-data-point probabilities for user-defined geological features."),
]

# Populated in main() so the home page can link to the navigation pages
_PAGES = {}


def home():
    st.markdown(
        """
        <div style="padding: 0.5rem 0 1.0rem 0;">
            <h1 style="margin-bottom: 0.2rem;">INTEGRATE</h1>
            <p style="color: #5A6B76; font-size: 1.1rem; margin: 0;">
                Localized probabilistic data integration in geophysics — Bayesian
                inversion of electromagnetic data via rejection sampling.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Module cards in a 3-column grid
    cols = st.columns(3)
    for i, (key, icon, title, desc) in enumerate(_MODULES):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"#### {icon} {title}")
                st.caption(desc)
                if key in _PAGES:
                    st.page_link(_PAGES[key], label="Open", icon=":material/arrow_forward:")

    st.markdown("")

    # Workspace overview
    st.subheader(":material/folder_open: Workspace")
    st.markdown(f"Working directory: `{os.getcwd()}`")

    h5_files = sorted(f for f in os.listdir('.') if f.endswith('.h5'))
    if h5_files:
        rows = []
        for f in h5_files:
            stat = os.stat(f)
            rows.append({
                "File": f,
                "Size (MB)": round(stat.st_size / 1e6, 2),
                "Modified": datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No H5 files found in current directory")


def main():
    st.set_page_config(
        page_title="INTEGRATE - Probabilistic Geophysical Data Integration",
        page_icon=":material/public:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    apply_style()

    home_page = st.Page(home, title="Home", icon=":material/home:", default=True)
    _PAGES["data"] = st.Page(data_analysis, title="Data Analysis", icon=":material/database:")
    _PAGES["prior"] = st.Page(prior_models, title="Prior Models", icon=":material/casino:")
    _PAGES["forward"] = st.Page(forward_modeling, title="Forward Modeling", icon=":material/bolt:")
    _PAGES["rejection"] = st.Page(rejection_sampling, title="Rejection Sampling", icon=":material/target:")
    _PAGES["plot"] = st.Page(visualization, title="Visualization", icon=":material/monitoring:")
    _PAGES["query"] = st.Page(query_tool, title="Query Tool", icon=":material/search:")

    nav = st.navigation({
        "INTEGRATE": [home_page],
        "Workflow": [_PAGES["data"], _PAGES["prior"], _PAGES["forward"], _PAGES["rejection"]],
        "Analysis": [_PAGES["plot"], _PAGES["query"]],
    })
    nav.run()


if __name__ == "__main__":
    main()
