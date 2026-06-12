"""
Shared visual styling for the INTEGRATE Streamlit apps.

Provides a common look across the main app (integrate_www.py) and the
standalone modules (ig_data.py, ig_prior.py, ...). The color palette is
defined in .streamlit/config.toml; this module adds web fonts and a few
reusable UI helpers (page headers, file-type badges).
"""

import streamlit as st

_CSS = """
/* Load web fonts only; the theme in .streamlit/config.toml applies them.
   Do NOT set font-family on Streamlit elements here - broad overrides break
   the Material Symbols icon font (icons render as overlapping raw text). */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

h1, h2, h3 {
    letter-spacing: -0.02em;
}

/* Page header block */
.ig-page-header {
    margin-bottom: 0.25rem;
}
.ig-page-header h2 {
    margin: 0 0 0.15rem 0;
    padding: 0;
    font-size: 1.7rem;
}
.ig-page-header p {
    margin: 0;
    color: #5A6B76;
    font-size: 0.95rem;
}

/* Pill badges for HDF5 file types */
.ig-badge {
    display: inline-block;
    padding: 2px 12px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.ig-badge-data      { background: #E3F2E5; color: #2E933C; }
.ig-badge-prior     { background: #DFEFF6; color: #1B6E9C; }
.ig-badge-posterior { background: #FBF3DC; color: #9A7400; }
.ig-badge-forward   { background: #F8E8DC; color: #B05E1E; }
.ig-badge-unknown   { background: #ECEFF1; color: #5A6B76; }
.ig-badge-error     { background: #F7E4E0; color: #C44536; }
"""


def apply_style():
    """Inject the shared CSS. Call once after st.set_page_config()."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def page_header(title, subtitle=""):
    """Render a consistent page header with title and muted subtitle."""
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="ig-page-header"><h2>{title}</h2>{subtitle_html}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")


def file_type_badge(file_type):
    """Return an HTML pill badge for an HDF5 file type (DATA, PRIOR, ...)."""
    kind = str(file_type).lower()
    if kind not in ("data", "prior", "posterior", "forward", "error"):
        kind = "unknown"
    return f'<span class="ig-badge ig-badge-{kind}">{file_type}</span>'
