"""
Shared Streamlit progress-bar helper for the INTEGRATE apps.

Bridges the integrate module's progress callback convention
(progress_callback(current, total, info_dict)) to a st.progress bar
with a status line. Used by the prior, forward, and rejection panes.
"""

import streamlit as st

_PHASE_LABELS = {
    'initializing': 'Initializing',
    'generating': 'Generating',
    'computing': 'Computing',
    'sampling': 'Sampling',
    'saving': 'Saving',
    'post_processing': 'Post-processing',
    'completed': 'Completed',
}


def make_progress_callback():
    """Create a progress bar + status line and return (callback, finish).

    callback(current, total, info_dict=None) follows the integrate module
    convention (see integrate_rejection and _report_progress in integrate.py);
    it also tolerates 2-argument calls without info_dict.
    finish(message) fills the bar to 100% and sets a final status message.
    """
    progress_bar = st.progress(0)
    status_text = st.empty()

    def callback(current, total, info_dict=None):
        if total > 0:
            progress_bar.progress(min(current / total, 1.0))
        if info_dict:
            phase = info_dict.get('phase', '')
            phase = _PHASE_LABELS.get(phase, phase)
            status = info_dict.get('status', '')
            status_text.text(f"{phase}: {status}" if status else phase)
        else:
            status_text.text(f"Progress: {current}/{total}")

    def finish(message="Done"):
        progress_bar.progress(1.0)
        status_text.text(message)

    return callback, finish
