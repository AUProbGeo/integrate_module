"""
INTEGRATE Query Tool Interface

Streamlit pane for computing per-data-point probabilities from posterior
realizations using natural-language queries via LLM translation.
"""

import os
import sys

import h5py
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ig_style import apply_style, page_header

try:
    import integrate as ig
except ImportError:
    st.error("Could not import integrate module. Please ensure it is properly installed.")
    st.stop()


def get_posterior_files():
    results = []
    for f in sorted(os.listdir('.')):
        if f.endswith('.h5'):
            try:
                with h5py.File(f, 'r') as hf:
                    if 'i_use' in hf:
                        results.append(f)
            except Exception:
                pass
    return results


def show_prior_model_info(f_prior_h5):
    try:
        with h5py.File(f_prior_h5, 'r') as f:
            model_keys = sorted([k for k in f.keys() if k.startswith('M') and k[1:].isdigit()])
    except Exception as e:
        st.warning(f"Could not read prior file: {e}")
        return

    if not model_keys:
        return

    COL_W = [1, 4, 2, 3]  # im | Name | Type | Depth range

    # Header row
    h = st.columns(COL_W)
    for col, label in zip(h, ['**im**', '**Name**', '**Type**', '**Depth range**']):
        col.markdown(label)
    st.divider()

    for key in model_keys:
        im = int(key[1:])
        info = ig.get_prior_model_info(f_prior_h5, im)
        z = info['z']
        kind = 'DISCRETE' if info['is_discrete'] else 'CONTINUOUS'
        name = str(info['name']) if info['name'] != key else key
        depth_range = f"{z[0]:.1f} – {z[-1]:.1f} m"

        row = st.columns(COL_W)
        row[0].markdown(str(im))
        row[1].markdown(name)
        row[2].markdown(kind)
        row[3].markdown(depth_range)

        if info['is_discrete'] and info['class_id'] is not None and info['class_name'] is not None:
            pairs = [f"**{int(cid)}** {cname}" for cid, cname in
                     zip(info['class_id'].flatten(), info['class_name'].flatten())]
            _, class_col = st.columns([1, sum(COL_W) - 1])
            class_col.caption("  ·  ".join(pairs))


def run_query_app():
    page_header("Query Tool",
                "Compute per-data-point probabilities that posterior realizations satisfy "
                "a plain-English geological query, translated by an LLM.")

    # --- LLM configuration (always first) ---
    st.subheader("LLM")
    env_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if env_key:
        st.caption("Using Claude (`anthropic/claude-sonnet-4-6`).")
        api_key = env_key
        model = 'anthropic/claude-sonnet-4-6'
    else:
        llm_choice = st.radio("LLM provider:", ["Claude", "Ollama"], horizontal=True)
        if llm_choice == "Claude":
            api_key = st.text_input("Anthropic API key:", type="password",
                                    placeholder="sk-ant-...")
            model = 'anthropic/claude-sonnet-4-6'
        else:
            api_key = None
            model = st.text_input("Ollama model:", value="ollama_chat/qwen3:latest",
                                  placeholder="ollama_chat/llama3:latest")
        if not (api_key or llm_choice == "Ollama"):
            st.info("Enter an API key or select Ollama to continue.")
            return

    st.markdown("---")

    # --- Posterior file selection ---
    st.subheader("Posterior File")
    post_files = get_posterior_files()
    if not post_files:
        st.warning("No posterior HDF5 files found in the current directory (files must contain an `i_use` dataset).")
        return

    filter_text = st.text_input("Filter filenames:", placeholder="e.g. SDR")
    if filter_text:
        filtered = [f for f in post_files if filter_text in f]
    else:
        filtered = post_files

    if not filtered:
        st.warning(f"No posterior files match '{filter_text}'.")
        return

    f_post_h5 = st.selectbox("Select posterior file:", filtered)

    # Read prior file path from posterior
    f_prior_h5 = ''
    try:
        with h5py.File(f_post_h5, 'r') as f:
            f_prior_h5 = str(f.attrs.get('f5_prior', ''))
    except Exception as e:
        st.error(f"Could not open posterior file: {e}")
        return

    if not f_prior_h5 or not os.path.isfile(f_prior_h5):
        st.warning(f"Prior file not found: '{f_prior_h5}'. Cannot display model info or run query.")
        return

    st.caption(f"Prior file: `{f_prior_h5}`")

    # --- Prior model info ---
    st.subheader("Available Prior Models")
    show_prior_model_info(f_prior_h5)

    st.markdown("---")

    # --- Query input ---
    st.subheader("Query")
    query_text = st.text_area(
        "Enter query in plain English:",
        placeholder="e.g. What is the probability that cumulative clay thickness exceeds 10 m within 0 to 30 m depth?",
        height=80,
    )

    if st.button("Query", type="primary"):
        if not query_text.strip():
            st.error("Please enter a query.")
            return

        with st.spinner("Translating query with LLM…"):
            try:
                query_dict, interp, sys_prompt = ig.query_from_text(
                    query_text, f_prior_h5, model=model, api_key=api_key
                )
            except Exception as e:
                st.error(f"LLM query translation failed: {e}")
                return

        st.info(f"**Interpretation:** {interp}")

        with st.spinner("Evaluating query over posterior realizations…"):
            try:
                P, meta = ig.query(f_post_h5, query_dict)
            except Exception as e:
                st.error(f"Query evaluation failed: {e}")
                return

        st.success(f"Done. Mean probability: {P.mean():.3f}  |  N locations: {meta['N_data']}")

        # Capture figures created by query_plot and display them
        before = set(plt.get_fignums())
        try:
            ig.query_plot(P, meta, query_text=query_text, interpretation=interp, text_panel=True)
        except Exception as e:
            st.error(f"Plotting failed: {e}")
        after = set(plt.get_fignums())
        for num in sorted(after - before):
            st.pyplot(plt.figure(num))
        plt.close('all')

        import json
        with st.expander("Query JSON"):
            st.code(json.dumps(query_dict, indent=2), language="json")

        with st.expander("System Prompt"):
            st.text(sys_prompt)


if __name__ == "__main__":
    st.set_page_config(page_title="INTEGRATE - Query Tool",
                       page_icon=":material/search:", layout="wide")
    apply_style()
    run_query_app()
