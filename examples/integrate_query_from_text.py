# %% [markdown]
# # query_from_text: LLM-powered natural language query translation
#
# `ig.query_from_text()` translates a plain-English description of a
# geological query into a valid query dict for `ig.query()`.
#
# **Requirements:**
# - `pip install anthropic`
# - `ANTHROPIC_API_KEY` environment variable set
# %%
try:
    get_ipython()
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except Exception:
    pass

# %%
import os
import integrate as ig

# %% [markdown]
# ## API key
#
# The key is read from the `ANTHROPIC_API_KEY` environment variable.
# Set it in your shell before running this script:
#   export ANTHROPIC_API_KEY="sk-ant-..."
# The key is passed directly to each `query_from_text()` call via `api_key=`.

# %%
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')  # set via environment variable

# %% [markdown]
# ## Setup
#
# Point to the posterior and prior HDF5 files.
# The prior file is used to extract model metadata (class names, depth range,
# discrete/continuous type) which is included in the LLM prompt automatically.

# %%
f_post_h5 = 'post_DAUGAARD_AVG_WF_id1_2_3_4_5_6_7_8_9_10_11_12_13.h5'

# The prior file path is also stored inside the posterior file, but we need
# it here to inspect available models before calling query_from_text.
import h5py
with h5py.File(f_post_h5, 'r') as f:
    f_prior_h5 = str(f.attrs.get('f5_prior', ''))

print(f"Prior file: {f_prior_h5}")

# %% [markdown]
# ## Inspect available models
#
# Before writing a query, use `get_prior_model_info()` to see what models
# and classes are available. This is the same information passed to the LLM.

# %%
import h5py as _h5
with _h5.File(f_prior_h5, 'r') as f:
    model_keys = sorted([k for k in f.keys() if k.startswith('M') and k[1:].isdigit()])

for key in model_keys:
    im = int(key[1:])
    info = ig.get_prior_model_info(f_prior_h5, im)
    z = info['z']
    kind = 'DISCRETE' if info['is_discrete'] else 'CONTINUOUS'
    print(f"  im={im}: {info['name']}  ({kind})  depth {z[0]:.1f}–{z[-1]:.1f} m")
    if info['is_discrete'] and info['class_id'] is not None:
        for cid, cname in zip(info['class_id'].flatten(), info['class_name'].flatten()):
            print(f"    class {int(cid)} = {cname}")

# %% [markdown]
# ## Example 1: Simple discrete query
#
# Ask in plain English. The LLM reads the available class names and depth
# range from the prior file and produces the correct JSON.

# %%
text1 = "What is the probability that the cumulative thickness of clay exceeds 10 m within 0 to 30 m depth?"
#text1 = "What is the probability that sand does not exist somewhere in the top 20m and that gravel exist in the top 3m?"
#text1 = "What is the probability the top 3m consist of pure clay?"
#text1 = "Hvad er sandynligheden for at finde mindst 10 meter grus i de øverste 30 meter, hvor der ikke er mere end max 3 meter ler i toppen?"
#text1 = "Hvad er sandsynligheden for at find max 10 meter ler?"
#text1 = "Hvor er det miocent sand i det øverste lag?"
#text1 = "Where is Miocene sand found in the first layer?"

query1, interp1 = ig.query_from_text(text1, f_prior_h5=f_prior_h5, api_key=ANTHROPIC_API_KEY)

print(f"Interpretation: {interp1}")
print("Generated query dict:")
import json
print(json.dumps(query1, indent=2))

# %%
P1, meta1 = ig.query(f_post_h5, query1)
print(f"N_data={meta1['N_data']}, mean P={P1.mean():.3f}")
ig.query_plot(P1, meta1, query_text=text1, interpretation=interp1)

# %% [markdown]
# ## Example 2: Continuous model query

# %%
text2 = "Probability that resistivity is below 100 ohm-m for a cumulative thickness of at least 25 m within 0 to 50 m depth."
query2, interp2 = ig.query_from_text(text2, f_prior_h5=f_prior_h5, api_key=ANTHROPIC_API_KEY)

print(f"Interpretation: {interp2}")
print("Generated query dict:")
print(json.dumps(query2, indent=2))

# %%
P2, meta2 = ig.query(f_post_h5, query2)
print(f"N_data={meta2['N_data']}, mean P={P2.mean():.3f}")
ig.query_plot(P2, meta2, query_text=text2, interpretation=interp2)

# %% [markdown]
# ## Example 3: Multi-constraint AND query

# %%
text3 = ("Probability that sand and gravel together have a cumulative thickness above 20 m "
         "within 0 to 30 m depth, AND the first non-sand/gravel layer at the top is less than 3 m thick.")
query3, interp3 = ig.query_from_text(text3, f_prior_h5=f_prior_h5, api_key=ANTHROPIC_API_KEY)

print(f"Interpretation: {interp3}")
print("Generated query dict:")
print(json.dumps(query3, indent=2))

# %%
P3, meta3 = ig.query(f_post_h5, query3)
print(f"N_data={meta3['N_data']}, mean P={P3.mean():.3f}")
ig.query_plot(P3, meta3, query_text=text3, interpretation=interp3)

# %% [markdown]
# ## Example 4: Save and reuse the generated query
#
# Once satisfied with the LLM-generated query, save it to JSON so it can
# be reused without another API call.

# %%
ig.save_query(query1, 'query_llm_clay10m.json')

# Reload and run
q_loaded = ig.load_query('query_llm_clay10m.json')
P_loaded, meta_loaded = ig.query(f_post_h5, q_loaded)
print(f"Reloaded query | mean P={P_loaded.mean():.3f}")

# %% [markdown]
# ## Example 5: Verbose mode — inspect the LLM prompt and response
#
# Use `verbose=True` to print the full system prompt sent to the LLM
# and the raw response. Useful for debugging or understanding what
# context the LLM receives.

# %%
query5, interp5 = ig.query_from_text(
    "Probability that clay is present for more than 5 m within 0 to 20 m depth.",
    f_prior_h5=f_prior_h5,
    api_key=ANTHROPIC_API_KEY,
    verbose=True,
)
print("Generated query dict:")
print(json.dumps(query5, indent=2))

# %% [markdown]
# ## Example 6: Unsupported query
#
# If the query cannot be expressed with the current constraint schema,
# the LLM responds with `UNSUPPORTED: <reason>` and a `ValueError` is raised.

# %%
try:
    query_bad, _ = ig.query_from_text(
        "What is the spatial correlation length of resistivity?",
        f_prior_h5=f_prior_h5,
        api_key=ANTHROPIC_API_KEY,
    )
except ValueError as e:
    print(f"Caught expected error: {e}")
