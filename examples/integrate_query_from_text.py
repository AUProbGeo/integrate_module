# %% [markdown]
# # query_from_text: LLM-powered natural language query translation
#
# `ig.query_from_text()` translates a plain-English geological query into a
# valid query dict for `ig.query()`. Any LiteLLM-supported model works,
# including local Ollama models.
#
# **Requirements:** `pip install litellm`

# %%
try:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except Exception:
    pass

import os
import json
import h5py
import integrate as ig

# %% [markdown]
# ## Setup
#
# Point to the posterior HDF5 file. The prior file path is read from it
# automatically and used to inject model metadata into the LLM prompt.

# %%
useDaugaard = False
if useDaugaard:
    ig.get_case_data(case='DAUGAARD', loadType='post')
    f_post_h5 = 'POST_DAUGAARD_AVG_prior_detailed_general_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu2000000_aT1.h5'

    with h5py.File(f_post_h5, 'r') as f:
        f_prior_h5 = str(f.attrs.get('f5_prior', ''))

    print(f"Prior file: {f_prior_h5}")
else:
    # Use generic f_post_h5  
    f_post_h5 = 'post_SDR_FEDL_ALL_id2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32_33_34_35_36_37.h5'
    f_post_h5 = 'post_SDR_FEDL_ALL_id1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32_33_34_35_36_37.h5'
    f_post_h5 = 'post_SDR_FEDL_ALL_id1.h5'

    with h5py.File(f_post_h5, 'r') as f:
        f_prior_h5 = str(f.attrs.get('f5_prior', ''))
        f_data_h5 = str(f.attrs.get('f5_data', ''))

# %% [markdown]
# ## Available models
#
# Lists the models and classes available in the prior file.
# This is exactly the information injected into the LLM prompt.

# %%
with h5py.File(f_prior_h5, 'r') as f:
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
# ## LLM selection
#
# Set `usellm` to `'claude'` or `'ollama'` and pick a model.
# Run `ig.query_test_llm()` to verify the connection before running queries.
#
# **Claude:** requires `ANTHROPIC_API_KEY` environment variable.
#
# **Ollama:** requires `ollama serve` running locally; see `ollama list` for
# available models.

# %%
usellm = 'ollama'  # 'claude' or 'ollama'
usellm = 'claude'  # 'claude' or 'ollama'

if usellm == 'claude':
    MODEL = 'anthropic/claude-sonnet-4-6'
    # Only set API_KEY if it is not already defined as a variable
    if 'API_KEY' not in dir():
        API_KEY = os.environ.get('ANTHROPIC_API_KEY')
elif usellm == 'ollama':
    #MODEL = 'ollama_chat/phi4:latest'
    #MODEL = 'ollama_chat/gemma4:latest'
    MODEL = 'ollama_chat/qwen3.6:latest'
    # Other options: ollama_chat/qwen3.6:latest, ollama_chat/gemma4:latest
    API_KEY = None
else:
    raise ValueError(f"Unsupported LLM: {usellm}")

ig.query_test_llm(model=MODEL, api_key=API_KEY)

# %% [markdown]
# ## Example 1: Simple discrete query

# %%
text1 = "What is the probability that Waterlevel is less than 8m"
#text1 = "What is the probability that the cumulative thickness of ANY clay exceeds 10 m within 0 to 30 m depth. Also, the clay MUST be located below the Waterlevel"
#text1 = "What is the probability that the cumulative thickness of meltwater clay exceeds 10 m within 0 to 30 m depth?"

query1, interp1, prompt1 = ig.query_from_text(text1, f_prior_h5=f_prior_h5, model=MODEL, api_key=API_KEY)
print(json.dumps(query1, indent=2))
print(interp1)

# save the prompt for reference
with open('query1_prompt.md', 'w') as f:
    f.write(prompt1)


P1, meta1 = ig.query(f_post_h5, query1)
print(f"N_data={meta1['N_data']}, mean P={P1.mean():.3f}")
ig.query_plot(P1, meta1, query_text=text1, interpretation=interp1,
              text_panel=True, hardcopy='query1_%s' % (MODEL)
)

# %% [markdown]
# ## Example 2: Continuous model query

# %%
text2 = "Probability that resistivity is below 100 ohm-m for a cumulative thickness of at least 25 m within 0 to 50 m depth."

query2, interp2, prompt2 = ig.query_from_text(text2, f_prior_h5=f_prior_h5, model=MODEL, api_key=API_KEY)
print(json.dumps(query2, indent=2))

P2, meta2 = ig.query(f_post_h5, query2)
print(f"N_data={meta2['N_data']}, mean P={P2.mean():.3f}")
ig.query_plot(P2, meta2, query_text=text2, interpretation=interp2,
              text_panel=True, hardcopy='query2_%s' % MODEL.replace('/', '_'))

# %% [markdown]
# ## Example 3: Multi-constraint AND query

# %%
text3 = ("Probability that sand and gravel together have a cumulative thickness above 20 m "
         "within 0 to 30 m depth, AND the first non-sand/gravel layer at the top is less than 3 m thick.")

query3, interp3, prompt3 = ig.query_from_text(text3, f_prior_h5=f_prior_h5, model=MODEL, api_key=API_KEY)
print(json.dumps(query3, indent=2))

P3, meta3 = ig.query(f_post_h5, query3)
print(f"N_data={meta3['N_data']}, mean P={P3.mean():.3f}")
ig.query_plot(P3, meta3, query_text=text3, interpretation=interp3,
              text_panel=True, hardcopy='query3_%s' % MODEL.replace('/', '_'))

# %% [markdown]
# ## Example 4: Save and reload a query
#
# Once satisfied, save the generated query dict to JSON so it can be reused
# without another LLM call.

# %%
ig.save_query(query1, 'query_llm_clay10m.json')

q_loaded = ig.load_query('query_llm_clay10m.json')
P_loaded, meta_loaded = ig.query(f_post_h5, q_loaded)
print(f"Reloaded query | mean P={P_loaded.mean():.3f}")

# %% [markdown]
# ## Example 5: Verbose mode
#
# Pass `verbose=True` to inspect the full system prompt and raw LLM response.
# Useful for debugging or understanding what context the model receives.

# %%
query5, interp5, prompt5 = ig.query_from_text(
    "Probability that any type of clay is present for more than 5 m within 0 to 20 m depth.",
    f_prior_h5=f_prior_h5,
    model=MODEL,
    api_key=API_KEY,
    verbose=True,
)
print(json.dumps(query5, indent=2))

# %% [markdown]
# ## Example 6: Unsupported query
#
# If the query cannot be expressed with the current constraint schema the LLM
# responds with `UNSUPPORTED: <reason>` and a `ValueError` is raised.

# %%
try:
    query_bad, _, _ = ig.query_from_text(
        "What is the spatial correlation length of resistivity?",
        f_prior_h5=f_prior_h5,
        model=MODEL,
        api_key=API_KEY,
    )
except ValueError as e:
    print(f"Caught expected error: {e}")

# %% [markdown]
# ## Probability vs Percentile queries
#
# `query_from_text()` returns **either** a probability query **or** a percentile
# query, depending on how the question is phrased:
#
# | Phrasing | Query type | dict key | Result shape | Plot function |
# |---|---|---|---|---|
# | *"What is the **probability** that X …"* | probability | `"constraints"` | `(N,)` in [0,1] | `ig.query_plot()` |
# | *"What is the **P50 / median / percentile** of X …"* | percentile | `"metric"` | `(N, n_pct)` in physical units | `ig.query_percentile_plot()` |
#
# `ig.query()` dispatches automatically. After the call, check `meta`:
# - `'percentiles' in meta` → percentile result
# - `'i_use_query' in meta` → probability result

# %% [markdown]
# ## Example 7: Percentile query via LLM
#
# Phrasing the question as "P5, P50, and P95" steers the LLM to emit a
# `"metric"` dict, which `ig.query()` routes to `query_percentile()`.

# %%
text7 = (
    "What are the P5, P50, and P95 of cumulative sand and gravel thickness "
    "within 0 to 30 m depth?"
)

query7, interp7, prompt7 = ig.query_from_text(
    text7, f_prior_h5=f_prior_h5, model=MODEL, api_key=API_KEY
)
print(json.dumps(query7, indent=2))
print(interp7)

# query() dispatches to query_percentile() because query7 has a "metric" key
pct7, meta7 = ig.query(f_post_h5, query7)
print(f"Percentiles requested: {meta7['percentiles']}")
print(f"Result shape: {pct7.shape}  (N_data × n_percentiles)")

ig.query_percentile_plot(
    pct7, meta7,
    query_text=text7,
    interpretation=interp7,
    text_panel=True,
    hardcopy="query7_pct_%s" % MODEL.replace("/", "_"),
)

# %% [markdown]
# ## Example 8: Percentile of a continuous parameter via LLM
#
# Percentile queries also work for continuous model parameters such as resistivity.
# Ask for the median (P50) depth at which resistivity drops below a threshold.

# %%
text8 = (
    "What is the median (P50) and uncertainty range (P10 to P90) of the cumulative "
    "thickness where resistivity is below 50 ohm-m within 0 to 60 m depth?"
)

query8, interp8, prompt8 = ig.query_from_text(
    text8, f_prior_h5=f_prior_h5, model=MODEL, api_key=API_KEY
)
print(json.dumps(query8, indent=2))
print(f"Percentiles in query: {query8.get('percentiles', [5, 50, 95])}")

pct8, meta8 = ig.query(f_post_h5, query8)
print(f"Percentiles computed: {meta8['percentiles']}")

ig.query_percentile_plot(
    pct8, meta8,
    query_text=text8,
    interpretation=interp8,
    text_panel=True,
    hardcopy="query8_pct_%s" % MODEL.replace("/", "_"),
)

# %% [markdown]
# ## Example 9: Generic dispatch pattern
#
# When you don't know in advance whether the LLM will return a probability or a
# percentile query, use `'percentiles' in meta` to route to the right plot function.

# %%
text9 = (
    "What is the probability that any type of clay exceeds 15 m cumulative "
    "thickness within 0 to 40 m depth?"
)

query9, interp9, _ = ig.query_from_text(
    text9, f_prior_h5=f_prior_h5, model=MODEL, api_key=API_KEY
)
print(json.dumps(query9, indent=2))

result9, meta9 = ig.query(f_post_h5, query9)

# Route to the correct plot function based on what came back
if "percentiles" in meta9:
    print(f"Percentile query — shape {result9.shape}, percentiles {meta9['percentiles']}")
    ig.query_percentile_plot(result9, meta9, query_text=text9, interpretation=interp9,
                             text_panel=True, hardcopy="query9_%s" % MODEL.replace("/", "_"))
else:
    print(f"Probability query — mean P={result9.mean():.3f}")
    ig.query_plot(result9, meta9, query_text=text9, interpretation=interp9,
                  text_panel=True, hardcopy="query9_%s" % MODEL.replace("/", "_"))

# %% [markdown]
# ## Example 10: Queries constrained by the water table (`depth_max_im`)
#
# Both queries below restrict the depth range to "above the water table".
# The LLM maps this to `depth_max_im` pointing to the water-level model
# parameter, so the depth cutoff varies **per posterior realization** rather
# than using a fixed depth.  Print the generated query dict to confirm that
# `depth_max_im` appears in it.
#
# **10a — Probability**: at least 5 m of raw material (gravel + sand) above the water table.
# **10b — Percentiles**: P2.5 / P50 / P97.5 of raw-material thickness above the water table.

# %%
# 10a — probability query
text10a = (
    "What is the probability that at least 5 m of raw material "
    "(gravel and sand combined) exists above the water table?"
)

query10a, interp10a, _ = ig.query_from_text(
    text10a, f_prior_h5=f_prior_h5, model=MODEL, api_key=API_KEY
)
print("Query 10a:")
print(json.dumps(query10a, indent=2))   # expect "depth_max_im" in the constraint
print(interp10a)

P10a, meta10a = ig.query(f_post_h5, query10a)
print(f"mean P={P10a.mean():.3f}")

ig.query_plot(P10a, meta10a, query_text=text10a, interpretation=interp10a,
              text_panel=True,
              hardcopy="query10a_%s" % MODEL.replace("/", "_"))

# %%
# 10b — percentile query
text10b = (
    "What are the P2.5, P50, and P97.5 percentiles of the cumulative thickness "
    "of raw material (gravel and sand combined) located above the water table?"
)

query10b, interp10b, _ = ig.query_from_text(
    text10b, f_prior_h5=f_prior_h5, model=MODEL, api_key=API_KEY
)
print("Query 10b:")
print(json.dumps(query10b, indent=2))   # expect "metric" key + "depth_max_im"
print(interp10b)

pct10b, meta10b = ig.query(f_post_h5, query10b)
print(f"Percentiles: {meta10b['percentiles']}  —  result shape: {pct10b.shape}")

ig.query_percentile_plot(pct10b, meta10b, query_text=text10b, interpretation=interp10b,
                         text_panel=True,
                         hardcopy="query10b_%s" % MODEL.replace("/", "_"))
