# %% [markdown]
# # query: Posterior Query Tool
#
# Computes the probability, per data point, that posterior realizations satisfy
# a user-defined feature constraint.
#
# The core function `query(f_post_h5, query)` takes a posterior HDF5
# file and a query definition (dict or JSON file path) and returns an array of
#
# probabilities – one value per data location.
# %%
try:
    get_ipython()
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except Exception:
    pass

# %%
import json

import h5py
import numpy as np
import matplotlib.pyplot as plt
import integrate as ig

# %% [markdown]
# ## Query structure
#
# A query is a dict (or JSON file) with a `"constraints"` list.
# Constraints are evaluated sequentially (implicit AND): a realization must
# pass **all** constraints to be counted.
#
# ### Discrete-model constraint  (e.g. lithology class, im=2)
# ```json
# {
#   "im": 2,
#   "classes": [1, 2],
#   "thickness_mode": "cumulative",
#   "thickness_comparison": ">",
#   "thickness_threshold": 10.0,
#   "depth_min": 0.0,
#   "depth_max": 30.0,
#   "negate": false
# }
# ```
#
# ### Continuous-model constraint  (e.g. resistivity, im=1)
# ```json
# {
#   "im": 1,
#   "value_comparison": "<",
#   "value_threshold": 500.0,
#   "thickness_mode": "cumulative",
#   "thickness_comparison": ">",
#   "thickness_threshold": 0.0,
#   "depth_min": 0.0,
#   "depth_max": 100.0,
#   "negate": false
# }
# ```
#
# **All fields:**
#
# | Field | Type | Description |
# |---|---|---|
# | `im` | int | Prior model index (1-based) |
# | `classes` | list[int] | Class IDs to match (discrete only) |
# | `value_comparison` | str | `"<"` or `">"` (continuous only) |
# | `value_threshold` | float | Value threshold for continuous condition |
# | `thickness_mode` | str | `"cumulative"` or `"first_occurrence"` |
# | `thickness_comparison` | str | `">"`, `"<"`, `">="`, `"<="` |
# | `thickness_threshold` | float | Thickness [m] to compare against |
# | `depth_min` | float | Optional lower depth bound [m] |
# | `depth_max` | float | Optional upper depth bound [m] |
# | `negate` | bool | If True, accept realizations that do NOT satisfy the constraint |

# %% [markdown]
# ---
# ## Core Functions
#
# The core query functions (query, query_plot, save_query, load_query, get_prior_model_info)
# are available from the integrate module.
# Access them as:
# - ig.query()
# - ig.query_plot()
# - ig.save_query()
# - ig.load_query()
# - ig.get_prior_model_info()
#
# All helper functions and implementation details are in integrate/integrate_query.py


# %% [markdown]
# ---
# ## Examples
#
# The examples below use the posterior and prior files from the `examples/`
# directory. They are guarded with `os.path.isfile` so the script can be
# imported without errors if the files are absent.

# %%
# Select posterior hdf5 file to query
# (here an example the outcome of integrate_workflow.py)
f_post_h5= 'post_DAUGAARD_AVG_WF_id1_2_3_4_5_6_7_8_9_10_11_12_13.h5'
f_post_h5 = 'post_daugaard_valley_new_N1000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nuse2000000_inflateNoise2.h5'
with h5py.File(f_post_h5, 'r') as f:
    f_prior_h5 = str(f.attrs.get('f5_prior', ''))
    f_data_h5 = str(f.attrs.get('f5_data', ''))

print(ig.get_prior_model_info(f_prior_h5, im=1))
print(ig.get_prior_model_info(f_prior_h5, im=2))

# Select data location to plot
ip = 1000

# %% [markdown]
# ### Example 0: Sand and gravel above 30 m depth
#
# Probability that the cumulative thickness of sand (class 2) and gravel (class 5)
# within 0–30 m depth exceeds 20 m.

# %%
query_ex0 = {
    "constraints": [
        {
            "im": 2,
            "classes": [2,5],
            "thickness_mode": "cumulative",
            "thickness_comparison": ">",
            "thickness_threshold": 20.0,
            "depth_min": 0.0,
            "depth_max": 30.0,
            "negate": False
        }
    ]
}

ig.save_query(query_ex0, 'query_ex0.json')

P0, meta0 = ig.query(f_post_h5, query_ex0)
print(f"Example 0 | N_data={meta0['N_data']}, mean P={P0.mean():.3f}")

query_ex0_title = ig.title_from_json(query_ex0, f_prior_h5)
ig.query_plot(P0, meta0, title="Q0: " + query_ex0_title, hardcopy='query_ex0')
ig.query_plot(P0, meta0, ip=ip, query_dict=query_ex0, f_post_h5=f_post_h5,
              title="Q0: " + query_ex0_title, hardcopy='query_ex0_ip%d' % ip)


# %% [markdown]
# ### Example 0b: Q0 with additional constraint on top layer thickness
#
# Same as Q0 (sand and gravel cumulative thickness > 20m within 0-30m)
# BUT with an additional constraint: any top layer that is NOT sand/gravel
# cannot be thicker than 3m.

# %%
query_ex0b = {
    "constraints": [
        {
            "im": 2,
            "classes": [2, 5],
            "thickness_mode": "cumulative",
            "thickness_comparison": ">",
            "thickness_threshold": 20.0,
            "depth_min": 0.0,
            "depth_max": 30.0,
            "negate": False
        },
        {
            "im": 2,
            "classes": [1, 3, 4, 6, 7, 8],  # All classes except sand (2) and gravel (5)
            "thickness_mode": "first_occurrence",
            "thickness_comparison": "<",
            "thickness_threshold": 3.0,
            "depth_min": 0.0,
            "depth_max": 30.0,
            "negate": False
        }
    ]
}

ig.save_query(query_ex0b, 'query_ex0b.json')

P0b, meta0b = ig.query(f_post_h5, query_ex0b)
print(f"Example 0b | N_data={meta0b['N_data']}, mean P={P0b.mean():.3f}")

ip_ex0b = np.argmax(P0b)
query_ex0b_title = ig.title_from_json(query_ex0b, f_prior_h5)
ig.query_plot(P0b, meta0b, title="Q0b: " + query_ex0b_title, hardcopy='query_ex0b')
ig.query_plot(P0b, meta0b, ip=ip_ex0b, query_dict=query_ex0b, f_post_h5=f_post_h5,
              title="Q0b: " + query_ex0b_title, hardcopy='query_ex0b_ip%d' % ip_ex0b)


# %% [markdown]
# ### Example 0c: Q0 with minimum overburden thickness constraint
#
# Same as Q0 (sand and gravel cumulative thickness > 20m within 0-30m)
# BUT with an additional constraint: there must be AT LEAST 2m of overburden
# (non-sand/gravel material) within 0-30m depth.
#
# Overburden = all classes except sand (2) and gravel (5)

# %%
query_ex0c = {
    "constraints": [
        {
            "im": 2,
            "classes": [2, 5],
            "thickness_mode": "cumulative",
            "thickness_comparison": ">",
            "thickness_threshold": 20.0,
            "depth_min": 0.0,
            "depth_max": 30.0,
            "negate": False
        },
        {
            "im": 2,
            "classes": [1, 3, 4, 6, 7, 8],  # Overburden: all classes except sand (2) and gravel (5)
            "thickness_mode": "cumulative",
            "thickness_comparison": ">",
            "thickness_threshold": 2.0,
            "depth_min": 0.0,
            "depth_max": 30.0,
            "negate": False
        }
    ]
}

ig.save_query(query_ex0c, 'query_ex0c.json')

P0c, meta0c = ig.query(f_post_h5, query_ex0c)
print(f"Example 0c | N_data={meta0c['N_data']}, mean P={P0c.mean():.3f}")

ip_ex0c = np.argmax(P0c)
query_ex0c_title = ig.title_from_json(query_ex0c, f_prior_h5)
ig.query_plot(P0c, meta0c, title="Q0c: " + query_ex0c_title, hardcopy='query_ex0c')
ig.query_plot(P0c, meta0c, ip=ip_ex0c, query_dict=query_ex0c, f_post_h5=f_post_h5,
              title="Q0c: " + query_ex0c_title, hardcopy='query_ex0c_ip%d' % ip_ex0c)


# %% [markdown]
# ### Example 0d: Q0 with NO overburden constraint
#
# Same as Q0 (sand and gravel cumulative thickness > 20m within 0-30m)
# BUT with an additional constraint: there can be NO overburden at the top.
# This means either sand (class 2) or gravel (class 5) must be the top layer.
#
# Overburden = all classes except sand (2) and gravel (5)

# %%
query_ex0d = {
    "constraints": [
        {
            "im": 2,
            "classes": [2, 5],
            "thickness_mode": "cumulative",
            "thickness_comparison": ">",
            "thickness_threshold": 20.0,
            "depth_min": 0.0,
            "depth_max": 30.0,
            "negate": False
        },
        {
            "im": 2,
            "classes": [1, 3, 4, 6, 7, 8],  # Overburden: all classes except sand (2) and gravel (5)
            "thickness_mode": "first_occurrence",
            "thickness_comparison": "<",
            "thickness_threshold": 0.1,  # Essentially 0 (with small numerical tolerance)
            "depth_min": 0.0,
            "depth_max": 30.0,
            "negate": False
        }
    ]
}

ig.save_query(query_ex0d, 'query_ex0d.json')

P0d, meta0d = ig.query(f_post_h5, query_ex0d)
print(f"Example 0d | N_data={meta0d['N_data']}, mean P={P0d.mean():.3f}")

query_ex0d_title = ig.title_from_json(query_ex0d, f_prior_h5)

# Comparison figure: all four Q0 variants side by side
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
for ax, P_i, meta_i, lbl in zip(
    axes.flat,
    [P0, P0b, P0c, P0d],
    [meta0, meta0b, meta0c, meta0d],
    [f"Q0: {query_ex0_title}", f"Q0b: {query_ex0b_title}",
     f"Q0c: {query_ex0c_title}", f"Q0d: {query_ex0d_title}"],
):
    ax.scatter(meta_i['X'], meta_i['Y'], c='black', s=2, alpha=0.5)
    sc = ax.scatter(meta_i['X'], meta_i['Y'], c=P_i, cmap='hot_r', vmin=0, vmax=1, s=1)
    plt.colorbar(sc, ax=ax, label='Probability')
    ax.set_xlabel('UTMX [m]')
    ax.set_ylabel('UTMY [m]')
    ax.set_title(lbl)
    ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('query_example0d_comparison.png', dpi=150)
plt.show()

ip_ex0d = np.argmax(P0d)
ig.query_plot(P0d, meta0d, title="Q0d: " + query_ex0d_title, hardcopy='query_ex0d')
ig.query_plot(P0d, meta0d, ip=ip_ex0d, query_dict=query_ex0d, f_post_h5=f_post_h5,
              title="Q0d: " + query_ex0d_title, hardcopy='query_ex0d_ip%d' % ip_ex0d)


# %%
query_ex1 = {
    "constraints": [
        {
            "im": 2,
            "classes": [7],
            "thickness_mode": "cumulative",
            "thickness_comparison": ">",
            "thickness_threshold": 23.0,
            "depth_min": 0.0,
            "depth_max": 100.0,
            "negate": False
        }
    ]
}

ig.save_query(query_ex1, 'query_ex1.json')

P1, meta1 = ig.query(f_post_h5, query_ex1)
print(f"Example 1 | N_data={meta1['N_data']}, mean P={P1.mean():.3f}")

query_ex1_title = ig.title_from_json(query_ex1, f_prior_h5)
ig.query_plot(P1, meta1, title="Q1: " + query_ex1_title, hardcopy='query_ex1')
ig.query_plot(P1, meta1, ip=ip, query_dict=query_ex1, f_post_h5=f_post_h5,
              title="Q1: " + query_ex1_title, hardcopy='query_ex1_ip%d' % ip)


# %% [markdown]
# ### Example 2: Continuous constraint
#
# Probability that resistivity (im=1) is less than 100 ohm-m
# for a cumulative thickness of at least 25 m within 0–50 m depth.

# %%
query_ex2 = {
    "constraints": [
        {
            "im": 1,
            "value_comparison": "<",
            "value_threshold": 100.0,
            "thickness_mode": "cumulative",
            "thickness_comparison": ">",
            "thickness_threshold": 25.0,
            "depth_min": 0.0,
            "depth_max": 50.0,
            "negate": False
        }
    ]
}

ig.save_query(query_ex2, 'query_ex2.json')

P2, meta2 = ig.query(f_post_h5, query_ex2)
print(f"Example 2 | N_data={meta2['N_data']}, mean P={P2.mean():.3f}")

query_ex2_title = ig.title_from_json(query_ex2, f_prior_h5)
ig.query_plot(P2, meta2, title="Q2: " + query_ex2_title, hardcopy='query_ex2')
ig.query_plot(P2, meta2, ip=ip, query_dict=query_ex2, f_post_h5=f_post_h5,
              title="Q2: " + query_ex2_title, hardcopy='query_ex2_ip%d' % ip)


# %% [markdown]
# ### Example 3: Combined constraint (AND)
#
# Probability that:
# 1. Clay (class 2) cumulative thickness > 5 m within 0–20 m, AND
# 2. Resistivity > 500 ohm-m for at least 1 m within 20–60 m.

# %%
query_ex3 = {
    "constraints": [
        {
            "im": 2,
            "classes": [2],
            "thickness_mode": "cumulative",
            "thickness_comparison": ">",
            "thickness_threshold": 5.0,
            "depth_min": 0.0,
            "depth_max": 20.0,
            "negate": False
        },
        {
            "im": 1,
            "value_comparison": ">",
            "value_threshold": 500.0,
            "thickness_mode": "cumulative",
            "thickness_comparison": ">",
            "thickness_threshold": 1.0,
            "depth_min": 20.0,
            "depth_max": 60.0,
            "negate": False
        }
    ]
}

ig.save_query(query_ex3, 'query_ex3.json')

P3, meta3 = ig.query(f_post_h5, query_ex3)
print(f"Example 3 | N_data={meta3['N_data']}, mean P={P3.mean():.3f}")

query_ex3_title = ig.title_from_json(query_ex3, f_prior_h5)
ig.query_plot(P3, meta3, title="Q3: " + query_ex3_title, hardcopy='query_ex3')
ig.query_plot(P3, meta3, ip=ip, query_dict=query_ex3, f_post_h5=f_post_h5,
              title="Q3: " + query_ex3_title, hardcopy='query_ex3_ip%d' % ip)


# %% [markdown]
# ### Example 4: First-occurrence thickness
#
# Probability that the first contiguous occurrence of clay (class 2)
# is less than 5 m thick within 0–30 m depth.

# %%
query_ex4 = {
    "constraints": [
        {
            "im": 2,
            "classes": [2],
            "thickness_mode": "first_occurrence",
            "thickness_comparison": "<",
            "thickness_threshold": 5.0,
            "depth_min": 0.0,
            "depth_max": 30.0,
            "negate": False
        }
    ]
}

ig.save_query(query_ex4, 'query_ex4.json')

P4, meta4 = ig.query(f_post_h5, query_ex4)
print(f"Example 4 | N_data={meta4['N_data']}, mean P={P4.mean():.3f}")

query_ex4_title = ig.title_from_json(query_ex4, f_prior_h5)
ig.query_plot(P4, meta4, title="Q4: " + query_ex4_title, hardcopy='query_ex4')
ig.query_plot(P4, meta4, ip=ip, query_dict=query_ex4, f_post_h5=f_post_h5,
              title="Q4: " + query_ex4_title, hardcopy='query_ex4_ip%d' % ip)


# %% [markdown]
# ### Example 5: Load a saved query from JSON and plot results

# %%
q = ig.load_query('query_ex1.json')
P5, meta5 = ig.query(f_post_h5, q)

query_ex5_title = ig.title_from_json(q, f_prior_h5)
ig.query_plot(P5, meta5, title="Q5 (loaded): " + query_ex5_title, hardcopy='query_ex5')
ig.query_plot(P5, meta5, ip=ip, query_dict=q, f_post_h5=f_post_h5,
              title="Q5 (loaded): " + query_ex5_title, hardcopy='query_ex5_ip%d' % ip)


# %% Query for watertable
f_post_h5 = 'post_SDR_FEDL_ALL_id1.h5'

with h5py.File(f_post_h5, 'r') as f:
    f_prior_h5 = str(f.attrs.get('f5_prior', ''))
    f_data_h5 = str(f.attrs.get('f5_data', ''))

# %%
query_ex_wt = {
    "constraints": [
        {
            "im": 2,
            "classes": [0,1, 2,3,4,5,6,7,8],
            "thickness_mode": "cumulative",
            "thickness_comparison": ">",
            "thickness_threshold": 5.0,
            "depth_min": 0.0,
            "depth_max_im": 3
        }
    ]
}

P_wt, meta_wt = ig.query(f_post_h5, query_ex_wt)

query_ex_wt_title = ig.title_from_json(query_ex_wt, f_prior_h5)
ip_wt = np.argmax(P_wt)
ig.query_plot(P_wt, meta_wt, title="Water table: " + query_ex_wt_title, hardcopy='query_ex_wt')
ig.query_plot(P_wt, meta_wt, ip=ip_wt, query_dict=query_ex_wt, f_post_h5=f_post_h5,
              title="Water table: " + query_ex_wt_title, hardcopy='query_ex_wt_ip%d' % ip_wt)


# %% [markdown]
# ### Example 6: Percentile query — cumulative thickness distribution
#
# Instead of asking "what is the *probability* that Sand+Grus exceeds X m?",
# ask "what are the *p5, p50, p95* of the cumulative Sand+Grus thickness?".
#
# The query dict uses a `"metric"` key instead of `"constraints"`.
# The thickness comparison fields are omitted — we want the raw distribution.

# %%
query_pct = {
    "metric": {
        "im": 2,
        "classes": [1, 2],          # Sand and Grus
        "thickness_mode": "cumulative",
        "depth_min": 0.0,
        "depth_max": 30.0
    },
    "percentiles": [5, 50, 95]
}

ig.save_query(query_pct, 'query_pct.json')

pct_values, meta_pct = ig.query_percentile(f_post_h5, query_pct)
# pct_values shape: (N_data, 3) — columns are P5, P50, P95

p5  = pct_values[:, 0]
p50 = pct_values[:, 1]
p95 = pct_values[:, 2]
print(f"Sand+Grus thickness (0-30 m) | "
      f"P5={p5.mean():.1f} m  P50={p50.mean():.1f} m  P95={p95.mean():.1f} m  (spatial mean)")

query_pct_title = ig.title_from_json(query_pct, f_prior_h5)
ig.query_percentile_plot(pct_values, meta_pct,
                         query_text=query_pct_title,
                         hardcopy='query_pct_sand_grus')


# %% [markdown]
# ### Example 7: Percentile query — cross-model depth bound
#
# P5/P50/P95 of the cumulative Sand+Grus thickness **above the water table**
# (im=3 is a scalar model storing the water-table depth per realization).
# `depth_max_im: 3` sets the upper depth cutoff to the water-table value of
# each realization individually.

# %%
query_pct_wl = {
    "metric": {
        "im": 2,
        "classes": [1, 2],          # Sand and Grus
        "thickness_mode": "cumulative",
        "depth_min": 0.0,
        "depth_max_im": 3           # per-realization depth cutoff = Waterlevel
    },
    "percentiles": [5, 50, 95]
}

pct_wl, meta_pct_wl = ig.query_percentile(f_post_h5, query_pct_wl)

p50_wl = pct_wl[:, 1]
print(f"Sand+Grus above water table | P50 spatial mean = {p50_wl.mean():.1f} m")

query_pct_wl_title = ig.title_from_json(query_pct_wl, f_prior_h5)
ig.query_percentile_plot(pct_wl, meta_pct_wl,
                         query_text=query_pct_wl_title,
                         hardcopy='query_pct_sand_grus_above_wl')

# %%
