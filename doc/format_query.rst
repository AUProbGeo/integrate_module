.. _format_query:

Query Tool
==========

Overview
--------

The INTEGRATE query tool evaluates geological queries against a posterior
ensemble, returning either a **probability** or a set of **percentiles** for
each survey data point.

Two query types are supported:

**Probability queries** answer *"what fraction of realizations satisfy condition X?"*

* *Probability that cumulative clay thickness exceeds 10 m within 0–30 m*
* *Probability that resistivity is below 100 Ω·m for at least 25 m*
* *Probability that the water table is shallower than 5 m*

**Percentile queries** answer *"what is the p5/p50/p95 of metric X?"*

* *P5/P50/P95 of cumulative Sand+Grus thickness within 0–30 m*
* *Median thickness of Sand above the water table*

Both query types can be written by hand as Python dicts / JSON files, or
translated automatically from plain English using an LLM via
:func:`ig.query_from_text`.


Core Functions
--------------

* ``ig.query()`` — dispatcher: routes to probability or percentile function based on dict structure
* ``ig.query_probability()`` — probability query (fraction of realizations satisfying constraints)
* ``ig.query_percentile()`` — percentile query (p5/p50/p95 of a metric across realizations)
* ``ig.query_from_text()`` — translate a plain-English query to a query dict using an LLM
* ``ig.title_from_json()`` — generate a plain-English title/description from an existing query dict using an LLM
* ``ig.query_plot()`` — plot the probability map *or* single-point detail view from a probability query
* ``ig.query_percentile_plot()`` — plot one map per percentile from a percentile query
* ``ig.save_query()`` / ``ig.load_query()`` — persist a query dict to/from JSON
* ``ig.get_prior_model_info()`` — inspect model names, types, depth ranges, and class labels for one model
* ``ig.prior_describe()`` — print a human-readable summary of all models in a prior HDF5 file
* ``ig.query_test_llm()`` — verify that an LLM model and API key are working


Query Dict Format
-----------------

Top-level structure
~~~~~~~~~~~~~~~~~~~

A query dict has a single key ``"constraints"`` whose value is a list of
constraint objects.  All constraints are combined with logical **AND**: a
realization is accepted only when it satisfies every constraint simultaneously.

.. code-block:: python

    query = {
        "constraints": [
            { ... },   # constraint 1
            { ... },   # constraint 2 — both must hold
        ]
    }

Constraint Fields
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 12 20 20 26

   * - Field
     - Type
     - Required
     - Valid values
     - Description
   * - ``im``
     - int
     - always
     - 1, 2, 3, …
     - Prior model index (see *Available Models*)
   * - ``classes``
     - list[int]
     - DISCRETE only
     - class IDs from the model
     - Match any of these class IDs
   * - ``value_comparison``
     - str
     - CONTINUOUS / SCALAR
     - ``"<"`` or ``">"``
     - Compare model value against threshold
   * - ``value_threshold``
     - float
     - CONTINUOUS / SCALAR
     - any float
     - Threshold for the value comparison
   * - ``thickness_mode``
     - str
     - depth models only
     - ``"cumulative"`` or ``"first_occurrence"``
     - How to aggregate thickness of matching layers
   * - ``thickness_comparison``
     - str
     - depth models only
     - ``">"``, ``"<"``, ``">="`` or ``"<="``
     - Operator applied to the computed thickness
   * - ``thickness_threshold``
     - float
     - depth models only
     - any float (metres)
     - Thickness threshold in metres
   * - ``depth_min``
     - float
     - optional
     - any float
     - Upper boundary of depth interval [m]
   * - ``depth_max``
     - float
     - optional
     - any float
     - Lower boundary of depth interval [m]
   * - ``depth_max_im``
     - int
     - optional
     - SCALAR model ``im``
     - Per-realization ``depth_max`` from a scalar model
   * - ``depth_min_im``
     - int
     - optional
     - SCALAR model ``im``
     - Per-realization ``depth_min`` from a scalar model
   * - ``negate``
     - bool
     - optional
     - ``true`` / ``false`` (default ``false``)
     - If true, invert the constraint result

**thickness_mode values:**

``"cumulative"``
    Sum the thickness of **all** matching layers within the depth interval.

``"first_occurrence"``
    Thickness of the **first** contiguous run of matching layers.


Model Types
~~~~~~~~~~~

DISCRETE models
    Store integer class IDs at each depth layer (e.g. lithology).
    Use the ``classes`` field to specify which class IDs to match.
    Do not use ``value_comparison`` / ``value_threshold``.

CONTINUOUS models
    Store floating-point values at each depth layer (e.g. resistivity).
    Use ``value_comparison`` + ``value_threshold`` together with the thickness
    fields to express conditions such as "resistivity < 100 Ω·m for >= 25 m".

SCALAR models  *(depth range = 0)*
    Store a **single value** per realization — no depth profile (e.g. a water
    table depth).  Use ``value_comparison`` and ``value_threshold`` only.
    **Omit all thickness and depth fields** — they have no meaning here.

Cross-model depth bounds
    ``depth_max_im`` and ``depth_min_im`` accept the ``im`` index of a SCALAR
    model.  For each realization, the value of that scalar model is used as
    the upper / lower depth boundary.  This enables constraints like "Sand
    above the water table" where the depth cutoff varies per realization.
    These may be combined with fixed ``depth_min`` / ``depth_max``.


Percentile Query Format
~~~~~~~~~~~~~~~~~~~~~~~

A percentile query has a ``"metric"`` key (instead of ``"constraints"``) and
an optional ``"percentiles"`` key.  The metric defines *what to measure* per
realization — the same fields as a constraint, minus the comparison fields
(``thickness_comparison``, ``thickness_threshold``, ``negate``).

.. code-block:: python

    query = {
        "metric": {
            "im": 2,
            "classes": [1, 2],        # Sand or Grus
            "thickness_mode": "cumulative",
            "depth_max": 30.0         # measure within 0–30 m
            # depth_max_im also supported for cross-model depth bounds
        },
        "percentiles": [5, 50, 95]    # optional; default [5, 50, 95]
    }

:func:`ig.query()` auto-detects the query type: dicts with ``"metric"`` are
routed to :func:`ig.query_percentile`; dicts with ``"constraints"`` are routed
to :func:`ig.query_probability`.

**Metric fields** (same as constraint fields minus comparisons):

``im``, ``classes``, ``value_comparison``, ``value_threshold``,
``thickness_mode``, ``depth_min``, ``depth_max``, ``depth_max_im``,
``depth_min_im``.  For SCALAR models, only ``im`` is needed (no thickness
fields).


Saving and Loading Queries
~~~~~~~~~~~~~~~~~~~~~~~~~~

Query dicts can be saved to and loaded from JSON files for reuse without
repeating an LLM call:

.. code-block:: python

    import integrate as ig

    # Save
    ig.save_query(query, 'clay_10m.json')

    # Load and execute (dispatcher routes automatically)
    query = ig.load_query('clay_10m.json')
    result, meta = ig.query(f_post_h5, query)


Running Queries
---------------

Discovering Available Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before writing a query it is useful to inspect which models exist in the prior
file, what type they are, their depth range, and (for discrete models) their
class IDs:

.. code-block:: python

    import integrate as ig
    import h5py

    # Read prior file path from the posterior file
    with h5py.File(f_post_h5, 'r') as f:
        f_prior_h5 = str(f.attrs['f5_prior'])

    # List all models
    with h5py.File(f_prior_h5, 'r') as f:
        model_keys = sorted([k for k in f.keys() if k.startswith('M') and k[1:].isdigit()])

    for key in model_keys:
        im   = int(key[1:])
        info = ig.get_prior_model_info(f_prior_h5, im)
        z    = info['z']
        kind = 'DISCRETE' if info['is_discrete'] else 'CONTINUOUS'
        print(f"  im={im}: {info['name']}  ({kind})  depth {z[0]:.1f}–{z[-1]:.1f} m")
        if info['is_discrete'] and info['class_id'] is not None:
            for cid, cname in zip(info['class_id'].flatten(), info['class_name'].flatten()):
                print(f"    class {int(cid)} = {cname}")

Example output::

    im=1: Resistivity  (CONTINUOUS)  depth 0.0–89.0 m
    im=2: Lithology    (DISCRETE)    depth 0.0–89.0 m
        class 1 = Sand
        class 2 = Grus
        class 3 = Moræneler
        class 4 = Miocene sand
        class 5 = Miocene clay
    im=3: Waterlevel   (CONTINUOUS)  depth 0.0–0.0 m


Executing a Probability Query
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import integrate as ig

    P, meta = ig.query_probability(f_post_h5, query)
    # or equivalently, using the dispatcher:
    P, meta = ig.query(f_post_h5, query)

    print(f"N locations : {meta['N_data']}")
    print(f"Mean P      : {P.mean():.3f}")

**Returns:**

``P``
    ``ndarray`` of shape ``(N_data,)`` — probability [0, 1] for each survey
    data point.

``meta``
    Dict with keys ``'X'``, ``'Y'``, ``'N_data'``, ``'N_post'``,
    ``'i_use'`` (all posterior indices), ``'i_use_query'`` (matching indices
    per location).


Executing a Percentile Query
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    pct_values, meta = ig.query_percentile(f_post_h5, query)
    # or equivalently:
    pct_values, meta = ig.query(f_post_h5, query)

    print(f"Median Sand+Grus thickness: {pct_values[:, 1].mean():.1f} m")

**Returns:**

``pct_values``
    ``ndarray`` of shape ``(N_data, n_percentiles)`` — one column per
    requested percentile, one row per survey location.

``meta``
    Dict with keys ``'X'``, ``'Y'``, ``'N_data'``, ``'N_post'``,
    ``'i_use'``, and ``'percentiles'`` (the list of requested percentile
    values, e.g. ``[5, 50, 95]``).


Visualising Results
~~~~~~~~~~~~~~~~~~~

``ig.query_plot()`` produces **one figure** depending on whether ``ip`` is set:

* **No** ``ip`` → XY probability map across all survey locations.
* **With** ``ip`` → single-point detail view (all posterior realizations + query-matching subset) for that data-point index.  The XY map is **not** shown.

This means ``hardcopy`` always saves exactly one figure regardless of which mode is used.

.. code-block:: python

    # XY probability map (no ip)
    ig.query_plot(P, meta)

    # With a custom title (auto-wrapped at 60 characters per line)
    ig.query_plot(P, meta, title='My Query Title')

    # With LLM-generated title from the query dict
    title = ig.title_from_json(query, f_prior_h5=f_prior_h5)
    ig.query_plot(P, meta, title=title, hardcopy='clay_query')

    # With query text and LLM interpretation in a side panel
    ig.query_plot(P, meta, query_text=text, interpretation=interp, text_panel=True)

    # Single-point detail view — XY map is skipped
    ig.query_plot(P, meta, ip=1000, query_dict=query, f_post_h5=f_post_h5,
                  title=title, hardcopy='clay_query_ip1000')

    # Percentile maps — one subplot per percentile
    ig.query_percentile_plot(pct_values, meta)

    # With text panel and hardcopy
    ig.query_percentile_plot(pct_values, meta,
                             query_text=text,
                             interpretation=interp,
                             text_panel=True,
                             hardcopy='sand_percentiles')


Examples
--------

Example 1: Discrete Cumulative Constraint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Probability that the cumulative thickness of clay (class 3) exceeds 10 m
within 0–30 m depth.*

.. code-block:: python

    import integrate as ig

    query = {
        "constraints": [
            {
                "im": 2,
                "classes": [3],
                "thickness_mode": "cumulative",
                "thickness_comparison": ">",
                "thickness_threshold": 10.0,
                "depth_min": 0.0,
                "depth_max": 30.0,
                "negate": False
            }
        ]
    }

    P, meta = ig.query(f_post_h5, query)
    print(f"Mean P = {P.mean():.3f}")
    ig.query_plot(P, meta)

To match **any** clay type (multiple class IDs), list them all:

.. code-block:: python

    "classes": [3, 5]   # Moræneler OR Miocene clay


Example 2: Continuous Cumulative Constraint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Probability that resistivity is below 100 Ω·m for a cumulative thickness of
at least 25 m within 0–50 m depth.*

.. code-block:: python

    query = {
        "constraints": [
            {
                "im": 1,
                "value_comparison": "<",
                "value_threshold": 100.0,
                "thickness_mode": "cumulative",
                "thickness_comparison": ">=",
                "thickness_threshold": 25.0,
                "depth_min": 0.0,
                "depth_max": 50.0,
                "negate": False
            }
        ]
    }

    P, meta = ig.query(f_post_h5, query)
    ig.query_plot(P, meta)


Example 3: Multi-Constraint AND
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Probability that Sand and Grus together exceed 20 m within 0–30 m depth
AND the first non-sand/gravel layer at the top is less than 3 m thick.*

Both constraints must hold simultaneously.

.. code-block:: python

    query = {
        "constraints": [
            {
                "im": 2,
                "classes": [1, 2],          # Sand or Grus
                "thickness_mode": "cumulative",
                "thickness_comparison": ">",
                "thickness_threshold": 20.0,
                "depth_min": 0.0,
                "depth_max": 30.0
            },
            {
                "im": 2,
                "classes": [1, 2],          # Sand or Grus — negated = "not sand/grus"
                "thickness_mode": "first_occurrence",
                "thickness_comparison": "<",
                "thickness_threshold": 3.0,
                "depth_min": 0.0,
                "depth_max": 30.0,
                "negate": True
            }
        ]
    }

    P, meta = ig.query(f_post_h5, query)
    ig.query_plot(P, meta)


Example 4: Scalar Model Query
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Probability that the water table (im=3) is shallower than 5 m.*

The Waterlevel model has depth range 0–0 m, meaning it stores a single value
per realization.  Thickness fields are not applicable.

.. code-block:: python

    query = {
        "constraints": [
            {
                "im": 3,
                "value_comparison": "<",
                "value_threshold": 5.0,
                "negate": False
            }
        ]
    }

    P, meta = ig.query(f_post_h5, query)
    ig.query_plot(P, meta)


Example 5: Cross-Model Depth Bound
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Probability that Sand and Grus have a cumulative thickness exceeding 5 m
in the zone above the water table.*

``depth_max_im: 3`` instructs the query engine to use the Waterlevel value
(im=3) of each realization as the upper depth cutoff for that realization.

.. code-block:: python

    query = {
        "constraints": [
            {
                "im": 2,
                "classes": [1, 2],          # Sand or Grus
                "thickness_mode": "cumulative",
                "thickness_comparison": ">",
                "thickness_threshold": 5.0,
                "depth_min": 0.0,
                "depth_max_im": 3,          # use Waterlevel per realization
                "negate": False
            }
        ]
    }

    P, meta = ig.query(f_post_h5, query)
    ig.query_plot(P, meta)

Use ``depth_min_im`` symmetrically to set a lower bound from a scalar model
(e.g. "below the water table").


Example 6: Percentile Query — Thickness Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*P5, P50, P95 of the cumulative thickness of Sand and Grus within 0–30 m.*

.. code-block:: python

    query = {
        "metric": {
            "im": 2,
            "classes": [1, 2],          # Sand or Grus
            "thickness_mode": "cumulative",
            "depth_min": 0.0,
            "depth_max": 30.0
        },
        "percentiles": [5, 50, 95]
    }

    pct_values, meta = ig.query_percentile(f_post_h5, query)
    # pct_values shape: (N_data, 3) — columns are P5, P50, P95

    ig.query_percentile_plot(pct_values, meta)

    # Access individual percentile maps
    p50 = pct_values[:, 1]   # median cumulative thickness
    print(f"Median Sand+Grus thickness — spatial mean: {p50.mean():.1f} m")


Example 7: Percentile Query — Cross-Model Depth Bound
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*P5, P50, P95 of the cumulative Sand+Grus thickness above the water table.*

.. code-block:: python

    query = {
        "metric": {
            "im": 2,
            "classes": [1, 2],
            "thickness_mode": "cumulative",
            "depth_min": 0.0,
            "depth_max_im": 3           # per-realization upper bound = Waterlevel
        },
        "percentiles": [5, 50, 95]
    }

    pct_values, meta = ig.query_percentile(f_post_h5, query)
    ig.query_percentile_plot(pct_values, meta,
                             query_text="Sand+Grus thickness above water table",
                             text_panel=True)


LLM-Powered Query Tools
-----------------------

Generating a Description from an Existing Query
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`ig.title_from_json` uses an LLM to produce a short plain-English
sentence describing what an existing query dict computes.  This is useful for
automatically labelling figures or log output without writing titles by hand.

.. code-block:: python

    import integrate as ig

    query = ig.load_query('clay_10m.json')

    # From a file path
    title = ig.title_from_json('clay_10m.json', f_prior_h5=f_prior_h5)

    # From a dict (e.g. returned by ig.load_query())
    title = ig.title_from_json(query, f_prior_h5=f_prior_h5)

    # Use as a figure title
    ig.query_plot(P, meta, title=title, hardcopy='clay_10m')

**Parameters:**

``file_json``
    Path to a JSON file **or** a query dict directly (e.g. from
    :func:`ig.load_query`).

``f_prior_h5`` *(optional)*
    Path to the prior HDF5 file.  When supplied, real model names, depth
    ranges, and class labels are included in the LLM prompt so the description
    uses geological names (e.g. "clay") rather than numeric IDs (e.g. "class 3").

``model``, ``api_key``
    Same as :func:`ig.query_from_text`.

``showInfo`` *(int, default 1)*
    Controls feedback when the LLM cannot be reached:

    * ``0`` — silent; empty string returned with no output.
    * ``1`` — one-line message including a hint to run :func:`ig.query_test_llm` *(default)*.
    * ``2`` — message plus full exception detail.

If the LLM is unavailable for any reason (missing ``litellm`` package, no API
key, network error) the function always returns an empty string — it never
raises — so it is safe to use in a pipeline without extra error handling.


Translating Plain English to a Query Dict
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`ig.query_from_text` uses `LiteLLM <https://docs.litellm.ai>`_ to
translate a plain-English geological question into a valid query dict.  The
LLM receives a structured system prompt that describes:

* both query types (probability and percentile) and when to use each
* the constraint and metric field schemas
* the available prior models for the specific prior file (names, types, depth
  ranges, class IDs)
* worked examples covering all query types

The LLM **auto-detects** the query type from the text:

* "What is the probability that …" → probability query (returns ``"constraints"``)
* "What are the p5/p50/p95 of …" → percentile query (returns ``"metric"`` + ``"percentiles"``)

The returned ``query_dict`` is ready to pass directly to :func:`ig.query`,
which dispatches to the correct function automatically.

Any LiteLLM-supported model works: Claude, GPT-4, or a locally running Ollama
model.


Requirements
~~~~~~~~~~~~

.. code-block:: bash

    pip install litellm

For Claude, set the environment variable before running::

    export ANTHROPIC_API_KEY=sk-ant-...


Testing the Connection
~~~~~~~~~~~~~~~~~~~~~~

Before running queries, verify that the chosen model and key are working:

.. code-block:: python

    import integrate as ig

    # Claude
    ig.query_test_llm(model='anthropic/claude-sonnet-4-6',
                      api_key=os.environ['ANTHROPIC_API_KEY'])

    # Local Ollama
    ig.query_test_llm(model='ollama_chat/qwen3:latest')

A successful test prints ``OK``.  A failed test prints the error message.


Translating a Query
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import integrate as ig, h5py

    with h5py.File(f_post_h5, 'r') as f:
        f_prior_h5 = str(f.attrs['f5_prior'])

    text = (
        "What is the probability that the cumulative thickness of any clay "
        "exceeds 10 m within 0 to 30 m depth?"
    )

    query_dict, interpretation, system_prompt = ig.query_from_text(
        text,
        f_prior_h5=f_prior_h5,
        model='anthropic/claude-sonnet-4-6',
        api_key=os.environ['ANTHROPIC_API_KEY'],
    )

    print("Interpretation:", interpretation)

**Return values:**

``query_dict``
    A valid query dict ready to pass directly to :func:`ig.query`.

``interpretation``
    A 1–2 sentence plain-English confirmation of what the LLM understood the
    query to mean, including the specific classes and thresholds used.
    **Always check this before running the query** — it catches misunderstandings
    cheaply.

``system_prompt``
    The full system prompt that was sent to the LLM.  Useful for auditing or
    debugging.  Can be saved to a file for inspection.


Full Workflow
~~~~~~~~~~~~~

.. code-block:: python

    import os, json
    import integrate as ig, h5py

    with h5py.File(f_post_h5, 'r') as f:
        f_prior_h5 = str(f.attrs['f5_prior'])

    # 1. Translate
    text = "Probability that sand and gravel above the water table exceed 5 m"
    query_dict, interpretation, system_prompt = ig.query_from_text(
        text,
        f_prior_h5=f_prior_h5,
        model='anthropic/claude-sonnet-4-6',
    )

    # 2. Inspect the generated query
    print("Interpretation:", interpretation)
    print(json.dumps(query_dict, indent=2))

    # 3. Execute
    P, meta = ig.query(f_post_h5, query_dict)
    print(f"Mean P = {P.mean():.3f}")

    # 4. Visualise
    ig.query_plot(P, meta,
                  query_text=text,
                  interpretation=interpretation,
                  text_panel=True,
                  hardcopy='sand_above_wl')

    # 5. Save the query for reuse (no LLM call needed next time)
    ig.save_query(query_dict, 'sand_above_wl.json')

Pass ``verbose=True`` to :func:`ig.query_from_text` to print the full system
prompt and raw LLM response — useful for debugging unexpected translations.


Supported Models
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Provider
     - Model string
     - Notes
   * - Anthropic Claude
     - ``'anthropic/claude-sonnet-4-6'``
     - Requires ``ANTHROPIC_API_KEY``
   * - OpenAI
     - ``'openai/gpt-4o'``
     - Requires ``OPENAI_API_KEY``
   * - Ollama (local)
     - ``'ollama_chat/qwen3:latest'``
     - Requires ``ollama serve`` running locally; no API key


Unsupported Queries
~~~~~~~~~~~~~~~~~~~

If the query cannot be expressed with the available constraint schema (for
example, "What is the spatial correlation length of resistivity?"), the LLM
responds with ``UNSUPPORTED: <reason>`` and :func:`ig.query_from_text` raises
a ``ValueError``:

.. code-block:: python

    try:
        query_dict, _, _ = ig.query_from_text(
            "What is the spatial correlation length of resistivity?",
            f_prior_h5=f_prior_h5,
        )
    except ValueError as e:
        print(f"Unsupported query: {e}")


API Reference
-------------

Quick Reference
~~~~~~~~~~~~~~~

.. code-block:: python

    from integrate import (
        query,                  # Dispatcher: routes to probability or percentile
        query_probability,      # Probability query (fraction satisfying constraints)
        query_percentile,       # Percentile query (p5/p50/p95 of a metric)
        query_from_text,        # Translate plain English to query dict via LLM
        title_from_json,        # Generate a plain-English description from a query dict via LLM
        query_plot,             # XY probability map or single-point detail view
        query_percentile_plot,  # Plot one map per percentile
        save_query,             # Save a query dict to a JSON file
        load_query,             # Load a query dict from a JSON file
        get_prior_model_info,   # Return metadata for one prior model
        prior_describe,         # Print a summary of all models in a prior file
        query_test_llm,         # Verify LLM model + API key connectivity
    )

**Key signatures:**

.. code-block:: python

    # Probability query
    P, meta = ig.query_probability(f_post_h5, query_dict)
    # meta keys: 'X', 'Y', 'N_data', 'N_post', 'i_use', 'i_use_query'

    # Percentile query
    pct_values, meta = ig.query_percentile(f_post_h5, query_dict)
    # pct_values shape: (N_data, n_percentiles)
    # meta adds 'percentiles' key

    # Dispatcher (auto-detects from dict structure)
    result, meta = ig.query(f_post_h5, query_dict)

    # LLM translation (auto-detects probability vs percentile from text)
    query_dict, interpretation, system_prompt = ig.query_from_text(
        text, f_prior_h5,
        model='anthropic/claude-sonnet-4-6',
        api_key=None, verbose=False,
    )

    # Plain-English description of an existing query dict (returns '' on LLM failure)
    description = ig.title_from_json(
        file_json,              # str path or dict (e.g. from ig.load_query())
        f_prior_h5=None,        # optional: adds real model/class names to prompt
        model='anthropic/claude-sonnet-4-6',
        api_key=None,
        showInfo=1,             # 0=silent, 1=warn on failure (default), 2=full detail
    )

    # query_plot: ip=None → XY probability map; ip=<int> → single-point detail only
    ig.query_plot(P, meta,
                  ip=None, query_dict=None,
                  f_prior_h5=None, f_post_h5=None,
                  title=None,            # auto-wrapped at 60 chars per line
                  query_text=None, interpretation=None,
                  text_panel=False, hardcopy=False)

    ig.query_percentile_plot(pct_values, meta,
                             query_text=None, interpretation=None,
                             text_panel=False, hardcopy=False)

    ig.save_query(query_dict, path)
    query_dict = ig.load_query(path)

    info = ig.get_prior_model_info(f_prior_h5, im)
    # info keys: 'name', 'is_discrete', 'z', 'class_id', 'class_name'

    # ig.query() returns (None, {}) with a printed message if f_post_h5 is missing
    result, meta = ig.query(f_post_h5, query_dict)

    result = ig.query_test_llm(model, api_key=None, verbose=1)
    # result keys: 'ok', 'model', 'response', 'error'


See Also
--------

* :doc:`format` — General HDF5 data format specifications
* :doc:`format_wells` — Borehole data format and integration workflow
* :doc:`workflow` — Complete inversion workflow
* :doc:`notebooks` — Jupyter notebook examples
