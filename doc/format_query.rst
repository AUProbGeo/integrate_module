.. _format_query:

Query Tool
==========

Overview
--------

The INTEGRATE query tool computes, for each survey data point, the probability
that the posterior realizations at that location satisfy a user-defined
geological constraint.  Constraints express conditions such as:

* *Cumulative thickness of clay exceeds 10 m within 0–30 m depth*
* *Resistivity is below 100 Ω·m for at least 25 m within 0–50 m depth*
* *Water table (scalar model) is shallower than 5 m*
* *Sand and gravel above the water table together exceed 5 m*

Queries can be written by hand as Python dicts / JSON files, or translated
automatically from plain English using an LLM via :func:`ig.query_from_text`.

The primary output is a probability array ``P`` of shape ``(N_data,)`` —
one value per survey location — together with a ``meta`` dict containing
coordinates and the indices of matching posterior realizations.


Core Functions
--------------

* ``ig.query()`` — evaluate a query dict against a posterior file
* ``ig.query_from_text()`` — translate a plain-English query to a query dict using an LLM
* ``ig.query_plot()`` — plot the resulting probability map
* ``ig.save_query()`` / ``ig.load_query()`` — persist a query dict to/from JSON
* ``ig.get_prior_model_info()`` — inspect model names, types, depth ranges, and class labels
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


Saving and Loading Queries
~~~~~~~~~~~~~~~~~~~~~~~~~~

Query dicts can be saved to and loaded from JSON files for reuse without
repeating an LLM call:

.. code-block:: python

    import integrate as ig

    # Save
    ig.save_query(query, 'clay_10m.json')

    # Load and execute
    query = ig.load_query('clay_10m.json')
    P, meta = ig.query(f_post_h5, query)


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


Executing a Query
~~~~~~~~~~~~~~~~~

.. code-block:: python

    import integrate as ig

    P, meta = ig.query(f_post_h5, query)

    print(f"N locations : {meta['N_data']}")
    print(f"Mean P      : {P.mean():.3f}")

**Returns:**

``P``
    ``ndarray`` of shape ``(N_data,)`` — probability [0, 1] for each survey
    data point.

``meta``
    Dict with keys:

    * ``'X'``, ``'Y'`` — coordinate arrays (or ``None``)
    * ``'N_data'``, ``'N_post'`` — number of locations and samples per location
    * ``'i_use'`` — ``ndarray (N_data, N_post)``, all posterior indices
    * ``'i_use_query'`` — list of ``N_data`` arrays, the subset of indices that
      satisfy the query at each location


Visualising Results
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Simple probability map
    ig.query_plot(P, meta)

    # With query text and LLM interpretation in a side panel
    ig.query_plot(P, meta,
                  query_text=text,
                  interpretation=interp,
                  text_panel=True)

    # Save figure to disk
    ig.query_plot(P, meta, hardcopy='clay_query')   # saves clay_query.png

    # Detailed view for one data point (shows posterior models)
    ig.query_plot(P, meta, ip=1000,
                  query_dict=query,
                  f_post_h5=f_post_h5)


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


LLM-Powered Query Translation
------------------------------

Overview
~~~~~~~~

:func:`ig.query_from_text` uses `LiteLLM <https://docs.litellm.ai>`_ to
translate a plain-English geological question into a valid query dict.  The
LLM receives a structured system prompt that describes:

* the constraint schema (all fields and their semantics)
* the available prior models for the specific prior file (names, types, depth
  ranges, class IDs)
* worked examples covering all constraint types

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
        query,               # Execute a query dict against a posterior file
        query_from_text,     # Translate plain English → query dict via LLM
        query_plot,          # Plot the probability map (and optional detail panel)
        save_query,          # Save a query dict to a JSON file
        load_query,          # Load a query dict from a JSON file
        get_prior_model_info,# Return metadata for one prior model (name, z, classes)
        query_test_llm,      # Verify LLM model + API key connectivity
    )

**Key signatures:**

.. code-block:: python

    P, meta = ig.query(f_post_h5, query_dict)

    query_dict, interpretation, system_prompt = ig.query_from_text(
        text, f_prior_h5,
        model='anthropic/claude-sonnet-4-6',
        api_key=None,
        verbose=False,
    )

    ig.query_plot(P, meta,
                  ip=None,
                  query_dict=None,
                  f_post_h5=None,
                  query_text=None,
                  interpretation=None,
                  text_panel=False,
                  hardcopy=False)

    ig.save_query(query_dict, path)
    query_dict = ig.load_query(path)

    info = ig.get_prior_model_info(f_prior_h5, im)
    # info keys: 'name', 'is_discrete', 'z', 'class_id', 'class_name'

    result = ig.query_test_llm(model, api_key=None, verbose=1)
    # result keys: 'ok', 'model', 'response', 'error'


See Also
--------

* :doc:`format` — General HDF5 data format specifications
* :doc:`format_wells` — Borehole data format and integration workflow
* :doc:`workflow` — Complete inversion workflow
* :doc:`notebooks` — Jupyter notebook examples
