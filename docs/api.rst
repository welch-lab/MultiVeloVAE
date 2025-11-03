.. automodule:: multivelovae

API
===

Import MultiVeloVAE as::

    import multivelovae as vv


Preprocessing
-------------

.. autosummary::
   :toctree: .

   aggregate_peaks_10x
   tfidf_norm
   knn_smooth_chrom
   velocity_graph
   is_outlier
   regress_out
   filter_genes_dispersion
   differential_dynamics

Tools
-----

.. autosummary::
   :toctree: .

   VAEChrom
   velocity
   velocity_graph

Plotting
--------

.. autosummary::
   :toctree: .

   dynamic_plot
   scatter_plot

Differential testing
--------------------

.. autosummary::
   :toctree: .

   differential_dynamics
   differential_dynamics_plot
   decoupling_plot
   velocity_embedding_stream
