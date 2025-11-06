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
   is_outlier
   regress_out
   filter_genes_dispersion

Velocity inference
-----

.. autosummary::
   :toctree: .

   VAEChrom
   velocity
   velocity_graph

Differential test
--------------------

.. autosummary::
   :toctree: .

   differential_dynamics

Plotting
--------

.. autosummary::
   :toctree: .

   dynamic_plot
   scatter_plot
   velocity_embedding_stream
   differential_dynamics_plot
   decoupling_plot
