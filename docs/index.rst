|Stars| |PyPI| |PyPIDownloads| |Docs|

MultiVeloVAE - Velocity inference from multi-lineage, multi-omic, and multi-sample single-cell data
===============================================================

Understanding how cells differentiate to their final specialized fates is a fundamental problem
in biomedical science. Single-cell multi-omic profiling provides an opportunity to identify dynamic
molecular changes, but new computational approaches are needed to realize this potential. In
particular, previous methods for RNA velocity inference lack support for multi-lineage, multi-sample,
and multi-omic single-cell data and cannot be used to identify differential dynamics. To overcome
these challenges, we introduce **MultiVeloVAE**, a probabilistic framework for multi-sample RNA velocity
inference that integrates single-cell RNA and multi-omic data. MultiVeloVAE models gene expression
and chromatin accessibility on a shared time scale, performs multi-sample inference from datasets
with partially overlapping modalities, accounts for lineage bifurcations, and enables statistical
testing of velocity parameters among cell types and over time. Using newly generated 10X Multiome
datasets from human embryoid bodies and differentiating macrophage cells, we demonstrate that
MultiVeloVAE provides novel insights into chromatin accessibility and gene expression dynamics
during development.

Check out the :doc:`usage` section for further information.


.. toctree::
   :caption: Main
   :maxdepth: 1
   :hidden:

   usage
   api

.. toctree::
   :caption: Notebooks
   :maxdepth: 1
   :hidden:

   Mouse_Brain_scRNA.ipynb
   EB_Multiome
   Mouse_Brain_Multiome
   HSPC_Integration
   HSPC_Macrophage_Integration
   HSPC_scRNA_Multiome_Integration
   Preprocessing_Example


.. |Stars| image:: https://img.shields.io/github/stars/welch-lab/multivelovae?logo=GitHub&color=yellow
   :target: https://github.com/welch-lab/multivelovae/stargazers

.. |PyPI| image:: https://img.shields.io/pypi/v/multivelovae?logo=PyPI
   :target: https://pypi.org/project/multivelovae

.. |PyPIDownloads| image:: https://pepy.tech/badge/multivelovae
   :target: https://pepy.tech/project/multivelovae

.. |Docs| image:: https://readthedocs.org/projects/multivelovae/badge/?version=latest
   :target: https://multivelovae.readthedocs.io
