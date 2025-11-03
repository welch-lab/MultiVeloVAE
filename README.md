# MultiVeloVAE - Velocity inference from multi-lineage, multi-omic, and multi-sample single-cell data
## Package Installation
The package depends on several popular packages in computational biology and machine learning, including [scanpy](https://scanpy.readthedocs.io/en/stable/), [scVelo](https://scvelo.readthedocs.io/en/stable/), [PyTorch](https://pytorch.org/), and [scikit-learn](https://scikit-learn.org/stable/). We suggest using a GPU to accelerate the training process.

To install the MultiVeloVAE package through PyPI:
```
pip install multivelovae
```
And import the package inside python:
```python
import multivelovae as vv
```

## Package Usage
The example notebooks of running the mouse brain and HSPC datasets are located in [paper-notebooks](https://github.com/welch-lab/MultiVeloVAE/tree/main/paper-notebooks).
Processed AnnData objects are shared directly through [figshare](https://figshare.com/articles/dataset/Post-processed_anndata_objects_for_MultiVeloVAE/30280333).
Expected runtimes using RTX3060-level graphics cards can be found inside each notebook.

[This file](https://github.com/welch-lab/MultiVeloVAE/blob/main/paper-notebooks/reproducible_package_versions.txt) lists the versions of packages used to generate manuscript figures.

## TODO
bioconda
