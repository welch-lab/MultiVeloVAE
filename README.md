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
Please feel free to test this method on our previously published 10X Multiome datasets. See https://multivelo.readthedocs.io/en/latest/MultiVelo_Demo.html. The example of running the mouse brain dataset is located in [paper-notebooks](https://github.com/welch-lab/MultiVeloVAE/tree/main/paper-notebooks). Alternatively, you can apply the same training and analysis steps on our single-sample HSPC dataset for which we provide the AnnData objects directly in [figshare](https://multivelo.readthedocs.io/en/latest/MultiVelo_Fig5.html). Expected runtimes can be found inside each notebook.

[This file](https://github.com/welch-lab/MultiVeloVAE/blob/main/paper-notebooks/reproducible_package_versions.txt) lists the versions of packages used to generate manuscript figures.

## TODO
bioconda
readthedocs