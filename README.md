# MultiVeloVAE - Velocity inference from multi-lineage, multi-omic, and multi-sample single-cell data
## Package Installation
The package depends on several packages in computational biology and machine learning, including [scanpy](https://scanpy.readthedocs.io/en/stable/), [scVelo](https://scvelo.readthedocs.io/en/stable/), [MultiVelo](https://multivelo.readthedocs.io/en/latest/), [PyTorch](https://pytorch.org/), [scikit-learn](https://scikit-learn.org/stable/). We suggest using a GPU to accelerate the training process.

We will make the package available on PyPI and Bioconda soon, but you can download and use it now by adding the path of the package to the system path:
```python
import sys
sys.path.append(< path to the package >)
import velovae as vv
```