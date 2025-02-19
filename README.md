# MultiVeloVAE - Velocity inference from multi-lineage, multi-omic, and multi-sample single-cell data
## Package Installation
The package depends on several popular packages in computational biology and machine learning, including [scanpy](https://scanpy.readthedocs.io/en/stable/), [scVelo](https://scvelo.readthedocs.io/en/stable/), [PyTorch](https://pytorch.org/), and [scikit-learn](https://scikit-learn.org/stable/). We suggest using a GPU to accelerate the training process.

We will make the package available on PyPI and Bioconda soon, but you can download and use it now by adding the path of the package to the system path:
```python
import sys
sys.path.append(< path to the package >)
import velovae as vv
```

Please feel free to test our method on our previously generated 10X Multiome datasets. See https://multivelo.readthedocs.io/en/latest/MultiVelo_Demo.html. The example to run the mouse brain dataset is located in [paper-notebooks](https://github.com/welch-lab/MultiVeloVAE/tree/main/paper-notebooks).
