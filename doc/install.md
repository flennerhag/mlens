# Installation

To install the package, you need the following dependencies: 

| Package | Version        |
| ------- | -------------- |
| scipy   | >= 0.17        |
| numpy   | >= 1.11.0      |
| pandas  | >= 0.19.0      |
| scikit-learn | >= 0.18.1 |
| matplotlib | >= 1.5.1    |
| seaborn |>= 0.7.1        |

You can install ``mlens`` either through PyPi or the ``python setyp.py install`` command:

# PyPI

### Stable version

```bash
pip install mlens  
```

### Bleeding edge

To ensure latest version is installed, fork the GitHub repository and install mlxtens using the symlink options.

```bash
git clone https://flennerhag/mlens.git; cd mlens;
pip install -e .
```

To update the package, pull the latest changes from the github repo

# Fork GitHub Repository

```bash
git clone https://flennerhag/mlens.git; cd mlens;
python setup.py install
```

Repeat to update package.
