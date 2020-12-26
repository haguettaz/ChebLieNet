# GroupEquivariantChebNets

To use this package, we recommend using a conda environment:

1. Create a new environment with python 3.6 and activate it
```
$ conda create -n chebnets python=3.6
$ conda activate chebnets
```

2. Install PyTorch and torch-scatter and torch-sparse from PyTorch Geometric:
```
pip3 install --user torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install --user torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.1+cu101.html
pip3 install --user torch-sparse -f https://https://pytorch-geometric.com/whl/torch-1.7.1+cu101.html
```

3. Install PyKeops modified version with more operations:
```
$ pip3 install -e PyKeops
```

3. Install GroupEquivariantChebNets:
```
$ pip3 install -e GroupEquivariantChebNets
```


