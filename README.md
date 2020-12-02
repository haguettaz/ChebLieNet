# GroupEquivariantChebNets

To use this code, please follow the procedure:

1. Create a new environment with python 3.8
```
$ conda create -n chebnets python=3.7
```

2. Install pytorch and pytorch-geometric on the new environment:

``` 
$ pip install torch==1.7.0+${CUDA} -f https://download.pytorch.org/whl/torch_stable.html

$ pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
$ pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
$ pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
$ pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
$ pip install torch-geometric
```

where ${CUDA} should be replaced by either cpu, cu92, cu101, cu102, or cu110 depending on your PyTorch installation.

3. Install the package GroupEquivariantChebNets
```
$ pip install -e GroupEquivariantChebNets
```


