# GroupEquivariantChebNets

To use this package, we recommend using a conda environment:

1. Create a new environment with python 3.8 and activate it
```
$ conda create -n chebnets python=3.8
$ conda activate chebnets
```

2. Install pytorch and pytorch-geometric on the new environment:

``` 
$ pip3 install torch==1.7.0+${CUDA} -f https://download.pytorch.org/whl/torch_stable.html

$ pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
$ pip3 install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
$ pip3 install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
$ pip3 install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
$ pip3 install torch-geometric
```

where `${CUDA}` should be replaced by either `cpu`, `cu92`, `cu101`, `cu102`, or `cu110` depending on your PyTorch installation.

3. Install the package GroupEquivariantChebNets
```
$ pip3 install -e GroupEquivariantChebNets
```


