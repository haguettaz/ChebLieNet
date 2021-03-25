# Group Equivariant ChebNets
[Hugo Aguettaz], [Erik J Bekkers], [Michaël Defferrard]

[Hugo Aguettaz]: https://people.epfl.ch/hugo.aguettaz
[Erik J Bekkers]: https://erikbekkers.bitbucket.io/
[Michaël Defferrard]: https://deff.ch/

> We introduce group manifold graph neural networks, a novel approach designed  to  perform  anisotropic  convolution  on  data  lying on  a  group  manifold.
> Recently, much  research  has  been  done  in  group  equivariant  neural  networks  and  graph  neural  networks. In this project, we combine both to create group graph equivariant neural networks, a stable andeasy to control algorithm. 
> Group equivariance is achieved via anisotropic spectral (Chebyshev) graph convolutions on graphs with anisotropic left-invariant Riemannian distance-based affinities encoded on the edges.  
> We show that this method gives promising results, performing better than isotropic kernels on CIFAR10, STL10, and ClimateNet while being highly adaptable.  We discuss the different hyper-parameters of this approach and gives the first ablation study of such neural networks.

[PyTorch]: https://pytorch.org

## Installation

To use this package, we recommend using a conda environment:

1. Create a new environment with python 3.7 and activate it
```
$ conda create -n chebnets python=3.7
$ conda activate chebnets
```
2. Install GroupEquivariantChebNets (and all dependencies):
```
$ pip3 install -e GroupEquivariantChebNets
```

## Notebooks

* [`group_manifold_graph.ipynb`]: group manifold graphs.
* [`nn_layers.ipynb`]: convolution, pooling and unpooling layers.
* [`eigen_space.ipynb`]: eigenmaps of the group manifold graphs.

[`group_manifold_graph.ipynb`]: https://github.com/ebekkers/GroupEquivariantChebNets/blob/main/notebooks/graph_manifold.ipynb
[`nn_layers.ipynb`]: https://github.com/github/ebekkers/GroupEquivariantChebNets/blob/main/notebooks/nn_layers.ipynb
[`eigen_space.ipynb`]: https://github.com/github/ebekkers/GroupEquivariantChebNets/blob/main/notebooks/eigen_space.ipynb

## Reproducing our results

Run the below to train a GroupEquivariantChebNet on MNIST, CIFAR10, STL10 or ClimateNet.
1. Create data folder to download data into and move on the GroupEquivariantChebNets folder
```
$ mkdir data
$ cd mkdir GroupEquivariantChebNets
```
3. Train WideResNet on MNIST with anisotropic kernels:
```
$ python3 -m scripts.train_mnist --path_to_graph "$HOME"/GroupEquivariantChebNets/graphs/saved_graphs --path_to_data "$HOME"/data --res_depth 2 --widen_factor 2 --anisotropic --coupled_sym
```
3. Train WideResNet on CIFAR10 with spatial rand pooling and anisotropic kernels:
```
$ python3 -m scripts.train_cifar10 --path_to_graph "$HOME"/GroupEquivariantChebNets/graphs/saved_graphs --path_to_data "$HOME"/data --res_depth 2 --widen_factor 4 --anisotropic --pool --reduction rand
```
4. Train WideResNet on STL10 with spatial rand pooling and anisotropic kernels:
```
$ python3 -m scripts.train_stl10 --path_to_graph "$HOME"/GroupEquivariantChebNets/graphs/saved_graphs --path_to_data "$HOME"/data --res_depth 3 --widen_factor 4 --anisotropic --pool --reduction rand
```
5. Train U-Net on ClimateNet with spatial max pooling, average unpooling and anisotropic kernels:
```
$ python3 -m scripts.train_artc --path_to_graph "$HOME"/GroupEquivariantChebNets/graphs/saved_graphs --path_to_data "$HOME"/data --anisotropic --reduction max --expansion avg
```


## License & citation

The content of this repository is released under the terms of the [MIT license](LICENSE.txt).
