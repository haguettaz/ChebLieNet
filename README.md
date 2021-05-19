# GEChebNet:  A group-equivariant neural network to exploit anisotropies via anisotropic manifolds
[Hugo Aguettaz], [Erik J Bekkers], [Michaël Defferrard]

[Hugo Aguettaz]: https://people.epfl.ch/hugo.aguettaz
[Erik J Bekkers]: https://erikbekkers.bitbucket.io/
[Michaël Defferrard]: https://deff.ch/

We introduce Group Equivariant ChebNets, a group-equivariant method on (anisotropic) manifolds. Surfing on the success of graph- and group-based neural networks, we take advantage of the recent developments in the geometric deep learning field to derive a new approach to exploit any anisotropies in data. Via discrete approximations of Lie groups, we develop a graph neural network made of anisotropic convolutional layers (Chebyshev convolutions), spatial pooling and unpooling layers, and global pooling layers. Group equivariance is achieved via equivariant and invariant operators on graphs with anisotropic left-invariant Riemannian distance-based affinities encoded on the edges. Thanks to its simple form, the Riemannian metric can be used to model any anisotropy's intensities, both in the spatial and orientation domains. Hence we open the doors to a better understanding of anisotropic properties. We empirically prove the existence of (data-dependent) sweet spots for anisotropic parameters on CIFAR10. This crucial result is an evidence to the necessity of using tunable anisotropic kernels. We also evaluate the scalability of this approach on STL10 and ClimateNet, showing its great adaptability to diverse tasks and the benefit of using anisotropic kernels.

[PyTorch]: https://pytorch.org

## Installation

We recommend using a virtual environment to install this package, as it does not need root privileges. The steps to follow are:
1. Create a new virtual environment:
```
$ python3 -m venv gechebnet
```
2. Activate it
```
$ source gechebnet/bin/activate
```
3. Install GroupEquivariantChebNets:
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
