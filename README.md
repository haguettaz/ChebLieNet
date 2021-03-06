# ChebLieNet: Invariant spectral graph NNs turned equivariant by Riemannian geometry on Lie groups

[Hugo Aguettaz](https://www.linkedin.com/in/hugo-aguettaz),
[Erik J. Bekkers](https://erikbekkers.bitbucket.io),
[Michaël Defferrard](https://deff.ch)

> We introduce ChebLieNet, a group-equivariant method on (anisotropic) manifolds.
> Surfing on the success of graph- and group-based neural networks, we take advantage of the recent developments in the geometric deep learning field to derive a new approach to exploit any anisotropies in data.
> Via discrete approximations of Lie groups, we develop a graph neural network made of anisotropic convolutional layers (Chebyshev convolutions), spatial pooling and unpooling layers, and global pooling layers.
> Group equivariance is achieved via equivariant and invariant operators on graphs with anisotropic left-invariant Riemannian distance-based affinities encoded on the edges.
> Thanks to its simple form, the Riemannian metric can model any anisotropies, both in the spatial and orientation domains.
> This control on anisotropies of the Riemannian metrics allows to balance equivariance (anisotropic metric) against invariance (isotropic metric) of the graph convolution layers.
> Hence we open the doors to a better understanding of anisotropic properties.
> Furthermore, we empirically prove the existence of (data-dependent) sweet spots for anisotropic parameters on CIFAR10.
> This crucial result is evidence of the benefice we could get by exploiting anisotropic properties in data.
> We also evaluate the scalability of this approach on STL10 (image data) and ClimateNet (spherical data), showing its remarkable adaptability to diverse tasks.

Paper: [`arXiv:2111.12139`](https://arxiv.org/abs/2111.12139), [`OpenReview:WsfXFxqZXRO`](https://openreview.net/forum?id=WsfXFxqZXRO).

## Installation

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/haguettaz/ChebLieNet/outputs?urlpath=lab)
&nbsp; Click the binder badge to run the code from your browser.

1. Optionally, create and activate a virtual environment.
    ```sh
    python -m venv cheblienet
    source cheblienet/bin/activate
    python -m pip install --upgrade pip setuptools wheel
    ```

2. Clone this repository.
    ```sh
    git clone https://github.com/haguettaz/ChebLieNet.git
    ```

3. Install the ChebLieNet package and its dependencies.
    ```sh
    python -m pip install -e ChebLieNet
    ```

## Notebooks

* [`graph_manifold.ipynb`]: building graphs from sampled Lie groups.
* [`eigen_space.ipynb`]: visualizing the Fourier modes (eigenspaces) of Lie groups.
* [`graph_diffusion.ipynb`]: heat diffusion on Lie groups.
* [`nn_layers.ipynb`]: convolution, pooling and unpooling layers.

[`graph_manifold.ipynb`]: https://nbviewer.jupyter.org/github/haguettaz/ChebLieNet/blob/outputs/notebooks/graph_manifold.ipynb
[`nn_layers.ipynb`]: https://nbviewer.jupyter.org/github/haguettaz/ChebLieNet/blob/outputs/notebooks/nn_layers.ipynb
[`eigen_space.ipynb`]: https://nbviewer.jupyter.org/github/haguettaz/ChebLieNet/blob/outputs/notebooks/eigen_space.ipynb
[`graph_diffusion.ipynb`]: https://nbviewer.jupyter.org/github/haguettaz/ChebLieNet/blob/outputs/notebooks/graph_diffusion.ipynb

## Reproducing our results

Train a WideResNet on MNIST with anisotropic kernels.
```sh
python -m train_mnist --path_to_graph ./saved_graphs --path_to_data ./data \
    --res_depth 2 --widen_factor 2 --anisotropic --coupled_sym --cuda
```

Train a WideResNet on CIFAR10 with spatial random pooling and anisotropic kernels.
```sh
python -m train_cifar10 --path_to_graph ./saved_graphs --path_to_data ./data \
    --res_depth 2 --widen_factor 4 --anisotropic --pool --reduction rand --cuda
```

Train a WideResNet on STL10 with spatial random pooling and anisotropic kernels.
```sh
python -m train_stl10 --path_to_graph ./saved_graphs --path_to_data ./data \
    --res_depth 3 --widen_factor 4 --anisotropic --reduction rand --cuda
```

Train a U-Net on ClimateNet with spatial max pooling, average unpooling, and anisotropic kernels.
```sh
python -m train_artc --path_to_graph ./saved_graphs --path_to_data ./data \
    --anisotropic --reduction max --expansion avg --cuda
```

## License & citation

The content of this repository is released under the terms of the [MIT license](LICENSE.txt).
Please cite our paper if you use it.

```
@inproceedings{cheblienet,
  title = {{ChebLieNet}: Invariant spectral graph {NN}s turned equivariant by Riemannian geometry on Lie groups},
  author = {Aguettaz, Hugo and Bekkers, Erik J. and Defferrard, Michaël},
  year = {2021},
  archivePrefix={arXiv},
  eprint={2111.12139},
  url = {https://arxiv.org/abs/2111.12139},
}
```
