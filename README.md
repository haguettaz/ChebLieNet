# GroupEquivariantChebNets

The code in this repo was developed and tested using the following conda environment (TODO add to a yaml file or something):

```

conda create --yes --name gechebnets numpy scipy matplotlib==3.2.2 scikit-learn jupyter pillow==6.2

conda activate gechebnets

conda install pytorch==1.7.0 cudatoolkit=10.1 -c pytorch --yes
conda install torchvision -c pytorch --yes


pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.7.0.html

pip install torch-geometric

pip install pytorch-ignite

pip install tensorboard

```
