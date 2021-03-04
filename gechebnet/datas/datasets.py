import itertools
import os
from typing import Optional

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive

from .transforms import Compose


class ARTCDataset(Dataset):
    """
    Dataset for reduced atmospheric river and tropical cyclone dataset.

    Credits. https://github.com/deepsphere/deepsphere-pytorch
    """

    resource = "http://island.me.berkeley.edu/ugscnn/data/climate_sphere_l5.zip"

    def __init__(
        self,
        path_to_data: str,
        indices: Optional[list] = None,
        transform_image: Optional[Compose] = None,
        transform_target: Optional[Compose] = None,
        download: Optional[bool] = False,
    ):
        """
        Initialization.

        Args:
            path_to_data (str): path to data directory.
            indices (list, optional): list of indices representing the subset of the data used for the current dataset.
            transform_image (:obj:`Compose`, optional): data' transformations.
            transform_target (:obj:`Compose`, optional): labels' transformations.
            download (bool, optional): if True, downloads the dataset from the internet and puts it in data directory.
                If dataset is already downloaded, it is not downloaded again.. Defaults to False.
        """
        self.path_to_data = path_to_data
        if download:
            self.download()
        self.files = indices if indices is not None else os.listdir(self.path_to_data)
        self.transform_image = transform_image
        self.transform_target = transform_target

    @property
    def indices(self):
        """
        Get files.

        Returns:
            (list): list of strings, which represent the files contained in the dataset.
        """
        return self.files

    def __len__(self):
        """
        Get length of dataset.

        Returns:
            (int): number of files contained in the dataset.
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): index of the desired datapoint.

        Returns:
            (tuple): (image, target) where target is index of the target class.
        """
        item = np.load(os.path.join(self.path_to_data, self.files[idx]))
        image, target = item["data"], item["labels"]
        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_target:
            target = self.transform_target(target)
        return image, target

    def get_runs(self, runs):
        """
        Get datapoints corresponding to specific runs.

        Args:
            runs (list): list of desired runs.

        Returns:
            (list): list of strings, which represents the files in the dataset, which belong to one of the desired runs.
        """
        files = []
        for file in self.files:
            for i in runs:
                if file.endswith("{}-mesh.npz".format(i)):
                    files.append(file)
        return files

    def download(self):
        """
        Download the dataset if it doesn't already exist.
        """
        if not self.check_exists():
            download_and_extract_archive(self.resource, download_root=os.path.split(self.path_to_data)[0])
        else:
            print("Data already exists")

    def check_exists(self):
        """
        Check if dataset already exists.
        """
        return os.path.exists(self.path_to_data)
