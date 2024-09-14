import torch

import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from dataset_processing import load_json, normalize_data, convert_gray2rgb, from_string_to_label


class DroneDataset(Dataset):
    def __init__(self, json_path: str, transform=None):
        """
               Custom dataset that loads one-dimensional images from a JSON file,
               where each pixel value is in the range [0, +inf), and each image is associated with a label.

               Parameters:
               -----------
               json_path : str
                   Path to the JSON file containing image data and corresponding labels.
               transform : optional
                Optional transform to be applied on an image.
       """

        self.data = load_json(json_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
            Returns a transformed sample from the dataset at the given index.

            Parameters:
            -----------
            index : int
                Index of the sample to fetch.

            Returns:
            --------
            image : torch.Tensor
                Transformed 3-channel grayscale image (values scaled to [0, 255]).
            label : int
                Numerical label associated with the image.
        """
        image_arr = np.array(self.data[index].get('coefs'))
        normalized_image = normalize_data(image_arr)
        normalized_rgb_image = convert_gray2rgb(normalized_image)

        object_label = from_string_to_label(self.data[index].get('object'))
        y_label = torch.tensor(int(object_label))

        if self.transform:
            normalized_rgb_image = self.transform(normalized_rgb_image)

        return normalized_rgb_image, y_label


def get_loader(dataset_path: str, batch=20, transform=None):
    """
        Returns a DataLoader for the custom dataset.

        Parameters:
        -----------
        dataset_path : str
            Path to the JSON file containing the dataset.
        batch_size : int, default is 20
            Number of samples per batch to load.
        transform : callable, optional
            Optional transform to be applied on an image.

        Returns:
        --------
        DataLoader
            A DataLoader object for the custom dataset.
    """
    dataset = DroneDataset(dataset_path, transform)

    return DataLoader(dataset=dataset, batch_size=batch, shuffle=True)
