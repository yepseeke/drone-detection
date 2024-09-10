import torch

import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from dataset_processing import load_json, normalize_scaleogram, convert_gray2rgb, from_string_to_label


class DroneDataset(Dataset):
    def __init__(self, json_path: str, transform=None):
        # json_path = os.path.join(root_dir, json_file)
        self.data = load_json(json_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_arr = np.array(self.data[index].get('coefs'))
        normalized_image = normalize_scaleogram(image_arr)
        normalized_rgb_image = convert_gray2rgb(normalized_image)

        object_label = from_string_to_label(self.data[index].get('object'))
        y_label = torch.tensor(int(object_label))

        if self.transform:
            normalized_rgb_image = self.transform(normalized_rgb_image)

        return normalized_rgb_image, y_label


def get_loader(dataset_path: str, batch=20, transform=None):
    dataset = DroneDataset(dataset_path, transform)

    return DataLoader(dataset=dataset, batch_size=batch, shuffle=True)
