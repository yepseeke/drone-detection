import os

from custom_model import CustomModel
from dataset_loader import get_loader

if __name__ == '__main__':
    cmodel = CustomModel('resnet18')

    train_json_path = os.path.join(os.getcwd(), r'datasets/seconds=0.2/train_cmor1.2-3.json')
    valid_json_path = os.path.join(os.getcwd(), r'datasets/seconds=0.2/valid_cmor1.2-3.json')

    train_loader = get_loader(train_json_path, batch=30)
    valid_loader = get_loader(valid_json_path, batch=30)

    cmodel.train_model(train_loader, valid_loader, epochs=1, learning_rate=0.001, is_accuracy=True, save=True,
                   save_config=True)
