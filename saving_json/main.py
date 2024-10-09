import os

from custom_model import CustomModel
from predict import predict_and_save

if __name__ == '__main__':
    cmodel = CustomModel(device='cuda:0')
    cmodel.load('models/6c6c79ab-231c-40ac-a9f8-18d1114f0b12.pth', device='cuda:0')

    folder_path = 'to_detect'
    files_to_detect = os.listdir(folder_path)
    for file_to_detect in files_to_detect:
        signal_path = f'{folder_path}/{file_to_detect}'
        predict_and_save(signal_path, cmodel, 'csv')
