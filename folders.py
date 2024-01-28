import os
import csv
import random
import shutil
import time
import torch
import numpy as np
import scaleogram as scg

from scipy.io import wavfile
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.models import resnet18

wavelet = 'cmor1-1.5'
coikw = {'alpha': 0.5, 'hatch': '/'}


def get_signal(file_path: str):
    samplerate, x = wavfile.read(file_path)
    return samplerate, x


def split_sound(step, folder, folder_to):
    index = 1
    for file in os.listdir(folder):
        file_path = folder + '/' + file
        samplerate, signal = get_signal(file_path)
        sample_step = int(samplerate * step)
        signal_shape = signal.shape[0]
        for i in range(0, signal_shape, sample_step):
            new_signal = signal[i:i + sample_step]
            new_path = folder_to + '/' + f'{index}.wav'
            wavfile.write(new_path, samplerate, new_signal)
            index += 1


def generate_sound_data():
    step_size = 0.2
    from_path = 'raw_data'
    from_folders = ['big_drone', 'bird', 'free_space', 'human', 'small_copter']
    to_folders = ['sound_big_drone', 'sound_bird', 'sound_free_space', 'sound_human', 'sound_small_copter']

    for i in range(5):
        start_time = time.time()

        old_path = from_path + '/' + from_folders[i]
        new_path = to_folders[i]
        split_sound(step_size, old_path, new_path)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("\nProcessing for: {:.2f} seconds".format(elapsed_time))


def save_scaleogram(file_path, signal, time, scales, wavelet):
    cwt = scg.CWT(time=time, signal=signal, scales=scales)
    scg.cws(cwt, figsize=(6, 4), coikw=coikw, wavelet=wavelet, yaxis='frequency', spectrum='amp', title='')
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()


def sound_scaleograms(sound_folder, scaleogram_folder):
    index = 1
    scales = scg.periods2scales(np.logspace(np.log10(2), np.log10(1000)), wavelet)

    for file in os.listdir(sound_folder):
        path_wav = sound_folder + '/' + file

        sample_rate, signal = get_signal(path_wav)

        signal_length = signal.shape[0] / sample_rate
        time = np.linspace(0, signal_length, signal.shape[0])

        path_jpg = scaleogram_folder + '/' + f'{index}.jpg'
        save_scaleogram(path_jpg, signal, time, scales, wavelet)

        index += 1


def generate_scaleograms():
    from_folders = ['sound_small_copter']
    to_folders = ['scaleogram_small_copter']
    for i in range(1):
        start_time = time.time()

        sound_path = from_folders[i]
        scaleogram_path = to_folders[i]
        sound_scaleograms(sound_path, scaleogram_path)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("\nProcessing for: {:.2f} seconds".format(elapsed_time))


def rename_scaleogram(folder, name):
    file_png = [file for file in os.listdir(folder)]
    print(file_png)

    for index, old in enumerate(file_png, start=1):
        new_name = folder + '/' + name + f'{index}.png'
        old_name = folder + '/' + old
        os.rename(old_name, new_name)


def select_number(n):
    number = list(range(1, n + 1))

    k = int(3 * n / 4)

    return random.sample(number, k)


def generate_dataset():
    path_train = 'dataset/train_data'
    path_test = 'dataset/test_data'

    paths = ['scaleogram_big_drone', 'scaleogram_bird', 'scaleogram_free_space', 'scaleogram_human',
             'scaleogram_small_copter']

    for path in paths:
        files = [file for file in os.listdir(path)]
        data_length = len(files)
        random_sample = select_number(data_length)

        for i in range(1, data_length + 1, 1):
            file_path_from = path + '/' + files[i - 1]
            if i in random_sample:
                shutil.copy(file_path_from, path_train)
            else:
                shutil.copy(file_path_from, path_test)


def get_class_number(filename):
    if "drone" in filename:
        return 0
    elif "bird" in filename:
        return 1
    elif "human" in filename:
        return 2
    elif "free_space" in filename:
        return 3
    elif "copter" in filename:
        return 4
    else:
        return -1


def create_csv():
    images_folder = 'dataset/data'

    csv_file_path = 'scaleogram.csv'

    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        csv_writer.writerow(['Sound', 'Label'])

        for filename in os.listdir(images_folder):
            file_path = os.path.join(images_folder, filename)

            if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                class_number = get_class_number(filename)

                csv_writer.writerow([filename, class_number])

    print(f'Created: {csv_file_path}')


def resize_images():
    image_folder_path = 'dataset/data'
    output_folder_path = 'dataset/data_resize'

    image_files = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]

    for image_file in image_files:
        image_path = os.path.join(image_folder_path, image_file)
        img = Image.open(image_path)
        resized_img = img.resize((590, 390), Image.LANCZOS)
        output_image_path = os.path.join(output_folder_path, image_file)

        resized_img.save(output_image_path)


def generate_test_data(sound_path):
    samplerate, signal = get_signal(sound_path)
    new_signal = signal[int(23.5 * samplerate):  int(24.5 * samplerate)]
    new_path = 'sound_bird_test1.wav'
    wavfile.write(new_path, samplerate, new_signal)
