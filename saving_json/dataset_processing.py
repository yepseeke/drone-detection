import os
import time

import matplotlib.pyplot as plt
import json

import numpy as np
import scaleogram as scg

from PIL import Image
from scipy.io import wavfile

object_listdir = [
    'big_drone', 'bird', 'free_space', 'human', 'small_copter'
]

raw_data_folder = r'raw_data'
cuts_folder_path = r'cut_folder_0.1_seconds'
wavelet = 'cmor1-2.5'
coikw = {'alpha': 0.5, 'hatch': '/'}


def get_signal(sound_path: str):
    samplerate, x = wavfile.read(sound_path)
    return samplerate, x


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def save_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


# time_step - time interval in seconds for each audio segment
# sound_path - path to the original audio file
# folder_path - folder where the audio segments will be saved
# object_name - name of the object recorded in the audio file
def cut_sound(time_step, sound_path, cuts_folder_path, object_name=''):
    if not os.path.exists(sound_path):
        raise Exception('No such sound path.')

    if not os.path.isdir(cuts_folder_path):
        os.mkdir(cuts_folder_path)

    sample_rate, signal = get_signal(sound_path)

    signal_length = signal.shape[0]
    time_step_rate = int(sample_rate * time_step)

    cut_index = 0
    for rates in range(0, signal_length, time_step_rate):
        cut_sounds = signal[rates: rates + time_step_rate]

        cut_index += 1
        numbered_object = f'{object_name}_{cut_index}.wav'

        cut_sound_path = os.path.join(cuts_folder_path, numbered_object)

        wavfile.write(cut_sound_path, sample_rate, cut_sounds)


# raw_data_folder/
# └── object_names/  # Folder named after the object recorded in the audio file
#     └── audio_segments/  # Folder where the original audio is stored
# cuts_folder_path/
# └── object_names/  # Folder named after the object recorded in the audio file
#     └── audio_segments/  # Folder where the audio segments will be stored
def cut_raw_data(time_step, raw_data_folder, cuts_folder_path):
    if not os.path.isdir(cuts_folder_path):
        os.mkdir(cuts_folder_path)

    for object_dir in object_listdir:
        object_sounds = os.path.join(os.getcwd(), raw_data_folder, object_dir)

        object_name = object_dir
        object_sound_number = 0
        object_folder_path = os.path.join(cuts_folder_path, object_name)

        for object_sound in os.listdir(object_sounds):
            object_sound_number += 1
            numbered_object_name = f'{object_name}_{object_sound_number}'
            object_sound_path = os.path.join(object_sounds, object_sound)

            cut_sound(time_step, object_sound_path, object_folder_path, numbered_object_name)


def get_scaleogram(sound_path, spectrum=None, wavelet=None, scales=None):
    sample_rate, signal = get_signal(sound_path)

    if not scales:
        scales = scg.periods2scales(np.logspace(np.log10(2), np.log10(1000)), wavelet)

    signal_length = signal.shape[0] / sample_rate
    time = np.linspace(0, signal_length, signal.shape[0])
    cwt = scg.CWT(time=time, signal=signal, scales=scales, wavelet=wavelet)

    if spectrum == 'amp':
        return np.abs(cwt.coefs), cwt.scales_freq
    elif spectrum == 'real':
        return np.real(cwt.coefs), cwt.scales_freq
    elif spectrum == 'imag':
        return np.imag(cwt.coefs), cwt.scales_freq
    return cwt.coefs, cwt.scales_freq


# the function takes every 96th sample of the scaleogram in time,
# creates an image from three spectra Real, Imaginary and Abs and saves it in json as an array
def save_scaleograms(cuts_folder_path, json_path):
    scaleograms = []
    for object_dir in object_listdir:
        object_sounds = os.path.join(os.getcwd(), cuts_folder_path, object_dir)

        for object_sound in os.listdir(object_sounds):
            object_sound_path = os.path.join(object_sounds, object_sound)

            coefs, _ = get_scaleogram(object_sound_path)
            coefs = coefs[len(coefs) // 3:, ::96]  # coefs[:, ::96]

            real, imag, absolute = np.real(coefs), np.imag(coefs), np.abs(coefs)
            values = np.concatenate((real, imag, absolute))

            to_json = {'object': object_dir, 'coefs': values.tolist()}
            scaleograms.append(to_json)

    save_json(json_path, scaleograms)


def separate_scaleograms(json_path, train_path, valid_path):
    data = load_json(json_path)
    train = []
    valid = []
    for dt in data:
        if np.random.rand() < 0.81:
            train.append(dt)
        else:
            valid.append(dt)

    save_json(train_path, train)
    save_json(valid_path, valid)


if __name__ == '__main__':
    # cut_raw_data(0.1, raw_data_folder, cuts_folder_path)
    json_path = r'json\cmor1-2.5_real_imag_abs\dataset_cmor1-2.5_triple_seconds=0.1.json'
    train_path = r'json\cmor1-2.5_real_imag_abs\train_cmor1-2.5_triple_seconds=0.1.json'
    valid_path = r'json\cmor1-2.5_real_imag_abs\valid_cmor1-2.5_triple_seconds=0.1.json'

    separate_scaleograms(json_path, train_path, valid_path)
    # save_scaleograms(cuts_folder_path, json_path)
    # coefs, _ = get_scaleogram('human_2_33.wav')
    # coefs = coefs[len(coefs) // 3:, ::96]
    #
    # real, imag, absolute = np.real(coefs), np.imag(coefs), np.abs(coefs)
    # values = np.concatenate((real, imag, absolute))
    #
    # plt.imshow(values)
    # plt.show()
