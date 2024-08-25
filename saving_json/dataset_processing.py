import os
import json
import shutil

import numpy as np
import scaleogram as scg

from scipy.io import wavfile

from main import wavelet

object_names = ['big_drone', 'bird', 'free_space', 'human', 'small_copter']

# This folder contains uncut audio recordings, in the following structure
# raw_data_folder/
# └── object_names/  # Folder named after the object recorded in the audio file
#     └── audio_recording/
not_cut_data_folder_path = r'raw_data'


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def save_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def get_signal(sound_path: str):
    samplerate, x = wavfile.read(sound_path)
    return samplerate, x


# Creates directories within the specified folder, where each directory will contain audio segments extracted from the original files.
# empty_folder/
# └── object_names/  # Directory named after the object mentioned in the audio file
def create_object_folders(empty_folder_path):
    if not os.path.isdir(empty_folder_path):
        os.mkdir(empty_folder_path)

    if os.listdir(empty_folder_path):
        print(os.listdir(empty_folder_path))
        raise Exception('The folder is not empty.')

    for object_name in object_names:
        object_path = os.path.join(empty_folder_path, object_name)
        os.mkdir(object_path)


# time_step - time interval in seconds for each audio segment
# sound_path - path to the original audio file
# object_cuts_folder_path - folder where the audio segments will be saved
# object_name - name of the object recorded in the audio file
def cut_sound(time_step, sound_path, object_cuts_folder_path, object_name=''):
    if not os.path.exists(sound_path):
        raise Exception('No such sound path.')

    if not os.path.isdir(object_cuts_folder_path):
        os.mkdir(object_cuts_folder_path)

    sample_rate, signal = get_signal(sound_path)
    signal_length = signal.shape[0]
    time_step_rate = int(sample_rate * time_step)

    cut_index = 0
    for rates in range(0, signal_length, time_step_rate):
        cut_sounds = signal[rates: rates + time_step_rate]
        cut_index += 1
        numbered_object = f'{object_name}_{cut_index}.wav'
        cut_sound_path = os.path.join(object_cuts_folder_path, numbered_object)
        wavfile.write(cut_sound_path, sample_rate, cut_sounds)


# raw_data_folder/
# └── object_names/  # Folder where the original audios are stored
#     └── audio_segments/
# cuts_folder_path/
# └── object_names/  # Folder where the audio segments will be stored
#     └── audio_segments/
def cut_raw_data(time_step, raw_data_folder, cuts_folder_path):
    create_object_folders(cuts_folder_path)

    for object_dir in object_names:
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


# The function takes every 96th sample of the scaleogram in time,
# creates an image from three spectrum Real, Imaginary and Abs and saves it in json as an array
def save_scaleograms(cuts_folder_path, json_path, wavelet=None, spectrum='amp'):
    scaleograms = []
    for object_dir in object_names:
        object_sounds = os.path.join(os.getcwd(), cuts_folder_path, object_dir)

        for object_sound in os.listdir(object_sounds):
            object_sound_path = os.path.join(object_sounds, object_sound)

            coefs, _ = get_scaleogram(object_sound_path, wavelet=wavelet)
            coefs = coefs[:, ::96]

            real, imag, absolute = np.real(coefs), np.imag(coefs), np.abs(coefs)

            values = np.zeros(1)
            if spectrum == 'amp':
                values = absolute
            elif spectrum == 'real':
                values = real
            elif spectrum == 'imag':
                values = imag
            elif spectrum == 'all':
                values = np.concatenate((real, imag, absolute))

            to_json = {'object': object_dir, 'coefs': values.tolist()}
            scaleograms.append(to_json)

    save_json(json_path, scaleograms)


def separate_sounds(cuts_folder_path, train_folder_path, valid_folder_path):
    for object_group in os.listdir(cuts_folder_path):
        object_group_path = os.path.join(cuts_folder_path, object_group)

        train_group_path = os.path.join(train_folder_path, object_group)
        valid_group_path = os.path.join(valid_folder_path, object_group)
        if not os.path.isdir(train_group_path):
            os.mkdir(train_group_path)
        if not os.path.isdir(valid_group_path):
            os.mkdir(valid_group_path)

        for object_name in os.listdir(object_group_path):
            object_path = os.path.join(object_group_path, object_name)
            if np.random.rand() < 0.8:
                shutil.copy(object_path, train_group_path)
            else:
                shutil.copy(object_path, valid_group_path)
