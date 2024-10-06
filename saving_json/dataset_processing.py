import os
import json
import shutil

import numpy as np

from scipy.io import wavfile
from signal_processing import get_signal, get_scaleogram, get_spectrogram, get_deltas_of_data

# This file contains functions specifically designed to create a dataset for model training
# by organizing audio recordings into the required folder structure.
#
# To create the dataset for training, ensure that your data is organized as follows:
#
# original_data_folder/
# └── object_name/        # Each folder should be named after the object represented in the audio files
#     └── audio_recording/ # Contains the audio recordings related to the specific object
#
# Note: The audio recordings in these folders are going to be cut into smaller clips.
# The objective is to segment these audio files into smaller clips, which will then be used
# for training the model on these cut audio segments.

original_data_folder_path = r'original_data'
object_names = ['big_drone', 'bird', 'free_space', 'human', 'small_copter']


def load_json(file_path):
    """
        Reads the contents of the JSON file

        :param file_path: The path to the JSON file that needs to be read.
        :return: The data extracted from the JSON file.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def save_json(file_path, data):
    """
        Saves the provided data to the JSON file at the specified file path.

        :param file_path: The path to the JSON file where the data will be saved.
        :param data: The data to be saved in the JSON file.
        :return: None
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def from_string_to_label(object_name: str):
    """
        Converts a given object name string to its corresponding numerical label.

        :param object_name: A string representing the name of the object.
        :return: An integer representing the numerical label of the object based on a predefined mapping.
            If the object name is not found in the mapping, returns -1.
    """
    label_map = {
        'big_drone': 0,
        'bird': 1,
        'free_space': 2,
        'human': 3,
        'small_copter': 4
    }

    return label_map.get(object_name, -1)


def from_label_to_string(class_num: int):
    labels = ['big_drone', 'bird', 'free_space', 'human', 'small_copter']
    return labels[class_num] if 0 <= class_num < len(labels) else 'unknown_class'


def convert_gray2rgb(image):
    """
        Converts a grayscale image to an RGB image by replicating the grayscale values across all three color channels.

        :param image: A 2D numpy array representing the grayscale image.
        :return: A 3D numpy array representing the RGB image, with shape (width, height, 3).
    """
    width, height = image.shape
    out = np.empty((width, height, 3), dtype=np.uint8)
    out[:, :, 0] = image
    out[:, :, 1] = image
    out[:, :, 2] = image

    return out


# Prepares directories for the storage of segmented signals.
def create_object_folders(empty_folder_path, object_names):
    """
        Creates directories within the specified folder, where each directory is named
        after an object and is intended to contain audio segments extracted from the original files.

        Structure:
        empty_folder_path/
        └── object_name/   Directories named after the objects, which will be created in the empty folder

        :param empty_folder_path: The path to the empty folder where new directories will be created.
        :param object_names: A list of names for the directories (representing objects).
        to be created inside `empty_folder_path`.
        :return: None
    """

    # If specified path does not exist, a new folder will be created
    if not os.path.isdir(empty_folder_path):
        os.mkdir(empty_folder_path)

    # If specified path exists
    if os.listdir(empty_folder_path):
        print(os.listdir(empty_folder_path))
        raise Exception('The folder is not empty.')

    for object_name in object_names:
        object_path = os.path.join(empty_folder_path, object_name)
        os.mkdir(object_path)


def cut_signal(time_step, signal_path, object_cuts_folder_path, object_name=''):
    """
        Cuts the specified '.wav' audio file into segments of a specified length and saves them.

        :param time_step: The length of each audio segment in seconds.
        :param signal_path: The file path to the original audio file.
        Signal is assumed to be a 1-dimensional array or list containing the audio data.
        :param object_cuts_folder_path: The folder where the audio segments will be saved.
        :param object_name: (Optional) The name of the object recorded in the audio file.
        :return: None
    """

    if not os.path.exists(signal_path):
        raise Exception('No such signal path.')

    if not os.path.isdir(object_cuts_folder_path):
        os.mkdir(object_cuts_folder_path)

    # `signal` is assumed to be a 1-dimensional array or list containing the audio data.
    sample_rate, signal = get_signal(signal_path)
    signal_length = signal.shape[0]
    time_step_rate = int(sample_rate * time_step)

    cut_index = 0
    for rates in range(0, signal_length, time_step_rate):
        cut_index += 1

        # The segments contents information for `time_step` seconds.
        cut_signal = signal[rates: rates + time_step_rate]

        numbered_object = f'{object_name}_{cut_index}.wav'
        cut_signal_path = os.path.join(object_cuts_folder_path, numbered_object)

        wavfile.write(cut_signal_path, sample_rate, cut_signal)


def cut_original_data(time_step, original_data_path, cuts_folder_path, object_names):
    """
        Cuts original signals into segments and saves them as '.wav' files.

        Original_data_path must have the following structure:
        original_data_path/
        └── object_names/   Folder where the original audios are stored
            └── audio_segments/
        Segments will be saved in the same structure:
        cuts_folder_path/
        └── object_names/   Folder where the audio segments will be stored
            └── audio_segments/

        :param time_step: The length of each audio segment in seconds.
        :param original_data_path: The directory where the original audios are stored.
        :param cuts_folder_path: The directory where the audio segments will be stored.
        :param object_names: A list of names for the directories (representing objects)
        to be created inside `empty_folder_path`.
        :return: None
    """

    create_object_folders(cuts_folder_path, object_names)

    for object_name in object_names:
        object_signals_path = os.path.join(original_data_path, object_name)

        object_signal_number = 0
        object_folder_path = os.path.join(cuts_folder_path, object_name)

        object_signal_filenames = os.listdir(object_signals_path)
        for object_signal in object_signal_filenames:
            object_signal_number += 1
            numbered_object_name = f'{object_name}_{object_signal_number}'
            object_signal_path = os.path.join(object_signals_path, object_signal)

            cut_signal(time_step, object_signal_path, object_folder_path, numbered_object_name)


def save_scaleograms(cuts_folder_path, json_path, wavelet=None, spectrum='amp', is_deltas=False):
    """
        Saves all scaleograms of segments into JSON file.
        The function takes every (sample_rate / 1000) sample to save memory.

        cuts_folder_path/
        └── object_names/   Folder where the audio segments will be stored
            └── audio_segments/

        :param cuts_folder_path: The directory where the audio segments are stored.
        :param json_path: Path to the JSON file where scaleogram will be saved.
        :param wavelet: (Optional) The type of wavelet to use for the wavelet transform.
            If None, a default wavelet will be selected.
        :param spectrum: (Optional) Specifies the type of spectrum to be used for the wavelet transform.
            It can be 'amp', 'real', 'imag' or 'all' which returns all three spectrums at once.
        :return: None
    """

    scaleograms = []
    for object_name in object_names:
        object_signals = os.path.join(cuts_folder_path, object_name)

        for object_signal in os.listdir(object_signals):
            object_signal_path = os.path.join(object_signals, object_signal)

            sample_rate, signal = get_signal(object_signal_path)
            coefs, _ = get_scaleogram(signal, sample_rate, wavelet=wavelet)

            step = sample_rate // 1000
            slices_step = coefs[:, ::step]

            real, imag, absolute = np.real(slices_step), np.imag(slices_step), np.abs(slices_step)

            values = coefs
            if spectrum == 'amp':
                values = absolute
            elif spectrum == 'real':
                values = real
            elif spectrum == 'imag':
                values = imag
            elif spectrum == 'all':
                values = np.concatenate((real, imag, absolute))

            if is_deltas:
                delta, delta_delta = get_deltas_of_data(values)
                values = np.concatenate((values, delta, delta_delta))

            to_json = {'object': object_name, 'coefs': values.tolist()}
            scaleograms.append(to_json)

    save_json(json_path, scaleograms)


def save_spectrograms(cuts_folder_path, json_path, is_deltas=False):
    """
        Saves all spectrograms of segments into JSON file.

        cuts_folder_path/
        └── object_names/   Folder where the audio segments will be stored
            └── audio_segments/

        :param cuts_folder_path: The directory where the audio segments are stored.
        :param json_path: Path to the JSON file where spectrogram will be saved.
        :param is_deltas: If True adds deltas arrays to the data that will be stored.
        :return: None
    """
    spectrograms = []
    for object_name in object_names:
        object_signals = os.path.join(cuts_folder_path, object_name)

        for object_signal in os.listdir(object_signals):
            object_signal_path = os.path.join(object_signals, object_signal)

            sample_rate, signal = get_signal(object_signal_path)
            spectrogram = get_spectrogram(sample_rate, signal)

            values = spectrogram[:80]

            if is_deltas:
                delta, delta_delta = get_deltas_of_data(values)
                values = np.concatenate((values.T, delta.T, delta_delta.T), axis=1)

            to_json = {'object': object_name, 'coefs': values.tolist()}
            spectrograms.append(to_json)

    save_json(json_path, spectrograms)


def save_mel_spectrograms(cuts_folder_path, json_path, is_deltas=False):
    """
        Saves all mel spectrograms of segments into JSON file.

        cuts_folder_path/
        └── object_names/   Folder where the audio segments will be stored
            └── audio_segments/

        :param cuts_folder_path: The directory where the audio segments are stored.
        :param json_path: Path to the JSON file where spectrogram will be saved.
        :param is_deltas: If True adds deltas arrays to the data that will be stored.
        :return: None
    """
    pass


def add_reversed_signals(object_cuts_folder_path):
    object_filenames = os.listdir(object_cuts_folder_path)
    for object_filename in object_filenames:
        object_path = os.path.join(object_cuts_folder_path, object_filename)
        sample_rate, signal = get_signal(object_path)

        reversed_signal = signal[::-1]
        reversed_signal_filename = f'reverse_{object_filename}'
        reversed_signal_path = os.path.join(object_cuts_folder_path, reversed_signal_filename)
        wavfile.write(reversed_signal_path, sample_rate, reversed_signal)
