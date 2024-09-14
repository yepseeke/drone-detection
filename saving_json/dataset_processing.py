import os
import json
import shutil

import numpy as np
from lmdb.tool import delta

from scipy.io import wavfile
from signal_processing import get_signal, get_scaleogram, get_spectrogram, get_deltas_spectrogram

object_names = ['big_drone', 'bird', 'free_space', 'human', 'small_copter']

# This directory contains uncut audio recordings, in the following structure
# original_data_folder/
# └── object_names/  # Folder named after the object recorded in the audio file
#     └── audio_recording/
original_data_folder_path = r'original_data'


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


def cut_sound(time_step, sound_path, object_cuts_folder_path, object_name=''):
    """
        Cuts the specified '.wav' audio file into segments of a specified length and saves them.

        :param time_step: The length of each audio segment in seconds.
        :param sound_path: The file path to the original audio file.
        Signal is assumed to be a 1-dimensional array or list containing the audio data.
        :param object_cuts_folder_path: The folder where the audio segments will be saved.
        :param object_name: (Optional) The name of the object recorded in the audio file.
        :return: None
    """

    if not os.path.exists(sound_path):
        raise Exception('No such sound path.')

    if not os.path.isdir(object_cuts_folder_path):
        os.mkdir(object_cuts_folder_path)

    # `signal` is assumed to be a 1-dimensional array or list containing the audio data.
    sample_rate, signal = get_signal(sound_path)
    signal_length = signal.shape[0]
    time_step_rate = int(sample_rate * time_step)

    cut_index = 0
    for rates in range(0, signal_length, time_step_rate):
        cut_index += 1

        # The segments contents information for `time_step` seconds.
        cut_sounds = signal[rates: rates + time_step_rate]

        numbered_object = f'{object_name}_{cut_index}.wav'
        cut_sound_path = os.path.join(object_cuts_folder_path, numbered_object)

        wavfile.write(cut_sound_path, sample_rate, cut_sounds)


def cut_original_data(time_step, original_data_path, cuts_folder_path, object_names):
    """
        Cuts original sounds into segments and saves them as '.wav' files.

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
        object_sounds_path = os.path.join(original_data_path, object_name)

        object_sound_number = 0
        object_folder_path = os.path.join(cuts_folder_path, object_name)

        for object_sound in os.listdir(object_sounds_path):
            object_sound_number += 1
            numbered_object_name = f'{object_name}_{object_sound_number}'
            object_sound_path = os.path.join(object_sounds_path, object_sound)

            cut_sound(time_step, object_sound_path, object_folder_path, numbered_object_name)


def save_scaleograms(cuts_folder_path, json_path, wavelet=None, spectrum='amp'):
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
        object_sounds = os.path.join(cuts_folder_path, object_name)

        for object_sound in os.listdir(object_sounds):
            object_sound_path = os.path.join(object_sounds, object_sound)

            sample_rate, signal = get_signal(object_sound_path)
            coefs, _ = get_scaleogram(signal, sample_rate, wavelet=wavelet)

            step = sample_rate // 1000
            coefs = coefs[:, ::step]

            real, imag, absolute = np.real(coefs), np.imag(coefs), np.abs(coefs)

            values = coefs
            if spectrum == 'amp':
                values = absolute
            elif spectrum == 'real':
                values = real
            elif spectrum == 'imag':
                values = imag
            elif spectrum == 'all':
                values = np.concatenate((real, imag, absolute))

            to_json = {'object': object_name, 'coefs': values.tolist()}
            scaleograms.append(to_json)

    save_json(json_path, scaleograms)


def split_sounds(cuts_folder_path, train_folder_path, valid_folder_path):
    """
        Splits whole audios into train data and valid data in a ratio of 4/1.

        cuts_folder_path/
        └── object_names/   Folder where the audio segments will be stored
            └── audio_segments/
        train_folder_path and valid_folder_path have the same structure.

        :param cuts_folder_path: The directory where the audio segments are stored.
        :param train_folder_path: The directory for audios that will be used as train data.
        :param valid_folder_path: The directory for audios that will be used as valid data.
        :return: None
    """
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


def normalize_data(coefs):
    """
        Normalizes the scaleogram/spectrogram data to a range of [0, 255].

        :param coefs: The scaleogram/spectrogram data.
        :return: Normalized scaleogram data.
    """
    min_coefs = np.min(coefs)
    max_coefs = np.max(coefs)
    normalized_coefs = np.int8(((coefs - min_coefs) / (max_coefs - min_coefs)) * 255)
    normalized_image = normalized_coefs.astype(np.uint8)

    return normalized_image


def save_spectrograms(cuts_folder_path, json_path, is_deltas=True):
    """
        Saves all spectrograms of segments into JSON file.

        cuts_folder_path/
        └── object_names/   Folder where the audio segments will be stored
            └── audio_segments/

        :param cuts_folder_path: The directory where the audio segments are stored.
        :param json_path: Path to the JSON file where scaleogram will be saved.
        :param is_deltas: If True adds deltas arrays to the data that will be stored.
        :return: None
    """
    spectrograms = []
    for object_name in object_names:
        object_sounds = os.path.join(cuts_folder_path, object_name)

        for object_sound in os.listdir(object_sounds):
            object_sound_path = os.path.join(object_sounds, object_sound)

            sample_rate, signal = get_signal(object_sound_path)
            spectrogram = get_spectrogram(sample_rate, signal)

            spectrogram_slice = spectrogram[:, :50]
            values = spectrogram_slice

            if is_deltas:
                delta, delta_delta = get_deltas_spectrogram(spectrogram_slice)
                values = np.concatenate((spectrogram_slice.T, delta.T, delta_delta.T), axis=1)
                print(values.shape)

            to_json = {'object': object_name, 'coefs': values.tolist()}
            spectrograms.append(to_json)

    save_json(json_path, spectrograms)
