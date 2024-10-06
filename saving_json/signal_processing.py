import scaleogram as scg
import numpy as np
import torch
import torchaudio.transforms as T

from scipy.io import wavfile
from scipy.signal import resample
from scipy.fft import fft


def get_signal(sound_path: str):
    """
        Returns samplerate of the signal and signal itself.

        :param sound_path: filepath to the `.wav` file.
        :return: samplerate and signal
    """
    samplerate, x = wavfile.read(sound_path)

    if len(x.shape) > 1:
        x = np.mean(x, axis=1)

    return samplerate, x


def get_scaleogram(sample_rate, signal, wavelet=None, scales=None):
    """
        Generates a scaleogram (a visual representation of the wavelet transform) for a given signal.

        :param signal: Signal is a 1-dimensional array or list containing the audio data.
        :param sample_rate: Sample rate of the signal.
        :param wavelet: (Optional) The type of wavelet to use for the wavelet transform.
        If None, a default wavelet will be selected.
        :param scales: (Optional) A list or array of scales to use for the wavelet transform.
        :return: The scaleogram data.
    """
    if scales is None:
        scales = []
    if len(scales) == 0:
        scales = scg.periods2scales(np.logspace(np.log10(2), np.log10(1000)), wavelet)

    signal_length = signal.shape[0] / sample_rate
    time = np.linspace(0, signal_length, signal.shape[0])
    cwt = scg.CWT(time=time, signal=signal, scales=scales, wavelet=wavelet)

    return cwt.coefs, cwt.scales_freq


def get_spectrogram(sample_rate, signal, target_sample_rate=48000):
    """
    Computes the spectrogram of the input signal with overlapping intervals.

    :param sample_rate: The sample rate of the input signal.
    :param signal: The input signal data.
    :param target_sample_rate: The sample rate to resample the signal to (default: 48 kHz).
    :return: A 2D array representing the spectrogram.
    """

    if sample_rate != target_sample_rate:
        num_samples = int(len(signal) * target_sample_rate / sample_rate)
        signal = resample(signal, num_samples)
        sample_rate = target_sample_rate

    epsilon = 1e-10
    interval = 4 * sample_rate // 125
    overlap = interval // 32
    signal_size = len(signal)

    num_slices = (signal_size - interval) // overlap + 1
    slices = np.empty((num_slices, interval // 2))

    for i in range(num_slices):
        start = i * overlap
        to_fft = signal[start:start + interval]
        transformed_frame = np.abs(fft(to_fft, interval))[:interval // 2] + epsilon
        slices[i] = transformed_frame

    return slices.T


def get_mel_spectrogram(sample_rate, signal, is_log=True):
    """
    Computes the Mel spectrogram of the input signal.

    :param sample_rate: The sample rate of the input signal.
    :param signal: The input signal data.
    :param is_log: Boolean flag to indicate if the output should be converted to logarithmic scale (default: True).
    :return: A 2D array representing the Mel spectrogram, optionally in logarithmic scale.
    """
    mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=128, n_fft=4096, hop_length=128)
    waveform = torch.tensor(signal, dtype=torch.float32)
    values = mel_transform(waveform)
    if is_log:
        db_transform = T.AmplitudeToDB(values)
        values = db_transform(values)

    return values.squeeze().numpy()


def get_deltas_of_data(coefs):
    """
    Computes the first and second derivatives (deltas) of the input coefficients.

    :param coefs: A 2D array of coefficients, which can represent a scaleogram, spectrogram, or mel spectrogram.
    :return: A tuple containing:
             - delta: The first derivative of the coefficients.
             - delta_delta: The second derivative of the coefficients.
    """
    delta = np.diff(coefs, axis=1)
    zero_column = np.zeros((coefs.shape[0], 1))
    delta = np.hstack([zero_column, delta])

    delta_delta = np.diff(delta, axis=1)
    delta_delta = np.hstack([zero_column, delta_delta])

    return delta, delta_delta


def save_signal_to_wav(signal, sample_rate, folder_path):
    """
    Normalizes the input signal and saves it as a WAV file.

    :param signal: A 1D array of audio signal data.
    :param sample_rate: The sample rate at which the signal was recorded.
    :param folder_path: The file path where the WAV file will be saved.
    """
    signal = signal / np.max(np.abs(signal))
    signal = np.int16(signal * 32767)

    wavfile.write(folder_path, sample_rate, signal)


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

    if max_coefs <= 255 and min_coefs >= 0:
        return coefs

    normalized_coefs = np.int8(((coefs - min_coefs) / (max_coefs - min_coefs)) * 255)
    normalized_image = normalized_coefs.astype(np.uint8)

    return normalized_image


def transform_data(coefs):
    """
        Normalizes data from the range [0, +inf) to [0, 255] and converts the 1D image into a 3-channel RGB image
        :param coefs: The scaleogram/spectrogram data.
        :return: Normalized 3d scaleogram/spectrogram data.
    """
    # Normalizes data from the range [0, +inf) to [0, 255]
    normalized_image = normalize_data(coefs)
    # Converts the 1D image into a 3-channel RGB image
    normalized_rgb_image = convert_gray2rgb(normalized_image)

    return normalized_rgb_image
