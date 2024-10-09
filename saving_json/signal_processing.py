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


def resample_signal(sample_rate, signal, target_sample_rate=48000):
    """
        Resamples the input signal to a target sample rate if necessary.

        :param sample_rate: The sample rate of the input signal.
        :param signal: The input signal as a 1-dimensional array or list containing signal data.
        :param target_sample_rate: (Optional) The sample rate to resample the signal to.
                                   Defaults to 48,000 Hz if not provided.
        :return: A tuple containing the new sample rate and the resampled signal.
                 If no resampling is required, returns the original sample rate and signal.
    """
    if sample_rate != target_sample_rate:
        num_samples = int(len(signal) * target_sample_rate / sample_rate)
        signal = resample(signal, num_samples)
        sample_rate = target_sample_rate

    return sample_rate, signal


def get_wavelet_transform(sample_rate, signal, wavelet=None, scales=None):
    """
        Returns wavelet transform for a given signal.

        :param signal: Signal is a 1-dimensional array or list containing the audio data.
        :param sample_rate: Sample rate of the signal.
        :param wavelet: (Optional) The type of wavelet to use for the wavelet transform.
        If None, a default wavelet will be selected.
        :param scales: (Optional) A list or array of scales to use for the wavelet transform.
        :return: The wavelet transform data.
    """

    if scales is None:
        scales = []
    if len(scales) == 0:
        scales = scg.periods2scales(np.logspace(np.log10(2), np.log10(1000)), wavelet)

    signal_length = signal.shape[0] / sample_rate
    time = np.linspace(0, signal_length, signal.shape[0])
    cwt = scg.CWT(time=time, signal=signal, scales=scales, wavelet=wavelet)

    return cwt.coefs, cwt.scales_freq


def downsample_scaleogram(sample_rate, coefs, step_size=1000, spectrum='amp'):
    """
        Downsamples the scaleogram coefficients and extracts the desired spectrum.

        :param sample_rate: The sample rate of the original signal.
        :param coefs: The scaleogram coefficients as a 2-dimensional array (complex values).
        :param step_size: (Optional) The factor for downsampling. Defaults to 1000, which determines
                      the number of samples to keep for each step in the downsampled coefficients.
        :param spectrum: (Optional) The type of spectrum to extract from the coefficients.
                     Can be 'amp' for amplitude, 'real' for the real part, 'imag' for the
                     imaginary part, or 'all' to combine all components. Defaults to 'amp'.
        :return: The downsampled spectrum values based on the specified type.
    """
    step = sample_rate // step_size
    downsampled_coefs = coefs[:, ::step]

    real, imag, absolute = np.real(downsampled_coefs), np.imag(downsampled_coefs), np.abs(downsampled_coefs)

    spectrum_options = {
        'amp': absolute,
        'real': real,
        'imag': imag,
        'all': np.concatenate((real, imag, absolute))
    }
    values = spectrum_options.get(spectrum, coefs)

    return values


def get_spectrogram(sample_rate, signal):
    """
        Computes the spectrogram of the input signal with overlapping intervals.

        :param sample_rate: The sample rate of the input signal.
        :param signal: The input signal data.
        :return: A 2D array representing the spectrogram.
    """

    RESAMPLE_INTERVAL_FACTOR = 4 / 125
    FFT_OVERLAP_FACTOR = 32

    interval = int(RESAMPLE_INTERVAL_FACTOR * sample_rate)
    overlap = interval // FFT_OVERLAP_FACTOR
    signal_size = len(signal)

    num_slices = (signal_size - interval) // overlap + 1
    slices = np.empty((num_slices, interval // 2))

    for i in range(num_slices):
        start = i * overlap
        to_fft = signal[start:start + interval]
        transformed_frame = np.abs(fft(to_fft, interval))[:interval // 2]
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

# spectrum?
def handle_wavelet_transformation(sample_rate, cut_signal, wavelet, deltas_flag):
    cut_signal_scaleogram, _ = get_wavelet_transform(sample_rate, cut_signal, wavelet=wavelet)
    values = downsample_scaleogram(sample_rate, cut_signal_scaleogram)

    if deltas_flag:
        delta, delta_delta = get_deltas_of_data(values)
        values = np.concatenate((values, delta, delta_delta))

    return transform_data(values)


def handle_fft_transformation(sample_rate, cut_signal, deltas_flag):
    cut_audio_spectrogram = get_spectrogram(sample_rate, cut_signal)
    values = cut_audio_spectrogram[:80]

    if deltas_flag:
        delta, delta_delta = get_deltas_of_data(values)
        values = np.concatenate((values.T, delta.T, delta_delta.T), axis=1)

    transformed_values = transform_data(values)

    return transformed_values


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
        Normalizes data from the range [0, +inf) to [0, 255] and converts 1D image into a 3-channel RGB image
        :param coefs: The scaleogram/spectrogram data.
        :return: Normalized 3d scaleogram/spectrogram data.
    """
    # Normalizes data from the range [0, +inf) to [0, 255]
    normalized_image = normalize_data(coefs)
    # Converts the 1D image into a 3-channel RGB image
    normalized_rgb_image = convert_gray2rgb(normalized_image)

    return normalized_rgb_image
