import scaleogram as scg
import numpy as np

from scipy.io import wavfile
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


def get_scaleogram(signal, sample_rate, spectrum=None, wavelet=None, scales=None):
    """
        Generates a scaleogram (a visual representation of the wavelet transform) for a given signal.

        :param signal: Signal is a 1-dimensional array or list containing the audio data.
        :param sample_rate: Sample rate of the signal.
        :param spectrum: (Optional) Specifies the type of spectrum to be used for the wavelet transform.
        It can be 'amp', 'real', 'imag'.
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

    if spectrum == 'amp':
        return np.abs(cwt.coefs), cwt.scales_freq
    elif spectrum == 'real':
        return np.real(cwt.coefs), cwt.scales_freq
    elif spectrum == 'imag':
        return np.imag(cwt.coefs), cwt.scales_freq
    return cwt.coefs, cwt.scales_freq


def get_spectrogram(sample_rate, signal):
    """
        Computes the spectrogram of the input signal with overlapping intervals.

        :param sample_rate: The sample rate of the input signal.
        :param signal: The input signal data.
        :return: A 2D array representing the spectrogram.
    """

    epsilon = 1e-10

    interval = 8 * sample_rate // 125

    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    signal_size = signal.shape[0]

    overlap = interval // 16
    slices = np.empty((signal_size // overlap, interval // 2))

    for i in range(0, signal_size, overlap):
        to_fft = signal[i: i + interval].copy()
        transformed_frame = np.abs(fft(to_fft, interval))[0:interval // 2] + epsilon
        slices[i // overlap] = transformed_frame

    return slices


def get_deltas_of_data(coeffs):
    delta = np.diff(coeffs, axis=1)
    zero_column = np.zeros((coeffs.shape[0], 1))
    delta = np.hstack([zero_column, delta])

    delta_delta = np.diff(delta, axis=1)
    delta_delta = np.hstack([zero_column, delta_delta])

    return delta, delta_delta
