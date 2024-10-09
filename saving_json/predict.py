import os
import csv

from custom_model import CustomModel
from dataset_processing import from_label_to_string
from signal_processing import get_signal, resample_signal, handle_fft_transformation, handle_wavelet_transformation


def perform_transformation(sample_rate, cut_signal, transformation_type, specific_wavelet, deltas_flag):
    if transformation_type == 'wavelet':
        return handle_wavelet_transformation(sample_rate, cut_signal, specific_wavelet, deltas_flag)
    elif transformation_type == 'fft':
        return handle_fft_transformation(sample_rate, cut_signal, deltas_flag)


def predict_and_save(signal_path: str, cmodel: CustomModel, csv_path: str, time_step=0.2):
    TARGET_SAMPLE_RATE = 48000

    transformation_type, specific_wavelet, deltas_flag = cmodel.parse_transform_type()

    sample_rate, signal = get_signal(signal_path)
    sample_rate, signal = resample_signal(sample_rate, signal, TARGET_SAMPLE_RATE)

    signal_length = signal.shape[0]
    time_in_seconds = signal_length / sample_rate
    time_step_rate = int(sample_rate * time_step)

    signal_filename = os.path.basename(signal_path)
    csv_file_path = os.path.join(csv_path, signal_filename.replace('.wav', '.csv'))
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['time', 'cls_n', 'cls_str'])

        for rates in range(0, signal_length, time_step_rate):
            cut_signal = signal[rates: rates + time_step_rate]
            cut_time = rates / signal_length * time_in_seconds
            values = perform_transformation(sample_rate, cut_signal, transformation_type, specific_wavelet, deltas_flag)

            cls = cmodel.predict(values)

            writer.writerow([f'{cut_time:.1f}', cls, from_label_to_string(cls)])
            print(f"Time: {cut_time:.1f} seconds | Class number: {cls} | Class name: {from_label_to_string(cls)}")
