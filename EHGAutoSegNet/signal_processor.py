import zipfile
import os, sys
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, decimate
import pandas as pd
import tensorflow as tf
import shutil
from .signal_post_processor import SignalPostProcessor

class SignalProcessor:
    def __init__(self, model_path=None):
        # Get the path of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to model.h5 relative to the script directory
        default_model_path = os.path.join(script_dir, 'model', 'model.h5')

        # Use the specified model_path if provided, otherwise use the default path
        model_path = model_path or default_model_path

        self.model = tf.keras.models.load_model(model_path, compile=False)

    @staticmethod
    def unzip_file(zip_file_path, extraction_directory, zip_name):
        """
        Unzips the contents of a ZIP file to the specified directory.

        Args:
            zip_file_path (str): Path to the ZIP file.
            extraction_directory (str): Directory to extract the contents to.
        """
        if not os.path.exists(extraction_directory):
            os.makedirs(extraction_directory, exist_ok=True)

        with zipfile.ZipFile(os.path.join(zip_file_path, zip_name), 'r') as zip_ref:
            zip_ref.extractall(extraction_directory)

    @staticmethod
    def zscore_normalization(data, axis=1):
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        normalized_data = (data - mean) / std
        return normalized_data

    @staticmethod
    def list_files_with_extension(directory, extension):
        """
        Returns a list of file paths with a specific extension in the specified directory
        and its subdirectories.
    
        Args:
            directory (str): Directory to start the search from.
            extension (str): File extension to filter the files.
    
        Returns:
            List[str]: List of file paths with the specified extension.
        """
        file_paths = []
        file_names = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(extension):
                    file_paths.append(root.split(directory)[-1])
                    file_names.append(file)
        return file_paths, file_names

    @staticmethod
    def read_mat_file(file_path):
        """
        Reads data from a MATLAB (.mat) file and returns it as a dictionary of NumPy arrays.

        Args:
            file_path (str): Path to the MATLAB file.

        Returns:
            dict: Dictionary of NumPy arrays containing the data from the file.
        """
        try:
            data = scipy.io.loadmat(file_path)
            numpy_data = {}
            for key in data:
                if isinstance(data[key], np.ndarray):
                    numpy_data[key] = data[key]
            return numpy_data
        except FileNotFoundError:
            print(f"File '{file_path}' not found.")
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")

    @staticmethod
    def downsample_numpy(signal, downsampling_factor):
        """
        Downsamples a signal using decimation and averaging.

        Args:
            signal (numpy.ndarray): Input signal to be downsampled.
            downsampling_factor (int): Downsampling factor.

        Returns:
            numpy.ndarray: Downsampled signal.
        """
        downsampled_signal = decimate(signal, 1, n=5)
        pad_size = np.ceil(float(downsampled_signal.size) / downsampling_factor) * downsampling_factor - downsampled_signal.size
        padded_signal = np.append(downsampled_signal, np.zeros(int(pad_size)) * np.NaN)
        return np.nanmean(padded_signal.reshape(-1, downsampling_factor), axis=1)

    @staticmethod
    def butter_bandpass_filter(signal, lowcut, highcut, sample_rate, order=5):
        """
        Applies a Butterworth bandpass filter to the input signal.

        Args:
            signal (numpy.ndarray): Input signal to be filtered.
            lowcut (float): Lower cutoff frequency.
            highcut (float): Upper cutoff frequency.
            sample_rate (float): Sample rate of the signal.
            order (int, optional): Filter order. Defaults to 5.

        Returns:
            numpy.ndarray: Filtered signal.
        """
        b, a = butter(order, [lowcut, highcut], fs=sample_rate, btype='band')
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal

    @staticmethod
    def divide_signal_into_windows(signal, sample_rate=10, window_duration=128, overlap = 64):
        """
        Divides a signal into windows with a specified duration and overlap.

        Args:
            signal (numpy.ndarray): Input signal to be divided into windows.
            sample_rate (int, optional): Sample rate of the signal. Defaults to 10 Hz.
            window_duration (int, optional): Duration of each window in seconds. Defaults to 128.
            overlap (int, optional): Overlap between windows in seconds. Default to 0.5.

        Returns:
            list: List of windows containing segments of the input signal.
        """
        # Calculate the window size and overlap in terms of number of samples
        window_size = int(window_duration * sample_rate)
        overlap = int(overlap * sample_rate)

        # Divide the signal into windows with overlap
        windows = []
        start = 0
        end = window_size

        while end <= len(signal):
            windows.append(signal[start:end])
            start += overlap
            end += overlap

        return windows

    @staticmethod
    def find_segment_boundaries(segmentation_array, sample_rate):
        """
        Finds the boundaries of each class segment in a segmentation array.

        Args:
            segmentation_array (numpy.ndarray): Segmentation array containing class labels.
            sample_rate (int, optional): Sample rate of the signal. Defaults to 10 Hz.

        Returns:
            list: List of tuples representing the start and end times of each class segment, along with the class type.
        """
        boundaries = []

        # Find the indices where the segmentation array changes
        changes = np.nonzero(np.diff(segmentation_array))[0] + 1

        if len(changes) == 0:
            boundaries = [(0, len(segmentation_array) / sample_rate, segmentation_array[0])]

        else:
    
            for i in range(len(changes)):
                start = changes[i - 1] if i > 0 else 0
                end = changes[i] - 1
    
                # Append segment boundaries with start, end, and class type
                boundaries.append((start / sample_rate, (end + 1) / sample_rate, segmentation_array[int((start + end) / 2)]))
        
            # Include final segment if it is different from the last change
            if segmentation_array[-1] != segmentation_array[changes[-1] - 1]:
                boundaries.append((changes[-1] / sample_rate, len(segmentation_array) / sample_rate, 
                                   segmentation_array[int((changes[-1] + len(segmentation_array)) / 2)]))

        return boundaries

    def process_signal_channels(self, signal_channels, original_sample_rate=20, target_sample_rate=10, 
                                window_duration = 128, overlap = 64, 
                                e_d_config=None, min_segment_length = 30,
                                lower_cutoff_freq=0.1, upper_cutoff_freq=4,
                                downsampling_factor=None, threshold=0.8, 
                                show_example=False, record_name=None, show_example_channels=None, show_example_w_len=None):
        """
        Processes signal channels by applying filtering, downsampling, segmentation, and post-processing.

        Args:
            signal_channels (list): List of signal channels.
            original_sample_rate (int): Original sample rate of the signals. Defaults to 20 Hz.
            target_sample_rate (int): Target sample rate after downsampling. Defaults to 5 Hz.
            window_duration (int): Windows length. Defaults to 128 s.
            overlap (int): Windows overlap in seconds. Defaults to 64 s.
            e_d_config (dict): Configuration dictionary for post-processing. Defaults to None.
            min_segment_length (int): Minimum windows length of the physiological class. Defaults to 30 s.
            lower_cutoff_freq (float): Lower cutoff frequency for bandpass filter. Defaults to 0.1 Hz.
            upper_cutoff_freq (float): Upper cutoff frequency for bandpass filter. Defaults to 2 Hz.
            downsampling_factor (int): Downsampling factor. If None, it will be calculated based on sample rates.
            threshold (float): Threshold value for segmentation (physiological class). Defaults to 0.8.
            show_example (bool): Whether to show an example plot. Defaults to False.
            record_name (str): Recording name to be show as figure title. Defaults to None.
            show_example_channels (list): channels to be show as example of each recordings. Defaults to None. Example [0, 5, 9]. 
            show_example_w_len (int): signal length to be shown as example. Defaults to None. Example 600s. 

        Returns:
            list: List of segmentation channels for each processed signal channel.
        """
        if downsampling_factor is None:
            downsampling_factor = int(original_sample_rate / target_sample_rate)

        if e_d_config is None:
            e_d_config = {
                1: {'n_e': 15, 'n_d': 10, 'strc_e': 5, 'strc_d': 5},
                2: {'n_e': 15, 'n_d': 10, 'strc_e': 5, 'strc_d': 5},
                3: {'n_e': 15, 'n_d': 10, 'strc_e': 5, 'strc_d': 5}
                }

        segmentation_channels = []

        for i in range(np.min(signal_channels.shape)):
            original_signal = signal_channels[i]

            # Filter a noisy signal.
            duration = 0.05
            num_samples = original_signal.shape[0]
            time = np.arange(0, num_samples) / original_sample_rate

            # Filtered signal
            filtered_signal = self.butter_bandpass_filter(original_signal, lower_cutoff_freq, upper_cutoff_freq,
                                                          original_sample_rate, order=5)


            # Downsampled signal
            downsampled_signal = self.downsample_numpy(filtered_signal, downsampling_factor)
            downsampled_time = np.arange(0, downsampled_signal.shape[0]) / target_sample_rate

            # Divide the signal into windows
            windows = self.divide_signal_into_windows(downsampled_signal, sample_rate=target_sample_rate,
                                                      window_duration=window_duration, overlap=overlap)

            # Expand dimensions of the windows list
            input_data = np.expand_dims(np.array(windows), axis=-1)

            # Perform z-score normalization on input_data
            normalized_input_data = self.zscore_normalization(input_data)

            # Predict the output using the model
            predictions = self.model.predict(normalized_input_data, verbose = 0)
            if type(predictions) == list:
                predictions = predictions[0]

            predicted_labels = predictions
            predicted_labels_thresholded = predictions
            id_test = np.ones((predictions.shape[0], 1))

            signal_processor = SignalPostProcessor()
            reconstructed_signals = signal_processor.reconstruct_signals(input_data, predicted_labels,
                                                                         predicted_labels_thresholded, id_test,
                                                                         overlap=overlap*target_sample_rate, show_rng=False, r_dict=True)
            
            reconstructed_signals['y_pred_test_th'] = signal_processor.threshold_segmentation_signals(
                reconstructed_signals['y_pred_test'], threshold=threshold)
            
            reconstructed_signals['y_pred_test_th_pp'] = signal_processor.process_segmentation_signals(
                reconstructed_signals['y_pred_test_th'], min_segment_length=min_segment_length*target_sample_rate, e_d_config=e_d_config)

            segmentation_channels.append(reconstructed_signals['y_pred_test_th_pp'][0].copy())

            if show_example and (i in show_example_channels):
                signal_processor.plot_signals(reconstructed_signals['x_test'], reconstructed_signals['y_test'],
                                              reconstructed_signals['y_pred_test_th_pp'], n_signals_show=1,
                                              fs = target_sample_rate, t_show_sec = show_example_w_len,
                                              rng_signals=False, title=record_name + ' - channel: ' + str(i))
                
        return segmentation_channels

    @staticmethod
    def save_segment_boundaries_to_excel(segmentation_channels, sample_rate, out_file_path, out_file_name):
        """
        Saves segment boundaries for each channel in separate sheets of an Excel file.

        Args:
            segmentation_channels (list): List of corresponding segmentation channels.
            sample_rate (int): Sample frequency.
            out_file_path (str): Output file path for the Excel file.
            out_file_name (str): Output file name for the Excel file.

        Returns:
            None
        """
        # Create a dictionary to store segment boundaries for each channel
        channel_boundaries = {}

        # Iterate over the channels
        for i, segmentation_array in enumerate(segmentation_channels):
            
            # Find the segment boundaries for the current channel
            boundaries = SignalProcessor.find_segment_boundaries(segmentation_array, sample_rate)
           
            # Store the boundaries for the current channel in the dictionary
            channel_boundaries[f'Channel_{i+1}'] = boundaries

        # Ensure the directory exists, creating it if necessary
        os.makedirs(out_file_path, exist_ok=True)

        # Create an Excel writer object to save the boundaries in separate sheets
        output_file = os.path.join(out_file_path, out_file_name)
        writer = pd.ExcelWriter(output_file)

        # Iterate over the channels and save the boundaries in separate sheets
        for channel, boundaries in channel_boundaries.items():
            df = pd.DataFrame(boundaries, columns=['Start Time', 'End Time', 'Class Type'])
            df.to_excel(writer, sheet_name=channel, index=False)

        # Save the Excel file and close the writer
        writer.close()

    @staticmethod
    def zip_folder(output_root_path, zip_file_name):
        """
        Compresses all files in a folder into a zip file.
    
        Args:
            output_root_path (str): Path to the folder containing the files.
            zip_file_name (str): Filename for the output zip file.
    
        Returns:
            None
        """
        output_path = os.path.join(output_root_path, zip_file_name)
        folder_path = os.path.join(output_root_path,'temp/')

        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, folder_path))
    
    @staticmethod
    def remove_folder_contents(folder):
        """
        Removes all directories and files within a folder.
        
        Args:
            folder_path (str): Path to the folder.
        
        Returns:
            None
        """
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


def process_files_and_zip(config):
    """
    Processes the files, saves the segment boundaries to Excel, and zips the temporary files.

    Args:
        config (dict): Configuration information. It should contain the following keys:

        - 'UPLOAD_FOLDER': Folder where the zip file is located.
        - 'PROCESS_FOLDER': Folder to save segmentation.
        - 'INPUT_FILE_NAME': If zip file: Input data file name. If None, UPLOAD_FOLDER containes unzipped files.
        - 'OUTPUT_FILE_NAME': File name to save the segmentation.
        - 'RECORDS_EXTENSION': Input file extension.
        - 'ORIGINAL_SAMPLE_RATE': Original sample rate for processing (in Hz).
        - 'TARGET_SAMPLE_RATE': Target sample rate for processing (in Hz).
        - 'WINDOWS_LENGTH': Length of windows for processing (in seconds).
        - 'OVERLAP': Overlap duration for windows (in seconds).
        - 'LOWER_CUTOFF_FREQ': Lower cutoff frequency for filtering (in Hz).
        - 'UPPER_CUTOFF_FREQ': Upper cutoff frequency for filtering (in Hz).
        - 'THRESHOLD': Threshold for segmentation.
        - 'MORPHOLOGICAL_STRUCTS': Dictionary of morphological structures for postprocessing.
        - 'MIN_SEGMENT_LENGHT': Minimum segment length for segmentation (in seconds).
        - 'SHOW_EXAMPLE': Boolean indicating whether to display examples.
        - 'SHOW_EXAMPLE_CHANNELS': List of channels to be plotted according to the .mat file.
        - 'SHOW_EXAMPLE_W_LEN': Duration of the examples to be displayed (in seconds).

    Returns:
        None
    """
    # Create an instance of SignalProcessor   
    signal_processor = SignalProcessor()

    # This try is used to show in the html if there is any error during data processing
    os.makedirs(os.path.join(config['PROCESS_FOLDER'], 'temp/'), exist_ok=True)

    # Unzip the file
    if config['INPUT_FILE_NAME'] is not None:
        extraction_directory = os.path.join(config['UPLOAD_FOLDER'], 'temp/')
        os.makedirs(extraction_directory, exist_ok=True)
        signal_processor.unzip_file(config['UPLOAD_FOLDER'], extraction_directory, config['INPUT_FILE_NAME'])
    else:
        extraction_directory = config['UPLOAD_FOLDER']


    # List files with the specified extension
    file_paths, file_names = signal_processor.list_files_with_extension(extraction_directory, config['RECORDS_EXTENSION'])

    for file_path, file_name in zip(file_paths, file_names):
        print( f"Processing file: {os.path.join(file_path, file_name)}... ")

        output_path = os.path.join(config['PROCESS_FOLDER'],  'temp', file_path)
        output_name = file_name.split('.mat')[0] + '.xlsx'

        mat_data = signal_processor.read_mat_file(os.path.join(extraction_directory, file_path, file_name))
        signal_channels = mat_data[list(mat_data.keys())[0]]

        segmentation_channels = signal_processor.process_signal_channels(signal_channels, config['ORIGINAL_SAMPLE_RATE'], config['TARGET_SAMPLE_RATE'], 
                                                                        window_duration = config['WINDOWS_LENGTH'], overlap=config['OVERLAP'],
                                                                        lower_cutoff_freq=config['LOWER_CUTOFF_FREQ'], upper_cutoff_freq=config['UPPER_CUTOFF_FREQ'], 
                                                                        threshold=config['THRESHOLD'],
                                                                        e_d_config=config['MORPHOLOGICAL_STRUCTS'], min_segment_length=config['MIN_SEGMENT_LENGHT'],
                                                                        record_name=file_name, show_example=True, 
                                                                        show_example_channels=config['SHOW_EXAMPLE_CHANNELS'],
                                                                        show_example_w_len = config['SHOW_EXAMPLE_W_LEN'])
        # Call the function to save the segment boundaries to Excel
        signal_processor.save_segment_boundaries_to_excel(segmentation_channels, config['TARGET_SAMPLE_RATE'], output_path, output_name)

        # socketio.emit('console', {'data': "Completed!"})

    # Zip the folder containing the temporary files
    signal_processor.zip_folder(config['PROCESS_FOLDER'], config['OUTPUT_FILE_NAME'])

    print("All segmentations zipped! Click Download!")

    # Restore stdout to its original value
    sys.stdout = sys.__stdout__

    # Remove files
    if config['INPUT_FILE_NAME'] is not None:
        signal_processor.remove_folder_contents(extraction_directory)
    signal_processor.remove_folder_contents(os.path.join(config['PROCESS_FOLDER'],'temp/'))

if __name__ == '__main__':

    config = {}
    config['UPLOAD_FOLDER'] = 'D:/202202/to_publish/uploaded_files'
    config['PROCESS_FOLDER'] = 'D:/202202/to_publish/processed_files'
    config['MODEL_FOLDER'] = "D:/202202/to_publish/model/model.h5"
    config['INPUT_FILE_NAME'] = "data.zip"
    config['OUTPUT_FILE_NAME'] = "segmentation.zip"

    config['RECORDS_EXTENSION'] = ".mat"
    config['ORIGINAL_SAMPLE_RATE'] = 20
    config['TARGET_SAMPLE_RATE'] = 10
    config['WINDOWS_LENGTH'] = 128
    config['OVERLAP'] = 64
    config['LOWER_CUTOFF_FREQ'] = 0.1
    config['UPPER_CUTOFF_FREQ'] = 4

    config['THRESHOLD'] = 0.8

    process_files_and_zip(config)

