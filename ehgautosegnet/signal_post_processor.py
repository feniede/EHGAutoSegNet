import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation
from tensorflow.keras.utils import to_categorical

class SignalPostProcessor:
    """
    Class for post-processing semantic segmentation signals.
    """

    def reconstruct_signals(self, x, y, p, ids, overlap=320, r_dict=False, show_rng=False, fs=10, t_show_min=10, n_signals_show=10, threshold=None):
        """
        Reconstruct signals from segmented windows.

        Parameters:
        - x: numpy array, input signals
        - y: numpy array, ground truth labels
        - p: numpy array, predicted labels
        - ids: numpy array, unique IDs for each signal
        - overlap: int, overlap between consecutive windows (default: 320)
        - r_dict: bool, whether to return signals as a dictionary (default: False)
        - show_rng: bool, whether to show a random selection of reconstructed signals (default: False)
        - fs: int, sampling rate of the signals (default: 5)
        - t_show_min: int, minimum duration (in minutes) to show for each signal (default: 10)
        - n_signals_show: int, number of signals to show if show_rng=True (default: 10)
        - threshold: float, threshold value for binary segmentation (optional)

        Returns:
        - If r_dict=True, returns a dictionary containing the reconstructed signals: {'x_test', 'y_test', 'y_pred_test'}
        - If r_dict=False, returns a tuple containing the reconstructed signals: (x_w, y_w, p_w)
        """
        x_w = []
        y_w = []
        p_w = []
        uniques, counts = np.unique(ids, return_counts=True)
        for u, c in zip(uniques, counts):
            idx, _ = np.where(ids == u)
            x_w.append(self.reconstruct_signal(x[idx], overlap))
            y_w.append(self.reconstruct_signal(y[idx], overlap))
            p_w.append(self.reconstruct_signal(p[idx], overlap))
            
        if show_rng:
            t_show = t_show_min * 60 * fs
            f, ax = plt.subplots(n_signals_show * 2, 1, sharex=True, figsize=(20, n_signals_show * 3.5)) 
    
            random_signals = np.random.randint(0, len(x_w), size=n_signals_show * 2)
            
            for i in range(0, n_signals_show * 2 - 1, 2):
                tmin = 0
                signal = x_w[random_signals[i]][tmin:tmin + t_show]
                ground_true = y_w[random_signals[i]][tmin:tmin + t_show]
                predictions = p_w[random_signals[i]][tmin:tmin + t_show]
                
                self._plot_semantic_seg(signal, ground_true, ax[i], threshold=threshold)
                ax[i].set_title('~~' * 64, color='m')
                ax[i].set_xlim([0, len(signal) / fs])
                ax[i].set_yticks([])
                
                self._plot_semantic_seg(signal, predictions, ax[i + 1], threshold=threshold)
                ax[i + 1].set_ylabel('--Predictions--')
                ax[i + 1].set_yticks([])
                ax[i].set_ylim([np.min(signal), np.max(signal)])
                ax[i + 1].set_ylim([np.min(signal), np.max(signal)])
                
        if r_dict:
            signals = {}
            signals['x_test'], signals['y_test'], signals['y_pred_test'] = x_w, y_w, p_w
            return signals
        else:
            return x_w, y_w, p_w

    def plot_signals(self, x, y_true, y_pred, fs=10, t_show_sec=600, n_signals_show=None, threshold=None, rng_signals=True, title=None):
        """
        Plot the original signal along with the ground truth and predicted semantic segmentation.

        Parameters:
        - x: numpy array, input signal
        - y_true: numpy array, ground truth segmentation
        - y_pred: numpy array, predicted segmentation
        - fs: float, sampling frequency of the signal
        - t_show_sec: int, duration in seconds to show in the plot
        - n_signals_show: int, number of signals to show (if None, all signals are shown)
        - threshold: float, threshold for binary visualization of the segmentation
        - rng_signals: bool, if True, randomly select signals to show; otherwise, select the first n_signals_show signals
        """

        x_w, y_w, p_w = x, y_true, y_pred

        t_show = t_show_sec * fs
        f, ax = plt.subplots(n_signals_show * 2, 1, sharex=True, figsize=(20, n_signals_show * 3.5))

        if rng_signals:
            n_signals_show = 10
            idx_signals = np.random.randint(0, len(x_w), size=n_signals_show * 2)
        else:
            if n_signals_show is not None:
                idx_signals = range(len(x))
            else:
                idx_signals = range(n_signals_show)

        for i in range(0, n_signals_show * 2 - 1, 2):
            tmin = 0
            signal = x_w[idx_signals[i]][tmin:tmin + t_show]
            ground_true = y_w[idx_signals[i]][tmin:tmin + t_show]
            predictions = p_w[idx_signals[i]][tmin:tmin + t_show]

            self._plot_semantic_seg(signal, ground_true, ax[i], threshold=threshold)
            ax[i].set_title(title, color='m')
            ax[i].set_xlim([0, len(signal) / fs])
            ax[i].set_ylabel('--Model Output--')
            ax[i].set_yticks([])

            self._plot_semantic_seg(signal, predictions, ax[i + 1], threshold=threshold)
            ax[i + 1].set_ylabel('--Postproccessing--')
            ax[i + 1].set_yticks([])
            ax[i].set_ylim([np.min(signal), np.max(signal)])
            ax[i + 1].set_ylim([np.min(signal), np.max(signal)])

    def threshold_segmentation_signals(self, y_pred, threshold=None):
        """
        Threshold the predicted segmentation signals.

        Parameters:
        - y_pred: list of numpy arrays, predicted segmentation signals
        - threshold: float, threshold value for binary segmentation

        Returns:
        - thresholded_segments: list of numpy arrays, thresholded segmentation signals
        """

        def thresholding(s_mask, threshold=None):
            masks = s_mask.copy()
            if threshold is not None:
                masks[..., 0][masks[..., 0] >= threshold] = 1
                masks[..., 0][masks[..., 0] < threshold] = 0
            masks = np.argmax(masks, axis=1)
            return masks

        return [thresholding(y_pred_i, threshold=threshold) for y_pred_i in y_pred]

    def erosion_dilation_operation_signal(self, segmentation, e_d_config=None):
        """
        Apply erosion and dilation operations to the segmentation signal.

        Parameters:
        - segmentation: numpy array, input segmentation signal
        - e_d_config: dict, erosion and dilation configuration

        Returns:
        - output: numpy array, processed segmentation signal
        """

        def e_d(segmentation_1channel, n_e, strc_e, n_d, strc_d):
            output = segmentation_1channel.copy()
            output = binary_dilation(output, structure=np.ones(strc_d), iterations=n_d)
            output = binary_erosion(output, structure=np.ones(strc_e), iterations=n_e)
            output = binary_dilation(output, structure=np.ones(strc_d), iterations=n_d)
            return output

        segmentation_1hot = to_categorical(segmentation)
        output = np.zeros(segmentation.shape)

        for i in reversed(range(1, segmentation_1hot.shape[-1])):
            if e_d_config is not None:
                n_e = e_d_config[i]['n_e']
                n_d = e_d_config[i]['n_d']
                strc_e = e_d_config[i]['strc_e']
                strc_d = e_d_config[i]['strc_d']
            aux = e_d(segmentation_1hot[:, i], n_e=n_e, strc_e=strc_e, n_d=n_d, strc_d=strc_d)
            output[aux == 1] = i

        return output

    def process_segmentation_signals(self, in_segmentations, min_segment_length, e_d_config=None):
        """
        Process the input segmentation signals.

        Parameters:
        - in_segmentations: list of numpy arrays, input segmentation signals
        - min_segment_length: int, minimum segment length for class replacement
        - e_d_config: dict, erosion and dilation configuration

        Returns:
        - processed_segments: list of numpy arrays, processed segmentation signals
        """

        return [self._process_segmentation_signal(ss, min_segment_length, e_d_config) for ss in in_segmentations]

    def _process_segmentation_signal(self, in_segmentation, min_segment_length, e_d_config=None):
        """
        Process a single segmentation signal.

        Parameters:
        - in_segmentation: numpy array, input segmentation signal
        - min_segment_length: int, minimum segment length for class replacement
        - e_d_config: dict, erosion and dilation configuration

        Returns:
        - output: numpy array, processed segmentation signal
        """

        segmentation = in_segmentation.copy()
        output = in_segmentation.copy()

        if e_d_config is not None:
            output = self.erosion_dilation_operation_signal(segmentation, e_d_config)

        segmentation = output.copy()

        start_idx = None  # Start index of the current segment
        prev_class = None  # Class of the previous segment

        for i in range(len(segmentation)):
            if segmentation[i] == 0 and start_idx is None:
                start_idx = i  # Start a new segment

                if i > 0:
                    prev_class = segmentation[i - 1]  # Get the class of the previous segment

            elif segmentation[i] != 0 and start_idx is not None:
                segment_length = i - start_idx  # Calculate the length of the segment

                if segment_length < min_segment_length:
                    if prev_class is not None and segmentation[i] == prev_class:
                        output[start_idx:i] = [prev_class] * segment_length
                    else:
                        output[start_idx:i] = [segmentation[i]] * segment_length

                start_idx = None  # Reset the start index
                prev_class = None  # Reset the previous class

        return output

    def _plot_semantic_seg(self, y, s_mask, ax, threshold=None):
        """
        Plot the semantic segmentation of a signal.

        Parameters:
        - y: numpy array, input signal
        - s_mask: numpy array, semantic segmentation mask
        - ax: matplotlib Axes object, plot axis
        - threshold: float, threshold value for binary segmentation (optional)

        Returns:
        None
        """
        edgecolors = ['k' ,'b', 'y','g']

        n_cat = s_mask.shape[-1]
        t = np.arange(0, y.shape[0] / 5, 1 / 5)

        if len(s_mask.shape) > 1:
            if threshold is not None:
                masks = s_mask.copy()
                masks[..., 0][masks[..., 0] >= threshold] = 1
                masks[..., 0][masks[..., 0] < threshold] = 0
                masks = np.argmax(masks, axis=1)
            else:
                masks = np.argmax(s_mask, axis=1)
        else:
            masks = s_mask

        # ax.plot(t, y, c='k', linewidth=0.5)

        for cat in range(n_cat):
            c = masks != cat
            if np.any(np.invert(c)):
                mask = np.ma.array(y, mask=c)
                if cat == 0:
                    ax.plot(t, mask, edgecolors[0], linewidth=0.75)
                    # continue
                elif cat == 1:
                    ax.plot(t, mask, edgecolors[1], linewidth=0.75)
                elif cat == 2:
                    ax.plot(t, mask, edgecolors[2], linewidth=0.75)
                elif cat == 3:
                    ax.plot(t, mask, edgecolors[3], linewidth=0.75)
                elif cat == 7:
                    ax.plot(t, mask, 'r', linewidth=0.5)
                else:
                    ax.plot(t, mask, 'm', linewidth=0.5)

    def reconstruct_signal(self, windows, overlap):
        """
        Reconstruct a signal from overlapping windows.

        Parameters:
        - windows: numpy array, windows of the signal
        - overlap: int, overlap between consecutive windows

        Returns:
        - reconstructed_signal: numpy array, reconstructed signal
        """
        # Calculate the number of samples in each window
        window_size = windows.shape[1]
        
        # Calculate the stride between consecutive windows
        stride = window_size - overlap
        
        # Calculate the number of overlapping samples
        overlap_samples = window_size - stride
        
        # Calculate the number of channels in the signal
        num_channels = windows.shape[2]
        
        # Calculate the length of the reconstructed signal
        signal_length = (windows.shape[0] - 1) * stride + window_size
        
        # Create an empty array to store the reconstructed signal
        reconstructed_signal = np.zeros((signal_length, num_channels))
        
        # Reconstruct the signal by overlapping and averaging the windows
        for i, window in enumerate(windows):
            start_index = i * stride
            end_index = start_index + window_size
            
            reconstructed_signal[start_index:end_index] += window
            
            # Average the overlapping samples
            if i > 0:
                reconstructed_signal[start_index:start_index + overlap_samples] /= 2.0
        
        return reconstructed_signal