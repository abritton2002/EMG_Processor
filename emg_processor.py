import os
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import logging
import re
import datetime

# Configure logging
logger = logging.getLogger(__name__)

class EMGProcessor:
    """
    Processes EMG data files and extracts features/metrics.
    """
    
    def __init__(self):
        """Initialize the EMG processor."""
        logger.info("EMG Processor initialized")
    
    def parse_filename(self, filename):
        """
        Parse EMG filename to extract metadata.
        Format: MMDDYYYY_TraqID_Name_sessiontype.csv
        
        Parameters:
        -----------
        filename : str
            EMG file name
            
        Returns:
        --------
        dict
            Parsed metadata
        """
        try:
            # Remove file extension and split by underscore
            base_name = os.path.splitext(os.path.basename(filename))[0]
            parts = base_name.split('_')
            
            if len(parts) < 4:
                logger.warning(f"Filename {filename} does not match expected format")
                return {
                    'date': None,
                    'traq_id': 'unknown',
                    'athlete_name': 'unknown',
                    'session_type': 'unknown'
                }
            
            # Parse date (MMDDYYYY)
            date_str = parts[0]
            try:
                date = datetime.datetime.strptime(date_str, '%m%d%Y').date()
            except ValueError:
                logger.warning(f"Invalid date format in filename: {date_str}")
                date = None
            
            return {
                'date': date,
                'traq_id': parts[1],
                'athlete_name': parts[2],
                'session_type': parts[3]
            }
        except Exception as e:
            logger.error(f"Error parsing filename {filename}: {e}")
            return {
                'date': None,
                'traq_id': 'unknown',
                'athlete_name': 'unknown',
                'session_type': 'unknown'
            }
    
    def load_delsys_emg_data(self, file_path):
        """
        Load EMG data from Delsys Trigno format with FCU and FCR channels.
        
        Parameters:
        -----------
        file_path : str
            Path to the EMG data file
            
        Returns:
        --------
        tuple
            (fcu_emg, fcr_emg, fcu_time, fcr_time, metadata)
        """
        logger.info(f"Loading file: {file_path}")
        
        # Extract metadata
        metadata = {}
        with open(file_path, 'r') as f:
            # Read metadata lines
            app_line = f.readline().strip().split(',')
            if len(app_line) >= 2:
                metadata['Application'] = app_line[1].strip()
            
            date_line = f.readline().strip().split(',')
            if len(date_line) >= 2:
                metadata['Date/Time'] = date_line[1].strip()
            
            length_line = f.readline().strip().split(',')
            if len(length_line) >= 2:
                metadata['Collection Length'] = length_line[1].strip()
            
            # Read muscle names (FCU and FCR)
            channel_line = f.readline().strip().split(',')
            if len(channel_line) >= 3:
                fcu_channel = channel_line[0].strip()
                fcr_channel = channel_line[2].strip()
                metadata['FCU'] = fcu_channel
                metadata['FCR'] = fcr_channel
            
            # Read sensor modes
            mode_line = f.readline().strip().split(',')
            if len(mode_line) >= 3:
                metadata['FCU_mode'] = mode_line[0].strip()
                metadata['FCR_mode'] = mode_line[2].strip()
            
            # Read column headers
            header_line = f.readline().strip().split(',')
            if len(header_line) >= 4:
                metadata['FCU_time_header'] = header_line[0].strip()
                metadata['FCU_emg_header'] = header_line[1].strip()
                metadata['FCR_time_header'] = header_line[2].strip()
                metadata['FCR_emg_header'] = header_line[3].strip()
            
            # Read sampling rates
            rate_line = f.readline().strip().split(',')
            if len(rate_line) >= 4:
                if 'Hz' in rate_line[1]:
                    fcu_fs = float(rate_line[1].replace('Hz', '').strip())
                    metadata['FCU_fs'] = fcu_fs
                if 'Hz' in rate_line[3]:
                    fcr_fs = float(rate_line[3].replace('Hz', '').strip())
                    metadata['FCR_fs'] = fcr_fs
        
        # Now read the actual data - using two separate arrays for FCU and FCR
        fcu_data = {'time': [], 'emg': []}
        fcr_data = {'time': [], 'emg': []}
        
        with open(file_path, 'r') as f:
            # Skip header lines (8 lines)
            for _ in range(8):
                f.readline()
            
            # Read data lines
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 4:  # Ensure we have all 4 expected columns
                    try:
                        # FCU data (first two columns)
                        fcu_time = float(parts[0])
                        fcu_emg = float(parts[1])
                        fcu_data['time'].append(fcu_time)
                        fcu_data['emg'].append(fcu_emg)
                    except (ValueError, IndexError):
                        pass  # Skip invalid FCU data
                    
                    try:
                        # FCR data (columns 3 and 4)
                        fcr_time = float(parts[2])
                        fcr_emg = float(parts[3])
                        fcr_data['time'].append(fcr_time)
                        fcr_data['emg'].append(fcr_emg)
                    except (ValueError, IndexError):
                        pass  # Skip invalid FCR data
        
        # Convert to numpy arrays
        fcu_time = np.array(fcu_data['time'])
        fcu_emg = np.array(fcu_data['emg'])
        fcr_time = np.array(fcr_data['time'])
        fcr_emg = np.array(fcr_data['emg'])
        
        logger.info(f"Data loaded: FCU: {len(fcu_emg)} samples, FCR: {len(fcr_emg)} samples")
        
        return fcu_emg, fcr_emg, fcu_time, fcr_time, metadata
    
    def preprocess_emg(self, emg_signal, fs):
        """
        Preprocess EMG data with multiple filtering approaches.
        
        Parameters:
        -----------
        emg_signal : numpy.ndarray
            Raw EMG signal
        fs : float
            Sampling frequency
            
        Returns:
        --------
        tuple
            (filtered, rectified, rms, envelope)
        """
        # High-pass filter (20 Hz cutoff to remove motion artifacts)
        sos_hp = signal.butter(4, 20, 'hp', fs=fs, output='sos')
        filtered = signal.sosfilt(sos_hp, emg_signal)
        
        # Notch filter for power line interference (60 Hz for US)
        notch_freq = 60.0
        notch_width = 4.0
        quality_factor = notch_freq / notch_width
        b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, fs)
        filtered = signal.filtfilt(b_notch, a_notch, filtered)
        
        # Rectification (full-wave)
        rectified = np.abs(filtered)
        
        # RMS smoothing
        rms_window_size = int(0.05 * fs)  # 50 ms RMS window
        if rms_window_size % 2 == 0:
            rms_window_size += 1  # Ensure window size is odd
        
        rms = np.sqrt(signal.convolve(rectified**2, np.ones(rms_window_size)/rms_window_size, mode='same'))
        
        # Envelope detection
        sos_env = signal.butter(4, 6, 'lp', fs=fs, output='sos')
        envelope = signal.sosfilt(sos_env, rectified)
        
        return filtered, rectified, rms, envelope
    
    def detect_throws_fcr(self, fcr_rms, time, fs, threshold_factor=1.2, min_duration=0.2, min_separation=1.5):
        """
        Detect throwing motions using adaptive thresholding on FCR EMG data.
        
        Parameters:
        -----------
        fcr_rms : numpy.ndarray
            RMS processed FCR EMG data
        time : numpy.ndarray
            Time array corresponding to FCR data
        fs : float
            Sampling frequency
        threshold_factor : float, optional
            Factor for adaptive threshold calculation
        min_duration : float, optional
            Minimum throw duration in seconds
        min_separation : float, optional
            Minimum separation between throws in seconds
            
        Returns:
        --------
        list
            List of (start_index, end_index) tuples for each detected throw
        """
        # Dynamic threshold based on signal statistics
        fcr_threshold = np.mean(fcr_rms) + threshold_factor * np.std(fcr_rms)
        
        # Find regions above threshold
        fcr_above = fcr_rms > fcr_threshold
        
        # Detect throw events
        throws = []
        start = None
        
        for i in range(1, len(fcr_above)):
            if fcr_above[i] and not fcr_above[i-1]:
                start = i
            elif not fcr_above[i] and fcr_above[i-1] and start is not None:
                duration = (i - start) / fs
                if duration >= min_duration:
                    throws.append((start, i))
                start = None
        
        # Filter throws by minimum separation
        if throws:
            filtered_throws = [throws[0]]
            for i in range(1, len(throws)):
                if (throws[i][0] - filtered_throws[-1][1]) / fs >= min_separation:
                    filtered_throws.append(throws[i])
            return filtered_throws
        return throws
    
    def map_throws_to_fcu(self, throws_fcr, fcr_time, fcu_time):
        """
        Map FCR throw indices to FCU indices for consistent analysis.
        
        Parameters:
        -----------
        throws_fcr : list
            List of FCR throw indices
        fcr_time : numpy.ndarray
            FCR time array
        fcu_time : numpy.ndarray
            FCU time array
            
        Returns:
        --------
        list
            List of FCU throw indices
        """
        throws_fcu = []
        for start_fcr, end_fcr in throws_fcr:
            # Convert FCR throw time points to FCU indices
            start_time = fcr_time[start_fcr]
            end_time = fcr_time[end_fcr]
            
            # Find closest FCU indices for these time points
            start_fcu = np.argmin(np.abs(fcu_time - start_time))
            end_fcu = np.argmin(np.abs(fcu_time - end_time))
            
            throws_fcu.append((start_fcu, end_fcu))
        
        return throws_fcu
    
    def calculate_comprehensive_metrics(self, emg_filtered, emg_rectified, emg_rms, emg_envelope, throws, time, fs):
        """
        Calculate comprehensive metrics for each throw.
        
        Parameters:
        -----------
        emg_filtered : numpy.ndarray
            Filtered EMG signal
        emg_rectified : numpy.ndarray
            Rectified EMG signal
        emg_rms : numpy.ndarray
            RMS processed EMG signal
        emg_envelope : numpy.ndarray
            Envelope of EMG signal
        throws : list
            List of (start_index, end_index) tuples for each throw
        time : numpy.ndarray
            Time array
        fs : float
            Sampling frequency
            
        Returns:
        --------
        dict
            Dictionary of metrics for each throw
        """
        # Initialize metrics arrays
        throw_count = len(throws)
        
        # Frequency domain metrics
        median_freqs = np.zeros(throw_count)
        mean_freqs = np.zeros(throw_count)
        bandwidth = np.zeros(throw_count)
        
        # Amplitude domain metrics
        peak_amplitudes = np.zeros(throw_count)
        rms_values = np.zeros(throw_count)
        
        # Temporal metrics
        rise_times = np.zeros(throw_count)
        contraction_times = np.zeros(throw_count)
        relaxation_times = np.zeros(throw_count)
        contraction_relaxation_ratios = np.zeros(throw_count)
        
        # Workload metrics
        throw_integrals = np.zeros(throw_count)
        throw_durations = np.zeros(throw_count)
        work_rates = np.zeros(throw_count)
        
        # Process each throw for comprehensive metrics
        for i, (start, end) in enumerate(throws):
            # Extract throw segments
            segment_filtered = emg_filtered[start:end]
            segment_rectified = emg_rectified[start:end]
            segment_rms = emg_rms[start:end]
            segment_envelope = emg_envelope[start:end]
            segment_time = time[start:end] - time[start]  # Normalize time to start at 0
            
            # Duration
            throw_durations[i] = segment_time[-1]
            
            # Amplitude domain metrics
            peak_amplitudes[i] = np.max(segment_rectified)
            rms_values[i] = np.sqrt(np.mean(segment_filtered**2))
            
            # Workload metrics - area under the curve (trapezoidal integration)
            throw_integrals[i] = np.trapz(segment_rectified, segment_time)
            work_rates[i] = throw_integrals[i] / throw_durations[i]
            
            # Temporal metrics
            peak_idx = np.argmax(segment_envelope)
            
            # Rise time (time to reach 90% of max from 10% of max)
            rise_threshold_low = 0.1 * segment_envelope[peak_idx]
            rise_threshold_high = 0.9 * segment_envelope[peak_idx]
            
            try:
                rise_start_idx = np.where(segment_envelope >= rise_threshold_low)[0][0]
                rise_end_idx = np.where(segment_envelope >= rise_threshold_high)[0][0]
                rise_times[i] = segment_time[rise_end_idx] - segment_time[rise_start_idx]
            except IndexError:
                rise_times[i] = np.nan
            
            # Contraction and relaxation times
            contraction_times[i] = segment_time[peak_idx] if peak_idx < len(segment_time) else np.nan
            relaxation_times[i] = segment_time[-1] - segment_time[peak_idx] if peak_idx < len(segment_time) else np.nan
            
            # Contraction-relaxation ratio
            if not np.isnan(contraction_times[i]) and not np.isnan(relaxation_times[i]) and relaxation_times[i] > 0:
                contraction_relaxation_ratios[i] = contraction_times[i] / relaxation_times[i]
            else:
                contraction_relaxation_ratios[i] = np.nan
            
            # Frequency domain metrics
            # Apply Hanning window to reduce spectral leakage
            windowed = segment_filtered * np.hanning(len(segment_filtered))
            
            # FFT
            fft_result = np.abs(fft(windowed))
            
            # Only first half of FFT is meaningful (Nyquist)
            n = len(segment_filtered)
            fft_result = fft_result[:n//2]
            
            # Frequency array
            freqs = fftfreq(n, 1/fs)[:n//2]
            
            # Only consider the physiologically relevant frequency range for EMG (20-450 Hz)
            freq_mask = (freqs >= 20) & (freqs <= 450)
            
            # Calculate frequency metrics
            if np.sum(fft_result[freq_mask]) > 0:  # Avoid division by zero
                power = fft_result[freq_mask]**2
                cumpower = np.cumsum(power)
                
                # Median frequency (point at which half the power is above and half below)
                median_idx = np.where(cumpower >= cumpower[-1]/2)[0][0]
                median_freqs[i] = freqs[freq_mask][median_idx]
                
                # Mean frequency
                mean_freqs[i] = np.sum(freqs[freq_mask] * power) / np.sum(power)
                
                # Bandwidth (frequency range containing 68% of power, ~1 std dev)
                lower_band_idx = np.where(cumpower >= cumpower[-1]*0.16)[0][0]
                upper_band_idx = np.where(cumpower >= cumpower[-1]*0.84)[0][0]
                bandwidth[i] = freqs[freq_mask][upper_band_idx] - freqs[freq_mask][lower_band_idx]
        
        # Calculate throw indices for plotting
        throw_indices = np.arange(1, throw_count + 1)
        
        # Calculate throw timestamps for time-based analysis
        throw_timestamps = np.array([time[start] for start, _ in throws])
        throw_end_times = np.array([time[end] for start, end in throws])
        
        # Results dictionary
        metrics = {
            'throw_indices': throw_indices,
            'throw_timestamps': throw_timestamps,
            'throw_end_times': throw_end_times,
            'throw_durations': throw_durations,
            
            # Frequency domain metrics
            'median_freqs': median_freqs,
            'mean_freqs': mean_freqs,
            'bandwidth': bandwidth,
            
            # Amplitude domain metrics
            'peak_amplitudes': peak_amplitudes,
            'rms_values': rms_values,
            
            # Temporal metrics
            'rise_times': rise_times,
            'contraction_times': contraction_times,
            'relaxation_times': relaxation_times,
            'contraction_relaxation_ratios': contraction_relaxation_ratios,
            
            # Workload metrics
            'throw_integrals': throw_integrals,
            'work_rates': work_rates
        }
        
        return metrics

# For direct testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    processor = EMGProcessor()
    # Test filename parsing
    test_filename = "03012025_TRAQ123_JohnDoe_mocap.csv"
    parsed = processor.parse_filename(test_filename)
    print(f"Parsed filename: {parsed}")