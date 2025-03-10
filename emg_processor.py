import os
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import logging
import re
import datetime
import pywt  # New import for wavelet analysis

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
            
            # Parse date - try multiple formats
            date_str = parts[0]
            date = None
            
            # Try different date formats
            date_formats = ['%m%d%Y', '%m%d%y']
            for fmt in date_formats:
                try:
                    date = datetime.datetime.strptime(date_str, fmt).date()
                    logger.info(f"Successfully parsed date as {fmt}: {date}")
                    break
                except ValueError:
                    continue
            
            # If all formats fail, try to manually parse
            if date is None:
                logger.warning(f"Standard date formats failed for: {date_str}")
                try:
                    # Assuming format is MMDDYY where YY is 2-digit year
                    if len(date_str) == 6:
                        month = int(date_str[0:2])
                        day = int(date_str[2:4])
                        year = int(date_str[4:6])
                        # Add 2000 to get full year
                        year += 2000
                        date = datetime.date(year, month, day)
                        logger.info(f"Manually parsed date as: {date}")
                except Exception as e:
                    logger.warning(f"Failed to manually parse date: {date_str}, error: {e}")
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
        Load EMG data from Delsys Trigno format with dynamic muscle channels.
        
        Parameters:
        -----------
        file_path : str
            Path to the EMG data file
            
        Returns:
        --------
        tuple
            (muscle1_emg, muscle2_emg, muscle1_time, muscle2_time, metadata)
        """
        logger.info(f"Loading file: {file_path}")
        
        # Extract metadata
        metadata = {}
        with open(file_path, 'r') as f:
            # Row 1: Application info
            line = f.readline().strip().split(',')
            if len(line) >= 2:
                metadata['Application'] = line[1].strip()
            
            # Row 2: Date/Time (MM/DD/YY HH:MM:SS)
            line = f.readline().strip().split(',')
            if len(line) >= 2:
                # Parse date and time
                date_time_str = line[1].strip()
                metadata['Collection_DateTime'] = date_time_str
                
                # Try to convert to datetime object for database
                try:
                    collection_date = datetime.datetime.strptime(date_time_str, '%m/%d/%Y %I:%M:%S %p')
                    metadata['Collection_Date'] = collection_date.date()
                    metadata['Start_Time'] = collection_date.time()
                except ValueError:
                    try:
                        # Try alternate format with 2-digit year
                        collection_date = datetime.datetime.strptime(date_time_str, '%m/%d/%y %I:%M:%S %p')
                        metadata['Collection_Date'] = collection_date.date()
                        metadata['Start_Time'] = collection_date.time()
                    except ValueError:
                        logger.warning(f"Could not parse date/time: {date_time_str}")
                        metadata['Collection_Date'] = None
                        metadata['Start_Time'] = None
            
            # Row 3: Collection length
            line = f.readline().strip().split(',')
            if len(line) >= 2:
                metadata['Collection_Length'] = line[1].strip()
            
            # Row 4: Muscle names with IDs (extract up to 4 muscles)
            line = f.readline().strip().split(',')
            muscle_count = 0
            
            # Dictionary to store muscle info
            muscles = {}
            
            # Process each column that might contain muscle info
            for i in range(0, min(len(line), 8), 2):  # Check columns A, C, E, G (0, 2, 4, 6)
                if i < len(line) and line[i].strip():
                    muscle_info = line[i].strip()
                    muscle_match = re.search(r'(\w+)\s*\((\d+)\)', muscle_info)
                    
                    if muscle_match:
                        muscle_name = muscle_match.group(1)
                        muscle_id = muscle_match.group(2)
                        
                        muscle_count += 1
                        muscles[f'muscle{muscle_count}_name'] = muscle_name
                        muscles[f'muscle{muscle_count}_id'] = muscle_id
            
            # Add muscle info to metadata
            metadata.update(muscles)
            metadata['muscle_count'] = muscle_count
            
            # Row 5: Sensor modes
            line = f.readline().strip().split(',')
            if len(line) >= 4:
                metadata['muscle1_mode'] = line[1].strip().replace('sensor mode:', '').strip()
                metadata['muscle2_mode'] = line[3].strip().replace('sensor mode:', '').strip()
            
            # Row 6: Column headers - just skip
            f.readline()
            
            # Row 7: Sampling rates
            line = f.readline().strip().split(',')
            if len(line) >= 4:
                muscle1_fs_str = line[1].strip()
                muscle2_fs_str = line[3].strip()
                
                # Extract numeric rates
                muscle1_fs_match = re.search(r'([\d\.]+)', muscle1_fs_str)
                if muscle1_fs_match:
                    metadata['muscle1_fs'] = float(muscle1_fs_match.group(1))
                
                muscle2_fs_match = re.search(r'([\d\.]+)', muscle2_fs_str)
                if muscle2_fs_match:
                    metadata['muscle2_fs'] = float(muscle2_fs_match.group(1))
        
        # Now read the actual data - using two separate arrays for muscle1 and muscle2
        muscle1_data = {'time': [], 'emg': []}
        muscle2_data = {'time': [], 'emg': []}
        
        with open(file_path, 'r') as f:
            # Skip header lines (7 lines)
            for _ in range(7):
                f.readline()
            
            # Read data lines
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 4:  # Ensure we have all 4 expected columns
                    try:
                        # Muscle1 data (first two columns)
                        muscle1_time = float(parts[0])
                        muscle1_emg = float(parts[1])
                        muscle1_data['time'].append(muscle1_time)
                        muscle1_data['emg'].append(muscle1_emg)
                    except (ValueError, IndexError):
                        pass  # Skip invalid muscle1 data
                    
                    try:
                        # Muscle2 data (columns 3 and 4)
                        muscle2_time = float(parts[2])
                        muscle2_emg = float(parts[3])
                        muscle2_data['time'].append(muscle2_time)
                        muscle2_data['emg'].append(muscle2_emg)
                    except (ValueError, IndexError):
                        pass  # Skip invalid muscle2 data
        
        # Convert to numpy arrays
        muscle1_time = np.array(muscle1_data['time'])
        muscle1_emg = np.array(muscle1_data['emg'])
        muscle2_time = np.array(muscle2_data['time'])
        muscle2_emg = np.array(muscle2_data['emg'])
        
        # Log what we found
        muscle1_name = metadata.get('muscle1_name', 'Muscle1')
        muscle2_name = metadata.get('muscle2_name', 'Muscle2')
        logger.info(f"Data loaded: {muscle1_name}: {len(muscle1_emg)} samples, {muscle2_name}: {len(muscle2_emg)} samples")
        logger.info(f"Found {metadata.get('muscle_count', 0)} muscles in file")
        
        return muscle1_emg, muscle2_emg, muscle1_time, muscle2_time, metadata
    
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
    
    def detect_throws_fcr(self, fcr_rms, time, fs, threshold_factor=1.5, min_duration=0.2, min_separation=5):
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
    
    def calculate_spectral_entropy(self, signal, fs, nperseg=256):
        """
        Calculate spectral entropy of an EMG signal.
        
        Parameters:
        -----------
        signal : numpy.ndarray
            EMG signal
        fs : float
            Sampling frequency
        nperseg : int
            Length of each segment for FFT calculation
            
        Returns:
        --------
        float
            Spectral entropy value
        """
        from scipy.signal import welch
        
        # Calculate power spectral density
        f, psd = welch(signal, fs=fs, nperseg=nperseg)
        
        # Normalize PSD to get probability distribution
        psd_norm = psd / np.sum(psd)
        
        # Calculate entropy (avoid log(0))
        psd_norm = psd_norm[psd_norm > 0]
        entropy = -np.sum(psd_norm * np.log2(psd_norm))
        
        return entropy
    
    def wavelet_analysis(self, signal, fs, wavelet='db4', max_level=5):
        """
        Perform wavelet decomposition on EMG signal and extract features.
        
        Parameters:
        -----------
        signal : numpy.ndarray
            EMG signal
        fs : float
            Sampling frequency
        wavelet : str
            Wavelet type (default: 'db4' - Daubechies 4)
        max_level : int
            Maximum decomposition level
            
        Returns:
        --------
        dict
            Dictionary of wavelet features
        """
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(signal, wavelet, level=max_level)
        
        # Extract features from each level
        features = {}
        for i, coeff in enumerate(coeffs):
            if i == 0:
                level_name = 'approximation'
            else:
                level_name = f'detail_{i}'
            
            # Energy at this level
            energy = np.sum(coeff**2)
            features[f'{level_name}_energy'] = energy
            
            # Mean absolute value
            features[f'{level_name}_mav'] = np.mean(np.abs(coeff))
            
            # Variance
            features[f'{level_name}_var'] = np.var(coeff)
        
        # Calculate relative energy in each band
        total_energy = sum(features[k] for k in features if k.endswith('_energy'))
        for i in range(max_level + 1):
            if i == 0:
                level_name = 'approximation'
            else:
                level_name = f'detail_{i}'
            features[f'{level_name}_rel_energy'] = features[f'{level_name}_energy'] / total_energy if total_energy > 0 else 0
        
        return features
    
    def calculate_coactivation_indices(self, muscle1_emg, muscle2_emg, fs):
        """
        Calculate coactivation indices between two muscles.
        
        Parameters:
        -----------
        muscle1_emg : numpy.ndarray
            EMG signal of first muscle
        muscle2_emg : numpy.ndarray
            EMG signal of second muscle
        fs : float
            Sampling frequency
            
        Returns:
        --------
        dict
            Dictionary of coactivation indices
        """
        # Ensure signals are of the same length
        min_len = min(len(muscle1_emg), len(muscle2_emg))
        muscle1_emg = muscle1_emg[:min_len]
        muscle2_emg = muscle2_emg[:min_len]
        
        # Normalize EMG signals
        muscle1_norm = muscle1_emg / (np.max(muscle1_emg) if np.max(muscle1_emg) > 0 else 1)
        muscle2_norm = muscle2_emg / (np.max(muscle2_emg) if np.max(muscle2_emg) > 0 else 1)
        
        # Rectify signals
        muscle1_rect = np.abs(muscle1_norm)
        muscle2_rect = np.abs(muscle2_norm)
        
        # Calculate coactivation index (CI) as per Falconer & Winter method
        # CI = 2 * common area between normalized EMG profiles / total area
        common_area = np.minimum(muscle1_rect, muscle2_rect).sum()
        total_area = muscle1_rect.sum() + muscle2_rect.sum()
        ci_falconer = 2 * common_area / total_area if total_area > 0 else 0
        
        # Calculate cross-correlation
        correlation = np.correlate(muscle1_rect, muscle2_rect, mode='valid')[0] / min_len
        
        # Calculate temporal overlap
        active_threshold = 0.1  # Threshold for considering muscle active
        muscle1_active = muscle1_rect > active_threshold
        muscle2_active = muscle2_rect > active_threshold
        both_active = np.logical_and(muscle1_active, muscle2_active)
        temporal_overlap = np.sum(both_active) / min_len
        
        # Calculate waveform similarity index
        diff_norm = np.sum((muscle1_rect - muscle2_rect) ** 2)
        sum_norm = np.sum(muscle1_rect ** 2) + np.sum(muscle2_rect ** 2)
        waveform_similarity = 1 - (diff_norm / sum_norm if sum_norm > 0 else 0)
        
        # Return all indices
        return {
            'ci_falconer': ci_falconer,
            'correlation': correlation,
            'temporal_overlap': temporal_overlap,
            'waveform_similarity': waveform_similarity
        }
    
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
        
        # Workload metrics
        throw_integrals = np.zeros(throw_count)
        throw_durations = np.zeros(throw_count)
        work_rates = np.zeros(throw_count)
        
        # NEW: Spectral entropy metrics
        spectral_entropies = np.zeros(throw_count)
        
        # NEW: Wavelet energy metrics
        wavelet_energy_low = np.zeros(throw_count)
        wavelet_energy_mid = np.zeros(throw_count)
        wavelet_energy_high = np.zeros(throw_count)
        
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
            
            # NEW: Calculate spectral entropy
            spectral_entropies[i] = self.calculate_spectral_entropy(segment_filtered, fs)
            
            # NEW: Calculate wavelet features
            try:
                wavelet_features = self.wavelet_analysis(segment_filtered, fs)
                
                # Map wavelet levels to frequency bands
                # Note: exact frequency bands depend on the sampling frequency and wavelet type
                # detail_1: ~fs/4 to fs/2 Hz (high frequency)
                # detail_2: ~fs/8 to fs/4 Hz (mid-high frequency)
                # detail_3: ~fs/16 to fs/8 Hz (mid frequency)
                # detail_4 & detail_5: lower frequencies
                
                # Low frequency band (detail levels 4-5)
                wavelet_energy_low[i] = (wavelet_features.get('detail_4_rel_energy', 0) + 
                                        wavelet_features.get('detail_5_rel_energy', 0) + 
                                        wavelet_features.get('approximation_rel_energy', 0))
                
                # Mid frequency band (detail levels 2-3)
                wavelet_energy_mid[i] = (wavelet_features.get('detail_2_rel_energy', 0) + 
                                        wavelet_features.get('detail_3_rel_energy', 0))
                
                # High frequency band (detail level 1)
                wavelet_energy_high[i] = wavelet_features.get('detail_1_rel_energy', 0)
            except Exception as e:
                logger.warning(f"Failed to calculate wavelet features for throw {i+1}: {e}")
                wavelet_energy_low[i] = 0
                wavelet_energy_mid[i] = 0
                wavelet_energy_high[i] = 0
        
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
            
            # Workload metrics
            'throw_integrals': throw_integrals,
            'work_rates': work_rates,
            
            # NEW: Spectral entropy metrics
            'spectral_entropies': spectral_entropies,
            
            # NEW: Wavelet energy metrics
            'wavelet_energy_low': wavelet_energy_low,
            'wavelet_energy_mid': wavelet_energy_mid,
            'wavelet_energy_high': wavelet_energy_high
        }
        
        return metrics
    
    def calculate_muscle_coactivation(self, muscle1_metrics, muscle2_metrics, 
                                     muscle1_throws, muscle2_throws, 
                                     muscle1_filtered, muscle2_filtered,
                                     muscle1_time, muscle2_time, fs):
        """
        Calculate coactivation metrics between two muscles for all throws.
        
        Parameters:
        -----------
        muscle1_metrics : dict
            Metrics dictionary for first muscle
        muscle2_metrics : dict
            Metrics dictionary for second muscle
        muscle1_throws : list
            List of throw indices for first muscle
        muscle2_throws : list
            List of throw indices for second muscle
        muscle1_filtered : numpy.ndarray
            Filtered EMG signal for first muscle
        muscle2_filtered : numpy.ndarray
            Filtered EMG signal for second muscle
        muscle1_time : numpy.ndarray
            Time array for first muscle
        muscle2_time : numpy.ndarray
            Time array for second muscle
        fs : float
            Sampling frequency
            
        Returns:
        --------
        dict
            Dictionary with coactivation metrics for each throw
        """
        throw_count = len(muscle1_throws)
        if throw_count != len(muscle2_throws):
            logger.warning(f"Throw count mismatch: {throw_count} vs {len(muscle2_throws)}")
            return None
        
        # Initialize coactivation metrics arrays
        ci_falconer_values = np.zeros(throw_count)
        correlation_values = np.zeros(throw_count)
        temporal_overlap_values = np.zeros(throw_count)
        waveform_similarity_values = np.zeros(throw_count)
        
        # Calculate coactivation indices for each throw
        for i in range(throw_count):
            # Extract segments for this throw
            m1_start, m1_end = muscle1_throws[i]
            m2_start, m2_end = muscle2_throws[i]
            
            m1_segment = muscle1_filtered[m1_start:m1_end]
            m2_segment = muscle2_filtered[m2_start:m2_end]
            
            # Calculate coactivation metrics
            coactivation = self.calculate_coactivation_indices(m1_segment, m2_segment, fs)
            
            # Store values
            ci_falconer_values[i] = coactivation['ci_falconer']
            correlation_values[i] = coactivation['correlation']
            temporal_overlap_values[i] = coactivation['temporal_overlap']
            waveform_similarity_values[i] = coactivation['waveform_similarity']
        
        # Return metrics dictionary
        return {
            'ci_falconer': ci_falconer_values,
            'correlation': correlation_values,
            'temporal_overlap': temporal_overlap_values,
            'waveform_similarity': waveform_similarity_values
        }