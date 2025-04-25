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
        Load EMG data from Delsys Trigno format with header rows.
        Handles the specific format of Delsys files with metadata.
        """
        metadata = {}
        muscles = {}
        muscle_count = 0
        
        logger.info(f"Loading file: {file_path}")
        file_size = os.path.getsize(file_path)
        logger.info(f"File size: {file_size} bytes")
        
        with open(file_path, 'r') as f:
            # Read and process header lines
            header_lines = []
            for i in range(7):  # Read first 7 lines
                line = f.readline().strip()
                header_lines.append(line)
            
            logger.info("First 7 header lines:")
            for i, line in enumerate(header_lines, 1):
                logger.info(f"Line {i}: {line}")
            
            # Parse metadata from header
            if header_lines[1].startswith('Date/Time:'):
                metadata['Collection_Date'] = header_lines[1].split(',')[1].strip()
            
            # Parse muscle names from line 4
            muscle_line = header_lines[3].split(',')
            muscle_names = [name.strip() for name in muscle_line if name.strip()]
            
            # Parse sampling rates from line 7
            sampling_rates = header_lines[6].split(',')
            for i, rate in enumerate(sampling_rates):
                if 'Hz' in rate:
                    try:
                        # Extract numeric value before 'Hz'
                        rate_value = float(rate.strip().split('Hz')[0].strip())
                        if i == 1:  # First muscle
                            metadata['muscle1_fs'] = rate_value
                        elif i == 3:  # Second muscle
                            metadata['muscle2_fs'] = rate_value
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Could not parse sampling rate from '{rate}'. Using default: 2000 Hz")
                        if i == 1:
                            metadata['muscle1_fs'] = 2000.0
                        elif i == 3:
                            metadata['muscle2_fs'] = 2000.0
            
            # Count data lines
            data_lines = sum(1 for _ in f)
            logger.info(f"Total data lines: {data_lines}")
            
            # Reset file pointer to start of data
            f.seek(0)
            for _ in range(7):  # Skip header lines
                next(f)
            
            # Read data into arrays
            time1, emg1, time2, emg2 = [], [], [], []
            for line in f:
                values = line.strip().split(',')
                if len(values) >= 4:
                    try:
                        if values[0]:  # Time1
                            time1.append(float(values[0]))
                        if values[1]:  # EMG1
                            emg1.append(float(values[1]))
                        if values[2]:  # Time2
                            time2.append(float(values[2]))
                        if values[3]:  # EMG2
                            emg2.append(float(values[3]))
                    except ValueError as e:
                        continue  # Skip invalid lines
            
            # Convert to numpy arrays
            time1 = np.array(time1)
            emg1 = np.array(emg1)
            time2 = np.array(time2)
            emg2 = np.array(emg2)
            
            # Log data statistics
            logger.info(f"Muscle1 Time - Length: {len(time1)}, First: {time1[0] if len(time1) > 0 else None}, Last: {time1[-1] if len(time1) > 0 else None}")
            logger.info(f"Muscle1 EMG - Length: {len(emg1)}, First: {emg1[0] if len(emg1) > 0 else None}, Last: {emg1[-1] if len(emg1) > 0 else None}")
            logger.info(f"Muscle2 Time - Length: {len(time2)}, First: {time2[0] if len(time2) > 0 else None}, Last: {time2[-1] if len(time2) > 0 else None}")
            logger.info(f"Muscle2 EMG - Length: {len(emg2)}, First: {emg2[0] if len(emg2) > 0 else None}, Last: {emg2[-1] if len(emg2) > 0 else None}")
            
            # Calculate actual sampling frequency from time data if available
            if len(time1) > 1:
                fs1 = round(1.0 / np.mean(np.diff(time1)), 1)
                logger.info(f"Sampling Frequency: {fs1} Hz")
                metadata['muscle1_fs'] = fs1
            if len(time2) > 1:
                fs2 = round(1.0 / np.mean(np.diff(time2)), 1)
                metadata['muscle2_fs'] = fs2
            
            # Store muscle names in metadata
            if len(muscle_names) >= 2:
                metadata['muscle1_name'] = muscle_names[0].split('(')[0].strip()
                metadata['muscle2_name'] = muscle_names[1].split('(')[0].strip()
                metadata['muscle_count'] = len(muscle_names)
            
            logger.info(f"Data loaded: {metadata.get('muscle1_name', 'FCU')}: {len(emg1)} samples, {metadata.get('muscle2_name', 'FCR')}: {len(emg2)} samples")
            logger.info(f"Found {metadata.get('muscle_count', 2)} muscles in file")
            
            return emg1, emg2, time1, time2, metadata
    
    def preprocess_emg(self, emg_signal, fs):
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

    def detect_throws_multi_muscle(self, fcr_rms, fcu_rms, time, fs, 
                                    threshold_factor_fcr=2.75, threshold_factor_fcu=1.25,
                                    min_duration=0.2, min_separation=10, 
                                    coincidence_window=0.2):
        """
        Detect throwing motions by considering both FCR and FCU EMG data.
        
        Parameters match those in your pipeline processing method.
        """
        # Diagnostic logging
        logger.info(f"FCR RMS - Mean: {np.mean(fcr_rms)}, Std: {np.std(fcr_rms)}")
        logger.info(f"FCU RMS - Mean: {np.mean(fcu_rms)}, Std: {np.std(fcu_rms)}")
        
        # Calculate dynamic thresholds for both muscles
        fcr_threshold = np.mean(fcr_rms) + threshold_factor_fcr * np.std(fcr_rms)
        fcu_threshold = np.mean(fcu_rms) + threshold_factor_fcu * np.std(fcu_rms)
        
        logger.info(f"FCR Threshold: {fcr_threshold}")
        logger.info(f"FCU Threshold: {fcu_threshold}")
        
        logger.info(f"FCR samples above threshold: {np.sum(fcr_rms > fcr_threshold)} / {len(fcr_rms)}")
        logger.info(f"FCU samples above threshold: {np.sum(fcu_rms > fcu_threshold)} / {len(fcu_rms)}")

        # Find regions above threshold for each muscle
        fcr_above = fcr_rms > fcr_threshold
        fcu_above = fcu_rms > fcu_threshold
        
        # Calculate a combined activation signal (both muscles above threshold)
        both_active = np.logical_and(fcr_above, fcu_above)
        
        # Also track activation from either muscle (helpful for detecting start and end)
        either_active = np.logical_or(fcr_above, fcu_above)
        
        # Detect throw events with focus on coincident activation
        throws = []
        start = None
        coincidence_samples = int(coincidence_window * fs)
        
        # Enhanced detection algorithm
        i = 1  # Start at index 1 to avoid negative indexing
        while i < len(time) - 1:
            # If we haven't found a start yet
            if start is None:
                # Look for either muscle starting to activate
                if either_active[i] and not either_active[i-1]:
                    # Mark this as a potential start
                    potential_start = i
                    
                    # Look ahead to see if both muscles are active within our coincidence window
                    found_coincidence = False
                    for j in range(i, min(i + coincidence_samples, len(both_active))):
                        if both_active[j]:
                            found_coincidence = True
                            break
                    
                    if found_coincidence:
                        # Confirmed: this is a throw start
                        start = potential_start
            # If we have a start already, look for the end
            elif not either_active[i] and either_active[i-1]:
                # Potential end found
                duration = (i - start) / fs
                if duration >= min_duration:
                    throws.append((start, i))
                start = None
                
            i += 1
        
        # Handle any active regions that extend to the end of the data
        if start is not None:
            duration = (len(time) - 1 - start) / fs
            if duration >= min_duration:
                throws.append((start, len(time) - 1))
        
        # Filter throws by minimum separation
        if throws:
            filtered_throws = [throws[0]]
            for i in range(1, len(throws)):
                if (throws[i][0] - filtered_throws[-1][1]) / fs >= min_separation:
                    filtered_throws.append(throws[i])
            return filtered_throws
        
        logger.warning(f"No throws detected. Check threshold factors and signal characteristics.")
        return throws

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
    
    def calculate_comprehensive_metrics(self, emg_filtered, emg_rectified, emg_rms, emg_envelope, throws, time, fs):
        """
        Calculate comprehensive metrics for each throw.
        """
        # If no throws detected, return empty metrics
        if not throws:
            logger.warning("No throws detected in the signal")
            return {
                'throw_indices': np.array([]),
                'throw_timestamps': np.array([]),
                'throw_end_times': np.array([]),
                'throw_durations': np.array([]),
                
                # Frequency domain metrics
                'median_freqs': np.array([]),
                'mean_freqs': np.array([]),
                'bandwidth': np.array([]),
                
                # Amplitude domain metrics
                'peak_amplitudes': np.array([]),
                'rms_values': np.array([]),
                
                # Temporal metrics
                'rise_times': np.array([]),
                'contraction_times': np.array([]),
                'relaxation_times': np.array([]),
                
                # Workload metrics
                'throw_integrals': np.array([]),
                'work_rates': np.array([]),
                
                # NEW: Spectral entropy metrics
                'spectral_entropies': np.array([]),
                
                # NEW: Wavelet energy metrics
                'wavelet_energy_low': np.array([]),
                'wavelet_energy_mid': np.array([]),
                'wavelet_energy_high': np.array([])
            }

        # Initialize metrics arrays
        throw_count = len(throws)
        
        # Preallocate arrays with proper checks
        median_freqs = np.full(throw_count, np.nan)
        mean_freqs = np.full(throw_count, np.nan)
        bandwidth = np.full(throw_count, np.nan)
        
        peak_amplitudes = np.full(throw_count, np.nan)
        rms_values = np.full(throw_count, np.nan)
        
        rise_times = np.full(throw_count, np.nan)
        contraction_times = np.full(throw_count, np.nan)
        relaxation_times = np.full(throw_count, np.nan)
        
        throw_integrals = np.full(throw_count, np.nan)
        throw_durations = np.full(throw_count, np.nan)
        work_rates = np.full(throw_count, np.nan)
        
        spectral_entropies = np.full(throw_count, np.nan)
        
        wavelet_energy_low = np.full(throw_count, np.nan)
        wavelet_energy_mid = np.full(throw_count, np.nan)
        wavelet_energy_high = np.full(throw_count, np.nan)
        
        # Process each throw for comprehensive metrics
        for i, (start, end) in enumerate(throws):
            try:
                # Validate input segments
                if start >= end or end > len(emg_filtered):
                    logger.warning(f"Invalid throw segment for throw {i}: start={start}, end={end}")
                    continue
                
                # Extract throw segments
                segment_filtered = emg_filtered[start:end]
                segment_rectified = emg_rectified[start:end]
                segment_rms = emg_rms[start:end]
                segment_envelope = emg_envelope[start:end]
                segment_time = time[start:end] - time[start]  # Normalize time to start at 0
                
                # Validate segments
                if len(segment_filtered) == 0 or len(segment_time) == 0:
                    logger.warning(f"Empty segment for throw {i}")
                    continue
                
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
                    wavelet_energy_low[i] = np.nan
                    wavelet_energy_mid[i] = np.nan
                    wavelet_energy_high[i] = np.nan
            
            except Exception as e:
                logger.error(f"Error processing throw {i}: {e}")
                continue
        
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