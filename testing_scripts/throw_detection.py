import sys
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from scipy import signal
import pandas as pd
import re

def detect_throws_multi_muscle(fcr_rms, fcu_rms, time, fs, 
                            threshold_factor_fcr=4, threshold_factor_fcu=2.0,
                            min_duration=0.2, min_separation=5, 
                            coincidence_window=0.1):
    """Detect throws using both FCR and FCU signals."""
    # Calculate thresholds
    fcr_threshold = np.mean(fcr_rms) + threshold_factor_fcr * np.std(fcr_rms)
    fcu_threshold = np.mean(fcu_rms) + threshold_factor_fcu * np.std(fcu_rms)
    
    # Find regions above threshold
    fcr_above = fcr_rms > fcr_threshold
    fcu_above = fcu_rms > fcu_threshold
    
    # Calculate combined signals
    both_active = np.logical_and(fcr_above, fcu_above)
    either_active = np.logical_or(fcr_above, fcu_above)
    
    # Detect throws
    throws = []
    start = None
    coincidence_samples = int(coincidence_window * fs)
    
    i = 1  # Start at index 1 to avoid negative indexing
    while i < len(time) - 1:
        # Look for start
        if start is None:
            if either_active[i] and not either_active[i-1]:
                potential_start = i
                
                # Check for coincidence
                found_coincidence = False
                for j in range(i, min(i + coincidence_samples, len(both_active))):
                    if both_active[j]:
                        found_coincidence = True
                        break
                
                if found_coincidence:
                    start = potential_start
        # Look for end
        elif not either_active[i] and either_active[i-1]:
            duration = (i - start) / fs
            if duration >= min_duration:
                throws.append((start, i))
            start = None
        
        i += 1
    
    # Handle case where activation continues to the end
    if start is not None:
        duration = (len(time) - 1 - start) / fs
        if duration >= min_duration:
            throws.append((start, len(time) - 1))
    
    # Filter by minimum separation
    if throws:
        filtered_throws = [throws[0]]
        for i in range(1, len(throws)):
            if (throws[i][0] - filtered_throws[-1][1]) / fs >= min_separation:
                filtered_throws.append(throws[i])
        return filtered_throws
    
    return throws

def preprocess_emg(emg_signal, fs):
    """Process raw EMG signal to get RMS."""
    # High-pass filter
    sos_hp = signal.butter(4, 20, 'hp', fs=fs, output='sos')
    filtered = signal.sosfilt(sos_hp, emg_signal)
    
    # Rectification
    rectified = np.abs(filtered)
    
    # RMS smoothing
    window_size = int(0.05 * fs)  # 50ms window
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window size
        
    rms = np.sqrt(signal.convolve(rectified**2, np.ones(window_size)/window_size, mode='same'))
    
    return rms

def load_delsys_emg_data(file_path):
    """
    Load EMG data from Delsys Trigno format with header rows.
    Handles the specific format of Delsys files with metadata.
    """
    print("Attempting to load as Delsys format file...")
    
    # Extract time and EMG data
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
    time = np.array(muscle1_data['time'])
    fcu_data = np.array(muscle1_data['emg'])
    fcr_data = np.array(muscle2_data['emg'])
    
    # Calculate sampling frequency
    if len(time) > 1:
        fs = 1.0 / (time[1] - time[0])
    else:
        fs = 2000  # Default value
    
    print(f"Successfully loaded as Delsys format: {len(time)} samples, Fs = {fs:.1f} Hz")
    return time, fcu_data, fcr_data, fs

def main():
    # Hide the main tkinter window
    root = Tk()
    root.withdraw()
    
    # Prompt for file
    file_path = filedialog.askopenfilename(title="Select EMG data file", 
                                          filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    
    if not file_path:
        print("No file selected. Exiting.")
        sys.exit(0)
    
    print(f"Loading {file_path}")
    
    # Try different loading methods
    time = None
    fcu_raw = None
    fcr_raw = None
    fs = None
    
    # Method 1: Try Delsys format loader
    try:
        time, fcu_raw, fcr_raw, fs = load_delsys_emg_data(file_path)
    except Exception as e:
        print(f"Could not load as Delsys format: {e}")
    
    # Method 2: Try standard CSV with header
    if time is None:
        try:
            print("Attempting to load as standard CSV...")
            data = pd.read_csv(file_path)
            
            # Assume first column is time, second and third are EMG channels
            time = data.iloc[:, 0].values
            fcu_raw = data.iloc[:, 1].values
            fcr_raw = data.iloc[:, 2].values
            
            # Calculate sampling frequency
            if len(time) > 1:
                fs = 1.0 / (time[1] - time[0])
            else:
                fs = 2000  # Default value
                
            print(f"Successfully loaded as standard CSV: {len(time)} samples, Fs = {fs:.1f} Hz")
        except Exception as e:
            print(f"Could not load as standard CSV: {e}")
    
    # Method 3: Try with no header
    if time is None:
        try:
            print("Attempting to load as headerless CSV...")
            data = pd.read_csv(file_path, header=None)
            
            # Assume first column is time, second and third are EMG channels
            time = data.iloc[:, 0].values
            fcu_raw = data.iloc[:, 1].values
            fcr_raw = data.iloc[:, 2].values
            
            # Calculate sampling frequency
            if len(time) > 1:
                fs = 1.0 / (time[1] - time[0])
            else:
                fs = 2000  # Default value
                
            print(f"Successfully loaded as headerless CSV: {len(time)} samples, Fs = {fs:.1f} Hz")
        except Exception as e:
            print(f"Could not load as headerless CSV: {e}")
    
    # Method 4: Try with custom delimiters
    if time is None:
        for delimiter in [',', '\t', ';', ' ']:
            try:
                print(f"Attempting to load with delimiter '{delimiter}'...")
                data = pd.read_csv(file_path, delimiter=delimiter, header=None)
                
                # Make sure we have at least 3 columns
                if data.shape[1] < 3:
                    continue
                
                # Assume first column is time, second and third are EMG channels
                time = data.iloc[:, 0].values
                fcu_raw = data.iloc[:, 1].values
                fcr_raw = data.iloc[:, 2].values
                
                # Calculate sampling frequency
                if len(time) > 1:
                    fs = 1.0 / (time[1] - time[0])
                else:
                    fs = 2000  # Default value
                    
                print(f"Successfully loaded with delimiter '{delimiter}': {len(time)} samples, Fs = {fs:.1f} Hz")
                break
            except Exception as e:
                pass
    
    # Method 5: Last resort - try to parse line by line
    if time is None:
        try:
            print("Attempting to parse file line by line...")
            times = []
            fcu_values = []
            fcr_values = []
            
            with open(file_path, 'r') as f:
                # Skip potential header lines
                for _ in range(10):  # Skip up to 10 header lines
                    line = f.readline()
                    # Check if this line contains numeric data
                    if re.search(r'^\s*[\d.-]+\s*[,;\t\s]', line):
                        # This appears to be data, not a header
                        f.seek(0)  # Go back to beginning
                        break
                
                # Parse data lines
                for line in f:
                    # Try to extract numbers from the line
                    numbers = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', line)
                    if len(numbers) >= 3:
                        times.append(float(numbers[0]))
                        fcu_values.append(float(numbers[1]))
                        fcr_values.append(float(numbers[2]))
            
            if len(times) > 0:
                time = np.array(times)
                fcu_raw = np.array(fcu_values)
                fcr_raw = np.array(fcr_values)
                
                # Calculate sampling frequency
                if len(time) > 1:
                    fs = 1.0 / (time[1] - time[0])
                else:
                    fs = 2000  # Default value
                    
                print(f"Successfully parsed line by line: {len(time)} samples, Fs = {fs:.1f} Hz")
        except Exception as e:
            print(f"Could not parse line by line: {e}")
    
    # Check if we successfully loaded the data
    if time is None or fcu_raw is None or fcr_raw is None:
        print("Failed to load data in any supported format. Exiting.")
        sys.exit(1)
    
    # Preprocess signals
    fcu_rms = preprocess_emg(fcu_raw, fs)
    fcr_rms = preprocess_emg(fcr_raw, fs)
    
    # Set detection parameters
    threshold_fcr = 2.75
    threshold_fcu = 1.25
    min_duration = 0.2
    min_separation = 10
    
    # Detect throws
    throws = detect_throws_multi_muscle(
        fcr_rms, fcu_rms, time, fs,
        threshold_factor_fcr=threshold_fcr,
        threshold_factor_fcu=threshold_fcu,
        min_duration=min_duration,
        min_separation=min_separation
    )
    
    # Print results
    print(f"\nDETECTED {len(throws)} THROWS")
    
    for i, (start_idx, end_idx) in enumerate(throws):
        start_time = time[start_idx]
        end_time = time[end_idx]
        duration = end_time - start_time
        print(f"Throw #{i+1}: Start={start_time:.2f}s, End={end_time:.2f}s, Duration={duration:.2f}s")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot raw signals
    ax1.plot(time, fcu_raw, 'b', label='FCU Raw', linewidth=0.5)
    ax1.plot(time, fcr_raw, 'r', label='FCR Raw', linewidth=0.5)
    ax1.set_title('Raw EMG Signals')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot RMS with thresholds
    ax2.plot(time, fcu_rms, 'b', label='FCU RMS')
    ax2.plot(time, fcr_rms, 'r', label='FCR RMS')
    
    # Add thresholds
    fcu_threshold = np.mean(fcu_rms) + threshold_fcu * np.std(fcu_rms)
    fcr_threshold = np.mean(fcr_rms) + threshold_fcr * np.std(fcr_rms)
    ax2.axhline(fcu_threshold, color='b', linestyle='--', alpha=0.5, label='FCU Threshold')
    ax2.axhline(fcr_threshold, color='r', linestyle='--', alpha=0.5, label='FCR Threshold')
    
    # Highlight throws
    for i, (start_idx, end_idx) in enumerate(throws):
        start_time = time[start_idx]
        end_time = time[end_idx]
        
        # Highlight in both plots
        ax1.axvspan(start_time, end_time, color='g', alpha=0.2)
        ax2.axvspan(start_time, end_time, color='g', alpha=0.3)
        
        # Add throw number
        mid_time = (start_time + end_time) / 2
        ax2.text(mid_time, ax2.get_ylim()[1] * 0.9, f"#{i+1}", 
                ha='center', va='center', fontweight='bold', color='green')
    
    ax2.set_title(f'RMS Signals with {len(throws)} Detected Throws')
    ax2.legend()
    ax2.set_xlabel('Time (seconds)')
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f"DETECTED {len(throws)} THROWS", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()