import os
import pandas as pd
import numpy as np
import logging
import datetime
from pathlib import Path
import argparse
from tqdm import tqdm
import pymysql  # Add this import
from emg_processor import EMGProcessor
from db_connector import DBConnector
from datetime import datetime, timedelta
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('emg_pipeline.log')
    ]
)

logger = logging.getLogger(__name__)

class EMGPipeline:
    """
    Main pipeline for processing EMG data files and loading them into a database.
    """
    
    def __init__(self, data_dir=None, db_config=None, batch_size=1000):
        """
        Initialize the EMG pipeline.
        
        Parameters:
        -----------
        data_dir : str, optional
            Directory containing EMG data files
        db_config : dict, optional
            Database configuration
        batch_size : int, optional
            Batch size for database inserts
        """
        self.data_dir = data_dir or os.getcwd()
        self.processor = EMGProcessor()
        self.db = DBConnector(db_config)
        self.batch_size = batch_size
        
        logger.info(f"EMG Pipeline initialized with data directory: {self.data_dir}")
        logger.info(f"Using batch size of {self.batch_size} for DB operations")
    
    def setup_database(self):
        """Set up database tables."""
        # Simply use the db_connector's create_tables method
        return self.db.create_tables()
    
    def process_file(self, file_path):
        """
        Process a single EMG data file and prepare data for database.
        
        Parameters:
        -----------
        file_path : str
            Path to the EMG data file
            
        Returns:
        --------
        dict
            Processed data ready for database insertion
        """
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Parse filename
            file_metadata = self.processor.parse_filename(file_path)
            session_id = os.path.splitext(os.path.basename(file_path))[0]
            
            # Check if this is an MVIC session
            is_mvic = 'mvic' in session_id.lower()
            
            # Load EMG data with dynamic muscle names
            muscle1_emg, muscle2_emg, muscle1_time, muscle2_time, metadata = self.processor.load_delsys_emg_data(file_path)
            
            # Extract muscle names for easier reference
            muscle1_name = metadata.get('muscle1_name', 'muscle1')
            muscle2_name = metadata.get('muscle2_name', 'muscle2')
            
            # Extract collection date and start time
            collection_date = metadata.get('Collection_Date')
            start_time = metadata.get('Start_Time')
            
            # Extract sampling rates
            muscle1_fs = metadata.get('muscle1_fs', 2000)
            muscle2_fs = metadata.get('muscle2_fs', 2000)
            
            logger.info(f"Loaded muscles: {muscle1_name} and {muscle2_name}")
            logger.info(f"Signal lengths: {len(muscle1_emg)} and {len(muscle2_emg)} samples")
            logger.info(f"Sampling rates: {muscle1_fs} and {muscle2_fs} Hz")
            
            # Recalculate proper time arrays for both muscles regardless of what's in the file
            # Since we know both recordings started and stopped simultaneously
            muscle1_samples = len(muscle1_emg)
            muscle2_samples = len(muscle2_emg)

            # Calculate actual durations based on samples and rates
            actual_duration_1 = muscle1_samples / muscle1_fs
            actual_duration_2 = muscle2_samples / muscle2_fs

            # Use the longer duration for both signals
            common_duration = max(actual_duration_1, actual_duration_2)
            logger.info(f"Common duration: {common_duration:.2f} seconds")

            # Generate consistent time arrays based on common duration
            muscle1_time_fixed = np.linspace(0, common_duration, muscle1_samples)
            muscle2_time_fixed = np.linspace(0, common_duration, muscle2_samples)

            # Create a common time base with sufficient resolution
            common_fs = max(muscle1_fs, muscle2_fs) 
            common_time = np.linspace(0, common_duration, int(common_duration * common_fs))
            logger.info(f"Common time base: {len(common_time)} samples at {common_fs} Hz")

            # Interpolate raw EMG signals to the common timeline
            muscle1_emg_aligned = np.interp(common_time, muscle1_time_fixed, muscle1_emg)
            muscle2_emg_aligned = np.interp(common_time, muscle2_time_fixed, muscle2_emg)
            
            # Preprocess the interpolated EMG data
            muscle1_filtered_aligned, muscle1_rectified_aligned, muscle1_rms_aligned, muscle1_envelope_aligned = self.processor.preprocess_emg(muscle1_emg_aligned, common_fs)
            muscle2_filtered_aligned, muscle2_rectified_aligned, muscle2_rms_aligned, muscle2_envelope_aligned = self.processor.preprocess_emg(muscle2_emg_aligned, common_fs)
            
            logger.info(f"Preprocessed aligned signals: {len(muscle1_rms_aligned)} and {len(muscle2_rms_aligned)} samples")
            logger.info(f"Signal maximums - Muscle1: {np.max(muscle1_rms_aligned):.2f}, Muscle2: {np.max(muscle2_rms_aligned):.2f}")

            # Prepare session data with additional metadata
            session_data = {
                'session_id': session_id,
                'date_recorded': file_metadata['date'],
                'collection_date': collection_date,
                'start_time': start_time,
                'traq_id': file_metadata['traq_id'],
                'athlete_name': file_metadata['athlete_name'],
                'session_type': file_metadata['session_type'],
                'muscle_count': metadata.get('muscle_count', 2),
                'muscle1_name': metadata.get('muscle1_name'),
                'muscle2_name': metadata.get('muscle2_name'),
                'muscle1_fs': muscle1_fs,
                'muscle2_fs': muscle2_fs,
                'file_path': file_path,
                'processed_date': datetime.now(),
                'is_mvic': is_mvic
            }
            
            # Format collection_date and start_time properly
            collection_date = None
            start_time = None
            if session_data.get('collection_date'):
                try:
                    # Parse the collection date string into a datetime object
                    collection_datetime = datetime.strptime(session_data['collection_date'], '%m/%d/%Y %I:%M:%S %p')
                    collection_date = collection_datetime.date()
                    start_time = collection_datetime.time()
                except Exception as e:
                    logger.warning(f"Could not parse collection date {session_data.get('collection_date')}: {e}")
                    collection_date = None
                    start_time = None
            
            # Check if this is an MVIC session
            is_mvic = session_data.get('is_mvic', False)
            
            if is_mvic:
                # For MVIC sessions, calculate reference values
                logger.info("Processing as MVIC session")
                
                # Calculate MVIC reference values
                muscle1_mvic_peak = np.max(muscle1_rectified_aligned)
                muscle1_mvic_rms = np.sqrt(np.mean(muscle1_filtered_aligned**2))
                muscle2_mvic_peak = np.max(muscle2_rectified_aligned)
                muscle2_mvic_rms = np.sqrt(np.mean(muscle2_filtered_aligned**2))
                
                # Add MVIC values to session data
                session_data['muscle1_mvic_peak'] = muscle1_mvic_peak
                session_data['muscle1_mvic_rms'] = muscle1_mvic_rms
                session_data['muscle2_mvic_peak'] = muscle2_mvic_peak
                session_data['muscle2_mvic_rms'] = muscle2_mvic_rms
                
                logger.info(f"MVIC Reference Values - Muscle1 Peak: {muscle1_mvic_peak:.2f}, RMS: {muscle1_mvic_rms:.2f}")
                logger.info(f"MVIC Reference Values - Muscle2 Peak: {muscle2_mvic_peak:.2f}, RMS: {muscle2_mvic_rms:.2f}")
                
                # For MVIC sessions, we don't need time series or throw data
                return {
                    'session_data': session_data,
                    'timeseries_data': None,
                    'throw_data': None
                }
            else:
                # For pitching sessions, continue with normal processing
                logger.info("Processing as pitching session")
                
                # Use the aligned signals for throw detection
                throws = self.processor.detect_throws_multi_muscle(
                    muscle2_rms_aligned, muscle1_rms_aligned, common_time, common_fs, 
                    threshold_factor_fcr=2.5,    # Reduced from 4.0 to be less strict
                    threshold_factor_fcu=1.2,    # Reduced from 1.6 to be less strict
                    min_duration=0.15,           # Reduced from 0.2 to catch shorter throws
                    min_separation=8,            # Reduced from 10 to allow closer throws
                    coincidence_window=0.15      # Increased from 0.1 to allow more timing variation
                )
                
                logger.info(f"Detected {len(throws)} throws from aligned signals")
                
                # Log threshold values used for debugging
                m1_mean = np.mean(muscle1_rms_aligned)
                m2_mean = np.mean(muscle2_rms_aligned)
                logger.info(f"Muscle1 (FCR) - Mean: {m1_mean:.4f}, Threshold: {m1_mean * 2.5:.4f}")
                logger.info(f"Muscle2 (FCU) - Mean: {m2_mean:.4f}, Threshold: {m2_mean * 1.2:.4f}")
                
                if len(throws) == 0:
                    logger.warning("No throws detected! This might indicate threshold values are still too strict")
                elif len(throws) < 10:
                    logger.warning(f"Only {len(throws)} throws detected. This seems low for a typical session.")
                
                # Convert throw indices to actual timestamps using common_time
                throw_timestamps = []
                for start_idx, end_idx in throws:
                    # Make sure indices are within bounds
                    if start_idx < len(common_time) and end_idx < len(common_time):
                        throw_timestamps.append((common_time[start_idx], common_time[end_idx]))
                    else:
                        logger.warning(f"Throw indices out of bounds: {start_idx}, {end_idx}, max: {len(common_time)-1}")
                        # Use valid indices
                        valid_start = min(start_idx, len(common_time)-1)
                        valid_end = min(end_idx, len(common_time)-1)
                        throw_timestamps.append((common_time[valid_start], common_time[valid_end]))

                # Calculate metrics using aligned signals
                muscle1_metrics = self.processor.calculate_comprehensive_metrics(
                    muscle1_filtered_aligned, muscle1_rectified_aligned, muscle1_rms_aligned, muscle1_envelope_aligned, 
                    throws, common_time, common_fs
                )

                muscle2_metrics = self.processor.calculate_comprehensive_metrics(
                    muscle2_filtered_aligned, muscle2_rectified_aligned, muscle2_rms_aligned, muscle2_envelope_aligned, 
                    throws, common_time, common_fs
                )

                coactivation_metrics = self.processor.calculate_muscle_coactivation(
                    muscle1_metrics, muscle2_metrics,
                    throws, throws,  # Same throws for both muscles
                    muscle1_filtered_aligned, muscle2_filtered_aligned,
                    common_time, common_time,  # Using the same timeline
                    common_fs
                )
            
                # Prepare time series data for pitching session
                timeseries_data = pd.DataFrame({
                    'emg_session_id': [session_id] * len(common_time),
                    'time_point': common_time,
                    'muscle1_emg': muscle1_emg_aligned,
                    'muscle2_emg': muscle2_emg_aligned
                })

                # Prepare throw data
                throw_data = []
                for i in range(len(throws)):
                    # Get throw start and end indices
                    start_idx, end_idx = throws[i]
                    
                    # Ensure indices are within bounds
                    start_idx = min(start_idx, len(common_time)-1)
                    end_idx = min(end_idx, len(common_time)-1)
                    
                    # Calculate relative start time
                    relative_start = common_time[start_idx]
                    
                    # Calculate absolute timestamp
                    absolute_timestamp = None
                    if collection_date and start_time:
                        try:
                            # start_time is now the finishing time
                            recording_end_time = datetime.combine(collection_date, start_time)
                            
                            # Extract total collection duration from metadata
                            total_duration = metadata.get('Collection_Length')
                            
                            # Convert to float if it's a string
                            if isinstance(total_duration, str):
                                # Remove any non-numeric characters (like "s" for seconds)
                                total_duration = float(''.join(c for c in total_duration if c.isdigit() or c == '.'))
                            
                            # Calculate recording start time by subtracting total duration
                            recording_start_time = recording_end_time - timedelta(seconds=total_duration)
                            
                            # Calculate absolute timestamp for this throw by adding relative start time to recording start
                            absolute_timestamp = recording_start_time + timedelta(seconds=relative_start)
                        except Exception as e:
                            logger.warning(f"Error calculating absolute timestamp: {e}")
                    
                    throw_row = {
                        'emg_session_id': session_id,
                        'trial_number': i + 1,
                        'start_time': throw_timestamps[i][0],  # Use the timestamp from common_time
                        'end_time': throw_timestamps[i][1],    # Use the timestamp from common_time
                        'duration': throw_timestamps[i][1] - throw_timestamps[i][0],
                        
                        # Timestamp and velocity matching columns
                        'relative_start_time': relative_start,
                        'absolute_timestamp': absolute_timestamp,
                        'session_trial': None,  # Will be populated during velocity matching
                        'pitch_speed_mph': None,  # Will be populated during velocity matching
                        'velocity_match_quality': 'no_match',  # Default match quality
                        
                        # Muscle1 metrics - original metrics
                        'muscle1_median_freq': muscle1_metrics['median_freqs'][i],
                        'muscle1_mean_freq': muscle1_metrics['mean_freqs'][i],
                        'muscle1_bandwidth': muscle1_metrics['bandwidth'][i],
                        'muscle1_peak_amplitude': muscle1_metrics['peak_amplitudes'][i],
                        'muscle1_rms_value': muscle1_metrics['rms_values'][i],
                        'muscle1_rise_time': muscle1_metrics['rise_times'][i],
                        'muscle1_throw_integral': muscle1_metrics['throw_integrals'][i],
                        'muscle1_work_rate': muscle1_metrics['work_rates'][i],
                        
                        # Muscle2 metrics - original metrics
                        'muscle2_median_freq': muscle2_metrics['median_freqs'][i],
                        'muscle2_mean_freq': muscle2_metrics['mean_freqs'][i],
                        'muscle2_bandwidth': muscle2_metrics['bandwidth'][i],
                        'muscle2_peak_amplitude': muscle2_metrics['peak_amplitudes'][i],
                        'muscle2_rms_value': muscle2_metrics['rms_values'][i],
                        'muscle2_rise_time': muscle2_metrics['rise_times'][i],
                        'muscle2_throw_integral': muscle2_metrics['throw_integrals'][i],
                        'muscle2_work_rate': muscle2_metrics['work_rates'][i],
                        
                        # Spectral entropy metrics
                        'muscle1_spectral_entropy': muscle1_metrics['spectral_entropies'][i],
                        'muscle2_spectral_entropy': muscle2_metrics['spectral_entropies'][i],
                        
                        # Wavelet energy metrics
                        'muscle1_wavelet_energy_low': muscle1_metrics['wavelet_energy_low'][i],
                        'muscle1_wavelet_energy_mid': muscle1_metrics['wavelet_energy_mid'][i],
                        'muscle1_wavelet_energy_high': muscle1_metrics['wavelet_energy_high'][i],
                        'muscle2_wavelet_energy_low': muscle2_metrics['wavelet_energy_low'][i],
                        'muscle2_wavelet_energy_mid': muscle2_metrics['wavelet_energy_mid'][i],
                        'muscle2_wavelet_energy_high': muscle2_metrics['wavelet_energy_high'][i],
                    }
                    
                    # Add coactivation metrics if available
                    if coactivation_metrics:
                        throw_row.update({
                            'coactivation_index': coactivation_metrics['ci_falconer'][i],
                            'coactivation_correlation': coactivation_metrics['correlation'][i],
                            'coactivation_temporal_overlap': coactivation_metrics['temporal_overlap'][i],
                            'coactivation_waveform_similarity': coactivation_metrics['waveform_similarity'][i],
                        })
                    
                    # Replace NaN values with None for database insertion
                    for key, value in throw_row.items():
                        if isinstance(value, float) and np.isnan(value):
                            throw_row[key] = None
                    
                    throw_data.append(throw_row)
                
                return {
                    'session_data': session_data,
                    'timeseries_data': timeseries_data,
                    'throw_data': throw_data
                }
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def match_throws_to_velocity(self, reprocess_all=False, excluded_sessions=None):
        """
        Match EMG throws to velocity data from trials and poi tables.
        By default, only processes sessions that haven't been matched yet.
        
        Parameters:
        -----------
        reprocess_all : bool, optional
            If True, reprocess all sessions even if they have matches.
            Default is False (only process unmatched sessions).
        excluded_sessions : list, optional
            List of session IDs to exclude from velocity matching
            
        Returns:
        --------
        bool
            True if matching was successful, False otherwise
        """
        conn = None
        
        # Default excluded sessions list
        if excluded_sessions is None:
            # Maintain a list of permanently excluded sessions here
            excluded_sessions = [55]  # Session 55 is excluded
        
        try:
            # Connect to the database
            conn = self.db.connect()
            if not conn:
                logger.error("Failed to connect to database")
                return False
            
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            
            # Clear velocity matching for excluded sessions if requested
            if excluded_sessions:
                excluded_ids_str = ','.join(map(str, excluded_sessions))
                logger.info(f"Clearing velocity matching for excluded sessions: {excluded_ids_str}")
                
                cursor.execute(f"""
                    UPDATE emg_throws 
                    SET session_trial = NULL,
                        pitch_speed_mph = NULL,
                        velocity_match_quality = 'excluded_manual'
                    WHERE emg_session_id IN ({excluded_ids_str})
                """)
                conn.commit()
            
            # Build the exclusion part of the query
            exclude_clause = ""
            if excluded_sessions:
                exclude_clause = f"AND es.emg_session_id NOT IN ({','.join(map(str, excluded_sessions))})"
            
            # Identify sessions that need processing
            if reprocess_all:
                # Process all sessions except excluded ones
                query = f"""
                    SELECT emg_session_id, athlete_name, date_recorded, filename 
                    FROM emg_sessions es
                    WHERE date_recorded IS NOT NULL
                    {exclude_clause}
                    ORDER BY date_recorded DESC
                """
                logger.info("Processing ALL sessions (including previously matched ones)")
            else:
                # Only process sessions with unmatched throws
                query = f"""
                    SELECT DISTINCT es.emg_session_id, es.athlete_name, es.date_recorded, es.filename 
                    FROM emg_sessions es
                    JOIN emg_throws et ON es.emg_session_id = et.emg_session_id
                    WHERE es.date_recorded IS NOT NULL
                    AND (
                        et.velocity_match_quality IS NULL 
                        OR et.velocity_match_quality = 'no_match'
                        OR et.pitch_speed_mph IS NULL
                    )
                    {exclude_clause}
                    ORDER BY es.date_recorded DESC
                """
                logger.info("Processing only sessions with unmatched throws")
            
            cursor.execute(query)
            emg_sessions = cursor.fetchall()
            logger.info(f"Found {len(emg_sessions)} EMG sessions to process")
            
            if len(emg_sessions) == 0:
                logger.info("No sessions need processing. All throws are already matched.")
                return True
            
            # Process each EMG session
            for emg_session in emg_sessions:
                session_id = emg_session['emg_session_id']
                athlete_name = emg_session['athlete_name']
                session_date = emg_session['date_recorded']
                
                logger.info(f"Processing session for {athlete_name} on {session_date}")
                
                # Format the athlete name - convert from "AlexBritton" to "Alex Britton"
                # Add space before capital letters except the first one
                formatted_name = ""
                for i, char in enumerate(athlete_name):
                    if i > 0 and char.isupper():
                        formatted_name += " " + char
                    else:
                        formatted_name += char
                
                logger.info(f"Searching for user with name: {formatted_name} on date: {session_date}")
                
                # Try matching on both name and date
                cursor.execute(f"""
                    SELECT sessions.session, sessions.date, users.traq, users.name
                    FROM sessions
                    INNER JOIN users ON users.user = sessions.user
                    WHERE users.name = '{formatted_name}'
                    AND DATE(sessions.date) = DATE('{session_date}')
                    ORDER BY sessions.date DESC
                    LIMIT 1
                """)
                
                pitching_session = cursor.fetchone()
                
                if not pitching_session:
                    logger.warning(f"No matching pitching session found for {athlete_name} on {session_date}")
                    continue
                    
                pitching_session_id = pitching_session['session']
                athlete_name_in_db = pitching_session['name']
                session_date_in_db = pitching_session['date']
                
                logger.info(f"Found matching session {pitching_session_id} for {athlete_name_in_db} on {session_date_in_db}")
                
                # Check if session already has matched throws (optimization)
                if not reprocess_all:
                    cursor.execute(f"""
                        SELECT COUNT(*) as matched_count
                        FROM emg_throws
                        WHERE emg_session_id = {session_id}
                        AND velocity_match_quality IS NOT NULL
                        AND velocity_match_quality != 'no_match'
                    """)
                    
                    matched_count = cursor.fetchone()['matched_count']
                    if matched_count > 0:
                        logger.info(f"Session {session_id} already has {matched_count} matched throws. Skipping.")
                        continue
                
                # Get EMG throws for this session
                cursor.execute(f"""
                    SELECT throw_id, trial_number, start_time, end_time, 
                        (end_time - start_time) as duration,
                        muscle1_peak_amplitude, muscle2_peak_amplitude
                    FROM emg_throws
                    WHERE emg_session_id = {session_id}
                    ORDER BY trial_number
                """)
                
                emg_throws = cursor.fetchall()
                logger.info(f"Found {len(emg_throws)} EMG throws to match")
                
                if not emg_throws:
                    continue
                
                # Get velocity data and extract trial numbers
                cursor.execute(f"""
                    SELECT session_trial, pitch_speed_mph
                    FROM poi
                    WHERE session_trial LIKE '{pitching_session_id}_%'
                    ORDER BY session_trial
                """)
                
                velocity_data = cursor.fetchall()
                logger.info(f"Found {len(velocity_data)} velocity records")
                
                if not velocity_data:
                    logger.warning(f"No velocity data found for session {pitching_session_id}")
                    continue
                
                # Extract trial numbers from session_trial strings
                # Format is typically: "pitching_session_id_trialNumber"
                velocity_trial_numbers = []
                for record in velocity_data:
                    session_trial = record['session_trial']
                    try:
                        # Extract the trial number after the last underscore
                        trial_number = int(session_trial.split('_')[-1])
                        velocity_trial_numbers.append(trial_number)
                    except (ValueError, IndexError):
                        logger.warning(f"Couldn't extract trial number from {session_trial}")
                
                logger.info(f"Extracted velocity trial numbers: {velocity_trial_numbers}")
                
                # Get timing information
                cursor.execute(f"""
                    SELECT t.session_trial, t.time, p.pitch_speed_mph
                    FROM trials t
                    JOIN poi p ON t.session_trial = p.session_trial
                    WHERE t.session_trial LIKE '{pitching_session_id}_%'
                    ORDER BY t.time
                """)
                
                trials = cursor.fetchall()
                logger.info(f"Found {len(trials)} trials with timing information")
                
                if not trials:
                    logger.warning(f"No trial timing data found for session {pitching_session_id}")
                    continue
                
                # Extract trial numbers and create a mapping
                trial_mapping = {}
                for trial in trials:
                    session_trial = trial['session_trial']
                    try:
                        # Extract the trial number after the last underscore
                        trial_number = int(session_trial.split('_')[-1])
                        trial_mapping[trial_number] = trial
                    except (ValueError, IndexError):
                        logger.warning(f"Couldn't extract trial number from {session_trial}")
                
                logger.info(f"Created mapping for {len(trial_mapping)} trials")
                
                # Clear any previous velocity mappings for this session
                cursor.execute(f"""
                    UPDATE emg_throws 
                    SET session_trial = NULL, pitch_speed_mph = NULL, velocity_match_quality = 'no_match'
                    WHERE emg_session_id = {session_id}
                """)
                conn.commit()
                
                # MATCHING STRATEGY: TRIAL NUMBER BASED
                # First try to match EMG throw trial numbers directly to velocity trial numbers
                match_count = 0
                matched_throw_ids = set()
                
                for throw in emg_throws:
                    emg_trial_number = throw['trial_number']
                    
                    # Check if this trial number exists in the trial_mapping
                    if emg_trial_number in trial_mapping:
                        trial = trial_mapping[emg_trial_number]
                        
                        cursor.execute(f"""
                            UPDATE emg_throws
                            SET 
                                session_trial = '{trial['session_trial']}',
                                pitch_speed_mph = {trial['pitch_speed_mph']},
                                velocity_match_quality = 'direct_trial_match'
                            WHERE throw_id = {throw['throw_id']}
                        """)
                        
                        match_count += 1
                        matched_throw_ids.add(throw['throw_id'])
                        
                if match_count > 0:
                    logger.info(f"Matched {match_count} throws using direct trial number mapping")
                    conn.commit()
                
                # FALLBACK STRATEGIES for throws that weren't directly matched
                
                # Get remaining unmatched throws
                remaining_throws = [throw for throw in emg_throws if throw['throw_id'] not in matched_throw_ids]
                remaining_trial_numbers = [trial_num for trial_num in trial_mapping.keys() 
                                        if trial_num not in [throw['trial_number'] for throw in emg_throws 
                                                            if throw['throw_id'] in matched_throw_ids]]
                
                if remaining_throws and remaining_trial_numbers:
                    logger.info(f"Attempting to match {len(remaining_throws)} remaining throws to {len(remaining_trial_numbers)} remaining trials")
                    
                    # STRATEGY 1: If remaining counts match, use sequence matching
                    if len(remaining_throws) == len(remaining_trial_numbers):
                        logger.info(f"Using sequence matching for remaining throws (counts match)")
                        
                        # Sort both lists by their respective trial numbers
                        remaining_throws.sort(key=lambda x: x['trial_number'])
                        remaining_trial_numbers.sort()
                        
                        fallback_match_count = 0
                        for i, throw in enumerate(remaining_throws):
                            if i < len(remaining_trial_numbers):
                                trial_num = remaining_trial_numbers[i]
                                trial = trial_mapping[trial_num]
                                
                                cursor.execute(f"""
                                    UPDATE emg_throws
                                    SET 
                                        session_trial = '{trial['session_trial']}',
                                        pitch_speed_mph = {trial['pitch_speed_mph']},
                                        velocity_match_quality = 'sequence_matched'
                                    WHERE throw_id = {throw['throw_id']}
                                """)
                                
                                fallback_match_count += 1
                        
                        logger.info(f"Matched {fallback_match_count} additional throws by sequence")
                        conn.commit()
                        match_count += fallback_match_count
                    
                    # STRATEGY 2: If counts differ, use time-ordered matching
                    else:
                        logger.info(f"Using time-order matching for remaining throws (counts differ)")
                        
                        # Sort both by trial numbers
                        remaining_throws.sort(key=lambda x: x['trial_number'])
                        remaining_trial_numbers.sort()
                        
                        # Match as many as possible in order
                        fallback_match_count = 0
                        match_limit = min(len(remaining_throws), len(remaining_trial_numbers))
                        
                        for i in range(match_limit):
                            throw = remaining_throws[i]
                            trial_num = remaining_trial_numbers[i]
                            trial = trial_mapping[trial_num]
                            
                            cursor.execute(f"""
                                UPDATE emg_throws
                                SET 
                                    session_trial = '{trial['session_trial']}',
                                    pitch_speed_mph = {trial['pitch_speed_mph']},
                                    velocity_match_quality = 'ordered_time_matched'
                                WHERE throw_id = {throw['throw_id']}
                            """)
                            
                            fallback_match_count += 1
                        
                        logger.info(f"Matched {fallback_match_count} additional throws by time order")
                        conn.commit()
                        match_count += fallback_match_count
                
                logger.info(f"Total of {match_count} throws matched for session {pitching_session_id}")
            
            # Check the results
            cursor.execute("""
                SELECT velocity_match_quality, COUNT(*) as count 
                FROM emg_throws 
                GROUP BY velocity_match_quality
            """)
            
            match_results = cursor.fetchall()
            logger.info("Velocity Matching Results:")
            for result in match_results:
                quality = result['velocity_match_quality'] if result['velocity_match_quality'] else 'None'
                logger.info(f"{quality}: {result['count']} throws")
            
            # Show some examples of matched throws
            cursor.execute("""
                SELECT et.throw_id, es.athlete_name, et.trial_number, 
                    et.session_trial, et.pitch_speed_mph, et.velocity_match_quality
                FROM emg_throws et
                JOIN emg_sessions es ON et.emg_session_id = es.emg_session_id
                WHERE et.velocity_match_quality IS NOT NULL AND et.velocity_match_quality != 'no_match'
                ORDER BY et.throw_id
                LIMIT 10
            """)
            matched_examples = cursor.fetchall()
            
            if matched_examples:
                logger.info("Examples of matched throws:")
                for example in matched_examples:
                    logger.info(f"Throw ID: {example['throw_id']}, Athlete: {example['athlete_name']}, " +
                            f"Trial: {example['trial_number']}, Session Trial: {example['session_trial']}, " +
                            f"Velocity: {example['pitch_speed_mph']} mph")
            
            # Report on excluded sessions
            if excluded_sessions:
                cursor.execute(f"""
                    SELECT COUNT(*) as count
                    FROM emg_throws
                    WHERE emg_session_id IN ({','.join(map(str, excluded_sessions))})
                    AND velocity_match_quality = 'excluded_manual'
                """)
                excluded_count = cursor.fetchone()['count']
                logger.info(f"{excluded_count} throws are marked as manually excluded from velocity matching")
            
            return True
        
        except Exception as e:
            logger.error(f"Error matching throws to velocity: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            if conn:
                conn.rollback()
            
            return False
        
        finally:
            # Ensure connection is closed
            if conn:
                conn.close()

    def validate_velocity_matching(self):
        """
        Analyze and report on the velocity matching results
        
        Returns:
        --------
        dict
            Detailed matching statistics
        """
        conn = None
        try:
            conn = self.db.connect()
            if not conn:
                logger.error("Failed to connect to database")
                return None
            
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            
            # Overall matching statistics
            cursor.execute("""
            SELECT 
                velocity_match_quality, 
                COUNT(*) as total_throws,
                ROUND(COUNT(*) / (SELECT COUNT(*) FROM emg_throws) * 100, 2) as percentage
            FROM emg_throws 
            GROUP BY velocity_match_quality
            """)
            match_quality_stats = cursor.fetchall()
            
            # Velocity distribution for matched throws
            cursor.execute("""
            SELECT 
                ROUND(AVG(pitch_speed_mph), 2) as avg_velocity,
                ROUND(MIN(pitch_speed_mph), 2) as min_velocity,
                ROUND(MAX(pitch_speed_mph), 2) as max_velocity,
                COUNT(*) as matched_throws
            FROM emg_throws
            WHERE pitch_speed_mph IS NOT NULL
            """)
            velocity_stats = cursor.fetchone()
            
            # Athlete-wise matching
            cursor.execute("""
            SELECT 
                es.athlete_name,
                COUNT(*) as total_throws,
                SUM(CASE WHEN et.velocity_match_quality = 'exact' THEN 1 ELSE 0 END) as exact_matches,
                ROUND(SUM(CASE WHEN et.velocity_match_quality = 'exact' THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) as match_percentage
            FROM emg_throws et
            JOIN emg_sessions es ON et.emg_session_id = es.emg_session_id
            GROUP BY es.athlete_name
            ORDER BY match_percentage DESC
            """)
            athlete_stats = cursor.fetchall()
            
            return {
                'match_quality': match_quality_stats,
                'velocity_stats': velocity_stats,
                'athlete_matching': athlete_stats
            }
        
        except Exception as e:
            logger.error(f"Error validating velocity matching: {e}")
            return None
        finally:
            if conn:
                conn.close()

    def save_to_database(self, processed_data):
        """Save processed data to the database with metadata."""
        if not processed_data:
            logger.error("No data to save to database")
            return False
        
        conn = self.db.connect()
        if not conn:
            logger.error("Failed to connect to database")
            return False
        
        try:
            cursor = conn.cursor()
            
            # Extract data components
            session_data = processed_data['session_data']
            timeseries_data = processed_data.get('timeseries_data')  # May be None for MVIC
            throw_data = processed_data.get('throw_data')  # May be None for MVIC
            
            # Start transaction
            conn.begin()
            
            # Get filename from session_id
            filename = session_data['session_id']
            
            # Check if session already exists
            cursor.execute(
                "SELECT numeric_id FROM emg_sessions WHERE filename = %s",
                (filename,)
            )
            
            session_result = cursor.fetchone()
            if session_result:
                emg_session_id = session_result[0]
                logger.info(f"Session {filename} already exists. Removing old data...")
                
                # Delete existing data
                cursor.execute("DELETE FROM emg_throws WHERE session_numeric_id = %s", (emg_session_id,))
                cursor.execute("DELETE FROM emg_timeseries WHERE session_numeric_id = %s", (emg_session_id,))
                cursor.execute("DELETE FROM emg_sessions WHERE numeric_id = %s", (emg_session_id,))
                cursor.execute("DELETE FROM emg_metadata WHERE filename = %s", (filename,))
            
            # Parse date fields
            date_recorded = session_data.get('date_recorded')
            collection_date = None
            start_time = None
            if session_data.get('collection_date'):
                try:
                    collection_datetime = datetime.strptime(session_data['collection_date'], '%m/%d/%Y %I:%M:%S %p')
                    collection_date = collection_datetime.date()
                    start_time = collection_datetime.time()
                except Exception as e:
                    logger.warning(f"Could not parse collection date {session_data.get('collection_date')}: {e}")
                    
            # First, insert into metadata table
            cursor.execute("""
            INSERT INTO emg_metadata (
                filename, date_recorded, collection_date, start_time,
                traq_id, session_type, muscle_count, muscle1_name, muscle2_name,
                muscle1_fs, muscle2_fs, file_path, processed_date, is_mvic
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                filename,
                date_recorded,
                collection_date,
                start_time,
                session_data['traq_id'],
                session_data['session_type'],
                session_data.get('muscle_count', 2),
                session_data.get('muscle1_name'),
                session_data.get('muscle2_name'),
                session_data.get('muscle1_fs'),
                session_data.get('muscle2_fs'),
                session_data['file_path'],
                session_data['processed_date'],
                session_data.get('is_mvic', False)
            ))
            
            # Get the auto-generated metadata ID
            cursor.execute("SELECT LAST_INSERT_ID()")
            metadata_id = cursor.fetchone()[0]
            
            # Check if this is an MVIC session
            is_mvic = session_data.get('is_mvic', False)
            
            # Insert into sessions table with appropriate MVIC values
            if is_mvic:
                cursor.execute("""
                INSERT INTO emg_sessions (
                    emg_session_id, filename, date_recorded, collection_date, start_time,
                    traq_id, athlete_name, session_type, muscle_count, muscle1_name, muscle2_name,
                    muscle1_fs, muscle2_fs, file_path, processed_date, is_mvic,
                    muscle1_mvic_peak, muscle1_mvic_rms, muscle2_mvic_peak, muscle2_mvic_rms
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    str(metadata_id),  # Use metadata ID as the emg_session_id
                    filename,
                    date_recorded,
                    collection_date,
                    start_time,
                    session_data['traq_id'],
                    session_data['athlete_name'],
                    session_data['session_type'],
                    session_data.get('muscle_count', 2),
                    session_data.get('muscle1_name'),
                    session_data.get('muscle2_name'),
                    session_data.get('muscle1_fs'),
                    session_data.get('muscle2_fs'),
                    session_data['file_path'],
                    session_data['processed_date'],
                    True,  # is_mvic
                    session_data.get('muscle1_mvic_peak'),
                    session_data.get('muscle1_mvic_rms'),
                    session_data.get('muscle2_mvic_peak'),
                    session_data.get('muscle2_mvic_rms')
                ))
            else:
                cursor.execute("""
                INSERT INTO emg_sessions (
                    emg_session_id, filename, date_recorded, collection_date, start_time,
                    traq_id, athlete_name, session_type, muscle_count, muscle1_name, muscle2_name,
                    muscle1_fs, muscle2_fs, file_path, processed_date, is_mvic
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    str(metadata_id),  # Use metadata ID as the emg_session_id
                    filename,
                    date_recorded,
                    collection_date,
                    start_time,
                    session_data['traq_id'],
                    session_data['athlete_name'],
                    session_data['session_type'],
                    session_data.get('muscle_count', 2),
                    session_data.get('muscle1_name'),
                    session_data.get('muscle2_name'),
                    session_data.get('muscle1_fs'),
                    session_data.get('muscle2_fs'),
                    session_data['file_path'],
                    session_data['processed_date'],
                    False  # is_mvic
                ))
            
            # Get the auto-generated session numeric ID
            cursor.execute("SELECT LAST_INSERT_ID()")
            session_numeric_id = cursor.fetchone()[0]
            
            # For pitching sessions, look for a matching MVIC session
            if not is_mvic:
                # Find matching MVIC session
                cursor.execute("""
                SELECT es.numeric_id, es.muscle1_mvic_peak, es.muscle1_mvic_rms, es.muscle2_mvic_peak, es.muscle2_mvic_rms 
                FROM emg_sessions es
                JOIN emg_metadata em ON es.emg_session_id = em.emg_session_id
                WHERE es.athlete_name = %s 
                AND em.date_recorded = %s 
                AND es.is_mvic = TRUE
                ORDER BY em.processed_date DESC
                LIMIT 1
                """, (session_data['athlete_name'], date_recorded))
                
                mvic_session = cursor.fetchone()
                
                if mvic_session:
                    # Associate this pitching session with the MVIC session
                    related_mvic_id = mvic_session[0]
                    logger.info(f"Found matching MVIC session: {related_mvic_id}")
                    
                    # Store MVIC reference values for normalization
                    muscle1_mvic_peak = mvic_session[1]
                    muscle1_mvic_rms = mvic_session[2]
                    muscle2_mvic_peak = mvic_session[3]
                    muscle2_mvic_rms = mvic_session[4]
                    
                    # Update the session with the MVIC reference
                    cursor.execute("""
                    UPDATE emg_sessions 
                    SET related_mvic_id = %s,
                        muscle1_mvic_peak = %s,
                        muscle1_mvic_rms = %s,
                        muscle2_mvic_peak = %s,
                        muscle2_mvic_rms = %s
                    WHERE numeric_id = %s
                    """, (
                        related_mvic_id,
                        muscle1_mvic_peak,
                        muscle1_mvic_rms,
                        muscle2_mvic_peak,
                        muscle2_mvic_rms,
                        session_numeric_id
                    ))
                    
                    # Calculate %MVIC for throws if we have throw data
                    if throw_data:
                        logger.info("Calculating %MVIC values for throws")
                        for throw in throw_data:
                            # Calculate %MVIC for Muscle 1
                            if muscle1_mvic_peak and muscle1_mvic_peak > 0:
                                throw['muscle1_peak_amplitude_pct_mvic'] = (throw['muscle1_peak_amplitude'] / muscle1_mvic_peak) * 100
                            if muscle1_mvic_rms and muscle1_mvic_rms > 0:
                                throw['muscle1_rms_value_pct_mvic'] = (throw['muscle1_rms_value'] / muscle1_mvic_rms) * 100
                                if 'muscle1_throw_integral' in throw:
                                    throw['muscle1_throw_integral_pct_mvic'] = (throw['muscle1_throw_integral'] / muscle1_mvic_rms) * 100
                            
                            # Calculate %MVIC for Muscle 2
                            if muscle2_mvic_peak and muscle2_mvic_peak > 0:
                                throw['muscle2_peak_amplitude_pct_mvic'] = (throw['muscle2_peak_amplitude'] / muscle2_mvic_peak) * 100
                            if muscle2_mvic_rms and muscle2_mvic_rms > 0:
                                throw['muscle2_rms_value_pct_mvic'] = (throw['muscle2_rms_value'] / muscle2_mvic_rms) * 100
                                if 'muscle2_throw_integral' in throw:
                                    throw['muscle2_throw_integral_pct_mvic'] = (throw['muscle2_throw_integral'] / muscle2_mvic_rms) * 100
                            
                            # Log sample values for verification
                            if throw['trial_number'] == 1:
                                logger.info(f"Sample %MVIC values for first throw:")
                                logger.info(f"Muscle1 Peak: {throw.get('muscle1_peak_amplitude_pct_mvic', 'N/A')}%")
                                logger.info(f"Muscle1 RMS: {throw.get('muscle1_rms_value_pct_mvic', 'N/A')}%")
                                logger.info(f"Muscle2 Peak: {throw.get('muscle2_peak_amplitude_pct_mvic', 'N/A')}%")
                                logger.info(f"Muscle2 RMS: {throw.get('muscle2_rms_value_pct_mvic', 'N/A')}%")
                else:
                    logger.warning(f"No matching MVIC session found for {session_data['athlete_name']} on {date_recorded}")
            
            # Insert throw data for pitching sessions
            if not is_mvic and throw_data:
                # Process throw data...
                for throw in throw_data:
                    column_names = []
                    placeholders = []
                    values = [session_numeric_id, str(metadata_id)]  # Add both IDs
                    
                    # Base columns
                    column_names.extend(['session_numeric_id', 'emg_session_id'])
                    placeholders.extend(['%s', '%s'])
                    
                    # Add all other throw data
                    for key, value in throw.items():
                        if key != 'emg_session_id':  # Skip this as we're using metadata_id
                            column_names.append(key)
                            placeholders.append('%s')
                            values.append(value if not pd.isna(value) else None)
                    
                    # Construct query
                    query = f"""
                    INSERT INTO emg_throws (
                        {', '.join(column_names)}
                    ) VALUES ({', '.join(placeholders)})
                    """
                    
                    # Execute insert
                    cursor.execute(query, values)
            
            # Insert time series data for pitching sessions
            if not is_mvic and timeseries_data is not None and not timeseries_data.empty:
                # Process time series data in batches...
                timeseries_data = timeseries_data.replace({np.nan: None})
                
                total_rows = len(timeseries_data)
                for start_idx in range(0, total_rows, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, total_rows)
                    batch = timeseries_data.iloc[start_idx:end_idx]
                    
                    # Create list of value tuples with both IDs
                    timeseries_values = list(zip(
                        [session_numeric_id] * len(batch),  # numeric_id
                        [str(metadata_id)] * len(batch),    # emg_session_id
                        batch['time_point'],
                        batch['muscle1_emg'],
                        batch['muscle2_emg']
                    ))
                    
                    # Insert batch
                    cursor.executemany("""
                    INSERT INTO emg_timeseries (
                        session_numeric_id, emg_session_id, time_point, muscle1_emg, muscle2_emg
                    ) VALUES (%s, %s, %s, %s, %s)
                    """, timeseries_values)
            
            # Commit transaction
            conn.commit()
            logger.info(f"Successfully saved {filename} to database")
            return True
            
        except Exception as e:
            logger.error(f"Error saving to database: {str(e)}")
            conn.rollback()
            return False
            
        finally:
            if conn:
                conn.close()
            
    def process_directory(self, directory=None, recursive=False):
        """Process all EMG data files in a directory."""
        directory = directory or self.data_dir
        logger.info(f"Processing directory: {directory} (recursive={recursive})")
        
        # Setup database
        if not self.setup_database():
            logger.error("Failed to setup database tables. Aborting.")
            return {'success': False, 'processed': 0, 'failed': 0, 'total': 0}
        
        # Get list of files to process
        files_to_process = []
        if recursive:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(('.csv', '.txt')):
                        files_to_process.append(os.path.join(root, file))
        else:
            files_to_process = [os.path.join(directory, f) for f in os.listdir(directory) 
                                if os.path.isfile(os.path.join(directory, f)) and 
                                f.endswith(('.csv', '.txt'))]
        
        if not files_to_process:
            logger.warning(f"No EMG data files found in {directory}")
            return {'success': True, 'processed': 0, 'failed': 0, 'total': 0}
        
        # Sort files so that MVIC files are processed first
        def file_sort_key(file_path):
            # Extract the base filename
            file_name = os.path.basename(file_path)
            # Check if it's an MVIC file
            is_mvic = 'mvic' in file_name.lower()
            # If it's an MVIC file, sort it first (0), otherwise sort it later (1)
            sort_order = 0 if is_mvic else 1
            # Return a tuple with sort order and filename for stable sorting
            return (sort_order, file_name)
        
        # Sort the files
        files_to_process.sort(key=file_sort_key)
        
        logger.info(f"Found {len(files_to_process)} files to process")
        logger.info(f"Files will be processed in this order: {[os.path.basename(f) for f in files_to_process]}")
        
        # Process each file
        processed_count = 0
        failed_count = 0
        
        for file_path in tqdm(files_to_process, desc="Processing EMG files"):
            try:
                # Process file
                processed_data = self.process_file(file_path)
                
                if processed_data:
                    # Save to database
                    if self.save_to_database(processed_data):
                        processed_count += 1
                    else:
                        failed_count += 1
                        logger.error(f"Failed to save {file_path} to database")
                else:
                    failed_count += 1
                    logger.error(f"Failed to process {file_path}")
            except Exception as e:
                failed_count += 1
                logger.error(f"Error processing {file_path}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Summary
        summary = {
            'success': True,
            'processed': processed_count,
            'failed': failed_count,
            'total': len(files_to_process)
        }
        
        logger.info(f"Directory processing complete. Successfully processed {processed_count} of {len(files_to_process)} files.")
        
        return summary
    
    def run_single_file(self, file_path):
        """
        Run the pipeline on a single file.
        
        Parameters:
        -----------
        file_path : str
            Path to the EMG data file
            
        Returns:
        --------
        bool
            Success status
        """
        logger.info(f"Processing single file: {file_path}")
        
        # Setup database
        if not self.setup_database():
            logger.error("Failed to setup database tables. Aborting.")
            return False
        
        # Process file
        processed_data = self.process_file(file_path)
        
        if not processed_data:
            logger.error(f"Failed to process {file_path}")
            return False
        
        # Save to database
        result = self.save_to_database(processed_data)
        
        if result:
            logger.info(f"Successfully processed and saved {file_path}")
        else:
            logger.error(f"Failed to save {file_path} to database")
        
        return result

def main():
    """Main entry point for the pipeline script."""
    parser = argparse.ArgumentParser(description='EMG Data Processing Pipeline')
    parser.add_argument('-d', '--directory', type=str, help='Directory containing EMG data files')
    parser.add_argument('-f', '--file', type=str, help='Single EMG data file to process')
    parser.add_argument('-r', '--recursive', action='store_true', help='Process directories recursively')
    parser.add_argument('-b', '--batch-size', type=int, default=1000, help='Batch size for database operations')
    
    args = parser.parse_args()
    
    parser.add_argument('--match-velocity', action='store_true', 
                        help='Match EMG throws to velocity data')
    
    args = parser.parse_args()
    
    if args.match_velocity:
        pipeline = EMGPipeline()
        success = pipeline.match_throws_to_velocity()
        exit(0 if success else 1)
    # Database configuration (hardcoded for simplicity)
    db_config = {
        'host': os.getenv('DB_HOST'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'database': os.getenv('DB_NAME')
    }
    
    # Initialize pipeline
    pipeline = EMGPipeline(
        data_dir=args.directory,
        db_config=db_config,
        batch_size=args.batch_size
    )
    
    # Run pipeline
    if args.file:
        # Process single file
        success = pipeline.run_single_file(args.file)
        exit(0 if success else 1)
    elif args.directory:
        # Process directory
        summary = pipeline.process_directory(args.directory, args.recursive)
        exit(0 if summary['success'] else 1)
    else:
        parser.print_help()
        exit(1)

if __name__ == "__main__":
    main()