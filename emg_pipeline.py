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
        """
        Set up database tables.
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        conn = self.db.connect()
        if not conn:
            logger.error("Failed to connect to database")
            return False
            
        try:
            cursor = conn.cursor()
            
            # Create or update EMG throws table with new columns for spectral entropy, wavelet, and coactivation metrics
            try:
                # Check if the new columns exist already
                cursor.execute("SHOW COLUMNS FROM emg_throws LIKE 'muscle1_spectral_entropy'")
                column_exists = cursor.fetchone()
                
                # Add new columns if they don't exist
                if not column_exists:
                    logger.info("Adding new metric columns to emg_throws table")
                    
                    # Add spectral entropy columns
                    cursor.execute("ALTER TABLE emg_throws ADD COLUMN muscle1_spectral_entropy FLOAT")
                    cursor.execute("ALTER TABLE emg_throws ADD COLUMN muscle2_spectral_entropy FLOAT")
                    
                    # Add wavelet energy columns
                    cursor.execute("ALTER TABLE emg_throws ADD COLUMN muscle1_wavelet_energy_low FLOAT")
                    cursor.execute("ALTER TABLE emg_throws ADD COLUMN muscle1_wavelet_energy_mid FLOAT")
                    cursor.execute("ALTER TABLE emg_throws ADD COLUMN muscle1_wavelet_energy_high FLOAT")
                    cursor.execute("ALTER TABLE emg_throws ADD COLUMN muscle2_wavelet_energy_low FLOAT")
                    cursor.execute("ALTER TABLE emg_throws ADD COLUMN muscle2_wavelet_energy_mid FLOAT")
                    cursor.execute("ALTER TABLE emg_throws ADD COLUMN muscle2_wavelet_energy_high FLOAT")
                    
                    # Add coactivation columns
                    cursor.execute("ALTER TABLE emg_throws ADD COLUMN coactivation_index FLOAT")
                    cursor.execute("ALTER TABLE emg_throws ADD COLUMN coactivation_correlation FLOAT")
                    cursor.execute("ALTER TABLE emg_throws ADD COLUMN coactivation_temporal_overlap FLOAT")
                    cursor.execute("ALTER TABLE emg_throws ADD COLUMN coactivation_waveform_similarity FLOAT")
                    
                    conn.commit()
                    logger.info("Successfully added new metric columns")
            except Exception as e:
                logger.error(f"Error adding new columns: {e}")
                conn.rollback()
                
            # Continue with regular table creation
            return self.db.create_tables()
                
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            return False
        finally:
            self.db.disconnect()
    
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
            
            # Preprocess EMG data
            muscle1_filtered, muscle1_rectified, muscle1_rms, muscle1_envelope = self.processor.preprocess_emg(muscle1_emg, muscle1_fs)
            muscle2_filtered, muscle2_rectified, muscle2_rms, muscle2_envelope = self.processor.preprocess_emg(muscle2_emg, muscle2_fs)

            # Align muscle2 data to muscle1 timeline if lengths are different
            if len(muscle1_time) != len(muscle2_time):
                # Use the muscle1 timeline as reference and interpolate muscle2 data to match
                aligned_muscle2_rms = np.interp(muscle1_time, muscle2_time, muscle2_rms)
                common_time = muscle1_time
                common_fs = muscle1_fs
            else:
                aligned_muscle2_rms = muscle2_rms
                common_time = muscle1_time
                common_fs = muscle1_fs

            # Use integrated dual-muscle detection
            throws = self.processor.detect_throws_multi_muscle(
                muscle1_rms, muscle2_rms, common_time, common_fs, 
                threshold_factor_fcr=2.75, threshold_factor_fcu=1.25,
                min_duration=0.2, min_separation=10, 
                coincidence_window=0.2
            )

            # Calculate metrics for both muscles using the same throw indices
            muscle1_metrics = self.processor.calculate_comprehensive_metrics(
                muscle1_filtered, muscle1_rectified, muscle1_rms, muscle1_envelope, 
                throws, common_time, muscle1_fs
            )

            muscle2_metrics = self.processor.calculate_comprehensive_metrics(
                muscle2_filtered, muscle2_rectified, muscle2_rms, muscle2_envelope, 
                throws, common_time, muscle2_fs
            )
            
            # Calculate coactivation metrics between muscles
            coactivation_metrics = self.processor.calculate_muscle_coactivation(
                muscle1_metrics, muscle2_metrics,
                throws, throws,  # Same throws for both muscles now
                muscle1_filtered, muscle2_filtered,
                common_time, common_time,  # Using the common timeline
                common_fs
            )
            
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
                'muscle3_name': metadata.get('muscle3_name'),
                'muscle4_name': metadata.get('muscle4_name'),
                'muscle1_fs': muscle1_fs,
                'muscle2_fs': muscle2_fs,
                'file_path': file_path,
                'processed_date': datetime.now()
            }
            
            # Prepare time series data with dynamic muscle names
            timeseries_data = pd.DataFrame({
                'session_id': [session_id] * len(muscle2_time),
                'time_point': muscle2_time,
                'muscle1_emg': np.interp(muscle2_time, muscle1_time, muscle1_emg),  # Align muscle1 data to muscle2 time points
                'muscle2_emg': muscle2_emg
            })
            
            # Prepare throw data with new timestamp fields
            throw_data = []
            for i in range(len(throws)):
                # Calculate relative start time
                relative_start = muscle2_time[throws[i][0]]
                
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
                    'session_id': session_id,
                    'throw_number': i + 1,
                    'start_time': muscle2_time[throws[i][0]],
                    'end_time': muscle2_time[throws[i][1]],
                    'duration': muscle2_metrics['throw_durations'][i],
                    
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
                    
                    # NEW: Spectral entropy metrics
                    'muscle1_spectral_entropy': muscle1_metrics['spectral_entropies'][i],
                    'muscle2_spectral_entropy': muscle2_metrics['spectral_entropies'][i],
                    
                    # NEW: Wavelet energy metrics
                    'muscle1_wavelet_energy_low': muscle1_metrics['wavelet_energy_low'][i],
                    'muscle1_wavelet_energy_mid': muscle1_metrics['wavelet_energy_mid'][i],
                    'muscle1_wavelet_energy_high': muscle1_metrics['wavelet_energy_high'][i],
                    'muscle2_wavelet_energy_low': muscle2_metrics['wavelet_energy_low'][i],
                    'muscle2_wavelet_energy_mid': muscle2_metrics['wavelet_energy_mid'][i],
                    'muscle2_wavelet_energy_high': muscle2_metrics['wavelet_energy_high'][i],
                    
                    # New timestamp-related fields
                    'relative_start_time': relative_start,
                    'absolute_timestamp': absolute_timestamp,
                    'session_trial': None,  # Will be populated during velocity matching
                    'pitch_speed_mph': None,  # Will be populated during velocity matching
                    'velocity_match_quality': 'no_match'  # Default match quality
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
    
    def match_throws_to_velocity(self):
        """
        Match EMG throws to velocity data from trials and poi tables.
        Uses trial numbers to provide more accurate matching between EMG throws and velocity data.
        
        Returns:
        --------
        bool
            True if matching was successful, False otherwise
        """
        conn = None
        try:
            # Connect to the database
            conn = self.db.connect()
            if not conn:
                logger.error("Failed to connect to database")
                return False
            
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            
            # First, get all EMG sessions with dates
            cursor.execute("""
                SELECT numeric_id, athlete_name, date_recorded, filename 
                FROM emg_sessions
                WHERE date_recorded IS NOT NULL
                ORDER BY date_recorded DESC
            """)
            emg_sessions = cursor.fetchall()
            logger.info(f"Found {len(emg_sessions)} EMG sessions to process")
            
            # Process each EMG session
            for emg_session in emg_sessions:
                session_id = emg_session['numeric_id']
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
                
                # Get EMG throws for this session
                cursor.execute(f"""
                    SELECT throw_id, trial_number, start_time, end_time, 
                        (end_time - start_time) as duration,
                        muscle1_peak_amplitude, muscle2_peak_amplitude
                    FROM emg_throws
                    WHERE session_numeric_id = {session_id}
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
                    WHERE session_numeric_id = {session_id}
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
                JOIN emg_sessions es ON et.session_numeric_id = es.numeric_id
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
            JOIN emg_sessions es ON et.session_numeric_id = es.numeric_id
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
        """
        Save processed data to the database.
        """
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
            timeseries_data = processed_data['timeseries_data']
            throw_data = processed_data['throw_data']
            
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
                numeric_id = session_result[0]
                logger.info(f"Session {filename} already exists. Removing old data...")
                cursor.execute("DELETE FROM emg_throws WHERE session_numeric_id = %s", (numeric_id,))
                cursor.execute("DELETE FROM emg_timeseries WHERE session_numeric_id = %s", (numeric_id,))
                cursor.execute("DELETE FROM emg_sessions WHERE numeric_id = %s", (numeric_id,))
            
            # Insert session data with new fields
            date_recorded = session_data['date_recorded']
            if date_recorded is None:
                logger.warning(f"No date parsed from filename. Using current date for session {filename}")
                date_recorded = datetime.now().date()
            
            cursor.execute("""
            INSERT INTO emg_sessions (
                filename, date_recorded, collection_date, start_time,
                traq_id, athlete_name, session_type,
                muscle_count, muscle1_name, muscle2_name, muscle3_name, muscle4_name,
                muscle1_fs, muscle2_fs, file_path, processed_date
            ) VALUES ( %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                filename,
                date_recorded,
                session_data.get('collection_date'),
                session_data.get('start_time'),
                session_data['traq_id'],
                session_data['athlete_name'],
                session_data['session_type'],
                session_data.get('muscle_count', 2),
                session_data.get('muscle1_name'),
                session_data.get('muscle2_name'),
                session_data.get('muscle3_name'),
                session_data.get('muscle4_name'),
                session_data.get('muscle1_fs'),
                session_data.get('muscle2_fs'),
                session_data['file_path'],
                session_data['processed_date']
            ))
            
            # Get the auto-generated numeric_id
            cursor.execute("SELECT LAST_INSERT_ID()")
            numeric_id = cursor.fetchone()[0]
            
            # Insert throw data with all metrics
            if throw_data:
                # Construct dynamic SQL query based on the first throw's keys
                first_throw = throw_data[0]
                throw_columns = []
                placeholders = []
                
                # Start with required columns
                throw_columns.append("session_numeric_id")
                throw_columns.append("trial_number")
                throw_columns.append("start_time")
                throw_columns.append("end_time")
                throw_columns.append("duration")
                
                # Add all metric columns dynamically
                for key in first_throw.keys():
                    if key not in ['session_id', 'throw_number', 'start_time', 'end_time', 'duration']:
                        throw_columns.append(key)
                
                # Create placeholders for all columns
                placeholders = ["%s"] * len(throw_columns)
                
                # Construct SQL query
                insert_query = f"""
                INSERT INTO emg_throws (
                    {', '.join(throw_columns)}
                ) VALUES ({', '.join(placeholders)})
                """
                
                # Prepare values for each throw
                throw_values = []
                for throw in throw_data:
                    values = [numeric_id, throw['throw_number'], throw['start_time'], throw['end_time'], throw['duration']]
                    
                    # Add all metric values in the same order as columns
                    for key in throw_columns[5:]:  # Skip the first 5 required columns
                        values.append(throw.get(key))
                    
                    throw_values.append(tuple(values))
                
                # Execute insert
                cursor.executemany(insert_query, throw_values)
            
            # Insert time series data in batches with numeric_id
            if not timeseries_data.empty:
                # Convert DataFrame to list of tuples for executemany
                total_rows = len(timeseries_data)
                for start_idx in range(0, total_rows, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, total_rows)
                    batch = timeseries_data.iloc[start_idx:end_idx]
                    
                    # Create list of value tuples
                    timeseries_values = list(zip(
                        [numeric_id] * len(batch),  # Use numeric_id instead of session_id
                        batch['time_point'],
                        batch['muscle1_emg'],
                        batch['muscle2_emg']
                    ))
                    
                    # Insert batch
                    cursor.executemany("""
                    INSERT INTO emg_timeseries (session_numeric_id, time_point, muscle1_emg, muscle2_emg)
                    VALUES (%s, %s, %s, %s)
                    """, timeseries_values)
                    
                    logger.debug(f"Inserted batch of {len(batch)} time series rows")
            
            # Commit the transaction
            conn.commit()
            logger.info(f"Successfully saved data for session {filename} to database")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            conn.rollback()
            return False
        finally:
            self.db.disconnect()
    
    def process_directory(self, directory=None, recursive=False):
        """
        Process all EMG data files in a directory.
        
        Parameters:
        -----------
        directory : str, optional
            Directory to process. If None, uses self.data_dir
        recursive : bool, optional
            Whether to recursively process files in subdirectories
            
        Returns:
        --------
        dict
            Summary of processed files
        """
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
        
        logger.info(f"Found {len(files_to_process)} files to process")
        
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