import os
import pandas as pd
import numpy as np
import logging
import datetime
from pathlib import Path
import argparse
from tqdm import tqdm

from emg_processor import EMGProcessor
from db_connector import DBConnector

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
            
            # Load EMG data with dynamic muscle names
            muscle1_emg, muscle2_emg, muscle1_time, muscle2_time, metadata = self.processor.load_delsys_emg_data(file_path)
            
            # Extract muscle names for easier reference
            muscle1_name = metadata.get('muscle1_name', 'muscle1')
            muscle2_name = metadata.get('muscle2_name', 'muscle2')
            
            # Extract sampling rates
            muscle1_fs = metadata.get('muscle1_fs', 2000)
            muscle2_fs = metadata.get('muscle2_fs', 2000)
            
            # Preprocess EMG data
            muscle1_filtered, muscle1_rectified, muscle1_rms, muscle1_envelope = self.processor.preprocess_emg(muscle1_emg, muscle1_fs)
            muscle2_filtered, muscle2_rectified, muscle2_rms, muscle2_envelope = self.processor.preprocess_emg(muscle2_emg, muscle2_fs)
            
            # Use muscle2 (formerly FCR) as primary for throw detection
            throws_muscle2 = self.processor.detect_throws_fcr(muscle2_rms, muscle2_time, muscle2_fs)
            
            # Map muscle2 throw indices to muscle1 indices
            throws_muscle1 = self.processor.map_throws_to_fcu(throws_muscle2, muscle2_time, muscle1_time)
            
            # Calculate metrics
            muscle2_metrics = self.processor.calculate_comprehensive_metrics(
                muscle2_filtered, muscle2_rectified, muscle2_rms, muscle2_envelope, 
                throws_muscle2, muscle2_time, muscle2_fs
            )
            
            muscle1_metrics = self.processor.calculate_comprehensive_metrics(
                muscle1_filtered, muscle1_rectified, muscle1_rms, muscle1_envelope,
                throws_muscle1, muscle1_time, muscle1_fs
            )
            
            # Prepare session data with additional metadata
            session_data = {
                'session_id': session_id,
                'date_recorded': file_metadata['date'],
                'collection_date': metadata.get('Collection_Date'),
                'start_time': metadata.get('Start_Time'),
                'traq_id': file_metadata['traq_id'],
                'athlete_name': file_metadata['athlete_name'],
                'session_type': file_metadata['session_type'],
                'muscle_count': metadata.get('muscle_count', 2),
                'muscle1_name': metadata.get('muscle1_name'),
                'muscle2_name': metadata.get('muscle2_name'),
                'muscle3_name': metadata.get('muscle3_name'),
                'muscle4_name': metadata.get('muscle4_name'),
                'muscle1_id': metadata.get('muscle1_id'),
                'muscle2_id': metadata.get('muscle2_id'),
                'muscle3_id': metadata.get('muscle3_id'),
                'muscle4_id': metadata.get('muscle4_id'),
                'muscle1_fs': muscle1_fs,
                'muscle2_fs': muscle2_fs,
                'file_path': file_path,
                'processed_date': datetime.datetime.now()
            }
            
            # Prepare time series data with dynamic muscle names
            timeseries_data = pd.DataFrame({
                'session_id': [session_id] * len(muscle2_time),
                'time_point': muscle2_time,
                'muscle1_emg': np.interp(muscle2_time, muscle1_time, muscle1_emg),  # Align muscle1 data to muscle2 time points
                'muscle2_emg': muscle2_emg
            })
            
            # Prepare throw data with dynamic muscle names
            throw_data = []
            for i in range(len(throws_muscle2)):
                start_idx, end_idx = throws_muscle2[i]
                throw_row = {
                    'session_id': session_id,
                    'throw_number': i + 1,
                    'start_time': muscle2_time[start_idx],
                    'end_time': muscle2_time[end_idx],
                    'duration': muscle2_metrics['throw_durations'][i],
                    
                    # Muscle2 metrics
                    'muscle2_median_freq': muscle2_metrics['median_freqs'][i],
                    'muscle2_mean_freq': muscle2_metrics['mean_freqs'][i],
                    'muscle2_bandwidth': muscle2_metrics['bandwidth'][i],
                    'muscle2_peak_amplitude': muscle2_metrics['peak_amplitudes'][i],
                    'muscle2_rms_value': muscle2_metrics['rms_values'][i],
                    'muscle2_rise_time': muscle2_metrics['rise_times'][i],
                    'muscle2_throw_integral': muscle2_metrics['throw_integrals'][i],
                    'muscle2_work_rate': muscle2_metrics['work_rates'][i],
                    
                    # Muscle1 metrics
                    'muscle1_median_freq': muscle1_metrics['median_freqs'][i],
                    'muscle1_mean_freq': muscle1_metrics['mean_freqs'][i],
                    'muscle1_bandwidth': muscle1_metrics['bandwidth'][i],
                    'muscle1_peak_amplitude': muscle1_metrics['peak_amplitudes'][i],
                    'muscle1_rms_value': muscle1_metrics['rms_values'][i],
                    'muscle1_rise_time': muscle1_metrics['rise_times'][i],
                    'muscle1_throw_integral': muscle1_metrics['throw_integrals'][i],
                    'muscle1_work_rate': muscle1_metrics['work_rates'][i]
                }
                
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
    
    def save_to_database(self, processed_data):
        """
        Save processed data to the database.
        
        Parameters:
        -----------
        processed_data : dict
            Processed data from process_file()
            
        Returns:
        --------
        bool
            Success status
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
            
            # Check if session already exists
            cursor.execute(
                "SELECT session_id FROM emg_sessions WHERE session_id = %s",
                (session_data['session_id'],)
            )
            
            if cursor.fetchone():
                logger.info(f"Session {session_data['session_id']} already exists. Removing old data...")
                cursor.execute("DELETE FROM emg_throws WHERE session_id = %s", (session_data['session_id'],))
                cursor.execute("DELETE FROM emg_timeseries WHERE session_id = %s", (session_data['session_id'],))
                cursor.execute("DELETE FROM emg_sessions WHERE session_id = %s", (session_data['session_id'],))
            
            # Insert session data with new fields
            # If date_recorded is None, use today's date
            date_recorded = session_data['date_recorded']
            if date_recorded is None:
                logger.warning(f"No date parsed from filename. Using current date for session {session_data['session_id']}")
                date_recorded = datetime.datetime.now().date()
            
            cursor.execute("""
            INSERT INTO emg_sessions (
                session_id, date_recorded, collection_date, start_time,
                traq_id, athlete_name, session_type,
                muscle_count, muscle1_name, muscle2_name, muscle3_name, muscle4_name,
                muscle1_id, muscle2_id, muscle3_id, muscle4_id,
                muscle1_fs, muscle2_fs, file_path, processed_date
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                session_data['session_id'],
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
                session_data.get('muscle1_id'),
                session_data.get('muscle2_id'),
                session_data.get('muscle3_id'),
                session_data.get('muscle4_id'),
                session_data.get('muscle1_fs'),
                session_data.get('muscle2_fs'),
                session_data['file_path'],
                session_data['processed_date']
            ))
            
            # Insert throw data with dynamic muscle names
            if throw_data:
                throw_values = []
                for throw in throw_data:
                    throw_values.append((
                        throw['session_id'],
                        throw['throw_number'],
                        throw['start_time'],
                        throw['end_time'],
                        throw['duration'],
                        
                        # Muscle1 metrics
                        throw['muscle1_median_freq'],
                        throw['muscle1_mean_freq'],
                        throw['muscle1_bandwidth'],
                        throw['muscle1_peak_amplitude'],
                        throw['muscle1_rms_value'],
                        throw['muscle1_rise_time'],
                        throw['muscle1_throw_integral'],
                        throw['muscle1_work_rate'],
                        
                        # Muscle2 metrics
                        throw['muscle2_median_freq'],
                        throw['muscle2_mean_freq'],
                        throw['muscle2_bandwidth'],
                        throw['muscle2_peak_amplitude'],
                        throw['muscle2_rms_value'],
                        throw['muscle2_rise_time'],
                        throw['muscle2_throw_integral'],
                        throw['muscle2_work_rate']
                    ))
                
                # Use executemany for better performance
                cursor.executemany("""
                INSERT INTO emg_throws (
                    session_id, throw_number, start_time, end_time, duration,
                    muscle1_median_freq, muscle1_mean_freq, muscle1_bandwidth, muscle1_peak_amplitude, muscle1_rms_value,
                    muscle1_rise_time, muscle1_throw_integral, muscle1_work_rate,
                    muscle2_median_freq, muscle2_mean_freq, muscle2_bandwidth, muscle2_peak_amplitude, muscle2_rms_value,
                    muscle2_rise_time, muscle2_throw_integral, muscle2_work_rate
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, throw_values)
            
            # Insert time series data in batches with dynamic muscle names
            if not timeseries_data.empty:
                # Convert DataFrame to list of tuples for executemany
                total_rows = len(timeseries_data)
                for start_idx in range(0, total_rows, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, total_rows)
                    batch = timeseries_data.iloc[start_idx:end_idx]
                    
                    # Create list of value tuples
                    timeseries_values = list(zip(
                        batch['session_id'],
                        batch['time_point'],
                        batch['muscle1_emg'],
                        batch['muscle2_emg']
                    ))
                    
                    # Insert batch
                    cursor.executemany("""
                    INSERT INTO emg_timeseries (session_id, time_point, muscle1_emg, muscle2_emg)
                    VALUES (%s, %s, %s, %s)
                    """, timeseries_values)
                    
                    logger.debug(f"Inserted batch of {len(batch)} time series rows")
            
            # Commit the transaction
            conn.commit()
            logger.info(f"Successfully saved data for session {session_data['session_id']} to database")
            
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
    
    # Database configuration (hardcoded for simplicity)
    db_config = {
        'host': '10.200.200.107',
        'user': 'scriptuser1',
        'password': 'YabinMarshed2023@#$',
        'database': 'theia_pitching_db'
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