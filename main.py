#!/usr/bin/env python
"""
Main entry point for the EMG data processing pipeline.
Run this script to process EMG data files and load them into the database.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv
import datetime

from emg_pipeline import EMGPipeline
from db_connector import DBConnector

# Load environment variables
load_dotenv()

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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='EMG Data Processing Pipeline')
    
    # Main action arguments
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('-d', '--directory', type=str, help='Directory containing EMG data files')
    action_group.add_argument('-f', '--file', type=str, help='Single EMG data file to process')
    action_group.add_argument('--test-db', action='store_true', help='Test database connection')
    
    # Optional arguments
    parser.add_argument('-r', '--recursive', action='store_true', help='Process directories recursively')
    parser.add_argument('-b', '--batch-size', type=int, default=int(os.getenv('BATCH_SIZE', '1000')), 
                        help='Batch size for database operations')
    parser.add_argument('--dry-run', action='store_true', help='Process files but don\'t save to database')
    
    return parser.parse_args()

def get_db_config():
    """Get database configuration from environment variables."""
    return {
        'host': os.getenv('DB_HOST'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'database': os.getenv('DB_NAME')
    }

def main():
    """Main entry point for the EMG data processing pipeline."""
    args = parse_args()
    
    # Check if environment variables are loaded
    if not all([os.getenv('DB_HOST'), os.getenv('DB_USER'), os.getenv('DB_PASSWORD'), os.getenv('DB_NAME')]):
        logger.warning("Some database environment variables are missing.")
        logger.warning("Please make sure you have a .env file with the required configuration.")
    
    # Database configuration from environment
    db_config = get_db_config()
    
    # Initialize pipeline
    pipeline = EMGPipeline(
        data_dir=args.directory if args.directory else os.getenv('DEFAULT_DATA_DIR', os.getcwd()),
        db_config=db_config,
        batch_size=args.batch_size
    )
    
    # Test database connection
    if args.test_db:
        db = DBConnector(db_config)
        if db.test_connection():
            logger.info("Database connection test successful!")
            return 0
        else:
            logger.error("Database connection test failed!")
            return 1
    
    # Process single file
    if args.file:
        file_path = os.path.abspath(args.file)
        logger.info(f"Processing single file: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return 1
        
        if args.dry_run:
            # Just process without saving to database
            processed_data = pipeline.process_file(file_path)
            if processed_data:
                session_id = processed_data['session_data']['session_id']
                throws_count = len(processed_data['throw_data'])
                timeseries_rows = len(processed_data['timeseries_data'])
                logger.info(f"Dry run successful for {session_id}: {throws_count} throws, {timeseries_rows} time points")
                return 0
            else:
                logger.error(f"Failed to process {file_path}")
                return 1
        else:
            # Process and save to database
            success = pipeline.run_single_file(file_path)
            return 0 if success else 1
    
    # Process directory
    if args.directory:
        dir_path = os.path.abspath(args.directory)
        logger.info(f"Processing directory: {dir_path} (recursive={args.recursive})")
        
        if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
            logger.error(f"Directory not found: {dir_path}")
            return 1
        
        # Count files before processing
        file_count = 0
        if args.recursive:
            for root, _, files in os.walk(dir_path):
                file_count += sum(1 for f in files if f.endswith(('.csv', '.txt')))
        else:
            file_count = sum(1 for f in os.listdir(dir_path) 
                          if os.path.isfile(os.path.join(dir_path, f)) and 
                          f.endswith(('.csv', '.txt')))
        
        logger.info(f"Found {file_count} files to process")
        
        if file_count == 0:
            logger.warning(f"No EMG data files found in {dir_path}")
            return 0
        
        if args.dry_run:
            logger.info("Dry run mode: files will be processed but not saved to database")
        
        # Process directory
        summary = pipeline.process_directory(dir_path, args.recursive)
        
        logger.info(f"Directory processing complete.")
        logger.info(f"Successfully processed {summary['processed']} of {summary['total']} files.")
        if summary['failed'] > 0:
            logger.warning(f"Failed to process {summary['failed']} files.")
        
        return 0 if summary['success'] else 1

    # If neither --file nor --directory is provided, process all files from yesterday
    # in the default data directory (using MMDDYYYY in filename)
    default_data_dir = os.getenv('DEFAULT_DATA_DIR', os.getcwd())
    yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%m%d%Y')
    logger.info(f"No --file or --directory specified. Looking for files from yesterday ({yesterday}) in {default_data_dir}")
    files = [
        os.path.join(default_data_dir, f)
        for f in os.listdir(default_data_dir)
        if yesterday in f and (f.endswith('.csv') or f.endswith('.txt'))
    ]
    if not files:
        logger.info(f"No files found for {yesterday} in {default_data_dir}")
        return 0
    processed = 0
    failed = 0
    for file_path in files:
        logger.info(f"Processing file: {file_path}")
        try:
            processed_data = pipeline.process_file(file_path)
            if processed_data:
                if args.dry_run:
                    logger.info(f"Dry run: processed {file_path} successfully")
                    processed += 1
                else:
                    if pipeline.save_to_database(processed_data):
                        logger.info(f"Successfully processed and saved {file_path}")
                        processed += 1
                    else:
                        logger.error(f"Failed to save {file_path} to database")
                        failed += 1
            else:
                logger.error(f"Failed to process {file_path}")
                failed += 1
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            failed += 1
    logger.info(f"Finished processing yesterday's files. Success: {processed}, Failed: {failed}, Total: {len(files)}")
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())