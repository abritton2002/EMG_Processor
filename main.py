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

from emg_pipeline import EMGPipeline

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
    parser.add_argument('-b', '--batch-size', type=int, default=1000, help='Batch size for database operations')
    parser.add_argument('--dry-run', action='store_true', help='Process files but don\'t save to database')
    
    return parser.parse_args()

def main():
    """Main entry point for the EMG data processing pipeline."""
    args = parse_args()
    
    # Database configuration
    db_config = {
        'host': '10.200.200.107',
        'user': 'scriptuser1',
        'password': 'YabinMarshed2023@#$',
        'database': 'theia_pitching_db'
    }
    
    # Initialize pipeline
    pipeline = EMGPipeline(
        data_dir=args.directory if args.directory else os.getcwd(),
        db_config=db_config,
        batch_size=args.batch_size
    )
    
    # Test database connection
    if args.test_db:
        from db_connector import DBConnector
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
    
    return 0

if __name__ == "__main__":
    sys.exit(main())