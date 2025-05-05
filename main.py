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
import traceback

from emg_pipeline import EMGPipeline
from db_connector import DBConnector

class PipelineError(Exception):
    """Custom exception for pipeline-specific errors."""
    pass

# Load environment variables
load_dotenv()

# Create logs directory if it doesn't exist
try:
    os.makedirs('logs', exist_ok=True)
except PermissionError:
    print("ERROR: Cannot create logs directory - permission denied")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Cannot create logs directory - {str(e)}")
    sys.exit(1)

# Get current timestamp for log filename
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join('logs', f'emg_pipeline_{timestamp}.log')

# Configure logging with immediate output to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # StreamHandler with explicit flush after each log
        logging.StreamHandler(stream=sys.stdout),
        logging.FileHandler(log_file)
    ]
)

# Force stdout to flush immediately
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# Set unbuffered output for Python
os.environ['PYTHONUNBUFFERED'] = '1'

# Create a custom handler that flushes after each emission
class FlushingStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

# Replace the default StreamHandler with our flushing version
for handler in logging.root.handlers:
    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
        logging.root.removeHandler(handler)
        logging.root.addHandler(FlushingStreamHandler(sys.stdout))

logger = logging.getLogger(__name__)
logger.info(f"Starting pipeline run at {datetime.datetime.now()}")

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

def check_environment():
    """Check if all required environment variables and paths are set up correctly."""
    required_vars = ['DB_HOST', 'DB_USER', 'DB_PASSWORD', 'DB_NAME']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise PipelineError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    data_dir = os.getenv('DEFAULT_DATA_DIR')
    if data_dir and not os.path.exists(data_dir):
        raise PipelineError(f"Default data directory does not exist: {data_dir}")

def process_single_file(pipeline, file_path, dry_run=False):
    """Process a single file with error handling."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        logger.info(f"Processing file: {file_path}")
        processed_data = pipeline.process_file(file_path)
        
        if not processed_data:
            raise PipelineError(f"Failed to process file: {file_path}")
            
        if dry_run:
            session_id = processed_data['session_data']['session_id']
            throws_count = len(processed_data['throw_data'])
            timeseries_rows = len(processed_data['timeseries_data'])
            logger.info(f"Dry run successful for {session_id}: {throws_count} throws, {timeseries_rows} time points")
            return True
        else:
            if not pipeline.save_to_database(processed_data):
                raise PipelineError(f"Failed to save {file_path} to database")
            logger.info(f"Successfully processed and saved {file_path}")
            return True
            
    except FileNotFoundError as e:
        logger.error(str(e))
        return False
    except PipelineError as e:
        logger.error(str(e))
        return False
    except Exception as e:
        logger.error(f"Unexpected error processing {file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Main entry point with enhanced error handling."""
    try:
        # Check environment setup
        check_environment()
        
        args = parse_args()
        db_config = get_db_config()
        
        # Initialize pipeline with error handling
        try:
            pipeline = EMGPipeline(
                data_dir=args.directory if args.directory else os.getenv('DEFAULT_DATA_DIR', os.getcwd()),
                db_config=db_config,
                batch_size=args.batch_size
            )
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {str(e)}")
            return 1
        
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
            success = process_single_file(pipeline, os.path.abspath(args.file), args.dry_run)
            return 0 if success else 1
        
        # Process directory
        if args.directory:
            dir_path = os.path.abspath(args.directory)
            
            if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
                logger.error(f"Directory not found: {dir_path}")
                return 1
            
            try:
                summary = pipeline.process_directory(dir_path, args.recursive)
                
                logger.info(f"Directory processing complete.")
                logger.info(f"Successfully processed {summary['processed']} of {summary['total']} files.")
                if summary['failed'] > 0:
                    logger.warning(f"Failed to process {summary['failed']} files.")
                
                return 0 if summary['success'] else 1
                
            except Exception as e:
                logger.error(f"Error processing directory {dir_path}: {str(e)}")
                logger.error(traceback.format_exc())
                return 1
        
        # Process yesterday's files
        try:
            default_data_dir = os.getenv('DEFAULT_DATA_DIR', os.getcwd())
            yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%m%d%Y')
            
            logger.info(f"Looking for files from yesterday ({yesterday}) in {default_data_dir}")
            
            if not os.path.exists(default_data_dir):
                raise PipelineError(f"Default data directory not found: {default_data_dir}")
            
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
                try:
                    if process_single_file(pipeline, file_path, args.dry_run):
                        processed += 1
                    else:
                        failed += 1
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    logger.error(traceback.format_exc())
                    failed += 1
            
            logger.info(f"Finished processing yesterday's files. Success: {processed}, Failed: {failed}, Total: {len(files)}")
            return 0 if failed == 0 else 1
            
        except Exception as e:
            logger.error(f"Error processing yesterday's files: {str(e)}")
            logger.error(traceback.format_exc())
            return 1
            
    except PipelineError as e:
        logger.error(f"Pipeline Error: {str(e)}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return 1
    finally:
        logger.info("Pipeline run completed at %s", datetime.datetime.now())

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.error("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)