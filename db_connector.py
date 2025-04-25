import pymysql
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class DBConnector:
    """
    Handles database connections and operations for the EMG pipeline.
    """
    
    def __init__(self, config=None):
        """
        Initialize database connector with configuration.
        
        Parameters:
        -----------
        config : dict, optional
            Database configuration dictionary. If None, uses environment variables.
        """
        # Use provided config or read from environment variables
        if config is None:
            self.db_config = {
                'host': os.getenv('DB_HOST'),
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD'),
                'database': os.getenv('DB_NAME')
            }
            
            # Check if environment variables are set
            missing_vars = []
            for key, value in self.db_config.items():
                if value is None:
                    missing_vars.append(key.upper())
            
            if missing_vars:
                logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
                logger.warning("Please create a .env file with the required database configuration")
        else:
            self.db_config = config
        
        self.conn = None
        logger.info("Database connector initialized")
    
    def connect(self):
        """
        Connect to the database.
        
        Returns:
        --------
        pymysql.Connection or None
            Database connection if successful, None otherwise
        """
        try:
            # Check if all required configuration is present
            if None in self.db_config.values():
                missing_keys = [k for k, v in self.db_config.items() if v is None]
                logger.error(f"Missing database configuration: {', '.join(missing_keys)}")
                return None
                
            self.conn = pymysql.connect(
                host=self.db_config['host'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database=self.db_config['database']
            )
            logger.info(f"Connected to database {self.db_config['database']}")
            return self.conn
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return None
    
    def disconnect(self):
        """Close the database connection if it exists."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed")
    
    def execute_query(self, query, params=None, fetch=False):
        """
        Execute a SQL query on the database.
        
        Parameters:
        -----------
        query : str
            SQL query to execute
        params : tuple or dict, optional
            Parameters for the query
        fetch : bool, optional
            Whether to fetch and return results
            
        Returns:
        --------
        list or None
            Query results if fetch=True, None otherwise
        """
        if not self.conn:
            self.connect()
            
        if not self.conn:
            logger.error("No database connection available")
            return None
            
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query, params)
                
                if fetch:
                    result = cursor.fetchall()
                    return result
                else:
                    self.conn.commit()
                    return None
                    
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            self.conn.rollback()
            return None
    
    def create_tables(self):
        """
        Create the necessary tables for the EMG pipeline with updated columns
        """
        try:
            # Create EMG Sessions table with renamed ID field and without muscle3/muscle4
            self.execute_query("""
            CREATE TABLE IF NOT EXISTS emg_sessions (
                emg_session_id INT AUTO_INCREMENT PRIMARY KEY,
                filename VARCHAR(100) UNIQUE,
                date_recorded DATE,
                collection_date DATE,
                start_time TIME,
                traq_id VARCHAR(50),
                athlete_name VARCHAR(100),
                session_type VARCHAR(50),
                muscle_count INT,
                muscle1_name VARCHAR(50),
                muscle2_name VARCHAR(50),
                muscle1_fs FLOAT,
                muscle2_fs FLOAT,
                file_path VARCHAR(255),
                processed_date DATETIME,
                is_mvic BOOLEAN DEFAULT FALSE,
                
                # MVIC reference values (if this is an MVIC session)
                muscle1_mvic_peak FLOAT,
                muscle1_mvic_rms FLOAT,
                muscle2_mvic_peak FLOAT,
                muscle2_mvic_rms FLOAT,
                
                # Related MVIC session (if this is a pitching session)
                related_mvic_id INT,
                
                INDEX idx_athlete_name (athlete_name),
                INDEX idx_date_recorded (date_recorded)
            )
            """)
            
            # Create raw time series data table with updated field names
            # Note: Time series data will only be stored for pitching sessions, not for MVIC
            self.execute_query("""
            CREATE TABLE IF NOT EXISTS emg_timeseries (
                id INT AUTO_INCREMENT PRIMARY KEY,
                emg_session_id INT,
                time_point FLOAT,
                muscle1_emg FLOAT,
                muscle2_emg FLOAT,
                INDEX idx_session_id (emg_session_id),
                FOREIGN KEY (emg_session_id) REFERENCES emg_sessions(emg_session_id) ON DELETE CASCADE
            )
            """)
            
            # Create throw details table with renamed fields and without muscle3/muscle4
            self.execute_query("""
            CREATE TABLE IF NOT EXISTS emg_throws (
                throw_id INT AUTO_INCREMENT PRIMARY KEY,
                emg_session_id INT,
                trial_number INT,
                start_time FLOAT,
                end_time FLOAT,
                duration FLOAT,
                
                # Timestamp and velocity matching columns
                relative_start_time FLOAT,
                absolute_timestamp DATETIME,
                session_trial VARCHAR(100),
                pitch_speed_mph FLOAT,
                velocity_match_quality VARCHAR(20),
                
                # Muscle1 metrics
                muscle1_median_freq FLOAT,
                muscle1_mean_freq FLOAT,
                muscle1_bandwidth FLOAT,
                muscle1_peak_amplitude FLOAT,
                muscle1_peak_amplitude_pct_mvic FLOAT,
                muscle1_rms_value FLOAT,
                muscle1_rms_value_pct_mvic FLOAT,
                muscle1_rise_time FLOAT,
                muscle1_throw_integral FLOAT,
                muscle1_throw_integral_pct_mvic FLOAT,
                muscle1_work_rate FLOAT,
                
                # Muscle2 metrics
                muscle2_median_freq FLOAT,
                muscle2_mean_freq FLOAT,
                muscle2_bandwidth FLOAT,
                muscle2_peak_amplitude FLOAT,
                muscle2_peak_amplitude_pct_mvic FLOAT,
                muscle2_rms_value FLOAT,
                muscle2_rms_value_pct_mvic FLOAT,
                muscle2_rise_time FLOAT,
                muscle2_throw_integral FLOAT,
                muscle2_throw_integral_pct_mvic FLOAT,
                muscle2_work_rate FLOAT,
                
                # Spectral entropy metrics
                muscle1_spectral_entropy FLOAT,
                muscle2_spectral_entropy FLOAT,
                
                # Wavelet energy metrics
                muscle1_wavelet_energy_low FLOAT,
                muscle1_wavelet_energy_mid FLOAT,
                muscle1_wavelet_energy_high FLOAT,
                muscle2_wavelet_energy_low FLOAT,
                muscle2_wavelet_energy_mid FLOAT,
                muscle2_wavelet_energy_high FLOAT,
                
                # Coactivation metrics
                coactivation_index FLOAT,
                coactivation_correlation FLOAT,
                coactivation_temporal_overlap FLOAT,
                coactivation_waveform_similarity FLOAT,
                
                INDEX idx_session_id (emg_session_id),
                FOREIGN KEY (emg_session_id) REFERENCES emg_sessions(emg_session_id) ON DELETE CASCADE
            )
            """)
            
            logger.info("Database tables created/verified successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            return False
            
    def test_connection(self):
        """
        Test the database connection and tables.
        
        Returns:
        --------
        bool
            True if connection is successful and tables exist, False otherwise
        """
        if not self.connect():
            return False
        
        try:
            # Check if tables exist
            tables = self.execute_query(
                "SHOW TABLES LIKE 'emg_%'", 
                fetch=True
            )
            
            if not tables or len(tables) < 3:  # We should have at least 3 tables
                logger.warning("EMG tables not found. Creating tables...")
                self.create_tables()
                
            # Check again after potential creation
            tables = self.execute_query(
                "SHOW TABLES LIKE 'emg_%'", 
                fetch=True
            )
            
            if not tables or len(tables) < 3:
                logger.error("Failed to create or verify EMG tables")
                return False
                
            logger.info(f"Database connection test successful. Found {len(tables)} EMG tables.")
            return True
            
        except Exception as e:
            logger.error(f"Error testing database connection: {e}")
            return False
        finally:
            self.disconnect()

# For direct testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db = DBConnector()
    result = db.test_connection()
    print(f"Connection test {'successful' if result else 'failed'}")