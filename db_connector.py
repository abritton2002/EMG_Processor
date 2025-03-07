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
        Create the necessary tables for the EMG pipeline if they don't exist.
        """
        try:
            # Create EMG Sessions table with numeric ID
            self.execute_query("""
            CREATE TABLE IF NOT EXISTS emg_sessions (
                numeric_id INT AUTO_INCREMENT PRIMARY KEY,
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
                muscle3_name VARCHAR(50),
                muscle4_name VARCHAR(50),
                muscle1_id VARCHAR(20),
                muscle2_id VARCHAR(20),
                muscle3_id VARCHAR(20),
                muscle4_id VARCHAR(20),
                muscle1_fs FLOAT,
                muscle2_fs FLOAT,
                file_path VARCHAR(255),
                processed_date DATETIME
            )
            """)
            
            # Create raw time series data table with updated references
            self.execute_query("""
            CREATE TABLE IF NOT EXISTS emg_timeseries (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_numeric_id INT,
                time_point FLOAT,
                muscle1_emg FLOAT,
                muscle2_emg FLOAT,
                INDEX idx_session_id (session_numeric_id),
                FOREIGN KEY (session_numeric_id) REFERENCES emg_sessions(numeric_id) ON DELETE CASCADE
            )
            """)
            
            # Create throw details table with 'trial' instead of 'throw'
            self.execute_query("""
            CREATE TABLE IF NOT EXISTS emg_throws (
                throw_id INT AUTO_INCREMENT PRIMARY KEY,
                session_numeric_id INT,
                trial_number INT,
                start_time FLOAT,
                end_time FLOAT,
                duration FLOAT,
                
                /* Muscle1 metrics */
                muscle1_median_freq FLOAT,
                muscle1_mean_freq FLOAT,
                muscle1_bandwidth FLOAT,
                muscle1_peak_amplitude FLOAT,
                muscle1_rms_value FLOAT,
                muscle1_rise_time FLOAT,
                muscle1_throw_integral FLOAT,
                muscle1_work_rate FLOAT,
                
                /* Muscle2 metrics */
                muscle2_median_freq FLOAT,
                muscle2_mean_freq FLOAT,
                muscle2_bandwidth FLOAT,
                muscle2_peak_amplitude FLOAT,
                muscle2_rms_value FLOAT,
                muscle2_rise_time FLOAT,
                muscle2_throw_integral FLOAT,
                muscle2_work_rate FLOAT,
                
                INDEX idx_session_id (session_numeric_id),
                FOREIGN KEY (session_numeric_id) REFERENCES emg_sessions(numeric_id) ON DELETE CASCADE
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