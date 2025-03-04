import pymysql
import logging

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
            Database configuration dictionary
        """
        # Default database configuration
        self.db_config = config or {
            'host': '10.200.200.107',
            'user': 'scriptuser1',
            'password': 'YabinMarshed2023@#$',
            'database': 'theia_pitching_db'
        }
        
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
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # Create EMG Sessions table
            self.execute_query("""
            CREATE TABLE IF NOT EXISTS emg_sessions (
                session_id VARCHAR(100) PRIMARY KEY,
                date_recorded DATE,
                traq_id VARCHAR(50),
                athlete_name VARCHAR(100),
                session_type VARCHAR(50),
                fcu_fs FLOAT,
                fcr_fs FLOAT,
                file_path VARCHAR(255),
                processed_date DATETIME
            )
            """)
            
            # Create raw time series data table
            self.execute_query("""
            CREATE TABLE IF NOT EXISTS emg_timeseries (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(100),
                time_point FLOAT,
                fcu_emg FLOAT,
                fcr_emg FLOAT,
                INDEX idx_session_id (session_id),
                FOREIGN KEY (session_id) REFERENCES emg_sessions(session_id) ON DELETE CASCADE
            )
            """)
            
            # Create throw details table
            self.execute_query("""
            CREATE TABLE IF NOT EXISTS emg_throws (
                throw_id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(100),
                throw_number INT,
                start_time FLOAT,
                end_time FLOAT,
                duration FLOAT,
                
                /* FCR metrics */
                fcr_median_freq FLOAT,
                fcr_mean_freq FLOAT,
                fcr_bandwidth FLOAT,
                fcr_peak_amplitude FLOAT,
                fcr_rms_value FLOAT,
                fcr_rise_time FLOAT,
                fcr_contraction_time FLOAT,
                fcr_relaxation_time FLOAT,
                fcr_contraction_relaxation_ratio FLOAT,
                fcr_throw_integral FLOAT,
                fcr_work_rate FLOAT,
                
                /* FCU metrics */
                fcu_median_freq FLOAT,
                fcu_mean_freq FLOAT,
                fcu_bandwidth FLOAT,
                fcu_peak_amplitude FLOAT,
                fcu_rms_value FLOAT,
                fcu_rise_time FLOAT,
                fcu_contraction_time FLOAT,
                fcu_relaxation_time FLOAT,
                fcu_contraction_relaxation_ratio FLOAT,
                fcu_throw_integral FLOAT,
                fcu_work_rate FLOAT,
                
                INDEX idx_session_id (session_id),
                FOREIGN KEY (session_id) REFERENCES emg_sessions(session_id) ON DELETE CASCADE
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