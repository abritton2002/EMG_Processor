import os
import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend to prevent "Starting a Matplotlib GUI outside of the main thread" warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging
import datetime
from io import BytesIO
import pymysql
from dotenv import load_dotenv
from db_connector import DBConnector
from cycler import cycler
import matplotlib as mpl

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set matplotlib style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['axes.prop_cycle'] = cycler(color=['#2C3E50', '#E74C3C', '#3498DB', '#2ECC71', '#F39C12'])
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9

class EMGProfessionalReport:
    """Generates simple EMG reports showing raw EMG data with throw markers."""
    
    def __init__(self, db_config=None, output_dir="reports"):
        """Initialize the report generator."""
        self.db = DBConnector(db_config)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Simple EMG Report Generator initialized")
        
        # Set colors for report
        self.colors = {
            'primary': '#2C3E50',
            'secondary': '#3498DB',
            'accent1': '#E74C3C',
            'accent2': '#2ECC71',
            'accent3': '#F39C12',
            'light_gray': '#f8f9fa',
            'medium_gray': '#e9ecef',
            'dark_gray': '#6c757d',
            'text': '#343a40'
        }
        
        # Set muscle colors
        self.muscle_colors = {
            'FCU': '#2C3E50',  # Dark blue
            'FCR': '#E74C3C',  # Red
        }

    def _resolve_session_id(self, session_id):
        """Resolve a session ID to its numeric ID in the database."""
        if isinstance(session_id, int) or (isinstance(session_id, str) and session_id.isdigit()):
            return session_id
        with self.db.connect() as conn:
            if not conn:
                logger.error("Failed to connect to database")
                return None
            cursor = conn.cursor()
            cursor.execute("SELECT numeric_id FROM emg_sessions WHERE filename = %s", (session_id,))
            result = cursor.fetchone()
            if not result:
                logger.error(f"Session {session_id} not found")
                return None
            return result[0]

    def get_session_info(self, session_id):
        """Get session information from the database."""
        with self.db.connect() as conn:
            if not conn:
                logger.error("Failed to connect to database")
                return {}
            try:
                cursor = conn.cursor(pymysql.cursors.DictCursor)
                query = "SELECT * FROM emg_sessions WHERE numeric_id = %s" if \
                        (isinstance(session_id, int) or (isinstance(session_id, str) and session_id.isdigit())) \
                        else "SELECT * FROM emg_sessions WHERE filename = %s"
                cursor.execute(query, (session_id,))
                return cursor.fetchone() or {}
            except Exception as e:
                logger.error(f"Error retrieving session info for {session_id}: {e}")
                return {}

    def get_throws_for_session(self, session_id):
        """Get throw data for a session from the database."""
        numeric_id = self._resolve_session_id(session_id)
        if numeric_id is None:
            return pd.DataFrame()
        with self.db.connect() as conn:
            if not conn:
                logger.error("Failed to connect to database")
                return pd.DataFrame()
            try:
                query = "SELECT * FROM emg_throws WHERE session_numeric_id = %s ORDER BY trial_number"
                cursor = conn.cursor(pymysql.cursors.DictCursor)
                cursor.execute(query, (numeric_id,))
                throws = cursor.fetchall()
                df = pd.DataFrame(throws)
                if not df.empty:
                    df.rename(columns={'trial_number': 'throw_number', 'session_numeric_id': 'session_id'}, inplace=True)
                return df
            except Exception as e:
                logger.error(f"Error retrieving throws for session {session_id}: {e}")
                return pd.DataFrame()

    def get_timeseries_data(self, session_id, max_rows=5000000):
        """Get time series data for a session from the database."""
        numeric_id = self._resolve_session_id(session_id)
        if numeric_id is None:
            return pd.DataFrame()
        with self.db.connect() as conn:
            if not conn:
                logger.error("Failed to connect to database")
                return pd.DataFrame()
            try:
                query = "SELECT * FROM emg_timeseries WHERE session_numeric_id = %s ORDER BY time_point LIMIT %s"
                cursor = conn.cursor(pymysql.cursors.DictCursor)
                cursor.execute(query, (numeric_id, max_rows))
                timeseries = cursor.fetchall()
                df = pd.DataFrame(timeseries)
                if not df.empty:
                    df.rename(columns={'session_numeric_id': 'session_id'}, inplace=True)
                logger.info(f"Fetched {len(df)} rows for session {session_id}")
                return df
            except Exception as e:
                logger.error(f"Error retrieving time series data for session {session_id}: {e}")
                return pd.DataFrame()
                
    def create_raw_emg_plot(self, session_id):
        """Create a PDF report showing raw EMG data with throw regions marked."""
        try:
            # Get session data
            session_info = self.get_session_info(session_id)
            if not session_info:
                logger.error(f"Session {session_id} not found")
                return None
                
            # Get throws data
            throws_df = self.get_throws_for_session(session_id)
            if throws_df.empty:
                logger.warning(f"No throws found for session {session_id}")
                
            # Get time series data
            timeseries_df = self.get_timeseries_data(session_id)
            if timeseries_df.empty:
                logger.error(f"No time series data found for session {session_id}")
                return None
            
            # Extract important information
            athlete_name = session_info.get('athlete_name', 'Unknown Athlete')
            session_type = session_info.get('session_type', 'Unknown Session')
            session_date = session_info.get('date_recorded')
            
            # Format the date
            if isinstance(session_date, datetime.date) or isinstance(session_date, datetime.datetime):
                date_str = session_date.strftime('%b %d, %Y')
            else:
                date_str = str(session_date)
            
            # Extract muscle names
            muscle1_name = session_info.get('muscle1_name', 'FCU')
            muscle2_name = session_info.get('muscle2_name', 'FCR')
            
            # Set up colors for muscles
            muscle1_color = self.muscle_colors.get(muscle1_name, '#2C3E50')
            muscle2_color = self.muscle_colors.get(muscle2_name, '#E74C3C')
            
      
            plot_df = timeseries_df
            
            # Create output filename
            filename = session_info.get('filename', f"session_{session_id}")
            output_path = os.path.join(self.output_dir, f"{filename}_emg_report.pdf")
            
            # Create the PDF with matplotlib
            plt.figure(figsize=(11, 8.5), dpi=100)
            
            # Create figure with professional styling
            fig = plt.figure(figsize=(11, 8.5), dpi=100)
            gs = fig.add_gridspec(nrows=1, ncols=1, left=0.05, right=0.98, bottom=0.2, top=0.85)
            ax = fig.add_subplot(gs[0])
            
            # Plot EMG data - make sure to use correct column names
            muscle1_col = 'fcu_emg' if 'fcu_emg' in plot_df.columns else 'muscle1_emg'
            muscle2_col = 'fcr_emg' if 'fcr_emg' in plot_df.columns else 'muscle2_emg'
            time_col = 'time_point' if 'time_point' in plot_df.columns else 'time'
            
            ax.plot(plot_df[time_col], plot_df[muscle1_col], 
                   color=muscle1_color, linewidth=0.8, alpha=0.8, label=muscle1_name)
            ax.plot(plot_df[time_col], plot_df[muscle2_col], 
                   color=muscle2_color, linewidth=0.8, alpha=0.8, label=muscle2_name)
            
            # Add throw markers
            if not throws_df.empty:
                for i, throw in throws_df.iterrows():
                    # Shade throw region
                    ax.axvspan(throw['start_time'], throw['end_time'], 
                              alpha=0.15, color='#2ECC71', zorder=1)
                    
                    # Add throw number label
                    throw_num = throw['throw_number'] if 'throw_number' in throw else i+1
                    midpoint = (throw['start_time'] + throw['end_time']) / 2
                    ymax = ax.get_ylim()[1]
                    ax.text(midpoint, 0.95*ymax, f"#{int(throw_num)}", 
                           fontsize=8, color='#2C3E50', ha='center', va='top',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
            # Set up plot styling
            ax.set_xlabel('Time (seconds)', fontsize=11)
            ax.set_ylabel('EMG Amplitude (mV)', fontsize=11)
            ax.set_title('Raw EMG Signal', fontsize=14, pad=10, fontweight='bold', color='#2C3E50')
            
            # Add legend with custom styling
            legend = ax.legend(loc='upper right', frameon=True, fontsize=10)
            legend.get_frame().set_facecolor('#f8f9fa')
            legend.get_frame().set_edgecolor('#e9ecef')
            
            # Improve grid appearance
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add session info in the corner
            session_details = (
                f"Athlete: {athlete_name}\n"
                f"Session Type: {session_type}\n"
                f"Throws: {len(throws_df)}"
            )
            ax.text(0.02, 0.02, session_details, transform=ax.transAxes, fontsize=9,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='#e9ecef', boxstyle='round,pad=0.5'),
                   verticalalignment='bottom', horizontalalignment='left')
            
            # Add report title and subtitle
            fig.text(0.04, 0.925, "Electromyography Report", ha='left', va='center', fontsize=25, 
                    fontweight='bold', color=self.colors['primary'])
            
            fig.text(0.04, 0.875, "Raw EMG Signal with Throw Regions", ha='left', va='center', 
                    fontsize=18, fontweight='bold', color=self.colors['primary'])
            
            # Add footer
            fig.text(0.21, 0.03, f"{athlete_name}  -  {date_str}", fontsize=8, color=self.colors['dark_gray'])
            fig.text(0.46, 0.03, 'EMG Raw Signal Overview', fontsize=8, color=self.colors['dark_gray'])
            fig.text(0.93, 0.03, "Page 1", fontsize=8, color=self.colors['dark_gray'])
            
            # Add logo placeholder (as in your original code)
            logo_ax = fig.add_axes([0.04, 0.01, 0.08, 0.08], anchor='SE', zorder=-1)
            logo_ax.imshow(np.ones((100, 300, 4)))  # Placeholder logo
            logo_ax.axis('off')
            
            # Save the PDF
            plt.savefig(output_path)
            plt.close(fig)
            
            logger.info(f"Created EMG report: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create EMG report: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def generate_report(self, session_id):
        """Alias for create_raw_emg_plot to maintain API compatibility."""
        return self.create_raw_emg_plot(session_id)

if __name__ == "__main__":
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate Simple EMG Reports')
    parser.add_argument('session_id', help='Session ID to generate report for')
    parser.add_argument('--output-dir', default='reports', help='Directory to save reports')
    
    args = parser.parse_args()
    
    # Load environment variables for database connection
    load_dotenv()
    
    # Create database config from environment variables
    db_config = {
        'host': os.getenv('DB_HOST'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'database': os.getenv('DB_NAME')
    }
    
    # Create report generator
    report_generator = EMGProfessionalReport(db_config=db_config, output_dir=args.output_dir)
    
    # Generate report
    pdf_path = report_generator.create_raw_emg_plot(args.session_id)
    
    if pdf_path:
        print(f"Report generated successfully: {pdf_path}")
    else:
        print("Failed to generate report")
        exit(1)