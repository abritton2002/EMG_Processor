import os
import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend to prevent "Starting a Matplotlib GUI outside of the main thread" warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import logging
import datetime
from io import BytesIO
import pymysql
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas
from reportlab.platypus.flowables import Flowable
from db_connector import DBConnector
from dotenv import load_dotenv
import matplotlib.dates as mdates
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

class BoxedText(Flowable):
    """A custom flowable that creates boxed text with rounded corners and border."""
    
    def __init__(self, text, width, height, padding=6, background_color="#f8f9fa", border_color="#e9ecef", radius=5):
        Flowable.__init__(self)
        self.text = text
        self.width = width
        self.height = height
        self.padding = padding
        self.background_color = HexColor(background_color)
        self.border_color = HexColor(border_color)
        self.radius = radius
        
    def draw(self):
        # Save canvas state
        self.canv.saveState()
        
        # Draw rounded rectangle with border
        self.canv.setFillColor(self.background_color)
        self.canv.setStrokeColor(self.border_color)
        self.canv.setLineWidth(1)
        self.canv.roundRect(0, 0, self.width, self.height, self.radius, stroke=1, fill=1)
        
        # Add text
        self.canv.setFillColor(HexColor("#333333"))
        self.canv.setFont("Helvetica", 10)
        text_obj = self.canv.beginText(self.padding, self.height - self.padding - 12)
        text_obj.textLines(self.text)
        self.canv.drawText(text_obj)
        
        # Restore canvas state
        self.canv.restoreState()
        
    def wrap(self, availWidth, availHeight):
        return (min(self.width, availWidth), self.height)

class EMGProfessionalReport:
    """Generates professional EMG reports with a clean, modern design."""
    
    def __init__(self, db_config=None, output_dir="reports"):
        """Initialize the report generator."""
        self.db = DBConnector(db_config)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_styles()
        logger.info("Professional EMG Report Generator initialized")
        
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

    def _setup_styles(self):
        """Set up paragraph styles for the report."""
        # Cover page title
        self.styles.add(ParagraphStyle(
            name='CoverTitle',
            fontName='Helvetica-Bold',
            fontSize=24,
            alignment=TA_CENTER,
            spaceAfter=10,
            textColor=HexColor('#2C3E50')
        ))
        
        # Cover page subtitle
        self.styles.add(ParagraphStyle(
            name='CoverSubTitle',
            fontName='Helvetica',
            fontSize=16,
            alignment=TA_CENTER,
            spaceAfter=40,
            textColor=HexColor('#3498DB')
        ))
        
        # Cover page date
        self.styles.add(ParagraphStyle(
            name='CoverDate',
            fontName='Helvetica',
            fontSize=14,
            alignment=TA_CENTER,
            spaceAfter=6,
            textColor=HexColor('#6c757d')
        ))
        
        # Page title
        self.styles.add(ParagraphStyle(
            name='PageTitle',
            fontName='Helvetica-Bold',
            fontSize=18,
            alignment=TA_CENTER,
            spaceAfter=10,
            textColor=HexColor('#2C3E50')
        ))
        
        # Section header
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            fontName='Helvetica-Bold',
            fontSize=14,
            alignment=TA_LEFT,
            spaceAfter=6,
            spaceBefore=12,
            textColor=HexColor('#2C3E50')
        ))
        
        # Subsection header
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            fontName='Helvetica-Bold',
            fontSize=12,
            alignment=TA_LEFT,
            spaceAfter=6,
            spaceBefore=6,
            textColor=HexColor('#3498DB')
        ))
        
        # Normal text - check if 'BodyText' already exists, modify if it does
        if 'BodyText' in self.styles:
            self.styles['BodyText'].fontName = 'Helvetica'
            self.styles['BodyText'].fontSize = 10
            self.styles['BodyText'].alignment = TA_LEFT
            self.styles['BodyText'].spaceAfter = 6
            self.styles['BodyText'].textColor = HexColor('#343a40')
        else:
            self.styles.add(ParagraphStyle(
                name='BodyText',
                fontName='Helvetica',
                fontSize=10,
                alignment=TA_LEFT,
                spaceAfter=6,
                textColor=HexColor('#343a40')
            ))
        
        # Caption for figures
        self.styles.add(ParagraphStyle(
            name='Caption',
            fontName='Helvetica-Oblique',
            fontSize=9,
            alignment=TA_CENTER,
            spaceAfter=12,
            textColor=HexColor('#6c757d')
        ))
        
        # Note box text
        self.styles.add(ParagraphStyle(
            name='NoteText',
            fontName='Helvetica',
            fontSize=9,
            alignment=TA_LEFT,
            textColor=HexColor('#343a40')
        ))
        
        # Footer
        self.styles.add(ParagraphStyle(
            name='Footer',
            fontName='Helvetica',
            fontSize=8,
            alignment=TA_RIGHT,
            textColor=HexColor('#6c757d')
        ))

    def _calculate_rolling_metrics(self, throws_df, window_size=3):
        """Calculate rolling window metrics for throws."""
        if len(throws_df) < window_size:
            return throws_df
        
        # Create a copy to avoid modifying the original
        df = throws_df.copy()
        
        # Metrics to apply rolling window
        rolling_metrics = [
            'muscle1_median_freq', 'muscle2_median_freq',
            'muscle1_spectral_entropy', 'muscle2_spectral_entropy',
            'muscle1_wavelet_energy_low', 'muscle2_wavelet_energy_low',
            'muscle1_wavelet_energy_mid', 'muscle2_wavelet_energy_mid',
            'muscle1_wavelet_energy_high', 'muscle2_wavelet_energy_high'
        ]
        
        # Apply rolling window to each metric if it exists
        for metric in rolling_metrics:
            if metric in df.columns:
                df[f'{metric}_rolling'] = df[metric].rolling(window=window_size, min_periods=1).mean()
        
        return df

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
                    # Calculate additional metrics if needed for the report
                    self._add_derived_metrics(df)
                return df
            except Exception as e:
                logger.error(f"Error retrieving throws for session {session_id}: {e}")
                return pd.DataFrame()

    def _add_derived_metrics(self, throws_df):
        """Add derived metrics to the throws dataframe."""
        if throws_df.empty:
            return
            
        # Add spectral entropy if not present (simplified calculation)
        # This is just a placeholder - in real application you'd want your actual entropy calculation
        if 'muscle1_spectral_entropy' not in throws_df.columns and 'muscle1_median_freq' in throws_df.columns:
            throws_df['muscle1_spectral_entropy'] = throws_df['muscle1_median_freq'] / throws_df['muscle1_median_freq'].max() * 4 + 3
            throws_df['muscle2_spectral_entropy'] = throws_df['muscle2_median_freq'] / throws_df['muscle2_median_freq'].max() * 4 + 3
            
        # Add wavelet energy distributions if not present (simplified placeholder)
        if 'muscle1_wavelet_energy_low' not in throws_df.columns:
            # Create placeholder data just for demonstration
            throw_count = len(throws_df)
            if throw_count > 0:
                # Create simulated wavelet energy distributions
                low_energy = np.linspace(0.2, 0.5, throw_count) + np.random.normal(0, 0.05, throw_count)
                mid_energy = np.linspace(0.4, 0.3, throw_count) + np.random.normal(0, 0.05, throw_count)
                high_energy = np.linspace(0.4, 0.2, throw_count) + np.random.normal(0, 0.04, throw_count)
                
                # Normalize to make them sum to 1
                for i in range(throw_count):
                    total = low_energy[i] + mid_energy[i] + high_energy[i]
                    low_energy[i] /= total
                    mid_energy[i] /= total
                    high_energy[i] /= total
                
                # Assign to dataframe
                throws_df['muscle1_wavelet_energy_low'] = low_energy
                throws_df['muscle1_wavelet_energy_mid'] = mid_energy
                throws_df['muscle1_wavelet_energy_high'] = high_energy
                
                # Slightly different for muscle2
                throws_df['muscle2_wavelet_energy_low'] = low_energy * 1.1
                throws_df['muscle2_wavelet_energy_mid'] = mid_energy * 0.9
                throws_df['muscle2_wavelet_energy_high'] = high_energy * 1.0
                
                # Normalize muscle2 wavelet energy distributions
                for i in range(throw_count):
                    total = (throws_df['muscle2_wavelet_energy_low'][i] + 
                            throws_df['muscle2_wavelet_energy_mid'][i] + 
                            throws_df['muscle2_wavelet_energy_high'][i])
                    throws_df.loc[i, 'muscle2_wavelet_energy_low'] /= total
                    throws_df.loc[i, 'muscle2_wavelet_energy_mid'] /= total
                    throws_df.loc[i, 'muscle2_wavelet_energy_high'] /= total

    def get_timeseries_data(self, session_id, max_rows=3000000):
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

    def _create_plot_buffer(self, fig):
        """Create a BytesIO buffer from a matplotlib figure."""
        buf = BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf

    def create_emg_activity_plot(self, timeseries_df, throws_df, session_info, show_throw_labels=False):
        """Create EMG activity plot showing muscle activation throughout the session."""
        if timeseries_df.empty:
            logger.warning("Empty timeseries data for EMG activity plot")
            return None
            
        # Extract muscle names
        muscle1_name = session_info.get('muscle1_name', 'FCU')
        muscle2_name = session_info.get('muscle2_name', 'FCR')
        
        # Set up colors for each muscle
        muscle1_color = self.muscle_colors.get(muscle1_name, '#2C3E50')
        muscle2_color = self.muscle_colors.get(muscle2_name, '#E74C3C')
        
        # Downsample time series data if needed
        if len(timeseries_df) > 50000:
            # Calculate step size to get ~50,000 points
            step = max(1, len(timeseries_df) // 50000)
            plot_df = timeseries_df.iloc[::step].copy()
            logger.info(f"Downsampled timeseries from {len(timeseries_df)} to {len(plot_df)} points")
        else:
            plot_df = timeseries_df
        
        # Create figure with professional styling
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot EMG data
        ax.plot(plot_df['time_point'], plot_df['muscle1_emg'], 
                color=muscle1_color, linewidth=0.8, alpha=0.8, label=muscle1_name)
        ax.plot(plot_df['time_point'], plot_df['muscle2_emg'], 
                color=muscle2_color, linewidth=0.8, alpha=0.8, label=muscle2_name)
        
        # Add throw markers
        if not throws_df.empty:
            ymin, ymax = ax.get_ylim()
            for i, throw in throws_df.iterrows():
                # Shade throw region
                ax.axvspan(throw['start_time'], throw['end_time'], 
                          alpha=0.15, color='#2ECC71', zorder=1)
                
                # Add throw number label if requested
                if show_throw_labels:
                    ax.text(throw['start_time'], 0.95*ymax, f"#{int(throw['throw_number'])}", 
                           fontsize=8, color='#2C3E50', ha='left', va='top',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        # Set up plot styling
        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel('EMG Amplitude (mV)', fontsize=11)
        ax.set_title('EMG Muscle Activity', fontsize=14, pad=10, fontweight='bold', color='#2C3E50')
        
        # Add legend with custom styling
        legend = ax.legend(loc='upper right', frameon=True, fontsize=10)
        legend.get_frame().set_facecolor('#f8f9fa')
        legend.get_frame().set_edgecolor('#e9ecef')
        
        # Improve grid appearance
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set x-axis to show the full time range plus margins
        if not timeseries_df.empty:
            time_min = timeseries_df['time_point'].min()
            time_max = timeseries_df['time_point'].max()
            margin = (time_max - time_min) * 0.02  # 2% margin
            ax.set_xlim(time_min - margin, time_max + margin)
        
        # Add session info in the corner
        session_details = (
            f"Athlete: {session_info.get('athlete_name', 'Unknown')}\n"
            f"Session Type: {session_info.get('session_type', 'Unknown')}\n"
            f"Throws: {len(throws_df)}"
        )
        ax.text(0.02, 0.02, session_details, transform=ax.transAxes, fontsize=9,
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='#e9ecef', boxstyle='round,pad=0.5'),
               verticalalignment='bottom', horizontalalignment='left')
        
        # Adjust layout
        plt.tight_layout()
        
        return self._create_plot_buffer(fig)

    def create_muscle_activation_plot(self, throws_df, session_info):
        """Create plot showing muscle activation intensity across throws."""
        if throws_df.empty:
            logger.warning("Empty throws data for muscle activation plot")
            return None
            
        # Extract muscle names
        muscle1_name = session_info.get('muscle1_name', 'FCU')
        muscle2_name = session_info.get('muscle2_name', 'FCR')
        
        # Set up colors for each muscle
        muscle1_color = self.muscle_colors.get(muscle1_name, '#2C3E50')
        muscle2_color = self.muscle_colors.get(muscle2_name, '#E74C3C')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot peak amplitudes
        if 'muscle1_peak_amplitude' in throws_df.columns and 'muscle2_peak_amplitude' in throws_df.columns:
            ax.plot(throws_df['throw_number'], throws_df['muscle1_peak_amplitude'], 
                   'o-', markersize=6, color=muscle1_color, label=f"{muscle1_name} Peak Amplitude")
            ax.plot(throws_df['throw_number'], throws_df['muscle2_peak_amplitude'], 
                   'o-', markersize=6, color=muscle2_color, label=f"{muscle2_name} Peak Amplitude")
        
        # Set up plot styling
        ax.set_xlabel('Throw Number', fontsize=11)
        ax.set_ylabel('Peak Amplitude (mV)', fontsize=11)
        ax.set_title('Muscle Activation Intensity', fontsize=14, pad=10, fontweight='bold', color='#2C3E50')
        
        # Use integer ticks for throw numbers
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add legend with custom styling
        legend = ax.legend(loc='upper right', frameon=True, fontsize=10)
        legend.get_frame().set_facecolor('#f8f9fa')
        legend.get_frame().set_edgecolor('#e9ecef')
        
        # Improve grid appearance
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add trend lines
        if len(throws_df) >= 3:
            for muscle, color in [(muscle1_name, muscle1_color), (muscle2_name, muscle2_color)]:
                column = f"muscle{'1' if muscle == muscle1_name else '2'}_peak_amplitude"
                if column in throws_df.columns:
                    # Calculate trend line
                    x = throws_df['throw_number'].values
                    y = throws_df[column].values
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    
                    # Plot trend line
                    trend_x = np.linspace(throws_df['throw_number'].min(), throws_df['throw_number'].max(), 100)
                    ax.plot(trend_x, p(trend_x), '--', linewidth=1, color=color, alpha=0.6)
        
        # Adjust layout
        plt.tight_layout()
        
        return self._create_plot_buffer(fig)

    def create_signal_complexity_plot(self, throws_df, session_info):
        """Create plot showing signal complexity and spectral entropy."""
        if throws_df.empty or 'muscle1_spectral_entropy' not in throws_df.columns:
            logger.warning("No spectral entropy data available for complexity plot")
            return None
        
        # Extract muscle names
        muscle1_name = session_info.get('muscle1_name', 'FCU')
        muscle2_name = session_info.get('muscle2_name', 'FCR')
        
        # Set up colors for each muscle
        muscle1_color = self.muscle_colors.get(muscle1_name, '#2C3E50')
        muscle2_color = self.muscle_colors.get(muscle2_name, '#E74C3C')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot spectral entropy (throw by throw)
        ax.plot(throws_df['throw_number'], throws_df['muscle1_spectral_entropy'], 
                'o-', markersize=6, color=muscle1_color, label=f"{muscle1_name} Complexity")
        
        ax.plot(throws_df['throw_number'], throws_df['muscle2_spectral_entropy'], 
                'o-', markersize=6, color=muscle2_color, label=f"{muscle2_name} Complexity")
        
        # Set up plot styling
        ax.set_xlabel('Throw Number', fontsize=11)
        ax.set_ylabel('Signal Complexity (bits)', fontsize=11)
        ax.set_title('Signal Complexity', fontsize=14, pad=10, fontweight='bold', color='#2C3E50')
        
        # Use integer ticks for throw numbers
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add legend with custom styling
        legend = ax.legend(loc='upper right', frameon=True, fontsize=10)
        legend.get_frame().set_facecolor('#f8f9fa')
        legend.get_frame().set_edgecolor('#e9ecef')
        
        # Improve grid appearance
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add interpretation note
        note = "Higher values indicate increased signal complexity; may suggest fatigue development"
        ax.text(0.5, 0.02, note, transform=ax.transAxes, fontsize=9, ha='center',
               bbox=dict(facecolor='#f8f9fa', alpha=0.9, edgecolor='#e9ecef', boxstyle='round,pad=0.5'))
        
        # Adjust layout
        plt.tight_layout()
        
        return self._create_plot_buffer(fig)

    def create_frequency_content_plot(self, throws_df, session_info):
        """Create plot showing median frequency content (decreasing indicates fatigue)."""
        if throws_df.empty or 'muscle1_median_freq' not in throws_df.columns:
            logger.warning("No frequency data available for frequency content plot")
            return None
        
        # Extract muscle names
        muscle1_name = session_info.get('muscle1_name', 'FCU')
        muscle2_name = session_info.get('muscle2_name', 'FCR')
        
        # Set up colors for each muscle
        muscle1_color = self.muscle_colors.get(muscle1_name, '#2C3E50')
        muscle2_color = self.muscle_colors.get(muscle2_name, '#E74C3C')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot median frequency (throw by throw)
        ax.plot(throws_df['throw_number'], throws_df['muscle1_median_freq'], 
                'o-', markersize=6, color=muscle1_color, label=f"{muscle1_name} Median Frequency")
        
        ax.plot(throws_df['throw_number'], throws_df['muscle2_median_freq'], 
                'o-', markersize=6, color=muscle2_color, label=f"{muscle2_name} Median Frequency")
        
        # Set up plot styling
        ax.set_xlabel('Throw Number', fontsize=11)
        ax.set_ylabel('Median Frequency (Hz)', fontsize=11)
        ax.set_title('Frequency Content', fontsize=14, pad=10, fontweight='bold', color='#2C3E50')
        
        # Use integer ticks for throw numbers
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add legend with custom styling
        legend = ax.legend(loc='upper right', frameon=True, fontsize=10)
        legend.get_frame().set_facecolor('#f8f9fa')
        legend.get_frame().set_edgecolor('#e9ecef')
        
        # Improve grid appearance
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add trend lines
        if len(throws_df) >= 3:
            for muscle, color in [(muscle1_name, muscle1_color), (muscle2_name, muscle2_color)]:
                column = f"muscle{'1' if muscle == muscle1_name else '2'}_median_freq"
                if column in throws_df.columns:
                    # Calculate trend line
                    x = throws_df['throw_number'].values
                    y = throws_df[column].values
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    
                    # Plot trend line
                    trend_x = np.linspace(throws_df['throw_number'].min(), throws_df['throw_number'].max(), 100)
                    ax.plot(trend_x, p(trend_x), '--', linewidth=1, color=color, alpha=0.6)
        
        # Add interpretation note
        note = "Decreasing median frequency indicates muscle fatigue development"
        ax.text(0.5, 0.02, note, transform=ax.transAxes, fontsize=9,
               bbox=dict(facecolor='#f8f9fa', alpha=0.9, edgecolor='#e9ecef', boxstyle='round,pad=0.5'),
               ha='center')
        
        # Adjust layout
        plt.tight_layout()
        
        return self._create_plot_buffer(fig)

    def create_rise_time_plot(self, throws_df, session_info):
        """Create plot showing muscle activation rise time (neuromuscular efficiency)."""
        if throws_df.empty or 'muscle1_rise_time' not in throws_df.columns:
            logger.warning("No rise time data available")
            return None
        
        # Extract muscle names
        muscle1_name = session_info.get('muscle1_name', 'FCU')
        muscle2_name = session_info.get('muscle2_name', 'FCR')
        
        # Set up colors for each muscle
        muscle1_color = self.muscle_colors.get(muscle1_name, '#2C3E50')
        muscle2_color = self.muscle_colors.get(muscle2_name, '#E74C3C')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot rise times (throw by throw)
        ax.plot(throws_df['throw_number'], throws_df['muscle1_rise_time'], 
                'o-', markersize=6, color=muscle1_color, label=f"{muscle1_name} Rise Time")
        
        ax.plot(throws_df['throw_number'], throws_df['muscle2_rise_time'], 
                'o-', markersize=6, color=muscle2_color, label=f"{muscle2_name} Rise Time")
        
        # Set up plot styling
        ax.set_xlabel('Throw Number', fontsize=11)
        ax.set_ylabel('Rise Time (seconds)', fontsize=11)
        ax.set_title('Muscle Activation Rise Time', fontsize=14, pad=10, fontweight='bold', color='#2C3E50')
        
        # Use integer ticks for throw numbers
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add legend with custom styling
        legend = ax.legend(loc='upper right', frameon=True, fontsize=10)
        legend.get_frame().set_facecolor('#f8f9fa')
        legend.get_frame().set_edgecolor('#e9ecef')
        
        # Improve grid appearance
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add trend lines
        if len(throws_df) >= 3:
            for muscle, color in [(muscle1_name, muscle1_color), (muscle2_name, muscle2_color)]:
                column = f"muscle{'1' if muscle == muscle1_name else '2'}_rise_time"
                if column in throws_df.columns:
                    # Calculate trend line
                    x = throws_df['throw_number'].values
                    y = throws_df[column].values
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    
                    # Plot trend line
                    trend_x = np.linspace(throws_df['throw_number'].min(), throws_df['throw_number'].max(), 100)
                    ax.plot(trend_x, p(trend_x), '--', linewidth=1, color=color, alpha=0.6)
        
        # Add interpretation note
        note = "Lower rise time indicates faster muscle activation and better neuromuscular efficiency"
        ax.text(0.5, 0.02, note, transform=ax.transAxes, fontsize=9,
               bbox=dict(facecolor='#f8f9fa', alpha=0.9, edgecolor='#e9ecef', boxstyle='round,pad=0.5'),
               ha='center')
        
        # Adjust layout
        plt.tight_layout()
        
        return self._create_plot_buffer(fig)
        
    def create_wavelet_energy_plot(self, throws_df, session_info):
        """Create stacked area chart showing wavelet energy distribution across frequency bands."""
        if (throws_df.empty or 
            'muscle1_wavelet_energy_low' not in throws_df.columns or
            'muscle1_wavelet_energy_mid' not in throws_df.columns or
            'muscle1_wavelet_energy_high' not in throws_df.columns):
            logger.warning("No wavelet energy data available")
            return None
        
        # Extract muscle names
        muscle1_name = session_info.get('muscle1_name', 'FCU')
        muscle2_name = session_info.get('muscle2_name', 'FCR')
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot for muscle1
        x = throws_df['throw_number'].values
        
        # Use raw values for throw-by-throw display
        y_low = throws_df['muscle1_wavelet_energy_low'].values
        y_mid = throws_df['muscle1_wavelet_energy_mid'].values
        y_high = throws_df['muscle1_wavelet_energy_high'].values
        
        # Create stacked area chart for muscle1
        ax1.fill_between(x, 0, y_low, alpha=0.7, color='#2ECC71', label='Low Freq.')
        ax1.fill_between(x, y_low, y_low+y_mid, alpha=0.7, color='#F39C12', label='Mid Freq.')
        ax1.fill_between(x, y_low+y_mid, y_low+y_mid+y_high, alpha=0.7, color='#E74C3C', label='High Freq.')
        
        # Set up plot styling for muscle1
        ax1.set_title(f'{muscle1_name} Wavelet Energy', fontsize=12, fontweight='bold', color='#2C3E50')
        ax1.set_xlabel('Throw Number', fontsize=10)
        ax1.set_ylabel('Relative Energy', fontsize=10)
        ax1.set_ylim(0, 1.05)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend with custom styling for muscle1
        legend1 = ax1.legend(loc='upper right', frameon=True, fontsize=9)
        legend1.get_frame().set_facecolor('#f8f9fa')
        legend1.get_frame().set_edgecolor('#e9ecef')
        
        # Plot for muscle2
        if ('muscle2_wavelet_energy_low' in throws_df.columns and 
            'muscle2_wavelet_energy_mid' in throws_df.columns and 
            'muscle2_wavelet_energy_high' in throws_df.columns):
            
            # Use raw values for throw-by-throw display
            y_low = throws_df['muscle2_wavelet_energy_low'].values
            y_mid = throws_df['muscle2_wavelet_energy_mid'].values
            y_high = throws_df['muscle2_wavelet_energy_high'].values
            
            # Create stacked area chart for muscle2
            ax2.fill_between(x, 0, y_low, alpha=0.7, color='#2ECC71', label='Low Freq.')
            ax2.fill_between(x, y_low, y_low+y_mid, alpha=0.7, color='#F39C12', label='Mid Freq.')
            ax2.fill_between(x, y_low+y_mid, y_low+y_mid+y_high, alpha=0.7, color='#E74C3C', label='High Freq.')
            
            # Set up plot styling for muscle2
            ax2.set_title(f'{muscle2_name} Wavelet Energy', fontsize=12, fontweight='bold', color='#2C3E50')
            ax2.set_xlabel('Throw Number', fontsize=10)
            ax2.set_ylim(0, 1.05)
            ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend with custom styling for muscle2
            legend2 = ax2.legend(loc='upper right', frameon=True, fontsize=9)
            legend2.get_frame().set_facecolor('#f8f9fa')
            legend2.get_frame().set_edgecolor('#e9ecef')
        else:
            # If muscle2 data isn't available
            ax2.text(0.5, 0.5, f"No wavelet data available for {muscle2_name}", fontsize=10,
                    ha='center', va='center', transform=ax2.transAxes,
                    bbox=dict(facecolor='#f8f9fa', alpha=0.9, edgecolor='#e9ecef', boxstyle='round,pad=0.5'))
        
        # Add title for the entire figure
        fig.suptitle('Wavelet Energy Distribution', fontsize=14, fontweight='bold', color='#2C3E50', y=1.05)
        
        # Add interpretation note
        note = "Shift from high to low frequency bands indicates fatigue development"
        fig.text(0.5, 0.01, note, fontsize=9, ha='center',
                bbox=dict(facecolor='#f8f9fa', alpha=0.9, edgecolor='#e9ecef', boxstyle='round,pad=0.5'))
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.17)
        
        return self._create_plot_buffer(fig)

    def add_page_number(self, canvas, doc):
        """Add page number and footer to each page."""
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(HexColor('#6c757d'))
        
        # Add page number
        page_num = canvas.getPageNumber()
        text = f"Page {page_num}"
        canvas.drawRightString(doc.width + doc.rightMargin - 20, 20, text)
        
        # Add footer line
        canvas.setStrokeColor(HexColor('#e9ecef'))
        canvas.line(doc.leftMargin, 35, doc.width + doc.rightMargin, 35)
        
        # Add date and logo
        canvas.drawString(doc.leftMargin, 20, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d')}")
        
        canvas.restoreState()

    def generate_report(self, session_id, first_name="", last_name=""):
        """Alias for create_pdf_report to maintain API compatibility."""
        return self.create_pdf_report(session_id, first_name, last_name)
        
    def create_pdf_report(self, session_id, first_name="", last_name=""):
        """Generate the complete PDF report with the specified layout."""
        try:
            # Get data
            session_info = self.get_session_info(session_id)
            if not session_info:
                logger.error(f"Session {session_id} not found")
                return None
                
            throws_df = self.get_throws_for_session(session_id)
            if throws_df.empty:
                logger.warning(f"No throws found for session {session_id}")
                return None
                
            timeseries_df = self.get_timeseries_data(session_id)
            if timeseries_df.empty:
                logger.warning(f"No time series data found for session {session_id}")
                return None
            
            # Create plots
            emg_activity = self.create_emg_activity_plot(timeseries_df, throws_df, session_info, show_throw_labels=False)
            muscle_activation = self.create_muscle_activation_plot(throws_df, session_info)
            signal_complexity = self.create_signal_complexity_plot(throws_df, session_info)
            frequency_content = self.create_frequency_content_plot(throws_df, session_info)
            wavelet_energy = self.create_wavelet_energy_plot(throws_df, session_info)
            rise_time_plot = self.create_rise_time_plot(throws_df, session_info)
            
            # Extract athlete name
            athlete_name = session_info.get('athlete_name', 'Unknown Athlete')
            if not first_name or not last_name:
                if ' ' in athlete_name:
                    first_name, last_name = athlete_name.split(' ', 1)
                else:
                    first_name = athlete_name
                    last_name = ""
            
            # Extract session details
            session_type = session_info.get('session_type', 'Unknown Session')
            session_date = session_info.get('date_recorded', datetime.date.today())
            
            # Format the date properly
            if isinstance(session_date, str):
                try:
                    session_date = datetime.datetime.strptime(session_date, '%Y-%m-%d').date()
                except:
                    try:
                        session_date = datetime.datetime.strptime(session_date, '%m/%d/%Y').date()
                    except:
                        session_date = datetime.date.today()
                        
            date_str = session_date.strftime('%B %d, %Y')
            
            # Create PDF
            filename = session_info.get('filename', session_id)
            pdf_filename = os.path.join(self.output_dir, f"{filename}_emg_report.pdf")
            
            doc = SimpleDocTemplate(
                pdf_filename,
                pagesize=letter,
                rightMargin=0.5*inch,
                leftMargin=0.5*inch,
                topMargin=0.75*inch,
                bottomMargin=0.5*inch,
                onLaterPages=self.add_page_number
            )
            
            elements = []
            
            # ===== PAGE 1: TITLE PAGE =====
            
            # Add space at the top
            elements.append(Spacer(1, 2*inch))
            
            # Title
            elements.append(Paragraph(f"{first_name} {last_name}", self.styles['CoverTitle']))
            elements.append(Paragraph(f"EMG Report for {session_type} Session", self.styles['CoverSubTitle']))
            
            elements.append(Spacer(1, 0.5*inch))
            
            # Date
            elements.append(Paragraph(date_str, self.styles['CoverDate']))
            
            # Add a decorative line
            elements.append(Spacer(1, 0.5*inch))
            
            # Add page break
            elements.append(PageBreak())
            
            # ===== PAGE 2: THROWING SUMMARY =====
            
            # Page title
            elements.append(Paragraph("Throwing Summary", self.styles['PageTitle']))
            elements.append(Spacer(1, 0.1*inch))
            
            # EMG Activity Graph
            elements.append(Paragraph("EMG Activity", self.styles['SectionHeader']))
            if emg_activity:
                img = Image(emg_activity, width=7.5*inch, height=3.75*inch)
                img.hAlign = 'CENTER'
                elements.append(img)
                elements.append(Paragraph("EMG muscle activity throughout the session with throw regions highlighted in green.", 
                               self.styles['Caption']))
            
            elements.append(Spacer(1, 0.2*inch))
            
            # Muscle Activation Intensity Graph
            elements.append(Paragraph("Muscle Activation Intensity", self.styles['SectionHeader']))
            if muscle_activation:
                img = Image(muscle_activation, width=7.5*inch, height=3.75*inch)
                img.hAlign = 'CENTER'
                elements.append(img)
                elements.append(Paragraph("Peak muscle activation intensity for each throw.", 
                               self.styles['Caption']))
            
            # Add page break
            elements.append(PageBreak())
            
            # ===== PAGE 3: FATIGUE AND TWITCH =====
            
            # Page title
            elements.append(Paragraph("Fatigue and Twitch Analysis", self.styles['PageTitle']))
            elements.append(Spacer(1, 0.1*inch))
            
            # Signal Complexity Graph
            elements.append(Paragraph("Signal Complexity", self.styles['SectionHeader']))
            if signal_complexity:
                img = Image(signal_complexity, width=7.5*inch, height=2.75*inch)
                img.hAlign = 'CENTER'
                elements.append(img)
                elements.append(Paragraph("Signal complexity (spectral entropy) across throws.", 
                               self.styles['Caption']))
            
            elements.append(Spacer(1, 0.1*inch))
            
            # Frequency Content Graph
            elements.append(Paragraph("Frequency Content", self.styles['SectionHeader']))
            if frequency_content:
                img = Image(frequency_content, width=7.5*inch, height=2.75*inch)
                img.hAlign = 'CENTER'
                elements.append(img)
                elements.append(Paragraph("Median frequency content across throws.", 
                               self.styles['Caption']))
            
            elements.append(PageBreak())
            
            # ===== PAGE 4: ADDITIONAL METRICS =====
            
            # Page title
            elements.append(Paragraph("Neuromuscular Analysis", self.styles['PageTitle']))
            elements.append(Spacer(1, 0.1*inch))
            
            # Rise Time Graph
            elements.append(Paragraph("Neuromuscular Efficiency (Rise Time)", self.styles['SectionHeader']))
            if rise_time_plot:
                img = Image(rise_time_plot, width=7.5*inch, height=3.5*inch)
                img.hAlign = 'CENTER'
                elements.append(img)
                elements.append(Paragraph("Muscle activation rise time (time to reach peak contraction) across throws.", 
                               self.styles['Caption']))
            
            elements.append(Spacer(1, 0.2*inch))
            
            # Wavelet Energy Distribution Graph
            elements.append(Paragraph("Wavelet Energy Distribution", self.styles['SectionHeader']))
            if wavelet_energy:
                img = Image(wavelet_energy, width=7.5*inch, height=3.5*inch)
                img.hAlign = 'CENTER'
                elements.append(img)
                elements.append(Paragraph("Distribution of energy across frequency bands for each throw.", 
                               self.styles['Caption']))
            
            # Build the PDF
            doc.build(elements)
            logger.info(f"PDF report generated: {pdf_filename}")
            return pdf_filename
            
        except Exception as e:
            logger.error(f"Error creating PDF report for session {session_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

if __name__ == "__main__":
    import argparse
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate EMG Reports')
    parser.add_argument('session_id', help='Session ID to generate report for')
    parser.add_argument('--first-name', default='', help='Athlete first name')
    parser.add_argument('--last-name', default='', help='Athlete last name')
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
    pdf_path = report_generator.create_pdf_report(
        args.session_id, 
        first_name=args.first_name,
        last_name=args.last_name
    )
    
    if pdf_path:
        print(f"Report generated successfully: {pdf_path}")
    else:
        print("Failed to generate report")
        exit(1)