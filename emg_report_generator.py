import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import datetime
from io import BytesIO
import pymysql
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.colors import HexColor
from db_connector import DBConnector
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

class SimpleEMGReportGenerator:
    """Generates simple, easy-to-understand PDF reports for EMG muscle activity data."""
    
    def __init__(self, db_config=None, output_dir="reports"):
        """Initialize the report generator."""
        self.db = DBConnector(db_config)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_styles()
        logger.info("Simple EMG Report Generator initialized")

    def _setup_styles(self):
        """Set up paragraph styles for the report."""
        # Title styles - avoid using 'Title' as it's already defined
        self.styles.add(ParagraphStyle(
            name='ReportTitle', 
            parent=self.styles['Title'],
            fontSize=18, 
            alignment=TA_CENTER, 
            spaceAfter=12
        ))
        
        # Section styles
        self.styles.add(ParagraphStyle(
            name='SectionHeading', 
            parent=self.styles['Heading2'],
            fontSize=14, 
            alignment=TA_LEFT, 
            spaceAfter=6,
            spaceBefore=12
        ))
        
        # Normal text - use existing style
        # self.styles['Normal'] is already defined
        
        # Caption text
        self.styles.add(ParagraphStyle(
            name='ImageCaption', 
            parent=self.styles['Italic'],
            fontSize=9, 
            alignment=TA_CENTER, 
            spaceAfter=12
        ))
        
        # Insight text
        self.styles.add(ParagraphStyle(
            name='InsightText', 
            parent=self.styles['Normal'],
            leftIndent=20, 
            spaceAfter=6
        ))

    def get_session_info(self, session_id):
        """Get session information from the database."""
        with self.db.connect() as conn:
            if not conn:
                logger.error("Failed to connect to database")
                return {}
            try:
                cursor = conn.cursor(pymysql.cursors.DictCursor)
                query = "SELECT * FROM emg_sessions WHERE numeric_id = %s" if \
                        isinstance(session_id, int) or (isinstance(session_id, str) and session_id.isdigit()) \
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
        plt.savefig(buf, format='png', dpi=300)
        plt.close(fig)
        buf.seek(0)
        return buf

    def create_session_overview_plot(self, timeseries_df, throws_df, session_info):
        """Create a simple overview plot of the entire session."""
        if timeseries_df.empty:
            logger.warning("Empty data provided for session overview plot")
            return None
            
        muscle1_name = session_info.get('muscle1_name', 'FCU')
        muscle2_name = session_info.get('muscle2_name', 'FCR')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(7.5, 4))
        
        # For very large datasets, use a smarter downsampling approach
        if len(timeseries_df) > 100000:
            # Calculate the entire time range
            min_time = timeseries_df['time_point'].min()
            max_time = timeseries_df['time_point'].max()
            
            logger.info(f"Full time range: {min_time} to {max_time} seconds")
            
            # Method 2: Time-based downsampling (better preservation of data shape)
            # Create evenly spaced time points across the entire range
            sample_size = 100000  # Number of points to target
            
            # Create time bins and take representative points from each bin
            time_range = max_time - min_time
            bin_size = time_range / sample_size
            
            # Use pandas cut to bin the data
            timeseries_df['time_bin'] = pd.cut(timeseries_df['time_point'], 
                                            bins=np.arange(min_time, max_time + bin_size, bin_size))
            
            # Take the first point from each bin
            plot_df = timeseries_df.groupby('time_bin').first().reset_index()
            
            # If we got fewer points than expected (due to empty bins), fall back to regular sampling
            if len(plot_df) < sample_size / 2:
                logger.warning(f"Time-based sampling yielded only {len(plot_df)} points. Falling back to regular sampling.")
                step = max(1, len(timeseries_df) // sample_size)
                plot_df = timeseries_df.iloc[::step].copy()
            
            logger.info(f"Downsampled from {len(timeseries_df)} to {len(plot_df)} points")
        else:
            plot_df = timeseries_df
        
        # Verify we have data across the full time range
        if not plot_df.empty:
            plot_min = plot_df['time_point'].min()
            plot_max = plot_df['time_point'].max()
            full_min = timeseries_df['time_point'].min()
            full_max = timeseries_df['time_point'].max()
            
            logger.info(f"Original data time range: {full_min} to {full_max}")
            logger.info(f"Downsampled data time range: {plot_min} to {plot_max}")
            
            # Ensure we're plotting the full range
            if abs(plot_min - full_min) > 1.0 or abs(plot_max - full_max) > 1.0:
                logger.warning("Downsampled data doesn't cover the full time range. Adjusting.")
                # Add the first and last points from the original dataframe
                first_point = timeseries_df.iloc[[0]].copy()
                last_point = timeseries_df.iloc[[-1]].copy()
                plot_df = pd.concat([first_point, plot_df, last_point]).reset_index(drop=True)
        
        # Plot EMG signals
        ax.plot(plot_df['time_point'], plot_df['muscle1_emg'], 
                color='blue', alpha=0.7, label=muscle1_name, linewidth=0.5)
        ax.plot(plot_df['time_point'], plot_df['muscle2_emg'], 
                color='red', alpha=0.7, label=muscle2_name, linewidth=0.5)
        
        # Highlight throws if available
        if not throws_df.empty:
            for _, throw in throws_df.iterrows():
                ax.axvspan(throw['start_time'], throw['end_time'], color='green', alpha=0.2)
                ax.text(throw['start_time'], ax.get_ylim()[1] * 0.9, 
                    f"#{throw['throw_number']}", fontsize=8, ha='left')
        
        # Add labels and legend
        ax.set_title('EMG Activity Throughout Session', fontsize=14)
        ax.set_xlabel('Time (seconds)', fontsize=10)
        ax.set_ylabel('EMG Amplitude (mV)', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Force x-axis to span the full time range with padding
        if not timeseries_df.empty:
            full_min = timeseries_df['time_point'].min()
            full_max = timeseries_df['time_point'].max()
            
            # Add a small buffer (5% of the range) on both sides
            time_range = full_max - full_min
            buffer = time_range * 0.05
            display_min = full_min - buffer
            display_max = full_max + buffer
            
            # Ensure all throws are visible
            if not throws_df.empty:
                throw_min = throws_df['start_time'].min()
                throw_max = throws_df['end_time'].max()
                display_min = min(display_min, throw_min - buffer)
                display_max = max(display_max, throw_max + buffer)
            
            # Set the limits
            ax.set_xlim(display_min, display_max)
            
            # Log the final plot limits
            logger.info(f"Plot x-axis limits set to: {ax.get_xlim()}")
        
        # Add session info
        session_text = (
            f"Athlete: {session_info.get('athlete_name', 'Unknown')}\n"
            f"Date: {session_info.get('date_recorded', 'Unknown')}\n"
            f"Type: {session_info.get('session_type', 'Unknown')}\n"
            f"Throws: {len(throws_df)}\n"
            f"Data Points: {len(timeseries_df):,}"
        )
        ax.text(0.02, 0.02, session_text, transform=ax.transAxes, fontsize=8,
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        return self._create_plot_buffer(fig)


    def create_key_metrics_plot(self, throws_df, session_info):
        """Create a plot showing the most important metrics across throws."""
        if throws_df.empty:
            logger.warning("Empty throws data provided for key metrics plot")
            return None
            
        muscle1_name = session_info.get('muscle1_name', 'FCU')
        muscle2_name = session_info.get('muscle2_name', 'FCR')
        
        # Create a figure with 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)
        
        # Plot 1: Peak amplitude
        if 'muscle1_peak_amplitude' in throws_df.columns and 'muscle2_peak_amplitude' in throws_df.columns:
            ax1 = axes[0]
            ax1.plot(throws_df['throw_number'], throws_df['muscle1_peak_amplitude'], 
                    'o-', color='blue', label=muscle1_name)
            ax1.plot(throws_df['throw_number'], throws_df['muscle2_peak_amplitude'], 
                    'o-', color='red', label=muscle2_name)
            ax1.set_ylabel('Peak Amplitude (mV)', fontsize=10)
            ax1.set_title('Muscle Activation Intensity', fontsize=12)
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Median frequency (fatigue indicator)
        if 'muscle1_median_freq' in throws_df.columns and 'muscle2_median_freq' in throws_df.columns:
            ax2 = axes[1]
            ax2.plot(throws_df['throw_number'], throws_df['muscle1_median_freq'], 
                    'o-', color='blue', label=muscle1_name)
            ax2.plot(throws_df['throw_number'], throws_df['muscle2_median_freq'], 
                    'o-', color='red', label=muscle2_name)
            ax2.set_ylabel('Median Frequency (Hz)', fontsize=10)
            ax2.set_title('Frequency Content (decreasing indicates fatigue)', fontsize=12)
            ax2.set_xlabel('Throw Number', fontsize=10)
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            
            # Set x-ticks to throw numbers
            ax2.set_xticks(throws_df['throw_number'])
        
        return self._create_plot_buffer(fig)

    def create_coordination_plot(self, throws_df, session_info):
        """Create a plot showing muscle coordination metrics."""
        if throws_df.empty or 'coactivation_index' not in throws_df.columns:
            logger.warning("No coordination data available")
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=(7.5, 4))
        
        # Plot coactivation index
        ax.plot(throws_df['throw_number'], throws_df['coactivation_index'], 
            'o-', color='green', label='Muscle Coordination')
        
        # Add trend line
        if len(throws_df) >= 3:
            x = throws_df['throw_number'].values
            y = throws_df['coactivation_index'].values
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), '--', color='green', alpha=0.6)
        
        # Add labels and styling
        ax.set_title('Muscle Coordination', fontsize=14)
        ax.set_xlabel('Throw Number', fontsize=10)
        ax.set_ylabel('Coordination Index', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(throws_df['throw_number'])
        
        # Add explanation
        explanation = "Higher values indicate better coordination between muscles"
        ax.text(0.5, 0.02, explanation, transform=ax.transAxes, fontsize=10,
            ha='center', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        return self._create_plot_buffer(fig)

    def create_wavelet_energy_plot(self, throws_df, session_info):
        """Create a plot showing wavelet energy distribution across frequency bands for both muscles."""
        if (throws_df.empty or 
            'muscle1_wavelet_energy_low' not in throws_df.columns or 
            'muscle1_wavelet_energy_mid' not in throws_df.columns or 
            'muscle1_wavelet_energy_high' not in throws_df.columns):
            logger.warning("No wavelet energy data available")
            return None
                
        muscle1_name = session_info.get('muscle1_name', 'FCU')
        muscle2_name = session_info.get('muscle2_name', 'FCR')
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 4))
        
        # Muscle 1 plot (left)
        x = throws_df['throw_number']
        y1 = throws_df['muscle1_wavelet_energy_low']
        y2 = throws_df['muscle1_wavelet_energy_mid']
        y3 = throws_df['muscle1_wavelet_energy_high']
        
        # Create stacked area chart for muscle 1
        ax1.fill_between(x, 0, y1, alpha=0.8, color='#2E8B57', label='Low Frequency')
        ax1.fill_between(x, y1, y1+y2, alpha=0.8, color='#FFA500', label='Mid Frequency')
        ax1.fill_between(x, y1+y2, y1+y2+y3, alpha=0.8, color='#B22222', label='High Frequency')
        
        # Add labels and styling for muscle 1
        ax1.set_title(f'{muscle1_name}', fontsize=12)
        ax1.set_xlabel('Throw Number', fontsize=9)
        ax1.set_ylabel('Relative Energy', fontsize=9)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(throws_df['throw_number'])
        
        # Muscle 2 plot (right)
        if ('muscle2_wavelet_energy_low' in throws_df.columns and 
            'muscle2_wavelet_energy_mid' in throws_df.columns and 
            'muscle2_wavelet_energy_high' in throws_df.columns):
            
            y1 = throws_df['muscle2_wavelet_energy_low']
            y2 = throws_df['muscle2_wavelet_energy_mid']
            y3 = throws_df['muscle2_wavelet_energy_high']
            
            # Create stacked area chart for muscle 2
            ax2.fill_between(x, 0, y1, alpha=0.8, color='#2E8B57', label='Low Frequency')
            ax2.fill_between(x, y1, y1+y2, alpha=0.8, color='#FFA500', label='Mid Frequency')
            ax2.fill_between(x, y1+y2, y1+y2+y3, alpha=0.8, color='#B22222', label='High Frequency')
            
            # Add labels and styling for muscle 2
            ax2.set_title(f'{muscle2_name}', fontsize=12)
            ax2.set_xlabel('Throw Number', fontsize=9)
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(throws_df['throw_number'])
        else:
            # If muscle 2 wavelet data isn't available
            ax2.text(0.5, 0.5, f"No wavelet data for {muscle2_name}", 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title(f'{muscle2_name}', fontsize=12)
        
        # Add main title
        fig.suptitle('Wavelet Energy Distribution', fontsize=14)
        
        # Add explanation
        fig.text(0.5, 0.01, "Shift from high to low frequency bands indicates fatigue development", 
                ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.85)  # Make room for title and explanation
        
        return self._create_plot_buffer(fig)


    # Add this method to your SimpleEMGReportGenerator class
    def create_fatigue_plot(self, throws_df, session_info):
        """Create a plot focused on fatigue indicators from spectral and wavelet analysis."""
        if throws_df.empty or len(throws_df) < 3:
            logger.warning("Not enough data for fatigue analysis")
            return None
            
        muscle1_name = session_info.get('muscle1_name', 'FCU')
        muscle2_name = session_info.get('muscle2_name', 'FCR')
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)
        
        # Plot 1: Spectral entropy (if available)
        if 'muscle1_spectral_entropy' in throws_df.columns and 'muscle2_spectral_entropy' in throws_df.columns:
            ax1 = axes[0]
            ax1.plot(throws_df['throw_number'], throws_df['muscle1_spectral_entropy'], 
                    'o-', color='blue', label=muscle1_name)
            ax1.plot(throws_df['throw_number'], throws_df['muscle2_spectral_entropy'], 
                    'o-', color='red', label=muscle2_name)
            ax1.set_ylabel('Spectral Entropy (bits)', fontsize=10)
            ax1.set_title('Signal Complexity (increasing indicates fatigue)', fontsize=12)
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            # Add trend lines if there are enough data points
            if len(throws_df) >= 3:
                # For muscle 1
                x = throws_df['throw_number']
                y = throws_df['muscle1_spectral_entropy']
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax1.plot(x, p(x), '--', color='blue', alpha=0.6)
                
                # For muscle 2
                y = throws_df['muscle2_spectral_entropy']
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax1.plot(x, p(x), '--', color='red', alpha=0.6)
        
        # Plot 2: Wavelet frequency bands (if available) or median frequency
        ax2 = axes[1]
        if ('muscle1_wavelet_energy_low' in throws_df.columns and 
            'muscle1_wavelet_energy_high' in throws_df.columns):
            # Calculate high to low frequency ratio
            throws_df['freq_ratio'] = throws_df['muscle1_wavelet_energy_high'] / throws_df['muscle1_wavelet_energy_low']
            
            ax2.plot(throws_df['throw_number'], throws_df['freq_ratio'], 
                    'o-', color='purple', label='High/Low Freq Ratio')
            ax2.set_ylabel('Frequency Ratio', fontsize=10)
            ax2.set_title('Frequency Shift (decreasing indicates fatigue)', fontsize=12)
            
            # Add trend line
            if len(throws_df) >= 3:
                x = throws_df['throw_number']
                y = throws_df['freq_ratio']
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax2.plot(x, p(x), '--', color='purple', alpha=0.6)
        else:
            # Use median frequency instead
            if 'muscle1_median_freq' in throws_df.columns:
                ax2.plot(throws_df['throw_number'], throws_df['muscle1_median_freq'], 
                        'o-', color='blue', label=f"{muscle1_name} Median Freq")
                ax2.set_ylabel('Median Frequency (Hz)', fontsize=10)
                ax2.set_title('Frequency Content (decreasing indicates fatigue)', fontsize=12)
                
                # Add trend line
                if len(throws_df) >= 3:
                    x = throws_df['throw_number']
                    y = throws_df['muscle1_median_freq']
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    ax2.plot(x, p(x), '--', color='blue', alpha=0.6)
        
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Throw Number', fontsize=10)
        ax2.set_xticks(throws_df['throw_number'])
        
        return self._create_plot_buffer(fig)

    def get_key_insights(self, throws_df, session_info):
        """Extract key insights from the throw data."""
        if throws_df.empty:
            return ["No throw data available for analysis."]
            
        insights = []
        muscle1_name = session_info.get('muscle1_name', 'FCU')
        muscle2_name = session_info.get('muscle2_name', 'FCR')
        
        # Basic session statistics
        insights.append(f"Total throws detected: {len(throws_df)}")
        
        # Peak activation insights
        if 'muscle1_peak_amplitude' in throws_df.columns:
            max_m1_throw = throws_df.loc[throws_df['muscle1_peak_amplitude'].idxmax()]
            insights.append(f"Highest {muscle1_name} activation: Throw #{int(max_m1_throw['throw_number'])} ({max_m1_throw['muscle1_peak_amplitude']:.2f} mV)")
        
        if 'muscle2_peak_amplitude' in throws_df.columns:
            max_m2_throw = throws_df.loc[throws_df['muscle2_peak_amplitude'].idxmax()]
            insights.append(f"Highest {muscle2_name} activation: Throw #{int(max_m2_throw['throw_number'])} ({max_m2_throw['muscle2_peak_amplitude']:.2f} mV)")
        
        # Fatigue insights
        if len(throws_df) >= 3:
            # Check for median frequency decline (fatigue indicator)
            if 'muscle1_median_freq' in throws_df.columns:
                x = throws_df['throw_number'].values
                y = throws_df['muscle1_median_freq'].values
                # Need to handle calculation without using full_true
                try:
                    slope, intercept, r_value, p_value, std_err = np.polyfit(x, y, 1, cov=True)[0]
                    
                    if p_value < 0.1 and slope < -0.5:
                        insights.append(f"Possible fatigue detected: {muscle1_name} frequency decreasing across throws")
                except:
                    # Simplified approach if the above fails
                    z = np.polyfit(x, y, 1)
                    slope = z[0]
                    if slope < -0.5:
                        insights.append(f"Possible fatigue: {muscle1_name} frequency trend decreasing")
            
            # Check for spectral entropy (simpler approach)
            if 'muscle1_spectral_entropy' in throws_df.columns:
                if len(throws_df) >= 3:
                    first_half = throws_df['muscle1_spectral_entropy'].iloc[:len(throws_df)//2].mean()
                    second_half = throws_df['muscle1_spectral_entropy'].iloc[len(throws_df)//2:].mean()
                    
                    if second_half > first_half * 1.1:  # 10% increase
                        insights.append(f"Signal complexity increasing in {muscle1_name}, indicating possible fatigue")
        
        # Coordination insights
        if 'coactivation_index' in throws_df.columns:
            avg_coact = throws_df['coactivation_index'].mean()
            
            if avg_coact > 0.7:
                insights.append("Excellent muscle coordination throughout the session")
            elif avg_coact > 0.5:
                insights.append("Good muscle coordination overall")
            else:
                insights.append("Potential for improved muscle coordination")
                
            # Check for trends in coordination (simplified)
            if len(throws_df) >= 3:
                first_half = throws_df['coactivation_index'].iloc[:len(throws_df)//2].mean()
                second_half = throws_df['coactivation_index'].iloc[len(throws_df)//2:].mean()
                
                if second_half < first_half * 0.9:  # 10% decrease
                    insights.append("Muscle coordination declining across throws")
                elif second_half > first_half * 1.1:  # 10% increase
                    insights.append("Muscle coordination improving throughout the session")
        
        return insights

    def create_metrics_table_data(self, throws_df, session_info):
        """Create a simple table of key metrics for each throw."""
        if throws_df.empty:
            return [["No throw data available"]]
            
        muscle1_name = session_info.get('muscle1_name', 'FCU')
        muscle2_name = session_info.get('muscle2_name', 'FCR')
        
        # Select the most important metrics for the table
        headers = ["Throw #", "Duration (s)", f"{muscle1_name} Peak (mV)", f"{muscle2_name} Peak (mV)"]
        
        # Add coordination if available - limit to just the most important metrics
        if 'coactivation_index' in throws_df.columns:
            headers.append("Coordination")
        
        # Create table rows
        rows = [headers]
        for _, throw in throws_df.iterrows():
            row = [
                int(throw['throw_number']),
                round(throw['duration'], 2),
                round(throw['muscle1_peak_amplitude'], 2) if 'muscle1_peak_amplitude' in throw else "N/A",
                round(throw['muscle2_peak_amplitude'], 2) if 'muscle2_peak_amplitude' in throw else "N/A"
            ]
            
            # Add coordination if available
            if 'coactivation_index' in throws_df.columns:
                row.append(round(throw['coactivation_index'], 2) if not pd.isna(throw['coactivation_index']) else "N/A")
            
            rows.append(row)
        
        return rows

    def add_page_number(self, canvas, doc):
        """Add page number to the canvas."""
        page_num = canvas.getPageNumber()
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(7.5*inch, 0.25*inch, f"Page {page_num}")

    def create_pdf_report(self, session_id):
        """Create a simple PDF report for a session."""
        try:
            # Get data from database
            session_info = self.get_session_info(session_id)
            if not session_info:
                logger.error(f"Session {session_id} not found")
                return None
                
            throws_df = self.get_throws_for_session(session_id)
            if throws_df.empty:
                logger.warning(f"No throws found for session {session_id}")
                return None
                
            timeseries_df = self.get_timeseries_data(session_id, max_rows=3000000)
            if timeseries_df.empty:
                logger.warning(f"No time series data found for session {session_id}")
                return None
            
            # Extract session details
            athlete_name = session_info.get('athlete_name', 'Unknown Athlete')
            session_date = session_info.get('date_recorded', 'Unknown Date')
            session_type = session_info.get('session_type', 'Unknown Type')
            muscle1_name = session_info.get('muscle1_name', 'FCU')
            muscle2_name = session_info.get('muscle2_name', 'FCR')
            
            # Create plots - add spectral entropy and wavelet energy plots
            overview_plot = self.create_session_overview_plot(timeseries_df, throws_df, session_info)
            metrics_plot = self.create_key_metrics_plot(throws_df, session_info)
            fatigue_plot = self.create_fatigue_plot(throws_df, session_info)
            
            # Create these plots only if data is available
            coordination_plot = None
            if 'coactivation_index' in throws_df.columns:
                coordination_plot = self.create_coordination_plot(throws_df, session_info)
                
            wavelet_plot = None
            if ('muscle1_wavelet_energy_low' in throws_df.columns and 
                'muscle1_wavelet_energy_mid' in throws_df.columns and 
                'muscle1_wavelet_energy_high' in throws_df.columns):
                wavelet_plot = self.create_wavelet_energy_plot(throws_df, session_info)
            
            # Get insights and table data
            insights = self.get_key_insights(throws_df, session_info)
            metrics_table_data = self.create_metrics_table_data(throws_df, session_info)
            
            # Create PDF
            filename = session_info.get('filename', session_id)
            pdf_filename = os.path.join(self.output_dir, f"{filename}_report.pdf")
            
            doc = SimpleDocTemplate(
                pdf_filename,
                pagesize=letter,
                rightMargin=0.5*inch,
                leftMargin=0.5*inch,
                topMargin=0.5*inch,
                bottomMargin=0.5*inch,
                onLaterPages=self.add_page_number
            )
            
            elements = []
            
            # Title
            title = f"EMG Analysis Report: {athlete_name}"
            elements.append(Paragraph(title, self.styles['ReportTitle']))
            elements.append(Spacer(1, 0.1*inch))
            
            # Session details
            session_details = f"Date: {session_date} | Session Type: {session_type} | Muscles: {muscle1_name}, {muscle2_name}"
            elements.append(Paragraph(session_details, self.styles['Normal']))
            elements.append(Spacer(1, 0.2*inch))
            
            # Key Insights Section
            elements.append(Paragraph("Key Insights", self.styles['SectionHeading']))
            for insight in insights[:4]:  # Limit to top 4 insights for brevity
                elements.append(Paragraph(f"• {insight}", self.styles['InsightText']))
            elements.append(Spacer(1, 0.2*inch))
            
            # Overview Section
            elements.append(Paragraph("Session Overview", self.styles['SectionHeading']))
            if overview_plot:
                img = Image(overview_plot, width=7*inch, height=3.5*inch)
                elements.append(img)
                elements.append(Paragraph("EMG activity throughout session with throw markers", self.styles['ImageCaption']))
            
            # Add page break to keep content organized
            elements.append(PageBreak())
            
            # Page 2: Performance and Fatigue Analysis
            elements.append(Paragraph("Performance Metrics", self.styles['SectionHeading']))
            if metrics_plot:
                img = Image(metrics_plot, width=7*inch, height=4*inch)  # Reduced height to fit more content
                elements.append(img)
                elements.append(Paragraph("Activation intensity and frequency content across throws", self.styles['ImageCaption']))
            
            # Add fatigue plot with spectral entropy and frequency analysis
            elements.append(Paragraph("Fatigue Analysis", self.styles['SectionHeading']))
            if fatigue_plot:
                img = Image(fatigue_plot, width=7*inch, height=4*inch)  # Reduced height to fit more
                elements.append(img)
                elements.append(Paragraph("Spectral entropy and frequency indicators of fatigue", self.styles['ImageCaption']))
            
            # Add page break before coordination and wavelet plots
            elements.append(PageBreak())
            
            # Page 3: Coordination, Wavelet and Data Summary
            # Add wavelet energy distribution plot
            if wavelet_plot:
                elements.append(Paragraph("Frequency Distribution", self.styles['SectionHeading']))
                img = Image(wavelet_plot, width=7*inch, height=3*inch)
                elements.append(img)
                elements.append(Paragraph("Distribution of energy across frequency bands", self.styles['ImageCaption']))
            
            # Add coordination plot if available
            if coordination_plot:
                elements.append(Paragraph("Muscle Coordination", self.styles['SectionHeading']))
                img = Image(coordination_plot, width=7*inch, height=3*inch)
                elements.append(img)
                elements.append(Paragraph("Muscle coordination across throws", self.styles['ImageCaption']))
            
            # Metrics Table - just a small summary of the most important metrics
            elements.append(Paragraph("Throw-by-Throw Summary", self.styles['SectionHeading']))
            
            # Create the metrics table
            col_widths = [0.6*inch] + [1.5*inch] * (len(metrics_table_data[0]) - 1)
            # Adjust if too many columns
            if len(metrics_table_data[0]) > 5:
                col_widths = [0.6*inch] + [1.0*inch] * (len(metrics_table_data[0]) - 1)
                
            table = Table(metrics_table_data, colWidths=col_widths)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            elements.append(table)
            
            # Conclusions Section - very brief
            elements.append(Spacer(1, 0.3*inch))
            elements.append(Paragraph("Summary", self.styles['SectionHeading']))
            
            # Add a concise summary paragraph - just 1-2 sentences
            if len(insights) > 0:
                # Extract the most important insight
                key_insight = insights[0]
                # Add one fatigue or coordination insight if available
                for insight in insights:
                    if "fatigue" in insight.lower() or "coordination" in insight.lower():
                        key_insight += " " + insight
                        break
                        
                elements.append(Paragraph(key_insight, self.styles['Normal']))
            else:
                elements.append(Paragraph("No significant patterns detected in this session.", self.styles['Normal']))
            
            # Build the PDF
            doc.build(elements)
            logger.info(f"PDF report generated: {pdf_filename}")
            return pdf_filename
            
        except Exception as e:
            logger.error(f"Error creating PDF report for session {session_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def generate_report(self, session_id):
        """Generate a report for a specific session."""
        return self.create_pdf_report(session_id)

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate EMG Reports')
    parser.add_argument('session_id', nargs='?', default=None, 
                      help='Session ID to generate report for (optional if using --directory)')
    parser.add_argument('--directory', '-d', action='store_true',
                      help='Process all sessions in the database')
    parser.add_argument('--output-dir', default='reports', 
                      help='Directory to save reports')
    parser.add_argument('--limit', type=int, default=100, 
                      help='Maximum number of sessions to process (for directory mode)')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
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
    generator = SimpleEMGReportGenerator(db_config=db_config, output_dir=args.output_dir)
    
    # Process sessions
    if args.directory:
        # Process all sessions
        session_ids = generator.get_all_session_ids(limit=args.limit)
        
        if not session_ids:
            print("No sessions found in the database")
            sys.exit(1)
            
        print(f"Found {len(session_ids)} sessions to process")
        success_count = 0
        
        for i, session_id in enumerate(session_ids):
            print(f"Processing session {i+1}/{len(session_ids)}: {session_id}")
            report_path = generator.generate_report(session_id)
            
            if report_path:
                success_count += 1
                print(f"  ✓ Report generated successfully: {report_path}")
            else:
                print(f"  ✗ Failed to generate report")
        
        print(f"Processed {len(session_ids)} sessions, {success_count} successful")
    elif args.session_id:
        # Process single session
        report_path = generator.generate_report(args.session_id)
        
        if report_path:
            print(f"Report generated successfully: {report_path}")
        else:
            print("Failed to generate report")
            sys.exit(1)
    else:
        parser.print_help()
        print("\nError: Either session_id or --directory must be specified")
        sys.exit(1)