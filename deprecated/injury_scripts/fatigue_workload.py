import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Add parent directory to Python path to find db_connector.py
# Assuming db_connector.py is in the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Alternatively, if db_connector.py is in the same directory as this script
# Create a custom DBConnector class
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
        import pymysql
        import os
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
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
                print(f"Warning: Missing environment variables: {', '.join(missing_vars)}")
                print("Please create a .env file with the required database configuration")
        else:
            self.db_config = config
        
        self.conn = None
        print("Database connector initialized")
    
    def connect(self):
        """
        Connect to the database.
        
        Returns:
        --------
        pymysql.Connection or None
            Database connection if successful, None otherwise
        """
        import pymysql
        
        try:
            # Check if all required configuration is present
            if None in self.db_config.values():
                missing_keys = [k for k, v in self.db_config.items() if v is None]
                print(f"Error: Missing database configuration: {', '.join(missing_keys)}")
                return None
                
            self.conn = pymysql.connect(
                host=self.db_config['host'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database=self.db_config['database']
            )
            print(f"Connected to database {self.db_config['database']}")
            return self.conn
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return None
    
    def disconnect(self):
        """Close the database connection if it exists."""
        if self.conn:
            self.conn.close()
            self.conn = None
            print("Database connection closed")

def calculate_fatigue_metrics(db_connector, plot_results=True):
    """
    Extract FCR/FCU fatigue metrics from existing EMG data.
    Uses actual data patterns to establish player-specific baselines.
    
    Parameters:
    -----------
    db_connector : DBConnector
        Database connector object from your pipeline
    plot_results : bool
        Whether to generate visualization plots
        
    Returns:
    --------
    dict
        Dictionary containing calculated metrics and baselines
    """
    # Connect to the database
    conn = db_connector.connect()
    if not conn:
        print("Failed to connect to the database. Check your connection settings.")
        return {}
    
    cursor = conn.cursor()
    
    # 1. Extract sequential throw data from the same sessions
    query = """
    SELECT 
        s.athlete_name,
        s.date_recorded,
        s.session_type,
        t.session_numeric_id,
        t.trial_number,
        t.throw_id,
        t.start_time,
        t.pitch_speed_mph,
        
        -- FCU (muscle1) metrics
        t.muscle1_median_freq,
        t.muscle1_peak_amplitude,
        t.muscle1_throw_integral,
        t.muscle1_spectral_entropy,
        t.muscle1_wavelet_energy_low,
        t.muscle1_wavelet_energy_mid,
        t.muscle1_wavelet_energy_high,
        
        -- FCR (muscle2) metrics
        t.muscle2_median_freq,
        t.muscle2_peak_amplitude, 
        t.muscle2_throw_integral,
        t.muscle2_spectral_entropy,
        t.muscle2_wavelet_energy_low,
        t.muscle2_wavelet_energy_mid,
        t.muscle2_wavelet_energy_high,
        
        -- Coactivation metrics
        t.coactivation_index
        
    FROM emg_throws t
    JOIN emg_sessions s ON t.session_numeric_id = s.numeric_id
    WHERE t.pitch_speed_mph IS NOT NULL
    ORDER BY s.athlete_name, s.date_recorded, t.session_numeric_id, t.trial_number
    """
    
    try:
        # Execute query and convert to DataFrame
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        df = pd.DataFrame(data, columns=columns)
        
        print(f"Loaded {len(df)} throws from {df['athlete_name'].nunique()} athletes")
        
        # Group sessions with at least 5 throws for valid fatigue analysis
        session_counts = df.groupby(['athlete_name', 'session_numeric_id']).size()
        valid_sessions = session_counts[session_counts >= 5].reset_index()
        
        print(f"Found {len(valid_sessions)} valid sessions with 5+ throws")
        
        # Initialize results storage
        fatigue_metrics = {
            'athlete_baselines': {},
            'session_fatigue_scores': {},
            'fatigue_indicators': []
        }
        
        # Process each athlete's sessions
        for athlete in df['athlete_name'].unique():
            athlete_data = df[df['athlete_name'] == athlete].copy()
            print(f"\nProcessing {len(athlete_data)} throws for {athlete}")
            
            # Calculate athlete's baseline (using first 3 throws of each session)
            baseline_throws = []
            for session_id in athlete_data['session_numeric_id'].unique():
                session_data = athlete_data[athlete_data['session_numeric_id'] == session_id].sort_values('trial_number')
                baseline_throws.append(session_data.head(3))
            
            baseline_df = pd.concat(baseline_throws)
            
            # Calculate baseline metrics for this athlete
            athlete_baseline = {
                'fcu_median_freq_baseline': baseline_df['muscle1_median_freq'].median(),
                'fcr_median_freq_baseline': baseline_df['muscle2_median_freq'].median(),
                'fcu_peak_amp_baseline': baseline_df['muscle1_peak_amplitude'].median(),
                'fcr_peak_amp_baseline': baseline_df['muscle2_peak_amplitude'].median(),
                'fcu_integral_baseline': baseline_df['muscle1_throw_integral'].median(),
                'fcr_integral_baseline': baseline_df['muscle2_throw_integral'].median(),
                'coactivation_baseline': baseline_df['coactivation_index'].median(),
                'fcu_entropy_baseline': baseline_df['muscle1_spectral_entropy'].median(),
                'fcr_entropy_baseline': baseline_df['muscle2_spectral_entropy'].median()
            }
            
            fatigue_metrics['athlete_baselines'][athlete] = athlete_baseline
            
            # Process each session for this athlete
            for session_id in athlete_data['session_numeric_id'].unique():
                session_data = athlete_data[athlete_data['session_numeric_id'] == session_id].sort_values('trial_number')
                
                if len(session_data) < 5:
                    continue  # Skip sessions with too few throws
                    
                # Calculate rolling metrics to detect fatigue onset
                # For sequences of fastball throws, these metrics show most reliable fatigue indicators
                
                # 1. Key fatigue marker: Median frequency decline
                fcu_freq_pct = session_data['muscle1_median_freq'] / athlete_baseline['fcu_median_freq_baseline']
                fcr_freq_pct = session_data['muscle2_median_freq'] / athlete_baseline['fcr_median_freq_baseline']
                
                # 2. Key fatigue marker: Peak amplitude changes (often increases with fatigue)
                fcu_amp_pct = session_data['muscle1_peak_amplitude'] / athlete_baseline['fcu_peak_amp_baseline']
                fcr_amp_pct = session_data['muscle2_peak_amplitude'] / athlete_baseline['fcr_peak_amp_baseline']
                
                # 3. Spectral entropy decreases with fatigue
                fcu_entropy_pct = session_data['muscle1_spectral_entropy'] / athlete_baseline['fcu_entropy_baseline']
                fcr_entropy_pct = session_data['muscle2_spectral_entropy'] / athlete_baseline['fcr_entropy_baseline']
                
                # 4. Energy shifts to lower frequencies with fatigue
                fcu_high_low_ratio = session_data['muscle1_wavelet_energy_high'] / (session_data['muscle1_wavelet_energy_low'] + 0.001)
                fcr_high_low_ratio = session_data['muscle2_wavelet_energy_high'] / (session_data['muscle2_wavelet_energy_low'] + 0.001)
                
                # Calculate composite fatigue score
                # Weighted to focus on the most reliable fatigue indicators for pitchers
                fatigue_score = (
                    (1.0 - fcu_freq_pct) * 0.25 +  # FCU frequency decrease
                    (1.0 - fcr_freq_pct) * 0.25 +  # FCR frequency decrease 
                    (1.0 - fcu_entropy_pct) * 0.15 +  # FCU entropy decrease
                    (1.0 - fcr_entropy_pct) * 0.15 +  # FCR entropy decrease
                    (1.0 - fcu_high_low_ratio / fcu_high_low_ratio.iloc[0]) * 0.1 +  # FCU frequency shift
                    (1.0 - fcr_high_low_ratio / fcr_high_low_ratio.iloc[0]) * 0.1    # FCR frequency shift
                )
                
                # Ensure fatigue score is positive and normalized to 0-100 scale
                fatigue_score = 100 * np.clip(fatigue_score, 0, 1)
                
                # Store for plotting
                results_df = pd.DataFrame({
                    'throw_number': session_data['trial_number'],
                    'fatigue_score': fatigue_score,
                    'velocity': session_data['pitch_speed_mph'],
                    'fcu_freq_pct': fcu_freq_pct * 100,
                    'fcr_freq_pct': fcr_freq_pct * 100,
                    'fcu_amp_pct': fcu_amp_pct * 100,
                    'fcr_amp_pct': fcr_amp_pct * 100,
                    'fcu_entropy_pct': fcu_entropy_pct * 100,
                    'fcr_entropy_pct': fcr_entropy_pct * 100
                })
                
                # Store session results
                session_date = session_data['date_recorded'].iloc[0]
                session_key = f"{athlete}_{session_date}_{session_id}"
                fatigue_metrics['session_fatigue_scores'][session_key] = results_df
                
                # Identify fatigue onset for this session
                # Define threshold: 15% increase in fatigue score and consistent rise
                baseline_fatigue = results_df['fatigue_score'].iloc[:3].mean()
                fatigue_threshold = baseline_fatigue * 1.15
                
                # Find point where fatigue exceeds threshold for 2+ consecutive throws
                fatigue_onset = None
                for i in range(3, len(results_df)):
                    if (results_df['fatigue_score'].iloc[i] > fatigue_threshold and 
                        results_df['fatigue_score'].iloc[i-1] > fatigue_threshold):
                        fatigue_onset = results_df['throw_number'].iloc[i-1]
                        break
                
                # Add to fatigue indicators collection
                if fatigue_onset is not None:
                    fatigue_metrics['fatigue_indicators'].append({
                        'athlete': athlete,
                        'session_id': session_id,
                        'session_date': session_date,
                        'fatigue_onset_throw': fatigue_onset,
                        'baseline_fatigue': baseline_fatigue,
                        'max_fatigue_score': results_df['fatigue_score'].max(),
                        'throw_count': len(results_df),
                        'velocity_drop_pct': (results_df['velocity'].iloc[0] - results_df['velocity'].iloc[-1]) / results_df['velocity'].iloc[0] * 100
                    })
                
                # Plot results for this session
                if plot_results:
                    # Create results directory if it doesn't exist
                    os.makedirs("emg_fatigue_results", exist_ok=True)
                    
                    plt.figure(figsize=(12, 8))
                    
                    # Plot 1: Fatigue score progression
                    plt.subplot(2, 1, 1)
                    plt.plot(results_df['throw_number'], results_df['fatigue_score'], 'o-', color='red', linewidth=2)
                    if fatigue_onset is not None:
                        plt.axvline(x=fatigue_onset, color='r', linestyle='--', alpha=0.7)
                        plt.text(fatigue_onset+0.5, results_df['fatigue_score'].max()*0.9, 
                                f"Fatigue onset: throw #{fatigue_onset}", color='r')
                    
                    plt.title(f"Forearm Fatigue Progression - {athlete} - Session {session_date}")
                    plt.ylabel("Fatigue Score (0-100)")
                    plt.grid(alpha=0.3)
                    
                    # Plot 2: Key metrics
                    plt.subplot(2, 1, 2)
                    plt.plot(results_df['throw_number'], results_df['fcu_freq_pct'], 'o-', label='FCU Freq %')
                    plt.plot(results_df['throw_number'], results_df['fcr_freq_pct'], 's-', label='FCR Freq %')
                    plt.plot(results_df['throw_number'], results_df['velocity'] * 1.1, '^-', label='Velocity (adjusted scale)')
                    
                    plt.title(f"Key Metrics Progression")
                    plt.xlabel("Throw Number")
                    plt.ylabel("Percentage of Baseline (%)")
                    plt.grid(alpha=0.3)
                    plt.legend()
                    
                    plt.tight_layout()
                    plt.savefig(f"emg_fatigue_results/fatigue_analysis_{session_key}.png", dpi=300)
                    plt.close()
        
        # Create summary of fatigue indicators
        if fatigue_metrics['fatigue_indicators']:
            fatigue_indicators_df = pd.DataFrame(fatigue_metrics['fatigue_indicators'])
            print("\nFatigue Indicator Summary:")
            print(fatigue_indicators_df.describe())
            
            fatigue_metrics['fatigue_indicators_df'] = fatigue_indicators_df
            
            # Plot distributions of key metrics
            if plot_results:
                os.makedirs("emg_fatigue_results", exist_ok=True)
                
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                sns.histplot(fatigue_indicators_df['fatigue_onset_throw'], kde=True)
                plt.title("Distribution of Fatigue Onset")
                plt.xlabel("Throw Number")
                
                plt.subplot(1, 2, 2)
                sns.scatterplot(data=fatigue_indicators_df, x='fatigue_onset_throw', y='velocity_drop_pct')
                plt.title("Fatigue Onset vs Velocity Drop")
                plt.xlabel("Fatigue Onset (Throw #)")
                plt.ylabel("Velocity Drop (%)")
                
                plt.tight_layout()
                plt.savefig("emg_fatigue_results/fatigue_patterns_summary.png", dpi=300)
                plt.close()
        
        return fatigue_metrics
    
    except Exception as e:
        import traceback
        print(f"Error in calculate_fatigue_metrics: {e}")
        traceback.print_exc()
        return {}
    
    finally:
        # Close database connection
        db_connector.disconnect()

def calculate_forearm_workload(db_connector, athlete_name=None):
    """
    Calculate enhanced forearm workload metrics from EMG data.
    Specifically designed for quick, high-intensity pitcher throws.
    
    Parameters:
    -----------
    db_connector : DBConnector
        Database connector object from your pipeline
    athlete_name : str, optional
        Filter to a specific athlete
        
    Returns:
    --------
    dict
        Dictionary containing workload metrics
    """
    # Connect to database
    conn = db_connector.connect()
    if not conn:
        print("Failed to connect to the database. Check your connection settings.")
        return {}
    
    cursor = conn.cursor()
    
    try:
        # Build query with optional athlete filter
        query = """
        SELECT 
            s.athlete_name,
            s.session_numeric_id,
            s.date_recorded,
            s.session_type,
            t.throw_id,
            t.trial_number,
            
            -- FCU (muscle1) metrics
            t.muscle1_peak_amplitude,
            t.muscle1_throw_integral,
            t.muscle1_work_rate,
            
            -- FCR (muscle2) metrics
            t.muscle2_peak_amplitude,
            t.muscle2_throw_integral,
            t.muscle2_work_rate,
            
            -- Coactivation metrics
            t.coactivation_index
        FROM emg_throws t
        JOIN emg_sessions s ON t.session_numeric_id = s.numeric_id
        """
        
        # Add athlete filter if specified
        if athlete_name:
            query += f" WHERE s.athlete_name = '{athlete_name}'"
        
        query += " ORDER BY s.athlete_name, s.date_recorded, t.trial_number"
        
        # Execute query
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        df = pd.DataFrame(data, columns=columns)
        
        print(f"Loaded {len(df)} throws for workload analysis")
        
        # Calculate enhanced workload metrics
        workload_metrics = {}
        
        # Process each athlete
        for athlete in df['athlete_name'].unique():
            athlete_data = df[df['athlete_name'] == athlete].copy()
            
            # Group by session
            sessions = athlete_data.groupby('session_numeric_id')
            
            athlete_workload = {
                'sessions': {},
                'cumulative_workload': 0
            }
            
            for session_id, session_data in sessions:
                # Create enhanced workload metric specifically for pitchers
                # This model weights quick, high-intensity activation patterns
                
                # 1. Base workload = FCU integral + FCR integral (basic workload)
                fcu_workload = session_data['muscle1_throw_integral'].sum()
                fcr_workload = session_data['muscle2_throw_integral'].sum()
                
                # 2. Apply intensity scaling - pitchers generate high peaks in short timeframes
                fcu_peak_factor = session_data['muscle1_peak_amplitude'].mean() / (session_data['muscle1_throw_integral'].mean() + 0.001)
                fcr_peak_factor = session_data['muscle2_peak_amplitude'].mean() / (session_data['muscle2_throw_integral'].mean() + 0.001)
                
                # Higher peaks in shorter time = more explosive = higher workload
                intensity_factor = (fcu_peak_factor + fcr_peak_factor) / 2
                
                # 3. Apply coactivation adjustment - higher coactivation = higher coordination demand
                coactivation_factor = 1 + (session_data['coactivation_index'].mean() * 0.2)
                
                # 4. Repetition factor - more throws = more workload
                repetition_factor = len(session_data)
                
                # Calculate session workload
                # Scale factors to keep final value in a reasonable range (0-1000)
                session_workload = (
                    (fcu_workload + fcr_workload) *  # Basic workload
                    intensity_factor *                # Intensity adjustment
                    coactivation_factor               # Coordination demand
                ) * 10  # Scaling factor
                
                # Store session workload details
                athlete_workload['sessions'][session_id] = {
                    'date': session_data['date_recorded'].iloc[0],
                    'throw_count': repetition_factor,
                    'fcu_workload': fcu_workload,
                    'fcr_workload': fcr_workload,
                    'intensity_factor': intensity_factor,
                    'coactivation_factor': coactivation_factor,
                    'session_workload': session_workload,
                    'workload_per_throw': session_workload / (repetition_factor or 1)  # Avoid division by zero
                }
                
                # Add to cumulative workload
                athlete_workload['cumulative_workload'] += session_workload
            
            # Calculate workload distribution
            session_workloads = [session['session_workload'] 
                                for session in athlete_workload['sessions'].values()]
            
            if session_workloads:
                athlete_workload['avg_session_workload'] = np.mean(session_workloads)
                athlete_workload['max_session_workload'] = np.max(session_workloads)
                athlete_workload['workload_variability'] = np.std(session_workloads) / (np.mean(session_workloads) or 1)  # Avoid division by zero
            
            # Store athlete's workload metrics
            workload_metrics[athlete] = athlete_workload
        
        # Visualize workload distribution
        os.makedirs("emg_fatigue_results", exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        
        # Prepare data for plotting
        athletes = []
        avg_workloads = []
        max_workloads = []
        
        for athlete, data in workload_metrics.items():
            if 'avg_session_workload' in data:  # Ensure we have workload data
                athletes.append(athlete)
                avg_workloads.append(data['avg_session_workload'])
                max_workloads.append(data['max_session_workload'])
        
        if athletes:  # Only create plot if we have data
            # Create bar chart
            x = np.arange(len(athletes))
            width = 0.35
            
            plt.bar(x - width/2, avg_workloads, width, label='Average Session Workload')
            plt.bar(x + width/2, max_workloads, width, label='Maximum Session Workload')
            
            plt.xlabel('Athlete')
            plt.ylabel('Forearm Workload (EMG Units)')
            plt.title('Forearm Workload Distribution by Athlete')
            plt.xticks(x, athletes, rotation=45, ha='right')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig("emg_fatigue_results/forearm_workload_distribution.png", dpi=300)
            plt.close()
        
        return workload_metrics
    
    except Exception as e:
        import traceback
        print(f"Error in calculate_forearm_workload: {e}")
        traceback.print_exc()
        return {}
    
    finally:
        # Close database connection
        db_connector.disconnect()

def get_pitcher_recommendations(fatigue_metrics, workload_metrics):
    """
    Generate actionable recommendations for pitchers based on EMG fatigue and workload data.
    
    Parameters:
    -----------
    fatigue_metrics : dict
        Output from calculate_fatigue_metrics function
    workload_metrics : dict
        Output from calculate_forearm_workload function
        
    Returns:
    --------
    dict
        Dictionary of recommendations by athlete
    """
    recommendations = {}
    
    # Process fatigue indicators if available
    if 'fatigue_indicators_df' in fatigue_metrics:
        fatigue_df = fatigue_metrics['fatigue_indicators_df']
        
        for athlete in fatigue_df['athlete'].unique():
            athlete_fatigue = fatigue_df[fatigue_df['athlete'] == athlete]
            
            # Get athlete's workload metrics
            athlete_workload = workload_metrics.get(athlete, {})
            
            # Initialize recommendations dict
            recommendations[athlete] = {
                'fatigue_profile': {},
                'workload_profile': {},
                'action_items': []
            }
            
            # Analyze fatigue patterns
            avg_fatigue_onset = athlete_fatigue['fatigue_onset_throw'].mean()
            avg_velocity_drop = athlete_fatigue['velocity_drop_pct'].mean()
            
            fatigue_profile = {
                'avg_fatigue_onset_throw': avg_fatigue_onset,
                'avg_velocity_drop_pct': avg_velocity_drop,
                'fatigue_onset_consistency': athlete_fatigue['fatigue_onset_throw'].std()
            }
            
            recommendations[athlete]['fatigue_profile'] = fatigue_profile
            
            # Analyze workload patterns
            if 'avg_session_workload' in athlete_workload:
                workload_profile = {
                    'avg_session_workload': athlete_workload['avg_session_workload'],
                    'max_session_workload': athlete_workload['max_session_workload'],
                    'workload_variability': athlete_workload.get('workload_variability', 0)
                }
                recommendations[athlete]['workload_profile'] = workload_profile
            
            # Generate specific recommendations
            # 1. Early fatigue onset
            if avg_fatigue_onset < 10:
                recommendations[athlete]['action_items'].append({
                    'category': 'Endurance',
                    'finding': f"Early forearm fatigue onset (avg throw #{avg_fatigue_onset:.1f})",
                    'action': "Implement progressive forearm endurance training with lower intensity, higher volume resistance exercises."
                })
            
            # 2. Inconsistent fatigue onset
            if fatigue_profile['fatigue_onset_consistency'] > 3:
                recommendations[athlete]['action_items'].append({
                    'category': 'Consistency',
                    'finding': "Highly variable fatigue patterns between sessions",
                    'action': "Focus on consistent pre-throwing routine and targeted forearm conditioning to stabilize fatigue response."
                })
            
            # 3. High velocity drop with fatigue
            if avg_velocity_drop > 3.0:
                recommendations[athlete]['action_items'].append({
                    'category': 'Performance',
                    'finding': f"Significant velocity drop with fatigue ({avg_velocity_drop:.1f}%)",
                    'action': "Implement in-session monitoring to track EMG fatigue markers and establish pitch count limits based on EMG-derived fatigue score."
                })
            
            # 4. Workload management (if workload data available)
            if 'workload_profile' in recommendations[athlete] and 'workload_variability' in recommendations[athlete]['workload_profile']:
                if recommendations[athlete]['workload_profile']['workload_variability'] > 0.3:
                    recommendations[athlete]['action_items'].append({
                        'category': 'Workload',
                        'finding': "High variability in forearm workload between sessions",
                        'action': "Implement more consistent workload progression with 10-15% weekly increases in forearm-specific training volume."
                    })
    
    return recommendations

def run_emg_fatigue_analysis(db_config=None):
    """
    Run the complete EMG-based fatigue and workload analysis for pitchers.
    
    Parameters:
    -----------
    db_config : dict, optional
        Database configuration
        
    Returns:
    --------
    dict
        Complete analysis results
    """
    # Initialize database connector
    # Using our embedded DBConnector class
    db = DBConnector(db_config)
    
    # Calculate fatigue metrics
    print("Calculating fatigue metrics...")
    fatigue_metrics = calculate_fatigue_metrics(db)
    
    # Calculate workload metrics
    print("\nCalculating workload metrics...")
    workload_metrics = calculate_forearm_workload(db)
    
    # Generate recommendations
    print("\nGenerating pitcher-specific recommendations...")
    recommendations = get_pitcher_recommendations(fatigue_metrics, workload_metrics)
    
    # Compile results
    results = {
        'fatigue_metrics': fatigue_metrics,
        'workload_metrics': workload_metrics,
        'recommendations': recommendations
    }
    
    # Print sample recommendations
    if recommendations:
        sample_athlete = list(recommendations.keys())[0]
        print(f"\nSample recommendations for {sample_athlete}:")
        
        if 'fatigue_profile' in recommendations[sample_athlete]:
            print("\nFatigue profile:")
            for key, value in recommendations[sample_athlete]['fatigue_profile'].items():
                print(f"  {key}: {value:.2f}")
        
        if 'action_items' in recommendations[sample_athlete]:
            print("\nAction items:")
            for item in recommendations[sample_athlete]['action_items']:
                print(f"  {item['category']}: {item['finding']}")
                print(f"    → {item['action']}")
                
        # Save recommendations to file
        os.makedirs("emg_fatigue_results", exist_ok=True)
        
        # Save as text file
        with open(f"emg_fatigue_results/recommendations.txt", 'w') as f:
            for athlete, rec in recommendations.items():
                f.write(f"Recommendations for {athlete}:\n")
                f.write("-" * 50 + "\n")
                
                if 'fatigue_profile' in rec:
                    f.write("\nFatigue Profile:\n")
                    for key, value in rec['fatigue_profile'].items():
                        f.write(f"  {key}: {value:.2f}\n")
                
                if 'workload_profile' in rec:
                    f.write("\nWorkload Profile:\n")
                    for key, value in rec['workload_profile'].items():
                        f.write(f"  {key}: {value:.2f}\n")
                
                if 'action_items' in rec:
                    f.write("\nAction Items:\n")
                    for item in rec['action_items']:
                        f.write(f"  {item['category']}: {item['finding']}\n")
                        f.write(f"    → {item['action']}\n")
                
                f.write("\n\n")
    
    print("\nAnalysis complete! Results saved to 'emg_fatigue_results' directory.")
    return results

# Add this main function to run the analysis when the script is executed directly
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # For demonstration, you can use hardcoded database info if needed
    # COMMENT OUT THIS BLOCK if you're using environment variables
    """
    db_config = {
        'host': 'localhost',
        'user': 'your_username',
        'password': 'your_password',
        'database': 'your_database'
    }
    """
    
    # Use environment variables from .env file
    db_config = {
        'host': os.getenv('DB_HOST'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'database': os.getenv('DB_NAME')
    }
    
    # Check if we have valid database configuration
    missing_config = [key for key, value in db_config.items() if value is None]
    if missing_config:
        print(f"ERROR: Missing database configuration: {missing_config}")
        print("Please create a .env file with the required database information:")
        print("DB_HOST=your_host")
        print("DB_USER=your_username")
        print("DB_PASSWORD=your_password")
        print("DB_NAME=your_database_name")
        exit(1)
    
    # Create output directory for analysis results
    os.makedirs("emg_fatigue_results", exist_ok=True)
    
    print("Starting EMG fatigue analysis for pitchers...")
    results = run_emg_fatigue_analysis(db_config)
    print("Analysis complete! Results saved to emg_fatigue_results directory.")