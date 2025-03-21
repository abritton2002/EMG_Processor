import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables for database connection
load_dotenv()

def get_db_connection():
    """Create database connection from environment variables"""
    try:
        user = os.getenv('DB_USER')
        password = os.getenv('DB_PASSWORD')
        if password:
            from urllib.parse import quote_plus
            password = quote_plus(password)
        host = os.getenv('DB_HOST')
        database = os.getenv('DB_NAME')
        
        if not all([user, password, host, database]):
            raise ValueError("Missing required environment variables")
            
        conn_str = f"mysql+pymysql://{user}:{password}@{host}/{database}"
        return create_engine(conn_str)
    except Exception as e:
        print(f"Failed to create DB connection: {str(e)}")
        return None

def load_data():
    """Load EMG and biomechanics data from database"""
    engine = get_db_connection()
    if not engine:
        print("Could not establish database connection")
        return None
        
    # Load EMG data
    emg_query = """
    SELECT 
        t.throw_id, t.session_numeric_id, t.session_trial, t.trial_number,
        t.muscle1_peak_amplitude, t.muscle2_peak_amplitude,
        t.pitch_speed_mph, t.velocity_match_quality,
        s.athlete_name, s.date_recorded
    FROM emg_throws t
    JOIN emg_sessions s ON t.session_numeric_id = s.numeric_id
    WHERE t.pitch_speed_mph IS NOT NULL
    ORDER BY s.athlete_name, s.date_recorded, t.trial_number
    """
    
    # Load biomechanics data
    bio_query = """
    SELECT 
        session_trial, 
        elbow_varus_moment
    FROM poi
    WHERE elbow_varus_moment IS NOT NULL
    """
    
    try:
        emg_df = pd.read_sql(emg_query, engine)
        bio_df = pd.read_sql(bio_query, engine)
        
        # Merge the datasets
        combined_df = pd.merge(emg_df, bio_df, on='session_trial', how='inner')
        return combined_df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_muscle_protection_ratio(df):
    """Calculate the Muscle Protection Ratio (FCU activation / elbow varus moment)"""
    if df is None or df.empty:
        return None
    
    # Calculate basic ratio (FCU activation / elbow varus moment)
    df['muscle_protection_ratio'] = df['muscle1_peak_amplitude'] / df['elbow_varus_moment']
    
    # Group by athlete and session
    grouped = df.groupby(['athlete_name', 'session_numeric_id'])
    
    results = []
    
    # Process each athlete's session
    for (athlete, session_id), group_df in grouped:
        # Sort by trial number
        session_df = group_df.sort_values('trial_number')
        
        # Get baseline from first 5 throws (or fewer if not available)
        baseline_throws = min(5, len(session_df))
        if baseline_throws < 3:  # Skip sessions with too few throws
            continue
            
        baseline_protection = session_df.iloc[:baseline_throws]['muscle_protection_ratio'].mean()
        
        # Calculate normalized ratio (percentage of baseline)
        session_df['normalized_protection'] = session_df['muscle_protection_ratio'] / baseline_protection * 100
        
        # Set risk levels based on normalized protection percentage
        def assign_risk(row):
            if row['normalized_protection'] >= 90:
                return 'Low Risk'
            elif row['normalized_protection'] >= 70:
                return 'Moderate Risk'
            else:
                return 'High Risk'
        
        session_df['risk_level'] = session_df.apply(assign_risk, axis=1)
        
        # Add to results
        results.append(session_df)
    
    if not results:
        return None
        
    # Combine all processed sessions
    result_df = pd.concat(results)
    return result_df

def analyze_protection_ratio():
    """Main function to analyze muscle protection ratio"""
    # Create output directory
    os.makedirs('protection_analysis', exist_ok=True)
    
    # Load and process data
    print("Loading data...")
    data = load_data()
    
    if data is None or data.empty:
        print("No data available for analysis")
        return
    
    print(f"Loaded {len(data)} throws from {data['athlete_name'].nunique()} athletes")
    
    # Calculate protection ratio
    print("Calculating muscle protection ratio...")
    protection_df = calculate_muscle_protection_ratio(data)
    
    if protection_df is None or protection_df.empty:
        print("Could not calculate protection ratio")
        return
    
    # Save processed data
    protection_df.to_csv('protection_analysis/protection_ratio_data.csv', index=False)
    
    # Basic statistics
    print("\nMuscle Protection Ratio Statistics:")
    stats = protection_df.groupby('athlete_name')['muscle_protection_ratio'].agg(
        ['mean', 'std', 'min', 'max']).reset_index()
    print(stats)
    
    # Risk distribution
    print("\nRisk Distribution by Athlete:")
    risk_counts = protection_df.groupby(['athlete_name', 'risk_level']).size().unstack(fill_value=0)
    risk_percent = risk_counts.div(risk_counts.sum(axis=1), axis=0) * 100
    print(risk_percent)
    
    # Create visualizations
    plot_protection_trends(protection_df)
    plot_protection_vs_velocity(protection_df)
    plot_risk_distribution(protection_df)
    
    print("\nAnalysis complete! Results saved to 'protection_analysis' directory")

def plot_protection_trends(df):
    """Plot protection ratio trends across throws for each athlete/session"""
    print("Creating protection trend plots...")
    
    # Group by athlete and session
    grouped = df.groupby(['athlete_name', 'session_numeric_id'])
    
    for (athlete, session_id), session_df in grouped:
        if len(session_df) < 5:  # Skip sessions with too few throws
            continue
            
        # Sort by trial number
        session_df = session_df.sort_values('trial_number')
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot normalized protection ratio
        plt.plot(session_df['trial_number'], session_df['normalized_protection'], 
                'o-', color='blue', linewidth=2, markersize=8)
        
        # Add risk zone shading
        plt.axhspan(0, 70, alpha=0.2, color='red', label='High Risk')
        plt.axhspan(70, 90, alpha=0.2, color='orange', label='Moderate Risk')
        plt.axhspan(90, max(110, session_df['normalized_protection'].max() * 1.1), 
                  alpha=0.2, color='green', label='Low Risk')
        
        # Add 100% baseline reference line
        plt.axhline(y=100, color='k', linestyle='--', alpha=0.7, label='Baseline')
        
        # Labels and title
        plt.xlabel('Throw Number', fontsize=12)
        plt.ylabel('Normalized Protection Ratio (%)', fontsize=12)
        plt.title(f'{athlete} - Session {session_id} Protection Trend', fontsize=14)
        
        # Add legend
        plt.legend(loc='best')
        
        # Y-axis limits
        plt.ylim(0, max(110, session_df['normalized_protection'].max() * 1.1))
        
        # Save figure
        filename = f'protection_analysis/{athlete}_session_{session_id}_protection_trend.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

def plot_protection_vs_velocity(df):
    """Plot protection ratio vs velocity for each athlete"""
    print("Creating protection vs velocity plots...")
    
    # Loop through each athlete
    for athlete, athlete_df in df.groupby('athlete_name'):
        if len(athlete_df) < 5:  # Skip athletes with too few throws
            continue
            
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot colored by risk level
        scatter = plt.scatter(
            athlete_df['pitch_speed_mph'], 
            athlete_df['normalized_protection'],
            c=athlete_df['risk_level'].map({'Low Risk': 0, 'Moderate Risk': 1, 'High Risk': 2}),
            cmap='RdYlGn_r',
            s=80,
            alpha=0.7
        )
        
        # Add risk zone shading
        plt.axhspan(0, 70, alpha=0.2, color='red')
        plt.axhspan(70, 90, alpha=0.2, color='orange')
        plt.axhspan(90, max(110, athlete_df['normalized_protection'].max() * 1.1), 
                  alpha=0.2, color='green')
        
        # Add baseline reference line
        plt.axhline(y=100, color='k', linestyle='--', alpha=0.7, label='Baseline')
        
        # Labels and title
        plt.xlabel('Pitch Velocity (mph)', fontsize=12)
        plt.ylabel('Normalized Protection Ratio (%)', fontsize=12)
        plt.title(f'{athlete} - Protection vs Velocity', fontsize=14)
        
        # Add legend
        categories = ['Low Risk', 'Moderate Risk', 'High Risk']
        legend1 = plt.legend(handles=scatter.legend_elements()[0], labels=categories,
                          title="Risk Level", loc="upper right")
        plt.gca().add_artist(legend1)
        
        # Add trend line
        if len(athlete_df) >= 5:
            try:
                z = np.polyfit(athlete_df['pitch_speed_mph'], athlete_df['normalized_protection'], 1)
                p = np.poly1d(z)
                plt.plot(np.linspace(athlete_df['pitch_speed_mph'].min(), athlete_df['pitch_speed_mph'].max(), 100), 
                       p(np.linspace(athlete_df['pitch_speed_mph'].min(), athlete_df['pitch_speed_mph'].max(), 100)), 
                       "r--", alpha=0.7)
            except:
                pass
        
        # Save figure
        filename = f'protection_analysis/{athlete}_protection_vs_velocity.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

def plot_risk_distribution(df):
    """Plot risk distribution for all athletes"""
    print("Creating risk distribution plots...")
    
    # Calculate risk percentages
    risk_counts = df.groupby(['athlete_name', 'risk_level']).size().unstack(fill_value=0)
    risk_percent = risk_counts.div(risk_counts.sum(axis=1), axis=0) * 100
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create stacked bar chart
    risk_percent.plot(kind='bar', stacked=True, 
                    color=['red', 'green', 'orange'], 
                    figsize=(12, 8))
    
    # Labels and title
    plt.xlabel('Athlete', fontsize=12)
    plt.ylabel('Percentage of Throws', fontsize=12)
    plt.title('Risk Distribution by Athlete', fontsize=14)
    
    # Add percentage labels on bars
    for i, athlete in enumerate(risk_percent.index):
        cumulative = 0
        for risk, percentage in risk_percent.loc[athlete].items():
            if percentage > 0:
                plt.text(i, cumulative + percentage/2, f'{percentage:.0f}%', 
                       ha='center', va='center', color='white', fontweight='bold')
            cumulative += percentage
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Save figure
    plt.tight_layout()
    plt.savefig('protection_analysis/risk_distribution.png', dpi=300)
    plt.close()
    
    # Create summary table
    summary = risk_percent.copy()
    summary['Total Throws'] = risk_counts.sum(axis=1)
    summary['Avg Protection Ratio'] = df.groupby('athlete_name')['muscle_protection_ratio'].mean()
    
    # Save summary table
    summary.to_csv('protection_analysis/risk_summary.csv')

if __name__ == "__main__":
    analyze_protection_ratio()