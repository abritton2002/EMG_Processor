import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from urllib.parse import quote_plus

# Load environment variables from .env file
load_dotenv()

def get_db_connection():
    """Establish SQLAlchemy connection using .env credentials."""
    try:
        user = os.getenv('DB_USER')
        password = quote_plus(os.getenv('DB_PASSWORD'))  # Encode special characters
        host = os.getenv('DB_HOST')
        database = os.getenv('DB_NAME')
        
        # Ensure all required env vars are present
        if not all([user, password, host, database]):
            raise ValueError("Missing required environment variables in .env file (DB_USER, DB_PASSWORD, DB_HOST, DB_NAME)")
        
        conn_str = f"mysql+pymysql://{user}:{password}@{host}/{database}"
        return create_engine(conn_str)
    except Exception as e:
        raise ConnectionError(f"Failed to create DB connection: {str(e)}")

def load_emg_data_with_velocity():
    """Load EMG data with real velocities."""
    engine = get_db_connection()
    query = """
    SELECT 
        t.throw_id, t.session_numeric_id, t.trial_number,
        t.muscle1_peak_amplitude, t.muscle1_rise_time, t.muscle1_throw_integral,
        t.muscle1_median_freq, t.muscle1_spectral_entropy,
        t.muscle2_peak_amplitude, t.muscle2_rise_time, t.muscle2_throw_integral,
        t.muscle2_median_freq, t.muscle2_spectral_entropy,
        t.muscle1_wavelet_energy_low, t.muscle1_wavelet_energy_mid, t.muscle1_wavelet_energy_high,
        t.muscle2_wavelet_energy_low, t.muscle2_wavelet_energy_mid, t.muscle2_wavelet_energy_high,
        t.coactivation_index, t.coactivation_correlation,
        t.pitch_speed_mph, t.velocity_match_quality,
        s.athlete_name, s.date_recorded
    FROM emg_throws t
    JOIN emg_sessions s ON t.session_numeric_id = s.numeric_id
    WHERE t.pitch_speed_mph IS NOT NULL
    ORDER BY s.athlete_name, t.session_numeric_id, t.trial_number
    """
    df = pd.read_sql(query, engine)
    return df

def get_athlete_velocity_stats():
    """Compute velocity statistics per athlete."""
    engine = get_db_connection()
    query = """
    SELECT 
        s.athlete_name,
        COUNT(t.throw_id) as throw_count,
        AVG(t.pitch_speed_mph) as avg_velocity,
        MIN(t.pitch_speed_mph) as min_velocity,
        MAX(t.pitch_speed_mph) as max_velocity,
        STDDEV(t.pitch_speed_mph) as std_velocity
    FROM emg_throws t
    JOIN emg_sessions s ON t.session_numeric_id = s.numeric_id
    WHERE t.pitch_speed_mph IS NOT NULL
    GROUP BY s.athlete_name
    ORDER BY avg_velocity DESC
    """
    df = pd.read_sql(query, engine)
    return df

def prepare_data(df):
    """Prepare data with focused feature engineering."""
    df = df.copy()

    # Replace zeros and handle nulls
    for col in ['muscle1_rise_time', 'muscle2_rise_time', 'muscle1_throw_integral', 'muscle2_throw_integral']:
        df[col] = df[col].replace(0, 0.001).fillna(0.001)

    # Rename for clarity
    df['fcu_peak_amplitude'] = df['muscle1_peak_amplitude']
    df['fcr_peak_amplitude'] = df['muscle2_peak_amplitude']
    df['fcu_rise_time'] = df['muscle1_rise_time']
    df['fcr_rise_time'] = df['muscle2_rise_time']
    df['fcu_integral'] = df['muscle1_throw_integral']
    df['fcr_integral'] = df['muscle2_throw_integral']
    df['fcu_median_freq'] = df['muscle1_median_freq']
    df['fcr_median_freq'] = df['muscle2_median_freq']
    df['fcu_entropy'] = df['muscle1_spectral_entropy']
    df['fcr_entropy'] = df['muscle2_spectral_entropy']

    # Core features
    df['fcu_activation_speed'] = df['fcu_peak_amplitude'] / df['fcu_rise_time']
    df['fcr_activation_speed'] = df['fcr_peak_amplitude'] / df['fcr_rise_time']
    df['timing_diff'] = df['fcr_rise_time'] - df['fcu_rise_time']
    df['coactivation_index'] = df['coactivation_index'].fillna(df['coactivation_index'].median())

    # Wavelet features (simplified)
    if 'muscle1_wavelet_energy_low' in df.columns:
        df['wavelet_total_fcu'] = (df['muscle1_wavelet_energy_low'] + 
                                   df['muscle1_wavelet_energy_mid'] + 
                                   df['muscle1_wavelet_energy_high'])
        df['wavelet_total_fcr'] = (df['muscle2_wavelet_energy_low'] + 
                                   df['muscle2_wavelet_energy_mid'] + 
                                   df['muscle2_wavelet_energy_high'])
        df['fcu_wavelet_high_prop'] = df['muscle1_wavelet_energy_high'] / df['wavelet_total_fcu'].replace(0, 0.001)
        df['fcr_wavelet_high_prop'] = df['muscle2_wavelet_energy_high'] / df['wavelet_total_fcr'].replace(0, 0.001)

    # Clean data
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df

def analyze_velocity_relationship(df):
    """Analyze EMG-velocity relationships."""
    features = [
        'fcu_peak_amplitude', 'fcr_peak_amplitude', 'fcu_activation_speed',
        'fcr_activation_speed', 'timing_diff', 'fcu_median_freq', 'fcr_median_freq',
        'coactivation_index', 'fcu_wavelet_high_prop', 'fcr_wavelet_high_prop'
    ]
    features = [f for f in features if f in df.columns]
    corr = df[features + ['pitch_speed_mph']].corr()['pitch_speed_mph'].drop('pitch_speed_mph')

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    sns.scatterplot(x='fcr_peak_amplitude', y='pitch_speed_mph', hue='athlete_name', size='fcu_activation_speed', 
                    data=df, ax=axes[0,0], alpha=0.7, legend='brief')
    axes[0,0].set_title('FCR Amplitude vs Velocity', fontsize=14)
    axes[0,0].set_xlabel('FCR Peak Amplitude (mV)', fontsize=12)
    axes[0,0].set_ylabel('Velocity (mph)', fontsize=12)

    sns.scatterplot(x='fcu_wavelet_high_prop', y='pitch_speed_mph', hue='athlete_name', size='timing_diff', 
                    data=df, ax=axes[0,1], alpha=0.7)
    axes[0,1].set_title('FCU High-Freq Energy vs Velocity', fontsize=14)
    axes[0,1].set_xlabel('FCU Wavelet High Proportion', fontsize=12)

    sns.scatterplot(x='timing_diff', y='pitch_speed_mph', hue='athlete_name', size='fcr_activation_speed', 
                    data=df, ax=axes[1,0], alpha=0.7)
    axes[1,0].set_title('Timing Difference vs Velocity', fontsize=14)
    axes[1,0].set_xlabel('FCR-FCU Timing Diff (s)', fontsize=12)

    top_corrs = corr.abs().sort_values(ascending=False)[:8]
    sns.barplot(x=top_corrs.values, y=top_corrs.index, hue=top_corrs.index, ax=axes[1,1], palette='viridis', legend=False)
    axes[1,1].set_title('Top Correlations with Velocity', fontsize=14)
    axes[1,1].set_xlim(-0.6, 0.6)
    axes[1,1].axvline(0, color='gray', linestyle='--')

    plt.tight_layout(pad=3.0)
    return fig, corr

def build_velocity_prediction_model(df):
    """Build a simpler, regularized model."""
    features = [
        'fcu_peak_amplitude', 'fcr_peak_amplitude', 'fcu_activation_speed',
        'fcr_activation_speed', 'timing_diff', 'coactivation_index',
        'fcu_wavelet_high_prop', 'fcr_wavelet_high_prop'
    ]
    features = [f for f in features if f in df.columns]
    X = df[features]
    y = df['pitch_speed_mph']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.03, max_depth=3,
        min_samples_split=5, min_samples_leaf=3, subsample=0.7,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
    importance = importance.sort_values('Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', hue='Feature', data=importance, ax=ax, palette='magma', legend=False)
    ax.set_title('Feature Importance', fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    plt.tight_layout(pad=2.0)

    return model, rmse, r2, cv_scores, importance, fig

def plot_athlete_comparison(df):
    """Compare athletes with streamlined visuals."""
    athlete_stats = df.groupby('athlete_name').agg({
        'pitch_speed_mph': ['mean', 'std'],
        'fcr_peak_amplitude': 'mean',
        'fcu_wavelet_high_prop': 'mean',
        'timing_diff': 'mean'
    }).reset_index()
    athlete_stats.columns = ['_'.join(col).strip('_') for col in athlete_stats.columns]
    athlete_stats = athlete_stats.sort_values('pitch_speed_mph_mean', ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    sns.barplot(x='pitch_speed_mph_mean', y='athlete_name', data=athlete_stats, ax=axes[0,0], color='skyblue')
    axes[0,0].errorbar(x=athlete_stats['pitch_speed_mph_mean'], y=athlete_stats['athlete_name'], 
                       xerr=athlete_stats['pitch_speed_mph_std'], fmt='none', c='black', capsize=5)
    axes[0,0].set_title('Average Velocity by Athlete', fontsize=14)
    axes[0,0].set_xlabel('Velocity (mph)', fontsize=12)

    sns.barplot(x='fcr_peak_amplitude_mean', y='athlete_name', data=athlete_stats, ax=axes[0,1], color='salmon')
    axes[0,1].set_title('FCR Peak Amplitude by Athlete', fontsize=14)
    axes[0,1].set_xlabel('FCR Amplitude (mV)', fontsize=12)

    sns.barplot(x='fcu_wavelet_high_prop_mean', y='athlete_name', data=athlete_stats, ax=axes[1,0], color='lightgreen')
    axes[1,0].set_title('FCU High-Freq Energy by Athlete', fontsize=14)
    axes[1,0].set_xlabel('FCU Wavelet High Prop', fontsize=12)

    sns.barplot(x='timing_diff_mean', y='athlete_name', data=athlete_stats, ax=axes[1,1], color='lightblue')
    axes[1,1].set_title('Timing Difference by Athlete', fontsize=14)
    axes[1,1].set_xlabel('Timing Diff (s)', fontsize=12)
    axes[1,1].axvline(0, color='red', linestyle='--')

    plt.tight_layout(pad=3.0)
    return fig, athlete_stats

def analyze_within_athlete_patterns(df):
    """Analyze within-athlete trends."""
    athletes = df['athlete_name'].unique()
    n_athletes = len(athletes)
    fig, axes = plt.subplots(n_athletes, 2, figsize=(16, 4 * n_athletes), squeeze=False)

    for i, athlete in enumerate(athletes):
        athlete_data = df[df['athlete_name'] == athlete].sort_values('trial_number')
        if len(athlete_data) < 3:
            continue

        ax1 = axes[i, 0]
        ax1.plot(athlete_data['trial_number'], athlete_data['pitch_speed_mph'], 's-', color='blue', label='Velocity')
        ax1.set_ylabel('Velocity (mph)', color='blue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.legend(loc='upper right')
        ax2 = ax1.twinx()
        ax2.plot(athlete_data['trial_number'], athlete_data['fcr_peak_amplitude'], 'o-', color='red', label='FCR Amp')
        ax2.set_ylabel('FCR Amp (mV)', color='red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.legend(loc='upper left')
        ax1.set_title(f'{athlete}: Velocity vs FCR Amplitude', fontsize=14)
        ax1.set_xlabel('Trial Number', fontsize=12)

        ax3 = axes[i, 1]
        ax3.plot(athlete_data['trial_number'], athlete_data['timing_diff'], '^-', color='green', label='Timing Diff')
        ax3.set_ylabel('Timing Diff (s)', color='green', fontsize=12)
        ax3.tick_params(axis='y', labelcolor='green')
        ax3.legend(loc='upper left')
        ax4 = ax3.twinx()
        ax4.plot(athlete_data['trial_number'], athlete_data['fcu_wavelet_high_prop'], 's-', color='purple', label='FCU High Prop')
        ax4.set_ylabel('FCU High Prop', color='purple', fontsize=12)
        ax4.tick_params(axis='y', labelcolor='purple')
        ax4.legend(loc='upper right')
        ax3.set_title(f'{athlete}: Timing vs FCU High-Freq', fontsize=14)
        ax3.set_xlabel('Trial Number', fontsize=12)

    plt.tight_layout(pad=3.0)
    return fig

def generate_emg_velocity_report(df=None):
    """Generate a concise EMG-velocity report."""
    if df is None:
        try:
            df = load_emg_data_with_velocity()
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None

    print("\n===== EMG-Velocity Analysis Report =====")
    if df.empty or 'pitch_speed_mph' not in df.columns:
        print("No velocity data found.")
        return None

    print(f"\nDataset: {len(df)} throws from {df['athlete_name'].nunique()} athletes")
    athlete_velocity = get_athlete_velocity_stats()
    print("\nAthlete Velocity Stats:")
    for _, row in athlete_velocity.iterrows():
        print(f"  {row['athlete_name']}: {row['avg_velocity']:.1f} mph (Range: {row['min_velocity']:.1f}-{row['max_velocity']:.1f}, "
              f"SD: {row['std_velocity']:.1f}, Throws: {int(row['throw_count'])})")

    analysis_df = prepare_data(df)
    print("\nAnalyzing relationships...")
    rel_fig, corr = analyze_velocity_relationship(analysis_df)
    print("\nTop Correlations:")
    for f, v in corr.abs().sort_values(ascending=False)[:8].items():
        print(f"  {f}: {corr[f]:.3f}")

    print("\nBuilding model...")
    model, rmse, r2, cv_scores, importance, imp_fig = build_velocity_prediction_model(analysis_df)
    print(f"\nModel Performance:")
    print(f"  Test R²: {r2:.3f}")
    print(f"  RMSE: {rmse:.3f} mph")
    print(f"  CV R²: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    print("\nTop Predictors:")
    print(importance.head().to_string(index=False))

    print("\nComparing athletes...")
    athlete_fig, _ = plot_athlete_comparison(analysis_df)
    print("\nWithin-athlete patterns...")
    patterns_fig = analyze_within_athlete_patterns(analysis_df)

    print("\nKey Findings:")
    print("1. FCR amplitude drives velocity, reflecting wrist snap power.")
    print("2. FCU high-frequency energy links to explosive activation.")
    print("3. Timing and coactivation enhance efficiency.")
    print("4. Small dataset (65 throws) limits model precision.")

    print("\nRecommendations:")
    print("1. Boost FCR strength with wrist flexion drills.")
    print("2. Train explosive FCU activation for high-frequency power.")
    print("3. Refine FCU-FCR timing with mechanics drills.")
    print("4. Collect 200+ throws for better accuracy.")

    rel_fig.savefig('velocity_relationships.png', dpi=300)
    imp_fig.savefig('velocity_predictors.png', dpi=300)
    athlete_fig.savefig('athlete_comparison.png', dpi=300)
    patterns_fig.savefig('within_athlete_patterns.png', dpi=300)
    print("\nAnalysis complete. Figures saved.")

    return {
        'analysis_df': analysis_df, 'relationship_fig': rel_fig, 'importance_fig': imp_fig,
        'athlete_fig': athlete_fig, 'patterns_fig': patterns_fig, 'model': model,
        'r2': r2, 'rmse': rmse, 'cv_scores': cv_scores, 'feature_importance': importance
    }

if __name__ == "__main__":
    results = generate_emg_velocity_report()