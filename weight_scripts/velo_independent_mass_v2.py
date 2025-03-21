import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# Assume your original functions are available
from no_bodyweight_velo import (
    load_emg_data_with_velocity_and_bodyweight,
    prepare_data,
    build_improved_model
)

# Set up plotting style
plt.style.use('seaborn-v0_8')

# Step 1: Load and Explore Raw Data
def analyze_raw_data_influence():
    """Check body weight's baseline influence on velocity and EMG features."""
    df = load_emg_data_with_velocity_and_bodyweight()
    print(f"Loaded {len(df)} throws from {df['athlete_name'].nunique()} athletes")

    # Key features to check
    features = [
        'pitch_speed_mph', 'muscle1_peak_amplitude', 'muscle2_throw_integral',
        'muscle1_median_freq', 'mass_kilograms'
    ]
    
    # Correlation matrix
    corr_matrix = df[features].corr()
    print("\nCorrelation Matrix (Raw Data):")
    print(corr_matrix)

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Heatmap (Raw Data)")
    plt.tight_layout()
    plt.savefig('raw_correlation_heatmap.png', dpi=300)
    plt.close()

    # Scatterplot: Body weight vs. Pitch speed
    plt.figure(figsize=(8, 6))
    plt.scatter(df['mass_kilograms'], df['pitch_speed_mph'], alpha=0.5)
    plt.xlabel('Body Weight (kg)')
    plt.ylabel('Pitch Speed (mph)')
    plt.title('Body Weight vs. Pitch Speed')
    plt.grid(True, alpha=0.3)
    plt.savefig('weight_vs_velocity.png', dpi=300)
    plt.close()

    return df

# Step 2 & 3: Test Normalization and Residualization
def test_normalization_and_residualization(df):
    """Compare raw, normalized, and residualized features."""
    analysis_df = prepare_data(df)
    
    # Debug: Print available columns to check what's there
    print("\nColumns in analysis_df after prepare_data():")
    print(analysis_df.columns.tolist())

    # Use muscle1-based names initially, as residualization happens before renaming
    base_feature = 'muscle1_peak_amplitude'  # Raw, before renaming to FCU
    norm_feature = 'muscle1_peak_amplitude_per_kg'  # Normalized
    resid_feature = 'muscle1_peak_amplitude_bodyweight_residuals'  # Residualized

    # Switch to FCU names for base and normalized if renaming has occurred
    if 'fcu_peak_amplitude' in analysis_df.columns:
        base_feature = 'fcu_peak_amplitude'
        norm_feature = 'fcu_peak_amplitude_per_kg'
        # Note: resid_feature stays as muscle1_ because no fcu_ residual exists

    # Verify columns exist before proceeding
    missing_cols = [col for col in [base_feature, norm_feature, resid_feature] 
                    if col not in analysis_df.columns]
    if missing_cols:
        raise KeyError(f"Missing expected columns: {missing_cols}")

    # Correlations with body weight
    correlations = {
        'Raw': pearsonr(analysis_df['mass_kilograms'], analysis_df[base_feature])[0],
        'Normalized': pearsonr(analysis_df['mass_kilograms'], analysis_df[norm_feature])[0],
        'Residualized': pearsonr(analysis_df['mass_kilograms'], analysis_df[resid_feature])[0]
    }
    print("\nCorrelations with Body Weight:")
    for method, corr in correlations.items():
        print(f"  {method}: {corr:.3f}")

    # Simple predictive power comparison
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        analysis_df[[base_feature, norm_feature, resid_feature]],
        analysis_df['pitch_speed_mph'],
        test_size=0.2,
        random_state=42
    )
    
    for feature in [base_feature, norm_feature, resid_feature]:
        model = LinearRegression()
        model.fit(X_train[[feature]], y_train)
        y_pred = model.predict(X_test[[feature]])
        r2 = r2_score(y_test, y_pred)
        print(f"  R² for {feature}: {r2:.3f}")

    # Scatterplot comparison
    plt.figure(figsize=(12, 4))
    for i, (feat, label) in enumerate([
        (base_feature, 'Raw'), (norm_feature, 'Normalized'), (resid_feature, 'Residualized')
    ], 1):
        plt.subplot(1, 3, i)
        plt.scatter(analysis_df['mass_kilograms'], analysis_df[feat], alpha=0.5)
        plt.xlabel('Body Weight (kg)')
        plt.ylabel(f'{label} Peak Amplitude')
        plt.title(f'{label} vs. Body Weight')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_adjustment_comparison.png', dpi=300)
    plt.close()

    return analysis_df

# Step 4: Analyze Full Model Behavior
def analyze_model_independence(analysis_df):
    """Check if the full model predictions are independent of body weight."""
    from sklearn.model_selection import train_test_split
    results = build_improved_model(analysis_df)
    
    # Define the same candidate features as in build_improved_model
    candidate_features = [
        'fcu_peak_amplitude', 'fcr_peak_amplitude', 'amplitude_sum', 'amplitude_ratio', 'amplitude_product',
        'fcu_rise_time', 'fcr_rise_time', 'timing_diff', 'timing_diff_abs', 'timing_ratio',
        'fcu_activation_speed', 'fcr_activation_speed',
        'fcu_integral', 'fcr_integral', 'integral_sum', 'integral_ratio', 'integral_product',
        'fcu_median_freq', 'fcr_median_freq', 'median_freq_diff', 'median_freq_ratio', 'median_freq_sum',
        'fcu_entropy', 'fcr_entropy', 'entropy_sum', 'entropy_ratio',
        'coactivation_index', 'coactivation_correlation', 'waveform_similarity', 'temporal_overlap',
        'fcu_low_energy_norm', 'fcu_mid_energy_norm', 'fcu_high_energy_norm',
        'fcr_low_energy_norm', 'fcr_mid_energy_norm', 'fcr_high_energy_norm',
        'low_energy_ratio', 'mid_energy_ratio', 'high_energy_ratio',
        'fcu_high_low_ratio', 'fcr_high_low_ratio'
    ]
    
    # Filter to available features
    features = [f for f in candidate_features if f in analysis_df.columns]
    X = analysis_df[features]

    # Scale and select features in the same way as build_improved_model
    X_scaled = results['scaler'].transform(X)
    X_selected = results['selector'].transform(X_scaled)
    y_pred = results['best_model'].predict(X_selected)
    y_actual = analysis_df['pitch_speed_mph']

    # Calculate errors
    errors = y_actual - y_pred
    analysis_df['prediction_error'] = errors

    # Check top features' correlations with body weight
    print("\nTop 5 Features and Their Correlations with Body Weight:")
    top_features = results['feature_importance'].head(5)
    for _, row in top_features.iterrows():
        feat = row['Feature']
        corr, pval = pearsonr(analysis_df['mass_kilograms'], analysis_df[feat])
        print(f"  {feat}: Corr = {corr:.3f}, p-value = {pval:.3f}")

    # Plot errors vs. body weight
    plt.figure(figsize=(8, 6))
    plt.scatter(analysis_df['mass_kilograms'], errors, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Body Weight (kg)')
    plt.ylabel('Prediction Error (mph)')
    plt.title('Prediction Error vs. Body Weight')
    plt.grid(True, alpha=0.3)
    plt.savefig('error_vs_weight.png', dpi=300)
    plt.close()

    # Baseline model with raw numeric features only
    raw_features = [f for f in analysis_df.columns if 'per_kg' not in f and 'residuals' not in f 
                    and f not in ['pitch_speed_mph', 'throw_id', 'session_numeric_id', 'trial_number', 
                                  'athlete_name', 'date_recorded', 'prediction_error', 'velocity_match_quality']]
    X_raw = analysis_df[raw_features].select_dtypes(include=[np.number])  # Ensure numeric only
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_actual, test_size=0.2, random_state=42)
    baseline_model = LinearRegression()
    baseline_model.fit(X_train, y_train)
    y_pred_baseline = baseline_model.predict(X_test)
    baseline_r2 = r2_score(y_test, y_pred_baseline)
    print(f"\nBaseline Model (Raw Features Only) Test R²: {baseline_r2:.3f}")

    return results

# Step 5: Athlete-Level Analysis
def analyze_athlete_level(df):
    """Examine body weight effects by athlete."""
    athlete_summary = df.groupby('athlete_name').agg({
        'mass_kilograms': 'mean',
        'pitch_speed_mph': 'mean',
        'muscle1_peak_amplitude': 'mean',
        'fcu_peak_amplitude_per_kg': 'mean',
        'muscle1_peak_amplitude_bodyweight_residuals': 'mean'  # Fixed to match existing column
    }).reset_index()
    
    print("\nAthlete-Level Summary:")
    print(athlete_summary)

    # Plot athlete-level trends
    plt.figure(figsize=(10, 6))
    for col, label in [
        ('muscle1_peak_amplitude', 'Raw FCU Amplitude'),
        ('fcu_peak_amplitude_per_kg', 'Normalized FCU Amplitude'),
        ('muscle1_peak_amplitude_bodyweight_residuals', 'Residualized FCU Amplitude')
    ]:
        plt.scatter(athlete_summary['mass_kilograms'], athlete_summary[col], label=label, alpha=0.7)
    plt.xlabel('Body Weight (kg)')
    plt.ylabel('FCU Amplitude Metrics')
    plt.title('Athlete-Level FCU Amplitude vs. Body Weight')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('athlete_level_analysis.png', dpi=300)
    plt.close()

# Main Execution
if __name__ == "__main__":
    print("Starting Body Weight Influence Analysis...")
    
    # Step 1
    raw_df = analyze_raw_data_influence()
    
    # Step 2 & 3
    analysis_df = test_normalization_and_residualization(raw_df)
    
    # Step 4
    results = analyze_model_independence(analysis_df)
    
    # Step 5
    analyze_athlete_level(analysis_df)
    
    print("\nAnalysis Complete! Check generated plots and console output.")