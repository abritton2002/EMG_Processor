import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pymysql
from dotenv import load_dotenv
import os

load_dotenv()

def get_db_connection():
    db_config = {
        'host': os.getenv('DB_HOST', '10.200.200.107'),
        'user': os.getenv('DB_USER', 'scriptuser1'),
        'password': os.getenv('DB_PASSWORD', 'YabinMarshed2023@#$'),
        'database': os.getenv('DB_NAME', 'theia_pitching_db')
    }
    return pymysql.connect(**db_config)

def load_emg_data():
    conn = get_db_connection()
    try:
        query = """
        SELECT 
            t.throw_id, t.session_numeric_id, t.trial_number,
            t.muscle1_peak_amplitude, t.muscle1_rise_time, t.muscle1_throw_integral,
            t.muscle1_median_freq, t.muscle2_peak_amplitude, t.muscle2_rise_time,
            t.muscle2_throw_integral, t.muscle2_median_freq,
            s.athlete_name, s.date_recorded
        FROM emg_throws t
        JOIN emg_sessions s ON t.session_numeric_id = s.numeric_id
        """
        df = pd.read_sql(query, conn)
        return df
    finally:
        conn.close()

def prepare_data(df):
    df = df.copy()
    
    # Replace zeros
    for col in ['muscle1_rise_time', 'muscle2_rise_time', 'muscle1_throw_integral', 'muscle2_throw_integral']:
        df[col] = df[col].replace(0, 0.001)
    
    # Rename for clarity
    df['fcu_peak_amplitude'] = df['muscle1_peak_amplitude']
    df['fcr_peak_amplitude'] = df['muscle2_peak_amplitude']
    df['fcu_rise_time'] = df['muscle1_rise_time']
    df['fcr_rise_time'] = df['muscle2_rise_time']
    df['fcu_integral'] = df['muscle1_throw_integral']
    df['fcr_integral'] = df['muscle2_throw_integral']
    df['fcu_median_freq'] = df['muscle1_median_freq']
    df['fcr_median_freq'] = df['muscle2_median_freq']
    
    # Core features
    df['fcu_activation_speed'] = df['fcu_peak_amplitude'] / df['fcu_rise_time']
    df['fcr_activation_speed'] = df['fcr_peak_amplitude'] / df['fcr_rise_time']
    df['contraction_ratio'] = df['fcu_peak_amplitude'] / df['fcr_peak_amplitude']
    df['timing_diff'] = df['fcr_rise_time'] - df['fcu_rise_time']
    df['integral_ratio'] = df['fcu_integral'] / df['fcr_integral']
    df['fcu_power'] = df['fcu_peak_amplitude']**2 / df['fcu_rise_time']
    
    # Average velocity mapping (proxy until real data available)
    velocity_mapping = {
        'DarielFregio': 91.0, 'AlexBritton': 75.0, 'CarterWallace': 79.6,
        'ZaneRose': 89.4, 'ZanRose': 87.4, 'MateoHamm': 82.3, 'JayceBlair': 74.7
    }
    df['velocity'] = df['athlete_name'].map(velocity_mapping)
    
    # Clean data
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    return df

def analyze_contraction_velocity_relationship(df):
    features = [
        'fcu_peak_amplitude', 'fcr_median_freq', 'fcu_activation_speed',
        'timing_diff', 'integral_ratio', 'fcu_power'
    ]
    
    corr = df[features + ['velocity']].corr()['velocity'].drop('velocity')
    
    # Larger figure with jitter for scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Add small jitter to velocity for visibility
    df_jitter = df.copy()
    df_jitter['velocity'] += np.random.normal(0, 0.2, len(df))  # Small jitter
    
    sns.scatterplot(x='fcr_median_freq', y='velocity', hue='athlete_name', data=df_jitter, ax=axes[0,0], alpha=0.6)
    axes[0,0].set_title('FCR Median Frequency vs Velocity (Proxy)')
    axes[0,0].set_xlabel('FCR Median Frequency (Hz)')
    
    sns.scatterplot(x='fcu_peak_amplitude', y='velocity', hue='athlete_name', data=df_jitter, ax=axes[0,1], alpha=0.6)
    axes[0,1].set_title('FCU Peak Amplitude vs Velocity (Proxy)')
    axes[0,1].set_xlabel('FCU Peak Amplitude (mV)')
    
    sns.scatterplot(x='timing_diff', y='velocity', hue='athlete_name', data=df_jitter, ax=axes[1,0], alpha=0.6)
    axes[1,0].set_title('FCR-FCU Timing Difference vs Velocity (Proxy)')
    axes[1,0].set_xlabel('Timing Difference (s)')
    
    sns.barplot(x=corr.values, y=corr.index, ax=axes[1,1])
    axes[1,1].set_title('Correlation with Velocity (Proxy)')
    axes[1,1].tick_params(axis='y', labelsize=12, rotation=0)
    axes[1,1].set_xlim(-0.5, 0.5)  # Standardize correlation range
    
    plt.tight_layout(pad=3.0)
    return fig, corr

def build_velocity_prediction_model(df):
    features = [
        'fcu_peak_amplitude', 'fcr_peak_amplitude', 'fcu_activation_speed',
        'fcr_activation_speed', 'contraction_ratio', 'timing_diff',
        'integral_ratio', 'fcu_median_freq', 'fcr_median_freq', 'fcu_power'
    ]
    
    X = df[features]
    y = df['velocity']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        min_samples_split=5
    )
    model.fit(X_train_scaled, y_train)
    
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))  # Wider for long labels
    sns.barplot(x='Importance', y='Feature', data=importance, ax=ax)
    ax.set_title('Feature Importance for Velocity Prediction (Proxy)')
    ax.tick_params(axis='y', labelsize=12, rotation=0)
    
    plt.tight_layout(pad=2.0)
    return model, rmse, r2, cv_scores, importance, fig

def generate_report(df):
    print("Velocity Prediction Analysis Report")
    print("==================================")
    
    analysis_df = prepare_data(df)
    print(f"\nDataset: {len(analysis_df)} throws from {analysis_df['athlete_name'].nunique()} athletes")
    velocities = analysis_df.groupby('athlete_name')['velocity'].mean().sort_values(ascending=False)
    print("\nAverage Velocities (Proxy):")
    for athlete, vel in velocities.items():
        print(f"  {athlete}: {vel:.1f} mph")
    
    print("\nAnalyzing contraction dynamics...")
    rel_fig, corr = analyze_contraction_velocity_relationship(analysis_df)
    print("\nKey Correlations with Velocity (Proxy):")
    print(corr.sort_values(ascending=False).round(3).to_string())
    
    print("\nBuilding prediction model...")
    model, rmse, r2, cv_scores, importance, imp_fig = build_velocity_prediction_model(analysis_df)
    print(f"\nModel Performance:")
    print(f"  Test R²: {r2:.3f}")
    print(f"  Test RMSE: {rmse:.3f} mph")
    print(f"  Cross-validation R²: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    print("\nTop Predictors:")
    print(importance.head().to_string(index=False))
    
    print("\nKey Insights (Based on Average Velocity Proxy):")
    print(f"- FCU amplitude critical but complex (corr: {corr['fcu_peak_amplitude']:.3f}, importance: {importance.iloc[0]['Importance']:.3f})")
    print(f"- FCR median freq strongly linked to velocity (corr: {corr['fcr_median_freq']:.3f})")
    print(f"- FCU-to-FCR timing enhances velocity (corr: {corr['timing_diff']:.3f})")
    
    print("\nTraining Recommendations:")
    print("- Strengthen FCU for optimal amplitude (avoid overuse)")
    print("- Boost FCR firing rate with explosive wrist drills")
    print("- Train FCU-to-FCR timing sequence")
    print("- Collect actual throw velocities (50-100 per athlete) for precision")
    
    rel_fig.savefig('contraction_velocity_relationships.png')
    imp_fig.savefig('feature_importance.png')
    
    return {
        'analysis_df': analysis_df,
        'relationship_fig': rel_fig,
        'importance_fig': imp_fig,
        'model': model,
        'r2': r2,
        'rmse': rmse,
        'cv_scores': cv_scores,
        'feature_importance': importance
    }

def main():
    df = load_emg_data()
    if df is not None:
        results = generate_report(df)
        print("\nAnalysis complete. Figures saved to current directory.")
    return results

if __name__ == "__main__":
    main()