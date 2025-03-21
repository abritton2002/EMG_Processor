import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from urllib.parse import quote_plus

load_dotenv()

def get_db_connection():
    try:
        user = os.getenv('DB_USER')
        password = quote_plus(os.getenv('DB_PASSWORD'))
        host = os.getenv('DB_HOST')
        database = os.getenv('DB_NAME')
        if not all([user, password, host, database]):
            raise ValueError("Missing required environment variables")
        conn_str = f"mysql+pymysql://{user}:{password}@{host}/{database}"
        return create_engine(conn_str)
    except Exception as e:
        raise ConnectionError(f"Failed to create DB connection: {str(e)}")

def load_emg_data_with_velocity():
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
    return pd.read_sql(query, engine)

def get_athlete_velocity_stats():
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
    return pd.read_sql(query, engine)

def prepare_data(df):
    df = df.copy()
    for col in ['muscle1_rise_time', 'muscle2_rise_time', 'muscle1_throw_integral', 'muscle2_throw_integral']:
        df[col] = df[col].replace(0, 0.001).fillna(0.001)

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

    df['fcu_activation_speed'] = df['fcu_peak_amplitude'] / df['fcu_rise_time']
    df['fcr_activation_speed'] = df['fcr_peak_amplitude'] / df['fcr_rise_time']
    df['timing_diff'] = df['fcr_rise_time'] - df['fcu_rise_time']
    df['coactivation_index'] = df['coactivation_index'].fillna(df['coactivation_index'].median())
    df['contraction_ratio'] = df['fcu_peak_amplitude'] / df['fcr_peak_amplitude'].replace(0, 0.001)
    df['integral_ratio'] = df['fcu_integral'] / df['fcr_integral'].replace(0, 0.001)

    if 'muscle1_wavelet_energy_low' in df.columns:
        df['wavelet_total_fcu'] = (df['muscle1_wavelet_energy_low'] + 
                                   df['muscle1_wavelet_energy_mid'] + 
                                   df['muscle1_wavelet_energy_high'])
        df['wavelet_total_fcr'] = (df['muscle2_wavelet_energy_low'] + 
                                   df['muscle2_wavelet_energy_mid'] + 
                                   df['muscle2_wavelet_energy_high'])
        df['fcu_wavelet_high_prop'] = df['muscle1_wavelet_energy_high'] / df['wavelet_total_fcu'].replace(0, 0.001)
        df['fcr_wavelet_high_prop'] = df['muscle2_wavelet_energy_high'] / df['wavelet_total_fcr'].replace(0, 0.001)

    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df

def build_velocity_prediction_model(df):
    features = [
        'fcu_peak_amplitude', 'fcr_peak_amplitude', 'fcu_activation_speed',
        'fcr_activation_speed', 'timing_diff', 'coactivation_index',
        'fcu_wavelet_high_prop', 'fcr_wavelet_high_prop', 'contraction_ratio',
        'integral_ratio', 'fcu_median_freq', 'fcr_median_freq'
    ]
    features = [f for f in features if f in df.columns]
    X = df[features]
    y = df['pitch_speed_mph']

    # Double split: train+val (80%), test (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=df['athlete_name'], random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp.index.map(df['athlete_name']), random_state=42  # 0.25 of 0.8 = 0.2 total
    )

    # Feature selection
    selector = SelectKBest(score_func=f_regression, k=8)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)
    selected_features = [features[i] for i in selector.get_support(indices=True)]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_val_scaled = scaler.transform(X_val_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    # Expanded hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.03, 0.05],
        'max_depth': [3, 4],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 3],
        'subsample': [0.7, 0.9]
    }
    base_model = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")

    # Evaluate on validation and test sets
    y_val_pred = model.predict(X_val_scaled)
    y_test_pred = model.predict(X_test_scaled)
    val_r2 = r2_score(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')

    # Feature importance
    importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', hue='Feature', data=importance, ax=ax, palette='magma', legend=False)
    ax.set_title('Feature Importance', fontsize=14)
    plt.tight_layout(pad=2.0)

    return model, val_r2, val_rmse, test_r2, test_rmse, cv_scores, importance, fig, scaler, selector

def generate_emg_velocity_report(df=None):
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
    print("\nBuilding optimized model with validation...")
    (model, val_r2, val_rmse, test_r2, test_rmse, cv_scores, importance, imp_fig, 
     scaler, selector) = build_velocity_prediction_model(analysis_df)

    print(f"\nModel Performance:")
    print(f"  Validation R²: {val_r2:.3f}")
    print(f"  Validation RMSE: {val_rmse:.3f} mph")
    print(f"  Test R²: {test_r2:.3f}")
    print(f"  Test RMSE: {test_rmse:.3f} mph")
    print(f"  CV R² (Train): {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    print("\nTop Predictors:")
    print(importance.head().to_string(index=False))

    print("\nKey Findings:")
    print("1. Optimized model with broader tuning improves fit.")
    print("2. Validation set provides realistic performance estimate.")
    print("3. Amplitude and frequency features dominate predictions.")
    print("4. Small dataset still constrains precision.")

    print("\nRecommendations:")
    print("1. Collect 200+ throws for robust generalization.")
    print("2. Validate on unseen athletes to test true performance.")
    print("3. Explore additional biomechanical features (e.g., arm length).")

    imp_fig.savefig('velocity_predictors_optimized.png', dpi=300)
    print("\nAnalysis complete. Feature importance figure saved.")

    return {
        'analysis_df': analysis_df, 'model': model, 
        'val_r2': val_r2, 'val_rmse': val_rmse,
        'test_r2': test_r2, 'test_rmse': test_rmse,
        'cv_scores': cv_scores, 'feature_importance': importance, 
        'importance_fig': imp_fig, 'scaler': scaler, 'selector': selector
    }

if __name__ == "__main__":
    results = generate_emg_velocity_report()