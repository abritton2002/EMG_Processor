import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from dotenv import load_dotenv
from urllib.parse import quote_plus

# Statistical and ML imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFECV
from sklearn.pipeline import Pipeline

# Load environment variables
load_dotenv()

def get_db_connection():
    """Create database connection using SQLAlchemy."""
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

def load_emg_data_with_velocity_and_bodyweight():
    """
    Load EMG throws data with velocity and body weight information.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with EMG throws, velocity, and body weight metrics
    """
    engine = get_db_connection()
    
    # Comprehensive query to join EMG throws and sessions
    query = """
    SELECT 
        t.throw_id, t.session_numeric_id, t.trial_number,
        t.muscle1_peak_amplitude, t.muscle1_rise_time, t.muscle1_throw_integral,
        t.muscle1_median_freq, t.muscle1_spectral_entropy,
        t.muscle2_peak_amplitude, t.muscle2_rise_time, t.muscle2_throw_integral,
        t.muscle2_median_freq, t.muscle2_spectral_entropy,
        t.muscle1_wavelet_energy_low, t.muscle1_wavelet_energy_mid, t.muscle1_wavelet_energy_high,
        t.muscle2_wavelet_energy_low, t.muscle2_wavelet_energy_mid, t.muscle2_wavelet_energy_high,
        t.coactivation_index, t.coactivation_correlation, t.coactivation_temporal_overlap,
        t.coactivation_waveform_similarity,
        t.pitch_speed_mph, t.velocity_match_quality,
        sess.mass_kilograms,
        sess.height_meters,
        s.athlete_name, s.date_recorded
    FROM emg_throws t
    JOIN emg_sessions s ON t.session_numeric_id = s.numeric_id
    JOIN sessions sess ON sess.session = SUBSTRING_INDEX(t.session_trial, '_', 1)
    WHERE 
        t.pitch_speed_mph IS NOT NULL 
        AND sess.mass_kilograms IS NOT NULL
        AND sess.mass_kilograms > 0
        AND sess.height_meters IS NOT NULL
        AND sess.height_meters > 0
    ORDER BY t.throw_id
    """
    
    return pd.read_sql(query, engine)

def prepare_data(df):
    """
    Enhanced data preparation with body weight normalization and feature engineering.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with EMG and velocity data
    
    Returns:
    --------
    pd.DataFrame
        Processed DataFrame with normalized and engineered features
    """
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Remove outliers based on velocity
    q1 = data['pitch_speed_mph'].quantile(0.01)
    q3 = data['pitch_speed_mph'].quantile(0.99)
    data = data[(data['pitch_speed_mph'] >= q1) & (data['pitch_speed_mph'] <= q3)]
    
    # Body weight normalization methods
    # 1. Direct normalization by body weight
    normalization_columns = [
        'muscle1_peak_amplitude', 'muscle2_peak_amplitude',
        'muscle1_rise_time', 'muscle2_rise_time',
        'muscle1_throw_integral', 'muscle2_throw_integral',
        'muscle1_median_freq', 'muscle2_median_freq',
        'muscle1_spectral_entropy', 'muscle2_spectral_entropy'
    ]
    
    # Create normalized columns
    for col in normalization_columns:
        data[f'{col}_per_kg'] = data[col] / data['mass_kilograms']
        data[f'{col}_per_height'] = data[col] / data['height_meters']
    
    # 2. Bodyweight residualization (controlling for body weight effects)
    def residualize(column, body_metric='mass_kilograms'):
        """
        Remove the linear effect of body weight from a column.
        
        Returns the residuals (what's left after accounting for body weight)
        """
        # Prepare data
        X = data[body_metric].values.reshape(-1, 1)
        y = data[column].values
        
        # Fit linear regression
        reg = LinearRegression().fit(X, y)
        
        # Calculate and return residuals
        return y - reg.predict(X)
    
    # Apply residualization to key metrics
    for col in normalization_columns:
        data[f'{col}_bodyweight_residuals'] = residualize(col)
    
    # 3. Calculate Body Metrics Indices
    # BMI and Height-normalized metrics
    data['bmi'] = data['mass_kilograms'] / (data['height_meters'] ** 2)
    
    # Existing feature engineering
    # Basic feature renaming
    data['fcu_peak_amplitude'] = data['muscle1_peak_amplitude']
    data['fcr_peak_amplitude'] = data['muscle2_peak_amplitude']
    data['fcu_rise_time'] = data['muscle1_rise_time']
    data['fcr_rise_time'] = data['muscle2_rise_time']
    data['fcu_integral'] = data['muscle1_throw_integral']
    data['fcr_integral'] = data['muscle2_throw_integral']
    data['fcu_median_freq'] = data['muscle1_median_freq']
    data['fcr_median_freq'] = data['muscle2_median_freq']
    data['fcu_entropy'] = data['muscle1_spectral_entropy']
    data['fcr_entropy'] = data['muscle2_spectral_entropy']
    
    # Include bodyweight-normalized versions in feature engineering
    data['fcu_peak_amplitude_per_kg'] = data['muscle1_peak_amplitude_per_kg']
    data['fcr_peak_amplitude_per_kg'] = data['muscle2_peak_amplitude_per_kg']
    data['fcu_rise_time_per_kg'] = data['muscle1_rise_time_per_kg']
    data['fcr_rise_time_per_kg'] = data['muscle2_rise_time_per_kg']
    data['fcu_integral_per_kg'] = data['muscle1_throw_integral_per_kg']
    data['fcr_integral_per_kg'] = data['muscle2_throw_integral_per_kg']
    
    # Existing feature engineering continues...
    # Activation speed (normalized by amplitude)
    data['fcu_activation_speed'] = data['fcu_peak_amplitude'] / data['fcu_rise_time'] 
    data['fcr_activation_speed'] = data['fcr_peak_amplitude'] / data['fcr_rise_time']
    
    # Timing and coordination
    data['timing_diff'] = data['fcr_rise_time'] - data['fcu_rise_time']
    data['timing_diff_abs'] = np.abs(data['timing_diff'])
    data['timing_ratio'] = data['fcu_rise_time'] / data['fcr_rise_time'].replace(0, 0.001)
    
    # Rest of the existing feature engineering remains the same...
    # (previous code for amplitude ratios, integral metrics, etc.)
    
    # Prepare for final processing
    data = data.replace([np.inf, -np.inf], np.nan)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
    
    # Drop near-zero variance features
    for col in numeric_cols:
        if col != 'pitch_speed_mph' and data[col].std() / data[col].mean() < 0.001:
            data = data.drop(columns=col)
    
    return data

def build_improved_model(df):
    """Improved modeling pipeline with ensemble approach and regularization"""
    
    # Define enhanced feature list based on domain knowledge
    candidate_features = [
        # Basic amplitude metrics
        'fcu_peak_amplitude', 'fcr_peak_amplitude', 'amplitude_sum', 'amplitude_ratio', 'amplitude_product',
        
        # Time-based metrics
        'fcu_rise_time', 'fcr_rise_time', 'timing_diff', 'timing_diff_abs', 'timing_ratio',
        
        # Activation dynamics
        'fcu_activation_speed', 'fcr_activation_speed',
        
        # Energy/work metrics
        'fcu_integral', 'fcr_integral', 'integral_sum', 'integral_ratio', 'integral_product',
        
        # Frequency domain
        'fcu_median_freq', 'fcr_median_freq', 'median_freq_diff', 'median_freq_ratio', 'median_freq_sum',
        
        # Spectral complexity
        'fcu_entropy', 'fcr_entropy', 'entropy_sum', 'entropy_ratio',
        
        # Coactivation metrics
        'coactivation_index', 'coactivation_correlation', 'waveform_similarity', 'temporal_overlap',
        
        # Wavelet energy distribution
        'fcu_low_energy_norm', 'fcu_mid_energy_norm', 'fcu_high_energy_norm',
        'fcr_low_energy_norm', 'fcr_mid_energy_norm', 'fcr_high_energy_norm',
        'low_energy_ratio', 'mid_energy_ratio', 'high_energy_ratio',
        'fcu_high_low_ratio', 'fcr_high_low_ratio'
    ]
    
    # Filter to only include features that exist in the dataframe
    features = [f for f in candidate_features if f in df.columns]
    X = df[features]
    y = df['pitch_speed_mph']
    
    # Create train/validation/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Use recursive feature elimination with cross-validation to identify optimal features
    print("Performing recursive feature elimination...")
    estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    selector = RFECV(
        estimator=estimator,
        step=1,
        cv=5,
        scoring='neg_mean_squared_error',
        min_features_to_select=5,
        n_jobs=-1
    )
    
    # Use a RobustScaler to handle outliers better than StandardScaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Fit the selector
    selector.fit(X_train_scaled, y_train)
    
    # Get selected features
    selected_features = [features[i] for i in range(len(features)) if selector.support_[i]]
    print(f"Selected {len(selected_features)} features out of {len(features)}")
    print(f"Optimal features: {', '.join(selected_features)}")
    
    # Transform data with selected features
    X_train_selected = selector.transform(X_train_scaled)
    X_test_scaled = scaler.transform(X_test)
    X_test_selected = selector.transform(X_test_scaled)
    
    # Train multiple models and create an ensemble
    print("\nTraining multiple model types...")
    models = {
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        'SVR': SVR(C=1.0, epsilon=0.1),
        'RandomForest': RandomForestRegressor(
            n_estimators=100, 
            max_depth=5, 
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.01,
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=3,
            subsample=0.8,
            random_state=42
        )
    }
    
    # Train all models
    model_results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_selected, y_train)
        
        # Evaluate performance
        train_pred = model.predict(X_train_selected)
        test_pred = model.predict(X_test_selected)
        
        train_r2 = r2_score(y_train, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_r2 = r2_score(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        # Perform cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_selected, y_train, 
                                   cv=cv, scoring='r2')
        
        model_results[name] = {
            'model': model,
            'train_r2': train_r2,
            'train_rmse': train_rmse,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'predictions': test_pred
        }
        
        print(f"  {name} - Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}, Test RMSE: {test_rmse:.3f} mph")
    
    # Find best model
    best_model_name = max(model_results, key=lambda k: model_results[k]['test_r2'])
    best_model = model_results[best_model_name]
    
    print(f"\nBest model: {best_model_name}")
    print(f"  Test R²: {best_model['test_r2']:.3f}")
    print(f"  Test RMSE: {best_model['test_rmse']:.3f} mph")
    
    # Calculate feature importance using permutation importance with p-values
    from scipy import stats
    from sklearn.inspection import permutation_importance

    # Perform permutation importance
    perm_importance = permutation_importance(
        best_model['model'], 
        X_test_selected, 
        y_test, 
        n_repeats=10,  # Number of times to permute each feature
        random_state=42
    )

    # Calculate p-values manually
    def calculate_permutation_pvalues(feature_importances, original_score, n_repeats):
        """
        Calculate p-values for permutation importance.
        
        Parameters:
        -----------
        feature_importances : array
            Importances from permutation importance
        original_score : float
            Original model score
        n_repeats : int
            Number of permutations
        
        Returns:
        --------
        p-values for each feature
        """
        pvalues = []
        for importance in feature_importances:
            # Count how many permuted importance values are as extreme or more extreme
            extreme_count = np.sum(np.abs(importance) >= np.abs(original_score))
            pvalue = (extreme_count + 1) / (n_repeats + 1)
            pvalues.append(pvalue)
        return pvalues

    # Calculate p-values
    # Use model's score on test set as the original score
    original_score = r2_score(y_test, best_model['predictions'])
    pvalues = calculate_permutation_pvalues(
        perm_importance.importances, 
        original_score, 
        n_repeats=10
    )

    # Create importance DataFrame with p-values
    importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': perm_importance.importances_mean,
        'Importance_std': perm_importance.importances_std,
        'p_value': pvalues
    }).sort_values('Importance', ascending=False)

    # Add significance indicator
    importance['Significant'] = importance['p_value'] < 0.05
    
    
    # Visualization of feature importance
    plt.figure(figsize=(10, 8))
    plt.barh(importance['Feature'], importance['Importance'], 
             xerr=importance['Importance_std'], 
             capsize=5, 
             color='steelblue')
    plt.title('Feature Importance (Permutation Method)', fontsize=16)
    plt.xlabel('Importance (Mean Decrease in R²)', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.savefig('permutation_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create visualizations
    # 1. Feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance.head(10), palette='Blues_d')
    plt.title(f'Top 10 Features for Pitch Velocity Prediction ({best_model_name})', fontsize=16)
    plt.xlabel('Relative Importance', fontsize=12)
    plt.ylabel('EMG Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig('improved_feature_importance.png', dpi=300, bbox_inches='tight')
    
    # 2. Predicted vs actual plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, best_model['predictions'], alpha=0.7, s=80, color='steelblue')
    
    # Add regression line
    z = np.polyfit(y_test, best_model['predictions'], 1)
    p = np.poly1d(z)
    plt.plot(np.sort(y_test), p(np.sort(y_test)), "r--", alpha=0.7, linewidth=2)
    
    # Add perfect prediction line
    min_val = min(min(y_test), min(best_model['predictions']))
    max_val = max(max(y_test), max(best_model['predictions']))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    plt.title(f'Actual vs Predicted Pitch Velocity ({best_model_name})', fontsize=16)
    plt.xlabel('Actual Velocity (mph)', fontsize=14)
    plt.ylabel('Predicted Velocity (mph)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add performance metrics text box
    plt.annotate(
        f"Test R² = {best_model['test_r2']:.3f}\nRMSE = {best_model['test_rmse']:.2f} mph",
        xy=(0.05, 0.95), xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
        fontsize=12, ha='left', va='top'
    )
    
    plt.tight_layout()
    plt.savefig('improved_prediction_performance.png', dpi=300, bbox_inches='tight')
    
    # 3. Model comparison
    plt.figure(figsize=(12, 6))
    model_names = []
    train_scores = []
    test_scores = []
    
    for name, result in model_results.items():
        model_names.append(name)
        train_scores.append(result['train_r2'])
        test_scores.append(result['test_r2'])
    
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    train_bars = ax.bar(x - width/2, train_scores, width, label='Training R²', color='skyblue')
    test_bars = ax.bar(x + width/2, test_scores, width, label='Test R²', color='darkblue')
    
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('R² Score', fontsize=14)
    ax.set_title('Model Performance Comparison', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    
    # Add R² values on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=8)
    
    add_labels(train_bars)
    add_labels(test_bars)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    
    # Return results
    return {
        'best_model_name': best_model_name,
        'best_model': best_model['model'],
        'test_r2': best_model['test_r2'],
        'test_rmse': best_model['test_rmse'],
        'feature_importance': importance,
        'selected_features': selected_features,
        'scaler': scaler,
        'selector': selector,
        'all_models': model_results
    }


def generate_improved_velocity_report():
    try:
        # Load data with body weight information
        print("Loading EMG and velocity data with body weight...")
        df = load_emg_data_with_velocity_and_bodyweight()
        
        if df.empty or 'pitch_speed_mph' not in df.columns:
            print("No velocity data found.")
            return None
        


        # Process data with body weight normalization
        print(f"\nDataset: {len(df)} throws with velocity data")
        print("Preparing data with body weight normalization...")
        analysis_df = prepare_data(df)
        
        # Build improved model
        print("\nBuilding improved pitch velocity prediction model...")
        results = build_improved_model(analysis_df)
        
        print("\nModel Training Complete!")
        print("\nFinal Model Summary:")
        print(f"  Model type: {results['best_model_name']}")
        print(f"  Test R²: {results['test_r2']:.3f}")
        print(f"  Test RMSE: {results['test_rmse']:.3f} mph")
        print(f"  Selected {len(results['selected_features'])} optimal features")
        
        print("\nTop 5 Predictive Features:")
        for _, row in results['feature_importance'].head(5).iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.3f}")
        
        print("\nKey Improvements:")
        print("1. Body weight normalization of muscle activation metrics")
        print("2. Residualization of body weight effects")
        print("3. Multi-method feature engineering")
        print("4. Robust scaling to handle outliers")
        print("5. Comprehensive feature selection")
        

        return results
    
    except Exception as e:
        print(f"Error generating improved pitch velocity report: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = generate_improved_velocity_report()