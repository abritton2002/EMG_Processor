import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Ensure output directory exists
output_dir = r"C:\Users\alex.britton\Documents\DelsysTesting\EMGanalysis\V2_REPO\model_analysis"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_and_prepare_data():
    """Load data using your existing functions"""
    from contraction_dynamics import load_emg_data_with_velocity, prepare_data
    raw_df = load_emg_data_with_velocity()
    df = prepare_data(raw_df)
    return df

def generate_robust_model_visualizations(df):
    """Generate comprehensive visualizations for the robust model"""
    
    features = [
        'fcu_peak_amplitude', 'fcr_peak_amplitude', 'fcu_activation_speed',
        'fcr_activation_speed', 'timing_diff', 'coactivation_index',
        'fcu_wavelet_high_prop', 'fcr_wavelet_high_prop'
    ]
    features = [f for f in features if f in df.columns]
    X = df[features]
    y = df['pitch_speed_mph']

    # Split data with stratification on athlete
    X_temp, X_holdout, y_temp, y_holdout, df_temp, df_holdout = train_test_split(
        X, y, df, test_size=0.2, random_state=42, 
        stratify=df['athlete_name'] if 'athlete_name' in df.columns else None
    )
    
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X_temp, y_temp, df_temp, test_size=0.25, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_holdout_scaled = scaler.transform(X_holdout)

    # Feature selection
    selector = SelectKBest(score_func=f_regression, k=6)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    X_holdout_selected = selector.transform(X_holdout_scaled)
    selected_features = [features[i] for i in selector.get_support(indices=True)]
    print("Selected features:", selected_features)

    # Train model with best params from your GridSearchCV
    model = GradientBoostingRegressor(
        learning_rate=0.01, max_depth=3, min_samples_leaf=3, 
        min_samples_split=10, n_estimators=200, subsample=0.7,
        random_state=42
    )
    model.fit(X_train_selected, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train_selected)
    y_test_pred = model.predict(X_test_selected)
    y_holdout_pred = model.predict(X_holdout_selected)
    
    # Calculate metrics for visualization
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    holdout_r2 = r2_score(y_holdout, y_holdout_pred)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    holdout_rmse = np.sqrt(mean_squared_error(y_holdout, y_holdout_pred))

    # Create feature importance dataframe
    importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    # Create visualization 1: Model Performance Comparison
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    
    # Prepare data for visualization
    performance_data = pd.DataFrame({
        'Dataset': ['Training', 'Test', 'Holdout'],
        'R²': [train_r2, test_r2, holdout_r2],
        'RMSE (mph)': [train_rmse, test_rmse, holdout_rmse]
    })
    
    # Plot R² bars
    sns.barplot(x='Dataset', y='R²', data=performance_data, ax=ax1, color='steelblue')
    
    # Create twin axis for RMSE
    ax2 = ax1.twinx()
    sns.pointplot(x='Dataset', y='RMSE (mph)', data=performance_data, ax=ax2, color='tomato', markers='D', scale=1.5)
    
    # Formatting
    ax1.set_title('Robust Model Performance Across Datasets', fontsize=16)
    ax1.set_ylabel('R² (Higher is Better)', fontsize=14, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax2.set_ylabel('RMSE in mph (Lower is Better)', fontsize=14, color='tomato')
    ax2.tick_params(axis='y', labelcolor='tomato')
    ax1.grid(True, alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate(performance_data['R²']):
        ax1.text(i, v/2, f"{v:.3f}", ha='center', fontsize=12, color='white', fontweight='bold')
    
    for i, v in enumerate(performance_data['RMSE (mph)']):
        ax2.text(i, v + 0.2, f"{v:.2f}", ha='center', fontsize=12, color='darkred', fontweight='bold')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'robust_model_performance.png'), dpi=300, bbox_inches='tight')
    
    # Create visualization 2: Feature Importance with Before/After Selection
    fig2, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate importance for all original features
    # We'll need to fit a simple model to get these values
    simple_model = GradientBoostingRegressor(random_state=42, n_estimators=100)
    simple_model.fit(X_train_scaled, y_train)
    
    # Create full importance dataframe
    full_importance = pd.DataFrame({
        'Feature': features,
        'Importance': simple_model.feature_importances_,
        'Selected': [f in selected_features for f in features]
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance
    sns.barplot(
        x='Importance', y='Feature', 
        hue='Selected', dodge=False,
        data=full_importance, 
        palette={True: 'seagreen', False: 'lightgray'},
        ax=ax
    )
    
    # Formatting
    ax.set_title('Feature Importance (Selected vs. Non-Selected)', fontsize=16)
    ax.set_xlabel('Relative Importance', fontsize=14)
    ax.set_ylabel('Feature', fontsize=14)
    
    # Add values on bars
    for i, v in enumerate(full_importance['Importance']):
        ax.text(v + 0.01, i, f"{v:.3f}", va='center', fontsize=10)
    
    plt.legend(title='Selected for Final Model')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance_selection.png'), dpi=300, bbox_inches='tight')
    
    # Create visualization 3: Actual vs Predicted by Dataset
    fig3, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Prepare data for visualization
    df_train_results = pd.DataFrame({
        'Actual': y_train,
        'Predicted': y_train_pred,
        'Dataset': 'Train',
        'Athlete': df_train['athlete_name'] if 'athlete_name' in df_train.columns else 'Unknown'
    })
    
    df_test_results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_test_pred,
        'Dataset': 'Test',
        'Athlete': df_test['athlete_name'] if 'athlete_name' in df_test.columns else 'Unknown'
    })
    
    df_holdout_results = pd.DataFrame({
        'Actual': y_holdout,
        'Predicted': y_holdout_pred,
        'Dataset': 'Holdout',
        'Athlete': df_holdout['athlete_name'] if 'athlete_name' in df_holdout.columns else 'Unknown'
    })
    
    all_results = pd.concat([df_train_results, df_test_results, df_holdout_results])
    
    # Plot each dataset
    datasets = ['Train', 'Test', 'Holdout']
    r2_values = [train_r2, test_r2, holdout_r2]
    rmse_values = [train_rmse, test_rmse, holdout_rmse]
    
    for i, (dataset, r2, rmse) in enumerate(zip(datasets, r2_values, rmse_values)):
        ax = axes[i]
        data_subset = all_results[all_results['Dataset'] == dataset]
        
        # Calculate min/max for the perfect line
        min_val = min(data_subset['Actual'].min(), data_subset['Predicted'].min())
        max_val = max(data_subset['Actual'].max(), data_subset['Predicted'].max())
        
        # Plot perfect prediction line
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        # Plot actual vs predicted
        sns.scatterplot(
            x='Actual', y='Predicted', 
            hue='Athlete', 
            data=data_subset,
            s=100, alpha=0.7,
            ax=ax
        )
        
        # Formatting
        ax.set_title(f'{dataset} Set (R²: {r2:.3f}, RMSE: {rmse:.2f} mph)', fontsize=14)
        ax.set_xlabel('Actual Velocity (mph)', fontsize=12)
        ax.set_ylabel('Predicted Velocity (mph)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Remove legend for first two plots to avoid redundancy
        if i < 2:
            ax.get_legend().remove()
    
    # Adjust legend for the last plot
    axes[2].legend(title='Athlete', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle('Actual vs Predicted Velocity Across Datasets', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'), dpi=300, bbox_inches='tight')
    
    # Create visualization 4: Per-Athlete Performance
    fig4, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Prepare data by athlete
    athlete_performance = all_results.groupby(['Athlete', 'Dataset']).agg({
        'Actual': 'mean',
        'Predicted': 'mean',
        'Actual': lambda x: np.mean(np.abs(x - all_results.loc[x.index, 'Predicted']))
    }).reset_index()
    
    athlete_performance.columns = ['Athlete', 'Dataset', 'Mean Absolute Error']
    
    # Pivot for easier plotting
    athlete_pivot = athlete_performance.pivot(index='Athlete', columns='Dataset', values='Mean Absolute Error')
    
    # Plot dataset comparison by athlete
    sns.heatmap(
        athlete_pivot, 
        annot=True, 
        fmt='.2f',
        cmap='YlOrRd_r',
        linewidths=0.5,
        ax=axes[0]
    )
    
    axes[0].set_title('Mean Absolute Error by Athlete and Dataset (Lower is Better)', fontsize=14)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    
    # Calculate per-athlete error metrics
    athlete_error = all_results.groupby('Athlete').apply(
        lambda x: pd.Series({
            'Mean Error (mph)': np.mean(x['Predicted'] - x['Actual']),
            'RMSE (mph)': np.sqrt(np.mean((x['Predicted'] - x['Actual'])**2)),
            'R²': r2_score(x['Actual'], x['Predicted']) if len(x) > 1 else np.nan,
            'Count': len(x)
        })
    ).reset_index()
    
    # Sort by RMSE
    athlete_error = athlete_error.sort_values('RMSE (mph)')
    
    # Plot
    bar_width = 0.35
    x = np.arange(len(athlete_error))
    
    # Bar chart with RMSE and +/- error
    axes[1].bar(
        x, 
        athlete_error['RMSE (mph)'], 
        width=bar_width, 
        label='RMSE', 
        color='indianred'
    )
    
    # Add R² as text
    for i, (_, row) in enumerate(athlete_error.iterrows()):
        if not np.isnan(row['R²']):
            axes[1].text(
                i, 
                row['RMSE (mph)'] + 0.2, 
                f"R²: {row['R²']:.2f}", 
                ha='center', 
                fontsize=9,
                fontweight='bold'
            )
    
    # Add sample size as text
    for i, (_, row) in enumerate(athlete_error.iterrows()):
        axes[1].text(
            i,
            0.3, 
            f"n={row['Count']}", 
            ha='center',
            fontsize=9
        )
    
    axes[1].set_title('Model Performance by Athlete', fontsize=14)
    axes[1].set_xlabel('Athlete', fontsize=12)
    axes[1].set_ylabel('RMSE (mph)', fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(athlete_error['Athlete'], rotation=45, ha='right')
    axes[1].grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'athlete_performance.png'), dpi=300, bbox_inches='tight')
    
    # Create visualization 5: Residual Analysis
    fig5, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Prepare residuals
    all_results['Residual'] = all_results['Actual'] - all_results['Predicted']
    
    # 1. Residuals by predicted value
    sns.scatterplot(
        x='Predicted', y='Residual', 
        hue='Dataset',
        style='Dataset',
        data=all_results,
        s=80, alpha=0.7,
        ax=axes[0]
    )
    
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0].set_title('Residuals vs Predicted Values', fontsize=14)
    axes[0].set_xlabel('Predicted Velocity (mph)', fontsize=12)
    axes[0].set_ylabel('Residual (Actual - Predicted)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Residuals by top feature
    top_feature = importance.iloc[0]['Feature']
    
    # Merge feature data back in
    all_results_with_features = pd.DataFrame()
    
    if 'athlete_name' in df_train.columns:
        # Training set
        train_features = pd.DataFrame(X_train[top_feature])
        train_features['Dataset'] = 'Train'
        train_features.index = y_train.index
        
        # Test set
        test_features = pd.DataFrame(X_test[top_feature])
        test_features['Dataset'] = 'Test'
        test_features.index = y_test.index
        
        # Holdout set
        holdout_features = pd.DataFrame(X_holdout[top_feature])
        holdout_features['Dataset'] = 'Holdout'
        holdout_features.index = y_holdout.index
        
        # Combine
        features_df = pd.concat([train_features, test_features, holdout_features])
        
        # Merge with results
        all_results_with_features = pd.merge(
            all_results.reset_index(), 
            features_df.reset_index(),
            left_on=['index', 'Dataset'],
            right_on=['index', 'Dataset']
        )
    
    if not all_results_with_features.empty:
        sns.scatterplot(
            x=top_feature, y='Residual',
            hue='Dataset',
            style='Dataset',
            data=all_results_with_features,
            s=80, alpha=0.7,
            ax=axes[1]
        )
        
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1].set_title(f'Residuals vs {top_feature}', fontsize=14)
        axes[1].set_xlabel(top_feature, fontsize=12)
        axes[1].set_ylabel('Residual (Actual - Predicted)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "Feature data unavailable", ha='center', va='center', fontsize=14)
        axes[1].set_title(f'Residuals vs {top_feature}', fontsize=14)
    
    # 3. Residuals by athlete
    sns.boxplot(
        x='Athlete', y='Residual',
        data=all_results,
        ax=axes[2]
    )
    
    axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[2].set_title('Residuals by Athlete', fontsize=14)
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha='right')
    axes[2].set_ylabel('Residual (Actual - Predicted)', fontsize=12)
    axes[2].grid(True, axis='y', alpha=0.3)
    
    # 4. Residual distribution
    sns.histplot(
        all_results['Residual'],
        kde=True,
        ax=axes[3]
    )
    
    # Add normal distribution for comparison
    from scipy.stats import norm
    residuals = all_results['Residual']
    mu, sigma = norm.fit(residuals)
    
    x = np.linspace(residuals.min(), residuals.max(), 100)
    y = norm.pdf(x, mu, sigma) * len(residuals) * (residuals.max() - residuals.min()) / 10
    
    axes[3].plot(x, y, 'r--', linewidth=2)
    axes[3].text(0.95, 0.95, f'μ = {mu:.2f}, σ = {sigma:.2f}', 
                transform=axes[3].transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[3].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[3].set_title('Residual Distribution', fontsize=14)
    axes[3].set_xlabel('Residual (Actual - Predicted)', fontsize=12)
    axes[3].set_ylabel('Count', fontsize=12)
    axes[3].grid(True, alpha=0.3)
    
    plt.suptitle('Residual Analysis', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'residual_analysis.png'), dpi=300, bbox_inches='tight')
    
    # Print status
    print(f"Generated 5 visualization sets and saved them to: {output_dir}")
    return {
        'performance_fig': fig1,
        'feature_importance_fig': fig2,
        'actual_vs_predicted_fig': fig3,
        'athlete_performance_fig': fig4,
        'residual_analysis_fig': fig5
    }
    
if __name__ == "__main__":
    try:
        print("Loading and preparing data...")
        df = load_and_prepare_data()
        
        print("Generating robust model visualizations...")
        figs = generate_robust_model_visualizations(df)
        
        print("Analysis complete!")
    except Exception as e:
        print(f"Error: {str(e)}")