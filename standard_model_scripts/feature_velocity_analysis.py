import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Ensure output directory exists
output_dir = r"C:\Users\alex.britton\Documents\DelsysTesting\EMGanalysis\V2_REPO\standard_model_scripts"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_data():
    """Load data from existing code."""
    from contraction_dynamics_model import load_emg_data_with_velocity, prepare_data
    raw_df = load_emg_data_with_velocity()
    df = prepare_data(raw_df)
    return df

def analyze_key_feature_relationships(df):
    """Create detailed visualizations of key feature-velocity relationships."""
    
    # Key features based on your analysis
    key_features = [
        'fcu_peak_amplitude', 
        'fcr_peak_amplitude', 
        'fcu_activation_speed',
        'fcr_activation_speed', 
        'timing_diff',
        'coactivation_index',
        'fcu_wavelet_high_prop'
    ]
    
    # Filter only features that exist in the dataframe
    key_features = [f for f in key_features if f in df.columns]
    
    # Create a grid of scatter plots with regression lines
    n_features = len(key_features)
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    axes = axes.flatten()
    
    # Add a title to the figure
    plt.suptitle('Key EMG Features vs. Throwing Velocity Relationships', fontsize=16, y=0.98)
    
    for i, feature in enumerate(key_features):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Calculate correlation
        corr, p_value = pearsonr(df[feature], df['pitch_speed_mph'])
        
        # Create scatterplot with regression line
        sns.regplot(
            x=feature, 
            y='pitch_speed_mph', 
            data=df,
            scatter_kws={'alpha': 0.7, 's': 80, 'color': 'dodgerblue'},
            line_kws={'color': 'red', 'lw': 2},
            ax=ax
        )
        
        # Overlay athlete points with different colors
        sns.scatterplot(
            x=feature,
            y='pitch_speed_mph',
            hue='athlete_name',
            data=df,
            ax=ax,
            alpha=0.7,
            s=100,
            legend=i == 0  # Only show legend on first plot
        )
        
        # Labels and title
        ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Velocity (mph)', fontsize=12)
        ax.set_title(f"{feature.replace('_', ' ').title()} vs Velocity\nr = {corr:.3f} (p = {p_value:.3f})", fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # If this is the first plot with legend, adjust the legend
        if i == 0:
            ax.legend(title="Athlete", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # If there are unused subplots, hide them
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure
    save_path = os.path.join(output_dir, 'feature_velocity_relationships.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature-velocity relationships saved to {save_path}")
    
    # Create correlation matrix heatmap
    plt.figure(figsize=(12, 10))
    features_for_corr = key_features + ['pitch_speed_mph']
    corr_matrix = df[features_for_corr].corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Generate a heatmap
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        annot=True, 
        fmt=".2f", 
        cmap="RdBu_r",
        vmin=-1, 
        vmax=1, 
        square=True,
        linewidths=0.5
    )
    
    plt.title('Correlation Matrix of Key EMG Features', fontsize=16)
    plt.tight_layout()
    
    # Save the correlation matrix
    corr_path = os.path.join(output_dir, 'feature_correlation_matrix.png')
    plt.savefig(corr_path, dpi=300, bbox_inches='tight')
    print(f"Correlation matrix saved to {corr_path}")
    
    return fig

def create_athlete_specific_analysis(df):
    """Create visualizations showing athlete-specific patterns."""
    
    # Get all unique athletes
    athletes = df['athlete_name'].unique()
    
    plt.figure(figsize=(15, 10))
    
    # Create a plot for FCR amplitude vs FCU amplitude colored by velocity
    scatter = plt.scatter(
        df['fcr_peak_amplitude'], 
        df['fcu_peak_amplitude'],
        c=df['pitch_speed_mph'],
        s=100,
        alpha=0.7,
        cmap='viridis'
    )
    
    # Add athlete labels
    for athlete in athletes:
        athlete_data = df[df['athlete_name'] == athlete]
        centroid_x = athlete_data['fcr_peak_amplitude'].mean()
        centroid_y = athlete_data['fcu_peak_amplitude'].mean()
        plt.annotate(
            athlete,
            (centroid_x, centroid_y),
            fontsize=12,
            weight='bold',
            ha='center',
            va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    plt.colorbar(scatter, label='Velocity (mph)')
    plt.xlabel('FCR Peak Amplitude', fontsize=14)
    plt.ylabel('FCU Peak Amplitude', fontsize=14)
    plt.title('Muscle Activation Patterns by Athlete and Velocity', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    save_path = os.path.join(output_dir, 'athlete_muscle_activation_patterns.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Athlete muscle activation patterns saved to {save_path}")
    
    return None

if __name__ == "__main__":
    try:
        df = load_data()
        _ = analyze_key_feature_relationships(df)
        _ = create_athlete_specific_analysis(df)
        print("EMG feature analysis visualizations complete!")
    except Exception as e:
        print(f"Error: {str(e)}")