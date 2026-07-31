import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for better-looking plots
sns.set_theme(style="whitegrid")

def load_data(results_dir="results"):
    """Loads all CSV files from the results directory into a single DataFrame."""
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    df_list = []
    
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not df_list:
        print(f"No CSV files found in {results_dir}")
        return None

    full_df = pd.concat(df_list, ignore_index=True)
    # Ensure resolution is treated as integer for proper sorting
    full_df['resolution'] = full_df['resolution'].astype(int)
    full_df.sort_values(by=['gpu', 'resolution'], inplace=True)
    return full_df

def plot_fps_vs_resolution(df, output_dir="plots"):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='resolution', y='fps_mean', hue='gpu', marker='o', linewidth=2, markersize=8)
    plt.title('FPS vs Resolution', fontsize=16, pad=15)
    plt.xlabel('Resolution', fontsize=14)
    plt.ylabel('FPS (Mean)', fontsize=14)
    plt.xticks(sorted(df['resolution'].unique()))
    plt.legend(title='GPU', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fps_vs_resolution.png'), dpi=300)
    plt.close()

def plot_latency_vs_resolution(df, output_dir="plots"):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='resolution', y='latency_mean_ms', hue='gpu', marker='o', linewidth=2, markersize=8)
    plt.title('Latency vs Resolution', fontsize=16, pad=15)
    plt.xlabel('Resolution', fontsize=14)
    plt.ylabel('Latency Mean (ms)', fontsize=14)
    plt.xticks(sorted(df['resolution'].unique()))
    plt.legend(title='GPU', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_vs_resolution.png'), dpi=300)
    plt.close()

def plot_fps_per_watt(df, output_dir="plots"):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='resolution', y='fps_per_watt', hue='gpu')
    plt.title('FPS per Watt vs Resolution', fontsize=16, pad=15)
    plt.xlabel('Resolution', fontsize=14)
    plt.ylabel('FPS / Watt', fontsize=14)
    plt.legend(title='GPU', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fps_per_watt.png'), dpi=300)
    plt.close()

def plot_gpu_memory_usage(df, output_dir="plots"):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='resolution', y='gpu_memory_MB', hue='gpu')
    plt.title('GPU Memory Usage vs Resolution', fontsize=16, pad=15)
    plt.xlabel('Resolution', fontsize=14)
    plt.ylabel('Memory Usage (MB)', fontsize=14)
    plt.legend(title='GPU', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gpu_memory_usage.png'), dpi=300)
    plt.close()

def plot_performance_scaling(df, output_dir="plots"):
    ref_gpu = 'NVIDIA GeForce RTX 4060 Ti'
    
    if ref_gpu not in df['gpu'].values:
        print(f"Reference GPU '{ref_gpu}' not found in data. Skipping scaling plot.")
        return
        
    scaling_data = []
    
    for res in df['resolution'].unique():
        # Get reference FPS for the current resolution
        ref_fps_series = df[(df['gpu'] == ref_gpu) & (df['resolution'] == res)]['fps_mean']
        if ref_fps_series.empty:
            continue
        ref_fps = ref_fps_series.values[0]
        
        for gpu in df['gpu'].unique():
            gpu_fps_series = df[(df['gpu'] == gpu) & (df['resolution'] == res)]['fps_mean']
            if gpu_fps_series.empty:
                continue
            gpu_fps = gpu_fps_series.values[0]
            
            scaling_data.append({
                'gpu': gpu,
                'resolution': res,
                'scaling': gpu_fps / ref_fps
            })
            
    scaling_df = pd.DataFrame(scaling_data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=scaling_df, x='resolution', y='scaling', hue='gpu')
    # Add a horizontal line at 1.0 representing the baseline (RTX 4060 Ti)
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Baseline (RTX 4060 Ti)')
    plt.title(f'Performance Scaling Relative to {ref_gpu}', fontsize=16, pad=15)
    plt.xlabel('Resolution', fontsize=14)
    plt.ylabel('Scaling Factor (x)', fontsize=14)
    
    # Update legend to include the baseline line
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, title='GPU', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_scaling.png'), dpi=300)
    plt.close()

def main():
    # Setup directories relative to the script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    output_dir = os.path.join(base_dir, "plots")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
        
    print(f"Loading data from: {results_dir}")
    df = load_data(results_dir)
    
    if df is not None and not df.empty:
        print(f"Data loaded successfully. Found {len(df)} records.")
        
        print("Generating FPS vs Resolution plot...")
        plot_fps_vs_resolution(df, output_dir)
        
        print("Generating Latency vs Resolution plot...")
        plot_latency_vs_resolution(df, output_dir)
        
        print("Generating FPS/Watt plot...")
        plot_fps_per_watt(df, output_dir)
        
        print("Generating GPU Memory Usage plot...")
        plot_gpu_memory_usage(df, output_dir)
        
        print("Generating Performance Scaling plot...")
        plot_performance_scaling(df, output_dir)
        
        print(f"\nAll plots have been successfully saved to the '{output_dir}' directory.")

if __name__ == "__main__":
    main()
