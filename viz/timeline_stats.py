import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_timeline_stats(
    df: pd.DataFrame,
    drift_scores: pd.DataFrame, # From audit.drift.analyze_label_drift
    timestamp_col: str = 'timestamp',
    output_path: str = 'timeline_stats.png'
) -> None:
    """
    Generates and saves a plot visualizing trends of annotation volume, label drift,
    and a simulated error rate over time.

    Args:
        df (pd.DataFrame): The main DataFrame containing annotation data.
                           Expected to have a timestamp_col.
        drift_scores (pd.DataFrame): DataFrame containing 'timestamp' and 'kl_divergence'
                                     from audit.drift.analyze_label_drift.
        timestamp_col (str): Name of the column containing timestamps (datetime objects).
        output_path (str): Full path including filename to save the generated plot.
    """
    if df.empty:
        print("Warning: Input DataFrame is empty. Cannot generate timeline statistics.")
        return
    if drift_scores.empty:
        print("Warning: Drift scores DataFrame is empty. Label drift will not be plotted.")

    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in main DataFrame.")

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Ensure drift_scores timestamp column is also datetime
    if not drift_scores.empty and timestamp_col in drift_scores.columns:
        drift_scores[timestamp_col] = pd.to_datetime(drift_scores[timestamp_col])

    # 1. Annotation Volume (e.g., daily task count)
    # Assuming 'task_id' exists and is unique per task
    if 'task_id' in df.columns:
        daily_volume = df.groupby(df[timestamp_col].dt.date)['task_id'].nunique().reset_index()
        daily_volume.columns = ['date', 'task_count']
        daily_volume['date'] = pd.to_datetime(daily_volume['date'])
    else:
        print("Warning: 'task_id' column not found. Cannot calculate unique task volume. Using row count.")
        daily_volume = df.groupby(df[timestamp_col].dt.date).size().reset_index(name='task_count')
        daily_volume.columns = ['date', 'task_count']
        daily_volume['date'] = pd.to_datetime(daily_volume['date'])


    # 2. Simulated Error Rate (for demonstration)
    # In a real scenario, this would come from a quality control module,
    # e.g., percentage of tasks flagged by speed_check or consistency issues.
    # For now, let's create a dummy error rate that fluctuates.
    # Align it with the daily_volume dates.
    if not daily_volume.empty:
        np.random.seed(42) # for reproducibility
        simulated_error_rate = np.random.rand(len(daily_volume)) * 0.1 + 0.02 # 2-12% error
        daily_volume['simulated_error_rate'] = simulated_error_rate
    else:
        daily_volume['simulated_error_rate'] = np.nan


    fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=True)
    fig.suptitle('Annotation Timeline Statistics', fontsize=16)

    # Plot 1: Annotation Volume
    if not daily_volume.empty:
        axes[0].plot(daily_volume['date'], daily_volume['task_count'], marker='o', linestyle='-', color='teal')
        axes[0].set_ylabel('Daily Annotation Volume')
        axes[0].set_title('Daily Annotation Volume (Tasks)')
        axes[0].grid(True, linestyle='--', alpha=0.6)
    else:
        axes[0].set_visible(False)

    # Plot 2: Label Drift (KL Divergence)
    if not drift_scores.empty:
        axes[1].plot(drift_scores[timestamp_col], drift_scores['kl_divergence'], marker='x', linestyle='--', color='purple')
        axes[1].set_ylabel('KL Divergence (Label Drift)')
        axes[1].set_title('Label Distribution Drift Over Time')
        axes[1].grid(True, linestyle='--', alpha=0.6)
    else:
        axes[1].set_visible(False)

    # Plot 3: Simulated Error Rate
    if not daily_volume.empty and 'simulated_error_rate' in daily_volume.columns:
        axes[2].plot(daily_volume['date'], daily_volume['simulated_error_rate'], marker='s', linestyle=':', color='red')
        axes[2].set_ylabel('Simulated Error Rate')
        axes[2].set_title('Simulated Daily Error Rate')
        axes[2].set_ylim(0, 0.2) # Assuming error rate between 0 and 20%
        axes[2].grid(True, linestyle='--', alpha=0.6)
    else:
        axes[2].set_visible(False)

    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent title overlap
    plt.savefig(output_path)
    plt.close(fig) # Close the figure to free memory

def test_plot_timeline_stats():
    """
    A simple test function for plot_timeline_stats.
    It creates dummy data and attempts to generate a plot.
    """
    print("\nRunning test_plot_timeline_stats...")

    # Create dummy main DataFrame
    data_df = {
        'task_id': range(1, 31),
        'annotator_id': ['A'] * 10 + ['B'] * 10 + ['C'] * 10,
        'label': ['pos', 'neg', 'neu'] * 10,
        'duration_seconds': np.random.randint(5, 60, 30),
        'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=30, freq='H'))
    }
    df = pd.DataFrame(data_df)

    # Create dummy drift scores DataFrame
    # Simulate some drift over time
    drift_data = {
        'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=10, freq='D')),
        'kl_divergence': [0.01, 0.02, 0.05, 0.1, 0.08, 0.15, 0.12, 0.07, 0.03, 0.01]
    }
    drift_scores = pd.DataFrame(drift_data)

    output_test_path = 'test_timeline_stats.png'

    try:
        plot_timeline_stats(df, drift_scores, output_path=output_test_path)
        print(f"Test plot saved to {output_test_path}")
        assert os.path.exists(output_test_path), "Test plot file was not created."
        print("Test Case 1 (Successful plot generation) passed.")
    except Exception as e:
        print(f"Test Case 1 failed: {e}")
        assert False, f"Plot generation failed: {e}"
    finally:
        if os.path.exists(output_test_path):
            os.remove(output_test_path) # Clean up test file

    # Test Case 2: Empty main DataFrame
    print("\n--- Test Case 2: Empty main DataFrame ---")
    df_empty = pd.DataFrame(columns=['task_id', 'annotator_id', 'label', 'duration_seconds', 'timestamp'])
    try:
        plot_timeline_stats(df_empty, drift_scores, output_path='temp.png')
        print("Test Case 2 (Empty main DataFrame) passed: No plot generated as expected.")
        assert not os.path.exists('temp.png'), "Plot should not be created for empty main DataFrame."
    except Exception as e:
        print(f"Test Case 2 failed: {e}")
        assert False, f"Expected graceful handling of empty DataFrame, but got error: {e}"

    # Test Case 3: Empty drift_scores DataFrame
    print("\n--- Test Case 3: Empty drift_scores DataFrame ---")
    drift_scores_empty = pd.DataFrame(columns=['timestamp', 'kl_divergence'])
    try:
        plot_timeline_stats(df, drift_scores_empty, output_path=output_test_path)
        print(f"Test Case 3 (Empty drift_scores DataFrame) passed: Plot generated without drift data.")
        assert os.path.exists(output_test_path), "Plot should still be created for empty drift_scores."
    except Exception as e:
        print(f"Test Case 3 failed: {e}")
        assert False, f"Plot generation failed: {e}"
    finally:
        if os.path.exists(output_test_path):
            os.remove(output_test_path) # Clean up test file

    # Test Case 4: Missing timestamp column in main df
    print("\n--- Test Case 4: Missing timestamp column in main df ---")
    df_missing_ts = df.drop(columns=['timestamp'])
    try:
        plot_timeline_stats(df_missing_ts, drift_scores, output_path='temp.png')
        assert False, "Test Case 4 failed: Expected ValueError for missing column"
    except ValueError as e:
        assert "Timestamp column 'timestamp' not found in main DataFrame." in str(e), "Test Case 4 failed: Incorrect error message"
    print("Test Case 4 (Missing timestamp column) passed as expected.")

    print("\nAll test cases for plot_timeline_stats completed.")

if __name__ == "__main__":
    test_plot_timeline_stats()
