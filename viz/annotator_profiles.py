import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_annotator_performance(
    df: pd.DataFrame,
    agreement_scores: dict, # From audit.consistency.compute_agreement
    workload_summary: pd.DataFrame, # From ethics.workload_analysis.analyze_workload
    annotator_id_col: str = 'annotator_id',
    output_path: str = 'annotator_profiles.png'
) -> None:
    """
    Generates and saves a plot visualizing annotator performance profiles,
    including total tasks, average agreement, and average task duration.

    Args:
        df (pd.DataFrame): The main DataFrame containing annotation data.
                           Expected to have 'annotator_id' and 'duration_seconds'.
        agreement_scores (dict): Dictionary of overall agreement scores (e.g., from consistency.py).
                                 Expected to contain 'cohen_kappa' or 'krippendorff_alpha'.
        workload_summary (pd.DataFrame): DataFrame summarizing annotator workload.
                                         Expected columns: 'annotator_id', 'total_tasks',
                                         'avg_task_duration_seconds'.
        annotator_id_col (str): Name of the column identifying annotators.
        output_path (str): Full path including filename to save the generated plot.
    """
    if df.empty or workload_summary.empty:
        print("Warning: Input DataFrame or workload summary is empty. Cannot generate annotator profiles.")
        return

    if annotator_id_col not in df.columns:
        raise ValueError(f"Annotator ID column '{annotator_id_col}' not found in main DataFrame.")
    if 'total_tasks' not in workload_summary.columns or 'avg_task_duration_seconds' not in workload_summary.columns:
        raise ValueError("Workload summary must contain 'total_tasks' and 'avg_task_duration_seconds' columns.")

    # Ensure annotator_id_col is consistent across dataframes
    workload_summary = workload_summary.rename(columns={workload_summary.columns[0]: annotator_id_col})

    # Merge agreement scores into workload summary for plotting
    # For simplicity, we'll use Cohen's Kappa if available, otherwise Krippendorff's Alpha
    # A more sophisticated approach might involve per-annotator agreement scores.
    
    # For this visualization, we'll calculate a simple average agreement per annotator
    # by joining with the original df and assuming a 'label' column exists.
    # This is a simplified proxy for individual annotator agreement.
    # A more robust approach would involve a dedicated function in consistency.py
    # to calculate agreement *per annotator* against a gold standard or majority.
    
    # For now, we'll just use the overall agreement score as a general indicator
    # or calculate a proxy if multiple labels per task are available.
    
    # Let's calculate a proxy for annotator agreement:
    # How often does an annotator's label match the majority vote for a task?
    # This requires re-running majority voting or having it pre-computed.
    
    # To avoid circular dependencies or re-computation, we'll assume `redundancy_check.resolve_disagreements`
    # has been run and we can use its output or simulate it.
    
    # For the purpose of this visualization, let's assume we have a 'resolved_label' column
    # or we can compute a simple agreement against the most frequent label per task.
    
    # Calculate agreement against majority for each annotator
    # This requires grouping by task_id and finding the majority label, then comparing.
    
    # Let's simplify for the plot: if overall agreement scores are provided, use them.
    # Otherwise, we'll just plot tasks and duration.
    
    # Create a temporary column for agreement score in workload_summary
    workload_summary['avg_agreement_score'] = np.nan
    
    if 'cohen_kappa' in agreement_scores and not np.isnan(agreement_scores['cohen_kappa']):
        workload_summary['avg_agreement_score'] = agreement_scores['cohen_kappa']
    elif 'krippendorff_alpha' in agreement_scores and not np.isnan(agreement_scores['krippendorff_alpha']):
        workload_summary['avg_agreement_score'] = agreement_scores['krippendorff_alpha']
    else:
        print("Warning: No valid overall agreement scores provided. Annotator agreement will not be plotted.")

    # Sort by total tasks for better visualization
    workload_summary = workload_summary.sort_values(by='total_tasks', ascending=False)

    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle('Annotator Performance Profiles', fontsize=16)

    # Plot 1: Total Tasks per Annotator
    axes[0].bar(workload_summary[annotator_id_col], workload_summary['total_tasks'], color='skyblue')
    axes[0].set_ylabel('Total Tasks')
    axes[0].set_title('Total Tasks Completed by Each Annotator')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Plot 2: Average Agreement Score per Annotator (if available)
    if not workload_summary['avg_agreement_score'].isnull().all():
        axes[1].bar(workload_summary[annotator_id_col], workload_summary['avg_agreement_score'], color='lightcoral')
        axes[1].set_ylabel('Average Agreement Score')
        axes[1].set_title('Average Agreement Score per Annotator')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].set_ylim(0, 1) # Agreement scores are typically between 0 and 1
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    else:
        axes[1].set_visible(False) # Hide subplot if no agreement data

    # Plot 3: Average Task Duration per Annotator
    axes[2].bar(workload_summary[annotator_id_col], workload_summary['avg_task_duration_seconds'], color='lightgreen')
    axes[2].set_ylabel('Avg Task Duration (seconds)')
    axes[2].set_title('Average Task Duration per Annotator')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent title overlap
    plt.savefig(output_path)
    plt.close(fig) # Close the figure to free memory

def test_plot_annotator_performance():
    """
    A simple test function for plot_annotator_performance.
    It creates dummy data and attempts to generate a plot.
    """
    print("\nRunning test_plot_annotator_performance...")

    # Create dummy dataframes
    data_df = {
        'task_id': range(1, 16),
        'annotator_id': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
        'label': ['pos', 'neg', 'pos', 'neg', 'pos', 'neg', 'pos', 'neg', 'pos', 'pos', 'neg', 'pos', 'neg', 'pos', 'neg'],
        'duration_seconds': [10, 15, 12, 8, 20, 11, 14, 16, 9, 13, 18, 10, 7, 22, 15],
        'timestamp': pd.to_datetime(['2023-01-01'] * 5 + ['2023-01-02'] * 5 + ['2023-01-03'] * 5)
    }
    df = pd.DataFrame(data_df)

    # Simulate agreement scores (from audit.consistency)
    # In a real scenario, you'd call consistency.compute_agreement(df, ...)
    agreement_scores = {
        "cohen_kappa": 0.75,
        "krippendorff_alpha": 0.70
    }

    # Simulate workload summary (from ethics.workload_analysis)
    # In a real scenario, you'd call workload_analysis.analyze_workload(df, ...)
    workload_summary_data = {
        'annotator_id': ['A', 'B', 'C'],
        'total_tasks': [5, 5, 5],
        'total_duration_seconds': [df[df['annotator_id'] == 'A']['duration_seconds'].sum(),
                                   df[df['annotator_id'] == 'B']['duration_seconds'].sum(),
                                   df[df['annotator_id'] == 'C']['duration_seconds'].sum()],
        'avg_task_duration_seconds': [df[df['annotator_id'] == 'A']['duration_seconds'].mean(),
                                      df[df['annotator_id'] == 'B']['duration_seconds'].mean(),
                                      df[df['annotator_id'] == 'C']['duration_seconds'].mean()]
    }
    workload_summary = pd.DataFrame(workload_summary_data)

    output_test_path = 'test_annotator_profiles.png'

    try:
        plot_annotator_performance(df, agreement_scores, workload_summary, output_path=output_test_path)
        print(f"Test plot saved to {output_test_path}")
        assert os.path.exists(output_test_path), "Test plot file was not created."
        print("Test Case 1 (Successful plot generation) passed.")
    except Exception as e:
        print(f"Test Case 1 failed: {e}")
        assert False, f"Plot generation failed: {e}"
    finally:
        if os.path.exists(output_test_path):
            os.remove(output_test_path) # Clean up test file

    # Test Case 2: Empty DataFrame
    print("\n--- Test Case 2: Empty DataFrame ---")
    df_empty = pd.DataFrame(columns=['task_id', 'annotator_id', 'label', 'duration_seconds', 'timestamp'])
    workload_summary_empty = pd.DataFrame(columns=['annotator_id', 'total_tasks', 'avg_task_duration_seconds'])
    try:
        plot_annotator_performance(df_empty, agreement_scores, workload_summary_empty, output_path='temp.png')
        print("Test Case 2 (Empty DataFrame) passed: No plot generated as expected.")
        assert not os.path.exists('temp.png'), "Plot should not be created for empty DataFrame."
    except Exception as e:
        print(f"Test Case 2 failed: {e}")
        assert False, f"Expected graceful handling of empty DataFrame, but got error: {e}"

    # Test Case 3: Missing required column in workload_summary
    print("\n--- Test Case 3: Missing required column in workload_summary ---")
    workload_summary_missing = pd.DataFrame({
        'annotator_id': ['A'],
        'total_tasks': [10]
        # 'avg_task_duration_seconds' is missing
    })
    try:
        plot_annotator_performance(df, agreement_scores, workload_summary_missing, output_path='temp.png')
        assert False, "Test Case 3 failed: Expected ValueError for missing column"
    except ValueError as e:
        assert "must contain 'total_tasks' and 'avg_task_duration_seconds' columns" in str(e), "Test Case 3 failed: Incorrect error message"
    print("Test Case 3 (Missing required column) passed as expected.")

    print("\nAll test cases for plot_annotator_performance completed.")

if __name__ == "__main__":
    test_plot_annotator_performance()
