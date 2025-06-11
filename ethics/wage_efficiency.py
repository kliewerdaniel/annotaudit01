import pandas as pd

def estimate_wage_efficiency(
    df: pd.DataFrame,
    annotator_id_col: str,
    duration_col: str,
    wage_per_hour_col: str
) -> pd.DataFrame:
    """
    Estimates annotator wage efficiency by calculating metrics like tasks per hour,
    average task duration, and estimated earnings per task.

    Args:
        df (pd.DataFrame): DataFrame containing annotation data.
                           Expected columns: annotator_id_col, duration_col, wage_per_hour_col.
                           'task_id' column is also expected for counting unique tasks.
        annotator_id_col (str): Name of the column identifying annotators.
        duration_col (str): Name of the column containing the duration of the task in seconds.
        wage_per_hour_col (str): Name of the column containing the annotator's wage per hour.

    Returns:
        pd.DataFrame: A summary DataFrame with annotator IDs and various efficiency metrics.
                      Returns an empty DataFrame if input is empty.
    """
    if df.empty:
        print("Warning: Input DataFrame is empty for wage efficiency analysis.")
        return pd.DataFrame(columns=[annotator_id_col, 'total_tasks', 'total_duration_hours',
                                     'avg_task_duration_seconds', 'estimated_total_earnings',
                                     'tasks_per_hour', 'cost_per_task'])

    required_cols = [annotator_id_col, duration_col, wage_per_hour_col, 'task_id']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame for wage efficiency analysis.")

    df = df.copy()

    # Calculate total tasks and total duration per annotator
    annotator_summary = df.groupby(annotator_id_col).agg(
        total_tasks=('task_id', 'nunique'),
        total_duration_seconds=(duration_col, 'sum'),
        wage_per_hour=(wage_per_hour_col, 'mean') # Assuming wage_per_hour is constant per annotator
    ).reset_index()

    annotator_summary['total_duration_hours'] = annotator_summary['total_duration_seconds'] / 3600
    annotator_summary['avg_task_duration_seconds'] = annotator_summary['total_duration_seconds'] / annotator_summary['total_tasks']

    # Estimated total earnings
    annotator_summary['estimated_total_earnings'] = annotator_summary['total_duration_hours'] * annotator_summary['wage_per_hour']

    # Tasks per hour
    # Handle division by zero if total_duration_hours is 0
    annotator_summary['tasks_per_hour'] = annotator_summary.apply(
        lambda row: row['total_tasks'] / row['total_duration_hours'] if row['total_duration_hours'] > 0 else 0,
        axis=1
    )

    # Cost per task
    annotator_summary['cost_per_task'] = annotator_summary.apply(
        lambda row: row['estimated_total_earnings'] / row['total_tasks'] if row['total_tasks'] > 0 else 0,
        axis=1
    )

    return annotator_summary

def test_estimate_wage_efficiency():
    """
    A simple test function for estimate_wage_efficiency.
    """
    print("\nRunning test_estimate_wage_efficiency...")

    # Test Case 1: Basic calculation
    data1 = {
        'task_id': [1, 2, 3, 4, 5, 6],
        'annotator_id': ['A', 'A', 'A', 'B', 'B', 'B'],
        'duration_seconds': [360, 180, 60, 120, 240, 60], # A: 600s (10min), B: 420s (7min)
        'wage_per_hour': [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]
    }
    df1 = pd.DataFrame(data1)
    efficiency1 = estimate_wage_efficiency(df1, 'annotator_id', 'duration_seconds', 'wage_per_hour')
    print(f"Test Case 1 (Basic Calculation):\n{efficiency1}")

    # Expected values for Annotator A:
    # total_tasks = 3
    # total_duration_seconds = 360 + 180 + 60 = 600
    # total_duration_hours = 600 / 3600 = 1/6 hours
    # avg_task_duration_seconds = 600 / 3 = 200
    # estimated_total_earnings = (1/6) * 20 = 3.333...
    # tasks_per_hour = 3 / (1/6) = 18
    # cost_per_task = 3.333... / 3 = 1.111...

    # Expected values for Annotator B:
    # total_tasks = 3
    # total_duration_seconds = 120 + 240 + 60 = 420
    # total_duration_hours = 420 / 3600 = 7/60 hours
    # avg_task_duration_seconds = 420 / 3 = 140
    # estimated_total_earnings = (7/60) * 25 = 2.916...
    # tasks_per_hour = 3 / (7/60) = 180/7 = 25.714...
    # cost_per_task = 2.916... / 3 = 0.972...

    assert efficiency1[efficiency1['annotator_id'] == 'A']['total_tasks'].iloc[0] == 3
    assert abs(efficiency1[efficiency1['annotator_id'] == 'A']['total_duration_hours'].iloc[0] - (600/3600)) < 1e-9
    assert abs(efficiency1[efficiency1['annotator_id'] == 'A']['estimated_total_earnings'].iloc[0] - (20.0 * 600/3600)) < 1e-9
    assert abs(efficiency1[efficiency1['annotator_id'] == 'A']['tasks_per_hour'].iloc[0] - 18.0) < 1e-9
    assert abs(efficiency1[efficiency1['annotator_id'] == 'A']['cost_per_task'].iloc[0] - (20.0 * 600/3600 / 3)) < 1e-9

    assert efficiency1[efficiency1['annotator_id'] == 'B']['total_tasks'].iloc[0] == 3
    assert abs(efficiency1[efficiency1['annotator_id'] == 'B']['total_duration_hours'].iloc[0] - (420/3600)) < 1e-9
    assert abs(efficiency1[efficiency1['annotator_id'] == 'B']['estimated_total_earnings'].iloc[0] - (25.0 * 420/3600)) < 1e-9
    assert abs(efficiency1[efficiency1['annotator_id'] == 'B']['tasks_per_hour'].iloc[0] - (3 / (420/3600))) < 1e-9
    assert abs(efficiency1[efficiency1['annotator_id'] == 'B']['cost_per_task'].iloc[0] - (25.0 * 420/3600 / 3)) < 1e-9
    print("Test Case 1 passed.")

    # Test Case 2: Empty DataFrame
    df2 = pd.DataFrame(columns=['task_id', 'annotator_id', 'duration_seconds', 'wage_per_hour'])
    efficiency2 = estimate_wage_efficiency(df2, 'annotator_id', 'duration_seconds', 'wage_per_hour')
    print(f"Test Case 2 (Empty DataFrame):\n{efficiency2}")
    assert efficiency2.empty, "Test Case 2 failed: Expected empty DataFrame for empty input"
    print("Test Case 2 passed.")

    # Test Case 3: Missing duration_seconds column
    data3 = {
        'task_id': [1, 2],
        'annotator_id': ['A', 'B'],
        'wage_per_hour': [20.0, 20.0]
    }
    df3 = pd.DataFrame(data3)
    try:
        estimate_wage_efficiency(df3, 'annotator_id', 'non_existent_duration', 'wage_per_hour')
        assert False, "Test Case 3 failed: Expected ValueError for missing column"
    except ValueError as e:
        assert "not found in DataFrame" in str(e), "Test Case 3 failed: Incorrect error message"
    print("Test Case 3 (Missing duration_seconds column) passed as expected.")

    print("All test cases for estimate_wage_efficiency passed!")

if __name__ == "__main__":
    test_estimate_wage_efficiency()
