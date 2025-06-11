import pandas as pd

def flag_fast_tasks(
    df: pd.DataFrame,
    duration_col: str,
    threshold_seconds: float = 5.0
) -> pd.DataFrame:
    """
    Flags tasks that were completed too quickly, potentially indicating annotator fatigue
    or low-quality annotations.

    Args:
        df (pd.DataFrame): DataFrame containing annotation data.
                           Expected to have a column for task duration.
        duration_col (str): Name of the column containing the duration of the task in seconds.
        threshold_seconds (float): The minimum acceptable duration for a task.
                                   Tasks with duration below this threshold will be flagged.

    Returns:
        pd.DataFrame: A DataFrame containing only the flagged tasks,
                      with an additional 'flag_reason' column.
                      Returns an empty DataFrame if no tasks are flagged.
    """
    if df.empty:
        print("Warning: Input DataFrame is empty for speed check.")
        return pd.DataFrame(columns=df.columns.tolist() + ['flag_reason'])

    if duration_col not in df.columns:
        raise ValueError(f"Duration column '{duration_col}' not found in DataFrame.")

    flagged_df = df[df[duration_col] < threshold_seconds].copy()
    if not flagged_df.empty:
        flagged_df['flag_reason'] = f"Task completed too quickly (duration < {threshold_seconds} seconds)"
    
    return flagged_df

def test_flag_fast_tasks():
    """
    A simple test function for flag_fast_tasks.
    """
    print("\nRunning test_flag_fast_tasks...")

    # Test Case 1: Some fast tasks
    data1 = {
        'task_id': [1, 2, 3, 4, 5],
        'annotator_id': ['A', 'B', 'A', 'C', 'B'],
        'duration_seconds': [3.0, 10.0, 4.5, 2.0, 15.0]
    }
    df1 = pd.DataFrame(data1)
    flagged1 = flag_fast_tasks(df1, 'duration_seconds', threshold_seconds=5.0)
    print(f"Test Case 1 (Some fast tasks):\n{flagged1}")
    assert len(flagged1) == 2, "Test Case 1 failed: Expected 2 flagged tasks"
    assert 1 in flagged1['task_id'].values, "Task 1 should be flagged"
    assert 4 in flagged1['task_id'].values, "Task 4 should be flagged"
    assert all(flagged1['flag_reason'].str.contains("too quickly")), "Flag reason incorrect"

    # Test Case 2: No fast tasks
    data2 = {
        'task_id': [10, 11, 12],
        'annotator_id': ['X', 'Y', 'Z'],
        'duration_seconds': [6.0, 8.0, 7.5]
    }
    df2 = pd.DataFrame(data2)
    flagged2 = flag_fast_tasks(df2, 'duration_seconds', threshold_seconds=5.0)
    print(f"Test Case 2 (No fast tasks):\n{flagged2}")
    assert flagged2.empty, "Test Case 2 failed: Expected no flagged tasks"

    # Test Case 3: All fast tasks
    data3 = {
        'task_id': [20, 21],
        'annotator_id': ['P', 'Q'],
        'duration_seconds': [1.0, 0.5]
    }
    df3 = pd.DataFrame(data3)
    flagged3 = flag_fast_tasks(df3, 'duration_seconds', threshold_seconds=5.0)
    print(f"Test Case 3 (All fast tasks):\n{flagged3}")
    assert len(flagged3) == 2, "Test Case 3 failed: Expected 2 flagged tasks"

    # Test Case 4: Empty DataFrame
    df4 = pd.DataFrame(columns=['task_id', 'annotator_id', 'duration_seconds'])
    flagged4 = flag_fast_tasks(df4, 'duration_seconds', threshold_seconds=5.0)
    print(f"Test Case 4 (Empty DataFrame):\n{flagged4}")
    assert flagged4.empty, "Test Case 4 failed: Expected empty DataFrame for empty input"

    # Test Case 5: Missing duration column
    data5 = {
        'task_id': [1, 2],
        'annotator_id': ['A', 'B']
    }
    df5 = pd.DataFrame(data5)
    try:
        flag_fast_tasks(df5, 'non_existent_col', threshold_seconds=5.0)
        assert False, "Test Case 5 failed: Expected ValueError for missing column"
    except ValueError as e:
        assert "not found in DataFrame" in str(e), "Test Case 5 failed: Incorrect error message"
    print("Test Case 5 (Missing duration column) passed as expected.")

    print("All test cases for flag_fast_tasks passed!")

if __name__ == "__main__":
    test_flag_fast_tasks()
