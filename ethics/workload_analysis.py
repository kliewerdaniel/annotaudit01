import pandas as pd

def analyze_workload(
    df: pd.DataFrame,
    annotator_id_col: str,
    timestamp_col: str,
    daily_hour_threshold: float = 10.0,
    weekly_hour_threshold: float = 50.0
) -> pd.DataFrame:
    """
    Analyzes annotator workload to detect potentially unhealthy working hours.
    Calculates total tasks, total duration, and flags annotators exceeding daily/weekly hour thresholds.

    Args:
        df (pd.DataFrame): DataFrame containing annotation data.
                           Expected columns: annotator_id_col, timestamp_col, and 'duration_seconds'.
        annotator_id_col (str): Name of the column identifying annotators.
        timestamp_col (str): Name of the column containing task submission timestamps (datetime objects).
        daily_hour_threshold (float): Maximum allowed working hours per day.
        weekly_hour_threshold (float): Maximum allowed working hours per week.

    Returns:
        pd.DataFrame: A summary DataFrame with annotator IDs, total tasks, total duration,
                      and flags for exceeding daily/weekly hour thresholds.
                      Returns an empty DataFrame if input is empty.
    """
    if df.empty:
        print("Warning: Input DataFrame is empty for workload analysis.")
        return pd.DataFrame(columns=[annotator_id_col, 'total_tasks', 'total_duration_hours',
                                     'avg_task_duration_seconds', 'flag_daily_overload', 'flag_weekly_overload'])

    required_cols = [annotator_id_col, timestamp_col, 'duration_seconds']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame for workload analysis.")

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Calculate total tasks and total duration per annotator
    annotator_summary = df.groupby(annotator_id_col).agg(
        total_tasks=('task_id', 'nunique'), # Assuming task_id is unique per task
        total_duration_seconds=('duration_seconds', 'sum')
    ).reset_index()
    annotator_summary['total_duration_hours'] = annotator_summary['total_duration_seconds'] / 3600
    annotator_summary['avg_task_duration_seconds'] = annotator_summary['total_duration_seconds'] / annotator_summary['total_tasks']

    # Analyze daily workload
    df['date'] = df[timestamp_col].dt.date
    daily_workload = df.groupby([annotator_id_col, 'date'])['duration_seconds'].sum().reset_index()
    daily_workload['daily_hours'] = daily_workload['duration_seconds'] / 3600

    # Flag annotators who exceeded daily threshold at least once
    over_daily_threshold = daily_workload[daily_workload['daily_hours'] > daily_hour_threshold]
    flagged_daily_annotators = over_daily_threshold[annotator_id_col].unique()
    annotator_summary['flag_daily_overload'] = annotator_summary[annotator_id_col].isin(flagged_daily_annotators)

    # Analyze weekly workload
    df['week'] = df[timestamp_col].dt.isocalendar().week.astype(int)
    df['year'] = df[timestamp_col].dt.year
    weekly_workload = df.groupby([annotator_id_col, 'year', 'week'])['duration_seconds'].sum().reset_index()
    weekly_workload['weekly_hours'] = weekly_workload['duration_seconds'] / 3600

    # Flag annotators who exceeded weekly threshold at least once
    over_weekly_threshold = weekly_workload[weekly_workload['weekly_hours'] > weekly_hour_threshold]
    flagged_weekly_annotators = over_weekly_threshold[annotator_id_col].unique()
    annotator_summary['flag_weekly_overload'] = annotator_summary[annotator_id_col].isin(flagged_weekly_annotators)

    return annotator_summary

def test_analyze_workload():
    """
    A simple test function for analyze_workload.
    """
    print("\nRunning test_analyze_workload...")

    # Test Case 1: Annotator A exceeds daily, Annotator B is fine
    data1 = {
        'task_id': range(10),
        'annotator_id': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'],
        'timestamp': pd.to_datetime([
            '2023-01-01 09:00:00', '2023-01-01 10:00:00', '2023-01-01 11:00:00', '2023-01-01 12:00:00', '2023-01-01 13:00:00',
            '2023-01-01 09:00:00', '2023-01-01 10:00:00', '2023-01-01 11:00:00', '2023-01-01 12:00:00', '2023-01-01 13:00:00'
        ]),
        'duration_seconds': [7200, 7200, 7200, 7200, 7200, # A: 5 tasks * 2 hours = 10 hours
                             3600, 3600, 3600, 3600, 3600]  # B: 5 tasks * 1 hour = 5 hours
    }
    df1 = pd.DataFrame(data1)
    # Add a task_id column for the test, as it's used in the function
    df1['task_id'] = range(len(df1))
    
    # Adjust thresholds for easier testing
    daily_threshold = 9.0
    weekly_threshold = 40.0 # Not relevant for this daily test

    summary1 = analyze_workload(df1, 'annotator_id', 'timestamp', daily_threshold, weekly_threshold)
    print(f"Test Case 1 (Daily Overload):\n{summary1}")
    
    assert summary1[summary1['annotator_id'] == 'A']['flag_daily_overload'].iloc[0] == True, "Annotator A should be flagged for daily overload"
    assert summary1[summary1['annotator_id'] == 'B']['flag_daily_overload'].iloc[0] == False, "Annotator B should not be flagged for daily overload"
    assert summary1[summary1['annotator_id'] == 'A']['total_duration_hours'].iloc[0] == 10.0, "Annotator A total hours incorrect"
    assert summary1[summary1['annotator_id'] == 'B']['total_duration_hours'].iloc[0] == 5.0, "Annotator B total hours incorrect"
    print("Test Case 1 passed.")

    # Test Case 2: Annotator C exceeds weekly, Annotator D is fine
    data2 = {
        'task_id': range(10),
        'annotator_id': ['C', 'C', 'C', 'C', 'C', 'D', 'D', 'D', 'D', 'D'],
        'timestamp': pd.to_datetime([
            '2023-01-01 09:00:00', '2023-01-02 09:00:00', '2023-01-03 09:00:00', '2023-01-04 09:00:00', '2023-01-05 09:00:00', # C: 5 days * 10 hours = 50 hours
            '2023-01-01 09:00:00', '2023-01-02 09:00:00', '2023-01-03 09:00:00', '2023-01-04 09:00:00', '2023-01-05 09:00:00'  # D: 5 days * 5 hours = 25 hours
        ]),
        'duration_seconds': [36000, 36000, 36000, 36000, 36000, # C: 10 hours/day
                             18000, 18000, 18000, 18000, 18000]  # D: 5 hours/day
    }
    df2 = pd.DataFrame(data2)
    df2['task_id'] = range(len(df2))

    daily_threshold = 15.0 # Not relevant for this weekly test
    weekly_threshold = 45.0

    summary2 = analyze_workload(df2, 'annotator_id', 'timestamp', daily_threshold, weekly_threshold)
    print(f"Test Case 2 (Weekly Overload):\n{summary2}")
    
    assert summary2[summary2['annotator_id'] == 'C']['flag_weekly_overload'].iloc[0] == True, "Annotator C should be flagged for weekly overload"
    assert summary2[summary2['annotator_id'] == 'D']['flag_weekly_overload'].iloc[0] == False, "Annotator D should not be flagged for weekly overload"
    assert summary2[summary2['annotator_id'] == 'C']['total_duration_hours'].iloc[0] == 50.0, "Annotator C total hours incorrect"
    assert summary2[summary2['annotator_id'] == 'D']['total_duration_hours'].iloc[0] == 25.0, "Annotator D total hours incorrect"
    print("Test Case 2 passed.")

    # Test Case 3: Empty DataFrame
    df3 = pd.DataFrame(columns=['annotator_id', 'timestamp', 'duration_seconds', 'task_id'])
    summary3 = analyze_workload(df3, 'annotator_id', 'timestamp')
    print(f"Test Case 3 (Empty DataFrame):\n{summary3}")
    assert summary3.empty, "Test Case 3 failed: Expected empty DataFrame for empty input"
    print("Test Case 3 passed.")

    # Test Case 4: Missing duration_seconds column
    data4 = {
        'task_id': [1, 2],
        'annotator_id': ['A', 'B'],
        'timestamp': pd.to_datetime(['2023-01-01', '2023-01-01'])
    }
    df4 = pd.DataFrame(data4)
    try:
        analyze_workload(df4, 'annotator_id', 'timestamp')
        assert False, "Test Case 4 failed: Expected ValueError for missing column"
    except ValueError as e:
        assert "duration_seconds' not found in DataFrame" in str(e), "Test Case 4 failed: Incorrect error message"
    print("Test Case 4 (Missing duration_seconds column) passed as expected.")

    print("All test cases for analyze_workload passed!")

if __name__ == "__main__":
    test_analyze_workload()
