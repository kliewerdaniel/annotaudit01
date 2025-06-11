import pandas as pd
from collections import Counter

def filter_noisy_labels(
    df: pd.DataFrame,
    task_id_col: str,
    label_col: str,
    annotator_id_col: str,
    agreement_threshold: float = 0.6 # A simple threshold for "agreement" within a task
) -> pd.DataFrame:
    """
    Filters out noisy or contradicting samples based on inter-annotator agreement within a task.
    A sample is considered noisy if the highest agreement for a label within a task
    falls below a specified threshold. This is a simplified approach for identifying
    tasks where annotators significantly disagree.

    Args:
        df (pd.DataFrame): DataFrame containing annotation data.
                           Expected columns: task_id_col, label_col, annotator_id_col.
        task_id_col (str): Name of the column identifying unique tasks.
        label_col (str): Name of the column containing the annotated labels.
        annotator_id_col (str): Name of the column identifying annotators.
        agreement_threshold (float): The minimum proportion of annotators agreeing on a label
                                     for a task to be considered "clean". Tasks where no label
                                     reaches this proportion will be filtered out.

    Returns:
        pd.DataFrame: A DataFrame with noisy samples removed.
                      Returns an empty DataFrame if input is empty or all samples are noisy.
    """
    if df.empty:
        print("Warning: Input DataFrame is empty for label filtering.")
        return pd.DataFrame(columns=df.columns)

    required_cols = [task_id_col, label_col, annotator_id_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame for label filtering.")

    clean_task_ids = set()

    # Group by task_id to check agreement within each task
    for task_id, group in df.groupby(task_id_col):
        labels = group[label_col].tolist()
        
        if not labels:
            continue

        label_counts = Counter(labels)
        total_annotations_for_task = len(labels)

        # Check if any label meets the agreement threshold
        is_clean = False
        if total_annotations_for_task > 0:
            for count in label_counts.values():
                if (count / total_annotations_for_task) >= agreement_threshold:
                    is_clean = True
                    break
        
        if is_clean:
            clean_task_ids.add(task_id)

    # Filter the original DataFrame to keep only clean tasks
    cleaned_df = df[df[task_id_col].isin(clean_task_ids)].copy()

    if cleaned_df.empty and not df.empty:
        print(f"Warning: All samples were filtered out based on agreement threshold {agreement_threshold}.")

    return cleaned_df

def test_filter_noisy_labels():
    """
    A simple test function for filter_noisy_labels.
    """
    print("\nRunning test_filter_noisy_labels...")

    # Test Case 1: Some noisy, some clean tasks
    data1 = {
        'task_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
        'annotator_id': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
        'label': ['pos', 'neg', 'neu', # Task 1: No clear majority (noisy)
                  'cat', 'cat', 'dog', # Task 2: 'cat' has 2/3 (0.66) > 0.6 (clean)
                  'red', 'red', 'red', # Task 3: 'red' has 3/3 (1.0) > 0.6 (clean)
                  'apple', 'banana']   # Task 4: No clear majority (noisy)
    }
    df1 = pd.DataFrame(data1)
    cleaned1 = filter_noisy_labels(df1, 'task_id', 'label', 'annotator_id', agreement_threshold=0.6)
    print(f"Test Case 1 (Mixed):\n{cleaned1}")
    
    expected_task_ids_1 = {2, 3}
    assert set(cleaned1['task_id'].unique()) == expected_task_ids_1, "Test Case 1 failed: Incorrect tasks filtered"
    assert len(cleaned1) == 6, "Test Case 1 failed: Incorrect number of rows"
    print("Test Case 1 passed.")

    # Test Case 2: All tasks clean (perfect agreement)
    data2 = {
        'task_id': [10, 10, 11, 11],
        'annotator_id': ['X', 'Y', 'X', 'Y'],
        'label': ['good', 'good', 'bad', 'bad']
    }
    df2 = pd.DataFrame(data2)
    cleaned2 = filter_noisy_labels(df2, 'task_id', 'label', 'annotator_id', agreement_threshold=0.6)
    print(f"Test Case 2 (All Clean):\n{cleaned2}")
    assert len(cleaned2) == len(df2), "Test Case 2 failed: Expected no filtering"
    print("Test Case 2 passed.")

    # Test Case 3: All tasks noisy
    data3 = {
        'task_id': [20, 20, 20, 21, 21, 21],
        'annotator_id': ['P', 'Q', 'R', 'P', 'Q', 'R'],
        'label': ['A', 'B', 'C', 'X', 'Y', 'Z']
    }
    df3 = pd.DataFrame(data3)
    cleaned3 = filter_noisy_labels(df3, 'task_id', 'label', 'annotator_id', agreement_threshold=0.6)
    print(f"Test Case 3 (All Noisy):\n{cleaned3}")
    assert cleaned3.empty, "Test Case 3 failed: Expected all tasks to be filtered"
    print("Test Case 3 passed.")

    # Test Case 4: Empty DataFrame
    df4 = pd.DataFrame(columns=['task_id', 'annotator_id', 'label'])
    cleaned4 = filter_noisy_labels(df4, 'task_id', 'label', 'annotator_id')
    print(f"Test Case 4 (Empty DataFrame):\n{cleaned4}")
    assert cleaned4.empty, "Test Case 4 failed: Expected empty DataFrame for empty input"
    print("Test Case 4 passed.")

    # Test Case 5: Missing label column
    data5 = {
        'task_id': [1, 2],
        'annotator_id': ['A', 'B']
    }
    df5 = pd.DataFrame(data5)
    try:
        filter_noisy_labels(df5, 'task_id', 'non_existent_label', 'annotator_id')
        assert False, "Test Case 5 failed: Expected ValueError for missing column"
    except ValueError as e:
        assert "not found in DataFrame" in str(e), "Test Case 5 failed: Incorrect error message"
    print("Test Case 5 (Missing label column) passed as expected.")

    print("All test cases for filter_noisy_labels passed!")

if __name__ == "__main__":
    test_filter_noisy_labels()
