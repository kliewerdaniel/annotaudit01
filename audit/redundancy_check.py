import pandas as pd
from collections import Counter

def resolve_disagreements(
    df: pd.DataFrame,
    task_id_col: str,
    label_col: str
) -> pd.DataFrame:
    """
    Resolves disagreements among annotators for the same task using majority voting.
    If there's a tie in votes, the label that appears first alphabetically among the tied labels is chosen.

    Args:
        df (pd.DataFrame): DataFrame containing annotation data.
                           Expected columns: task_id_col, label_col.
        task_id_col (str): Name of the column identifying unique tasks.
        label_col (str): Name of the column containing the annotated labels.

    Returns:
        pd.DataFrame: A DataFrame with unique tasks and their resolved labels.
                      Columns: task_id_col, 'resolved_label'.
                      Returns an empty DataFrame if input is empty.
    """
    if df.empty:
        print("Warning: Input DataFrame is empty for redundancy check.")
        return pd.DataFrame(columns=[task_id_col, 'resolved_label'])

    if task_id_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Required columns '{task_id_col}' or '{label_col}' not found in DataFrame.")

    resolved_labels = []

    # Group by task_id and apply majority voting
    for task_id, group in df.groupby(task_id_col):
        labels = group[label_col].tolist()
        
        # Count occurrences of each label
        label_counts = Counter(labels)
        
        if not label_counts:
            # Should not happen if group is not empty, but for safety
            resolved_labels.append({task_id_col: task_id, 'resolved_label': None})
            continue

        # Find the maximum vote count
        max_count = 0
        for count in label_counts.values():
            if count > max_count:
                max_count = count

        # Find all labels that have the maximum vote count (potential ties)
        majority_labels = [label for label, count in label_counts.items() if count == max_count]
        
        # Tie-breaking: choose the first label alphabetically among the tied ones
        resolved_label = sorted(majority_labels)[0]
        
        resolved_labels.append({task_id_col: task_id, 'resolved_label': resolved_label})

    return pd.DataFrame(resolved_labels)

def test_resolve_disagreements():
    """
    A simple test function for resolve_disagreements.
    """
    print("\nRunning test_resolve_disagreements...")

    # Test Case 1: Clear majority
    data1 = {
        'task_id': [1, 1, 1, 2, 2, 2, 3, 3],
        'annotator_id': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
        'label': ['cat', 'cat', 'dog', 'pos', 'neg', 'pos', 'neutral', 'positive']
    }
    df1 = pd.DataFrame(data1)
    resolved1 = resolve_disagreements(df1, 'task_id', 'label')
    print(f"Test Case 1 (Clear Majority):\n{resolved1}")
    expected1 = pd.DataFrame({
        'task_id': [1, 2, 3],
        'resolved_label': ['cat', 'pos', 'neutral'] # 'neutral' comes before 'positive' alphabetically
    })
    pd.testing.assert_frame_equal(resolved1.sort_values('task_id').reset_index(drop=True), 
                                  expected1.sort_values('task_id').reset_index(drop=True), 
                                  check_dtype=False)
    print("Test Case 1 passed.")

    # Test Case 2: Tie-breaking (alphabetical)
    data2 = {
        'task_id': [10, 10, 10, 10],
        'annotator_id': ['X', 'Y', 'Z', 'W'],
        'label': ['apple', 'banana', 'apple', 'banana']
    }
    df2 = pd.DataFrame(data2)
    resolved2 = resolve_disagreements(df2, 'task_id', 'label')
    print(f"Test Case 2 (Tie-breaking):\n{resolved2}")
    expected2 = pd.DataFrame({
        'task_id': [10],
        'resolved_label': ['apple'] # 'apple' comes before 'banana'
    })
    pd.testing.assert_frame_equal(resolved2, expected2, check_dtype=False)
    print("Test Case 2 passed.")

    # Test Case 3: Single annotation per task (no disagreement)
    data3 = {
        'task_id': [20, 21, 22],
        'annotator_id': ['P', 'Q', 'R'],
        'label': ['red', 'blue', 'green']
    }
    df3 = pd.DataFrame(data3)
    resolved3 = resolve_disagreements(df3, 'task_id', 'label')
    print(f"Test Case 3 (Single Annotation):\n{resolved3}")
    expected3 = pd.DataFrame({
        'task_id': [20, 21, 22],
        'resolved_label': ['red', 'blue', 'green']
    })
    pd.testing.assert_frame_equal(resolved3.sort_values('task_id').reset_index(drop=True), 
                                  expected3.sort_values('task_id').reset_index(drop=True), 
                                  check_dtype=False)
    print("Test Case 3 passed.")

    # Test Case 4: Empty DataFrame
    df4 = pd.DataFrame(columns=['task_id', 'annotator_id', 'label'])
    resolved4 = resolve_disagreements(df4, 'task_id', 'label')
    print(f"Test Case 4 (Empty DataFrame):\n{resolved4}")
    assert resolved4.empty, "Test Case 4 failed: Expected empty DataFrame for empty input"
    print("Test Case 4 passed.")

    # Test Case 5: Missing label column
    data5 = {
        'task_id': [1, 2],
        'annotator_id': ['A', 'B']
    }
    df5 = pd.DataFrame(data5)
    try:
        resolve_disagreements(df5, 'task_id', 'non_existent_label')
        assert False, "Test Case 5 failed: Expected ValueError for missing column"
    except ValueError as e:
        assert "not found in DataFrame" in str(e), "Test Case 5 failed: Incorrect error message"
    print("Test Case 5 (Missing label column) passed as expected.")

    print("All test cases for resolve_disagreements passed!")

if __name__ == "__main__":
    test_resolve_disagreements()
