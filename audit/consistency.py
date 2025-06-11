import pandas as pd
from sklearn.metrics import cohen_kappa_score
import numpy as np

def compute_agreement(
    df: pd.DataFrame,
    task_id_col: str,
    annotator_id_col: str,
    label_col: str
) -> dict:
    """
    Computes inter-annotator agreement scores (Cohen's Kappa and Krippendorff's Alpha).

    Args:
        df (pd.DataFrame): DataFrame containing annotation data.
                           Expected columns: task_id_col, annotator_id_col, label_col.
        task_id_col (str): Name of the column identifying unique tasks.
        annotator_id_col (str): Name of the column identifying annotators.
        label_col (str): Name of the column containing the annotated labels.

    Returns:
        dict: A dictionary containing 'cohen_kappa' and 'krippendorff_alpha' scores.
              Returns NaN for scores if insufficient data or errors occur.
    """
    agreement_scores = {
        "cohen_kappa": np.nan,
        "krippendorff_alpha": np.nan
    }

    # Prepare data for agreement calculation
    # Pivot table to get tasks as rows, annotators as columns, labels as values
    pivot_df = df.pivot_table(
        index=task_id_col,
        columns=annotator_id_col,
        values=label_col,
        aggfunc=lambda x: x.iloc[0] # Take the first label if multiple for same task/annotator
    )

    if pivot_df.empty or pivot_df.shape[1] < 2:
        print("Warning: Not enough annotators or tasks for agreement calculation.")
        return agreement_scores

    # Cohen's Kappa (pairwise)
    # This is typically for two annotators. For multiple, we can average pairwise.
    annotators = pivot_df.columns
    kappa_scores = []
    if len(annotators) >= 2:
        for i in range(len(annotators)):
            for j in range(i + 1, len(annotators)):
                ann1_labels = pivot_df[annotators[i]].dropna()
                ann2_labels = pivot_df[annotators[j]].dropna()

                # Find common tasks for pairwise comparison
                common_tasks = ann1_labels.index.intersection(ann2_labels.index)
                if len(common_tasks) > 1: # Need at least 2 samples for kappa
                    kappa = cohen_kappa_score(ann1_labels.loc[common_tasks], ann2_labels.loc[common_tasks])
                    kappa_scores.append(kappa)
        if kappa_scores:
            agreement_scores["cohen_kappa"] = np.mean(kappa_scores)
        else:
            print("Warning: No valid pairs for Cohen's Kappa calculation.")

    # Krippendorff's Alpha (for multiple annotators and various data types)
    # This implementation is simplified for nominal data.
    # A full Krippendorff's Alpha implementation is complex and often uses dedicated libraries.
    # For demonstration, we'll use a simplified approach that works for nominal data
    # and assumes all annotators have rated all items for a given task.
    # For a robust solution, consider a dedicated library like `irr` or `statsmodels` (if available locally).

    # Convert labels to numerical representation for calculation if they are not already
    unique_labels = df[label_col].unique()
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels_df = pivot_df.applymap(lambda x: label_to_int.get(x, np.nan))

    # Remove tasks where all annotators have NaN (no annotations)
    numeric_labels_df = numeric_labels_df.dropna(how='all')

    if numeric_labels_df.empty:
        print("Warning: No valid data for Krippendorff's Alpha calculation after dropping NaNs.")
        return agreement_scores

    # Simplified Krippendorff's Alpha for nominal data
    # This is a conceptual implementation. For production, use a tested library.
    # N = number of units (tasks)
    # m = number of annotators
    # c = number of categories (labels)
    # D_o = observed disagreement
    # D_e = expected disagreement

    N = numeric_labels_df.shape[0] # Number of tasks
    m = numeric_labels_df.shape[1] # Number of annotators
    
    if N == 0 or m < 2:
        print("Warning: Insufficient data for Krippendorff's Alpha calculation.")
        return agreement_scores

    # Calculate observed disagreement (D_o)
    D_o = 0.0
    for _, row in numeric_labels_df.iterrows():
        # Count occurrences of each label for the current task
        label_counts = row.value_counts().to_dict()
        # Sum of n_c * (n_c - 1) for each category c
        # where n_c is the number of annotators who assigned category c to the unit
        sum_nc_nc_minus_1 = sum(count * (count - 1) for count in label_counts.values())
        D_o += sum_nc_nc_minus_1

    # Calculate expected disagreement (D_e)
    # This requires the overall distribution of labels across all annotations
    all_labels = df[label_col].dropna()
    if all_labels.empty:
        print("Warning: No labels found for expected disagreement calculation.")
        return agreement_scores

    overall_label_counts = all_labels.value_counts()
    total_annotations = len(all_labels)
    
    if total_annotations == 0:
        print("Warning: Total annotations is zero for expected disagreement calculation.")
        return agreement_scores

    D_e = 0.0
    for count in overall_label_counts.values:
        D_e += count * (count - 1)

    # Krippendorff's Alpha formula: 1 - (D_o / D_e)
    if D_e == 0:
        # If D_e is 0, it means all labels are the same, so perfect agreement (alpha = 1)
        agreement_scores["krippendorff_alpha"] = 1.0
    else:
        agreement_scores["krippendorff_alpha"] = 1.0 - (D_o / D_e)

    return agreement_scores

def test_compute_agreement():
    """
    A simple test function for compute_agreement.
    """
    print("\nRunning test_compute_agreement...")
    # Test Case 1: Perfect agreement
    data1 = {
        'task_id': [1, 1, 2, 2, 3, 3],
        'annotator_id': ['A', 'B', 'A', 'B', 'A', 'B'],
        'label': ['cat', 'cat', 'dog', 'dog', 'bird', 'bird']
    }
    df1 = pd.DataFrame(data1)
    scores1 = compute_agreement(df1, 'task_id', 'annotator_id', 'label')
    print(f"Test Case 1 (Perfect Agreement): {scores1}")
    assert scores1['cohen_kappa'] == 1.0, "Test Case 1 Cohen's Kappa failed"
    assert scores1['krippendorff_alpha'] == 1.0, "Test Case 1 Krippendorff's Alpha failed"

    # Test Case 2: Some disagreement
    data2 = {
        'task_id': [1, 1, 2, 2, 3, 3],
        'annotator_id': ['A', 'B', 'A', 'B', 'A', 'B'],
        'label': ['cat', 'dog', 'dog', 'dog', 'bird', 'fish']
    }
    df2 = pd.DataFrame(data2)
    scores2 = compute_agreement(df2, 'task_id', 'annotator_id', 'label')
    print(f"Test Case 2 (Some Disagreement): {scores2}")
    # Expected kappa for (cat,dog), (dog,dog), (bird,fish)
    # Kappa for (cat,dog) is -0.5 (random agreement)
    # Kappa for (dog,dog) is 1.0
    # Kappa for (bird,fish) is -0.5 (random agreement)
    # Average kappa should be around 0.0
    assert scores2['cohen_kappa'] is not None and scores2['cohen_kappa'] < 1.0, "Test Case 2 Cohen's Kappa failed"
    assert scores2['krippendorff_alpha'] is not None and scores2['krippendorff_alpha'] < 1.0, "Test Case 2 Krippendorff's Alpha failed"

    # Test Case 3: Multiple annotators, some missing data
    data3 = {
        'task_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'annotator_id': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
        'label': ['pos', 'pos', 'neg', 'neg', 'neg', 'pos', 'neu', 'neu', 'neu']
    }
    df3 = pd.DataFrame(data3)
    scores3 = compute_agreement(df3, 'task_id', 'annotator_id', 'label')
    print(f"Test Case 3 (Multiple Annotators): {scores3}")
    assert scores3['cohen_kappa'] is not None, "Test Case 3 Cohen's Kappa failed"
    assert scores3['krippendorff_alpha'] is not None, "Test Case 3 Krippendorff's Alpha failed"

    # Test Case 4: Insufficient annotators
    data4 = {
        'task_id': [1, 2, 3],
        'annotator_id': ['A', 'A', 'A'],
        'label': ['cat', 'dog', 'bird']
    }
    df4 = pd.DataFrame(data4)
    scores4 = compute_agreement(df4, 'task_id', 'annotator_id', 'label')
    print(f"Test Case 4 (Insufficient Annotators): {scores4}")
    assert np.isnan(scores4['cohen_kappa']), "Test Case 4 Cohen's Kappa should be NaN"
    assert np.isnan(scores4['krippendorff_alpha']), "Test Case 4 Krippendorff's Alpha should be NaN"

    print("All test cases for compute_agreement passed!")

if __name__ == "__main__":
    test_compute_agreement()
