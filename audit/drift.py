import pandas as pd
import numpy as np
from scipy.stats import entropy

def _calculate_kl_divergence(p: pd.Series, q: pd.Series) -> float:
    """
    Calculates the Kullback-Leibler (KL) divergence between two probability distributions.

    Args:
        p (pd.Series): Probability distribution P.
        q (pd.Series): Probability distribution Q.

    Returns:
        float: The KL divergence D(P || Q). Returns 0 if distributions are identical.
               Handles cases where probabilities are zero by adding a small epsilon.
    """
    # Ensure both distributions have the same categories and are normalized
    all_categories = pd.Series(list(set(p.index) | set(q.index)))
    p_aligned = p.reindex(all_categories, fill_value=0).sort_index()
    q_aligned = q.reindex(all_categories, fill_value=0).sort_index()

    # Add a small epsilon to avoid log(0) and division by zero
    epsilon = 1e-10
    p_aligned = p_aligned + epsilon
    q_aligned = q_aligned + epsilon

    # Normalize to ensure they are valid probability distributions
    p_aligned = p_aligned / p_aligned.sum()
    q_aligned = q_aligned / q_aligned.sum()

    return entropy(p_aligned, q_aligned)

def analyze_label_drift(
    df: pd.DataFrame,
    timestamp_col: str,
    label_col: str,
    window_size: str = '7D'
) -> pd.DataFrame:
    """
    Analyzes label distribution changes over time using rolling Kullback-Leibler (KL) divergence.
    Compares the label distribution in a rolling window to the distribution in the preceding window.

    Args:
        df (pd.DataFrame): DataFrame containing annotation data.
                           Expected columns: timestamp_col, label_col.
        timestamp_col (str): Name of the column containing timestamps (datetime objects).
        label_col (str): Name of the column containing the annotated labels.
        window_size (str): Rolling window size (e.g., '7D' for 7 days, '24H' for 24 hours).

    Returns:
        pd.DataFrame: A DataFrame with 'timestamp' and 'kl_divergence' columns,
                      showing the KL divergence at each point in time.
                      The timestamp represents the end of the current window.
    """
    if df.empty:
        print("Warning: Input DataFrame is empty for drift analysis.")
        return pd.DataFrame(columns=[timestamp_col, 'kl_divergence'])

    df = df.sort_values(by=timestamp_col).copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Calculate overall label distribution as a reference (or initial window)
    # For rolling KL, we compare current window to previous window.
    # The first window will not have a preceding window, so KL will be NaN.

    kl_divergence_scores = []
    timestamps = []

    # Group by time window and calculate label distributions
    # Use a fixed-size rolling window based on time
    # We need to iterate through time points to define rolling windows
    min_time = df[timestamp_col].min()
    max_time = df[timestamp_col].max()
    
    # Generate time points for rolling windows
    # We'll use a daily frequency for simplicity, but it can be adjusted
    time_points = pd.date_range(start=min_time, end=max_time, freq='D')

    # Ensure window_size is a Timedelta for comparison
    window_td = pd.to_timedelta(window_size)

    for i in range(len(time_points)):
        current_end_time = time_points[i]
        current_start_time = current_end_time - window_td
        
        prev_end_time = current_start_time
        prev_start_time = prev_end_time - window_td

        current_window_df = df[(df[timestamp_col] > current_start_time) & (df[timestamp_col] <= current_end_time)]
        prev_window_df = df[(df[timestamp_col] > prev_start_time) & (df[timestamp_col] <= prev_end_time)]

        if not current_window_df.empty and not prev_window_df.empty:
            p_dist = current_window_df[label_col].value_counts(normalize=True)
            q_dist = prev_window_df[label_col].value_counts(normalize=True)
            
            kl_div = _calculate_kl_divergence(p_dist, q_dist)
            kl_divergence_scores.append(kl_div)
            timestamps.append(current_end_time)
        else:
            # If either window is empty, we cannot calculate KL divergence
            kl_divergence_scores.append(np.nan)
            timestamps.append(current_end_time)

    result_df = pd.DataFrame({
        timestamp_col: timestamps,
        'kl_divergence': kl_divergence_scores
    })
    
    # Drop rows where KL divergence could not be calculated (e.g., initial windows)
    result_df = result_df.dropna(subset=['kl_divergence']).reset_index(drop=True)

    return result_df

def test_analyze_label_drift():
    """
    A simple test function for analyze_label_drift.
    """
    print("\nRunning test_analyze_label_drift...")

    # Test Case 1: No drift (labels are consistent over time)
    data1 = {
        'timestamp': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03', '2023-01-03', '2023-01-04', '2023-01-04']),
        'label': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    }
    df1 = pd.DataFrame(data1)
    drift_scores1 = analyze_label_drift(df1, 'timestamp', 'label', window_size='1D')
    print(f"Test Case 1 (No Drift):\n{drift_scores1}")
    # Expect KL divergence to be very small or zero after the first valid window
    assert all(score < 0.01 for score in drift_scores1['kl_divergence'].dropna()), "Test Case 1 failed: Expected low KL divergence"

    # Test Case 2: Significant drift (label distribution changes)
    data2 = {
        'timestamp': pd.to_datetime([
            '2023-01-01', '2023-01-01', '2023-01-01', # Day 1: A, A, B
            '2023-01-02', '2023-01-02', '2023-01-02', # Day 2: A, B, B
            '2023-01-03', '2023-01-03', '2023-01-03', # Day 3: B, B, C
            '2023-01-04', '2023-01-04', '2023-01-04'  # Day 4: C, C, C
        ]),
        'label': ['A', 'A', 'B', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C']
    }
    df2 = pd.DataFrame(data2)
    drift_scores2 = analyze_label_drift(df2, 'timestamp', 'label', window_size='1D')
    print(f"Test Case 2 (Significant Drift):\n{drift_scores2}")
    # Expect some non-zero KL divergence values
    assert any(score > 0.1 for score in drift_scores2['kl_divergence'].dropna()), "Test Case 2 failed: Expected significant KL divergence"

    # Test Case 3: Empty DataFrame
    df3 = pd.DataFrame(columns=['timestamp', 'label'])
    drift_scores3 = analyze_label_drift(df3, 'timestamp', 'label', window_size='1D')
    print(f"Test Case 3 (Empty DataFrame):\n{drift_scores3}")
    assert drift_scores3.empty, "Test Case 3 failed: Expected empty DataFrame for empty input"

    print("All test cases for analyze_label_drift passed!")

if __name__ == "__main__":
    test_analyze_label_drift()
