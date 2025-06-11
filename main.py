import argparse
import pandas as pd
import json
import os
import sys

# Add parent directory to path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from audit import consistency, drift, speed_check, redundancy_check
from ethics import workload_analysis, wage_efficiency
from cleansing import label_filter, auto_relabel
from viz import annotator_profiles, timeline_stats

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads data from a CSV or JSONL file into a pandas DataFrame.

    Args:
        filepath (str): The path to the input data file.

    Returns:
        pd.DataFrame: The loaded data.

    Raises:
        ValueError: If the file format is not supported.
    """
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.jsonl'):
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return pd.DataFrame(data)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .jsonl")

def main():
    """
    Main function for the annotaudit CLI tool.
    Parses arguments and runs the specified audit pipelines.
    """
    parser = argparse.ArgumentParser(
        description="AnnotAudit: A toolkit for auditing human-annotated datasets."
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the input dataset (CSV or JSONL format)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="annotaudit_results",
        help="Directory to save audit results and visualizations."
    )
    parser.add_argument(
        "--run_all",
        action="store_true",
        help="Run all available audit, ethics, and cleansing modules."
    )
    parser.add_argument(
        "--consistency",
        action="store_true",
        help="Run inter-annotator consistency checks."
    )
    parser.add_argument(
        "--drift",
        action="store_true",
        help="Analyze label distribution drift over time."
    )
    parser.add_argument(
        "--speed_check",
        action="store_true",
        help="Flag tasks submitted too quickly."
    )
    parser.add_argument(
        "--redundancy_check",
        action="store_true",
        help="Resolve disagreements using majority voting."
    )
    parser.add_argument(
        "--workload_analysis",
        action="store_true",
        help="Detect annotators working unhealthy hours."
    )
    parser.add_argument(
        "--wage_efficiency",
        action="store_true",
        help="Estimate annotator pay alignment with effort."
    )
    parser.add_argument(
        "--label_filter",
        action="store_true",
        help="Remove noisy or contradicting samples."
    )
    parser.add_argument(
        "--auto_relabel",
        action="store_true",
        help="Use Ollama LLM to relabel samples locally (requires Ollama server running)."
    )
    parser.add_argument(
        "--annotator_profiles",
        action="store_true",
        help="Generate annotator performance profiles."
    )
    parser.add_argument(
        "--timeline_stats",
        action="store_true",
        help="Plot trends of annotation volume, drift, and errors."
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data from {args.data_path}...")
    try:
        df = load_data(args.data_path)
        print("Data loaded successfully.")
    except ValueError as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Simulate required columns for demonstration
    # In a real scenario, these would be part of a configuration or inferred.
    if 'annotator_id' not in df.columns:
        df['annotator_id'] = df.index % 5 # Example: 5 annotators
    if 'task_id' not in df.columns:
        df['task_id'] = df.index
    if 'label' not in df.columns:
        df['label'] = ['A', 'B', 'C', 'A', 'B'] * (len(df) // 5 + 1)
        df['label'] = df['label'].head(len(df))
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.to_datetime(pd.Series(range(len(df))), unit='s')
    if 'duration_seconds' not in df.columns:
        df['duration_seconds'] = [10, 20, 5, 15, 30] * (len(df) // 5 + 1)
        df['duration_seconds'] = df['duration_seconds'].head(len(df))
    if 'wage_per_hour' not in df.columns:
        df['wage_per_hour'] = 20.0 # Example wage

    # Ensure 'timestamp' is datetime for time-based analyses
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    if args.run_all or args.consistency:
        print("\nRunning Consistency Audit...")
        agreement_scores = consistency.compute_agreement(df, 'task_id', 'annotator_id', 'label')
        print("Agreement Scores:", agreement_scores)
        # Save results or pass to viz
        pd.DataFrame([agreement_scores]).to_csv(os.path.join(args.output_dir, "consistency_scores.csv"), index=False)

    if args.run_all or args.drift:
        print("\nRunning Drift Analysis...")
        drift_scores = drift.analyze_label_drift(df, 'timestamp', 'label', window_size='1D')
        print("Drift Scores (first 5):", drift_scores.head())
        drift_scores.to_csv(os.path.join(args.output_dir, "drift_scores.csv"), index=False)

    if args.run_all or args.speed_check:
        print("\nRunning Speed Check...")
        flagged_tasks = speed_check.flag_fast_tasks(df, 'duration_seconds', threshold_seconds=5)
        print(f"Flagged {len(flagged_tasks)} tasks for being too fast.")
        if not flagged_tasks.empty:
            flagged_tasks.to_csv(os.path.join(args.output_dir, "flagged_fast_tasks.csv"), index=False)

    if args.run_all or args.redundancy_check:
        print("\nRunning Redundancy Check (Majority Voting)...")
        resolved_labels = redundancy_check.resolve_disagreements(df, 'task_id', 'label')
        print("Resolved Labels (first 5):", resolved_labels.head())
        resolved_labels.to_csv(os.path.join(args.output_dir, "resolved_labels.csv"), index=False)

    if args.run_all or args.workload_analysis:
        print("\nRunning Workload Analysis...")
        workload_summary = workload_analysis.analyze_workload(df, 'annotator_id', 'timestamp')
        print("Workload Summary (first 5):", workload_summary.head())
        workload_summary.to_csv(os.path.join(args.output_dir, "workload_summary.csv"), index=False)

    if args.run_all or args.wage_efficiency:
        print("\nRunning Wage Efficiency Analysis...")
        efficiency_report = wage_efficiency.estimate_wage_efficiency(df, 'annotator_id', 'duration_seconds', 'wage_per_hour')
        print("Wage Efficiency Report (first 5):", efficiency_report.head())
        efficiency_report.to_csv(os.path.join(args.output_dir, "wage_efficiency_report.csv"), index=False)

    if args.run_all or args.label_filter:
        print("\nRunning Label Filter...")
        cleaned_df = label_filter.filter_noisy_labels(df, 'task_id', 'label', 'annotator_id')
        print(f"Original samples: {len(df)}, Cleaned samples: {len(cleaned_df)}")
        cleaned_df.to_csv(os.path.join(args.output_dir, "cleaned_labels.csv"), index=False)
        df = cleaned_df # Update df for subsequent steps if filtering occurred

    if args.run_all or args.auto_relabel:
        print("\nRunning Auto-Relabeling with Ollama...")
        # This requires an Ollama server running and a model pulled (e.g., 'llama2')
        # For demonstration, we'll use a dummy relabeling if Ollama is not available.
        try:
            # Assuming 'text_column' exists in your data for relabeling
            if 'text' not in df.columns:
                df['text'] = "This is a sample text for relabeling."
            relabel_results = auto_relabel.relabel_samples_with_ollama(df.head(5), 'text', 'label', model_name='llama2')
            print("Auto-Relabeling Results (first 5):", relabel_results)
            relabel_results.to_csv(os.path.join(args.output_dir, "auto_relabel_results.csv"), index=False)
        except ImportError:
            print("Ollama Python package not found. Skipping auto-relabeling.")
        except Exception as e:
            print(f"Error during auto-relabeling: {e}. Make sure Ollama server is running and model is pulled.")

    if args.run_all or args.annotator_profiles:
        print("\nGenerating Annotator Profiles...")
        # Assuming 'agreement_scores' and 'workload_summary' are available from previous steps
        # For a standalone run, these would need to be loaded or computed.
        if 'agreement_scores' not in locals():
            agreement_scores = consistency.compute_agreement(df, 'task_id', 'annotator_id', 'label')
        if 'workload_summary' not in locals():
            workload_summary = workload_analysis.analyze_workload(df, 'annotator_id', 'timestamp')

        annotator_profiles.plot_annotator_performance(
            df,
            agreement_scores,
            workload_summary,
            output_path=os.path.join(args.output_dir, "annotator_profiles.png")
        )
        print(f"Annotator profiles saved to {os.path.join(args.output_dir, 'annotator_profiles.png')}")

    if args.run_all or args.timeline_stats:
        print("\nGenerating Timeline Statistics...")
        # Assuming 'drift_scores' is available
        if 'drift_scores' not in locals():
            drift_scores = drift.analyze_label_drift(df, 'timestamp', 'label', window_size='1D')
        timeline_stats.plot_timeline_stats(
            df,
            drift_scores,
            output_path=os.path.join(args.output_dir, "timeline_stats.png")
        )
        print(f"Timeline statistics saved to {os.path.join(args.output_dir, 'timeline_stats.png')}")

    print("\nAnnotAudit process completed.")

if __name__ == "__main__":
    main()
