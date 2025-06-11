# annotaudit

A developer-friendly Python toolkit designed to audit, cleanse, and analyze human-annotated datasets within machine learning pipelines. `annotaudit` helps ensure data quality, identify annotation inconsistencies, and provide insights into annotator performance and behavior.

## Features

-   **⚖️ Detect Annotation Drift and Fatigue:** Algorithms to identify shifts in annotation patterns over time and signs of annotator fatigue.
-   **🛡️ Identify Rushed or Fabricated Task Completion:** Tools to flag tasks completed unusually quickly or exhibiting suspicious patterns.
-   **👁️ Visualize Annotator Profiles and Work Trends:** Generate visual reports on individual annotator performance, consistency, and workload distribution.
-   **🤖 Optional LLM-Assisted Relabeling:** Integrate with Large Language Models to assist in the relabeling process for improved accuracy and efficiency.
-   **🧹 Cleansing Pipeline:** A robust pipeline to filter inconsistent or noisy labels, ensuring a high-quality dataset for model training.

## Installation

To get started with `annotaudit`, clone the repository and install the required dependencies:

```bash
git clone https://github.com/kliewerdaniel/annotaudit01.git
cd annotaudit01
pip install -r requirements.txt
```

## Usage

`annotaudit` can be run via the `main.py` script with various command-line arguments to perform different auditing and cleansing tasks.

### Basic Audit

To perform a basic audit on your annotation data:

```bash
python main.py --audit path/to/your_annotations.csv --config config/schema.yaml
```

-   `--audit`: Specifies the path to your input CSV file containing annotations.
-   `--config`: Specifies the path to your configuration schema file (e.g., `config/schema.yaml`) which defines your annotation task and label structure.

### Generating Reports

After running an audit, various reports and visualizations are generated in the `annotaudit_results/` directory.

For example, to view annotator profiles:

```bash
# This is automatically generated after an audit
open annotaudit_results/annotator_profiles.png
```

### Cleansing Labels

To apply cleansing operations to your labels:

```bash
python main.py --cleanse path/to/your_annotations.csv --config config/schema.yaml --output cleaned_labels.csv
```

-   `--cleanse`: Initiates the cleansing pipeline.
-   `--output`: Specifies the output file path for the cleaned labels.

### Configuration

The `config/schema.yaml` file is crucial for defining your annotation project's structure, including label types, relationships, and any specific rules for auditing. An example schema might look like this:

```yaml
# config/schema.yaml
dataset_schema:
  id_column: "task_id"
  annotator_column: "annotator_id"
  timestamp_column: "timestamp"
  label_columns:
    - "category"
    - "sentiment"
  label_definitions:
    category:
      type: "categorical"
      values: ["A", "B", "C"]
    sentiment:
      type: "categorical"
      values: ["positive", "negative", "neutral"]
```

## Detailed Explanation of `main.py`

The `main.py` script serves as the command-line interface (CLI) for the `annotaudit` toolkit. It handles argument parsing, data loading, and orchestrates the execution of various auditing, ethics, cleansing, and visualization modules based on user-specified flags.

### How it Works

1.  **Argument Parsing**: Uses `argparse` to define and parse command-line arguments. Users can specify the input data path, output directory, and which specific audit/analysis modules to run.
2.  **Module Imports**: Dynamically adds the parent directory to `sys.path` to allow importing modules from `audit`, `ethics`, `cleansing`, and `viz` subdirectories.
3.  **Data Loading (`load_data` function)**:
    *   Supports loading data from `.csv` and `.jsonl` file formats into a pandas DataFrame.
    *   It's designed to be flexible for different annotation data structures.
4.  **Main Execution Flow (`main` function)**:
    *   Creates the output directory (`annotaudit_results` by default) if it doesn't exist.
    *   Loads the input dataset using `load_data`.
    *   **Simulated Column Creation**: For demonstration purposes, if certain key columns (`annotator_id`, `task_id`, `label`, `timestamp`, `duration_seconds`, `wage_per_hour`) are not present in the loaded DataFrame, it generates dummy data for them. In a real-world scenario, these columns would be expected in your input dataset or defined via a configuration.
    *   **Conditional Module Execution**: Based on the command-line arguments (`--run_all`, `--consistency`, `--drift`, etc.), it calls the relevant functions from the imported modules:
        *   **`audit` modules**:
            *   `consistency.compute_agreement`: Calculates inter-annotator agreement scores.
            *   `drift.analyze_label_drift`: Analyzes shifts in label distribution over time.
            *   `speed_check.flag_fast_tasks`: Identifies tasks completed unusually quickly.
            *   `redundancy_check.resolve_disagreements`: Resolves label disagreements, typically using majority voting.
        *   **`ethics` modules**:
            *   `workload_analysis.analyze_workload`: Detects annotators working unhealthy hours.
            *   `wage_efficiency.estimate_wage_efficiency`: Estimates how annotator pay aligns with effort.
        *   **`cleansing` modules**:
            *   `label_filter.filter_noisy_labels`: Removes inconsistent or noisy samples from the dataset.
            *   `auto_relabel.relabel_samples_with_ollama`: (Optional) Uses an Ollama LLM to relabel samples. This requires an Ollama server running locally.
        *   **`viz` modules**:
            *   `annotator_profiles.plot_annotator_performance`: Generates visual profiles of annotator performance.
            *   `timeline_stats.plot_timeline_stats`: Plots trends of annotation volume, drift, and errors over time.
    *   Each module's results (e.g., consistency scores, drift scores, flagged tasks) are printed to the console and saved as CSV files in the specified output directory. Visualizations are saved as PNG images.

### Command-Line Arguments

*   `data_path` (positional argument): Path to the input dataset (CSV or JSONL format).
*   `--output_dir` (optional, default: `annotaudit_results`): Directory to save all generated audit results and visualizations.
*   `--run_all` (flag): If set, executes all available audit, ethics, and cleansing modules.
*   `--consistency` (flag): Runs inter-annotator consistency checks.
*   `--drift` (flag): Analyzes label distribution drift over time.
*   `--speed_check` (flag): Flags tasks submitted too quickly.
*   `--redundancy_check` (flag): Resolves disagreements using majority voting.
*   `--workload_analysis` (flag): Detects annotators working unhealthy hours.
*   `--wage_efficiency` (flag): Estimates annotator pay alignment with effort.
*   `--label_filter` (flag): Removes noisy or contradicting samples.
*   `--auto_relabel` (flag): Uses Ollama LLM to relabel samples locally (requires Ollama server running).
*   `--annotator_profiles` (flag): Generates annotator performance profiles.
*   `--timeline_stats` (flag): Plots trends of annotation volume, drift, and errors.

This structure allows users to run a comprehensive suite of analyses or target specific aspects of their annotation data quality and annotator performance.

## Project Structure

```
.
├── main.py                     # Main entry point for running audits and cleansing
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── annotaudit_results/         # Directory for generated reports and visualizations
│   ├── annotator_profiles.png
│   ├── auto_relabel_results.csv
│   ├── cleaned_labels.csv
│   ├── consistency_scores.csv
│   ├── drift_scores.csv
│   ├── flagged_fast_tasks.csv
│   ├── resolved_labels.csv
│   ├── timeline_stats.png
│   ├── wage_efficiency_report.csv
│   └── workload_summary.csv
├── audit/                      # Modules for auditing annotation quality
│   ├── consistency.py
│   ├── drift.py
│   ├── redundancy_check.py
│   └── speed_check.py
├── cleansing/                  # Modules for cleansing and relabeling
│   ├── auto_relabel.py
│   └── label_filter.py
├── config/                     # Configuration files
│   └── schema.yaml             # Example schema for dataset definition
├── ethics/                     # Modules for ethical considerations in annotation
│   ├── wage_efficiency.py
│   └── workload_analysis.py
├── examples/                   # Example usage and sample data
│   ├── audit_example.ipynb
│   └── sample_data.csv
├── tests/                      # Unit and integration tests
│   └── test_consistency.py
└── viz/                        # Visualization scripts
    ├── annotator_profiles.py
    └── timeline_stats.py
```
