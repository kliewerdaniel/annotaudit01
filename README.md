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

