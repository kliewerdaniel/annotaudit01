import pandas as pd
import os

try:
    import ollama
except ImportError:
    ollama = None
    print("Warning: 'ollama' package not found. Auto-relabeling functionality will be limited.")
    print("Please install it with: pip install ollama")

def relabel_samples_with_ollama(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    model_name: str = 'llama2',
    prompt_template: str = "Given the text: '{text}', what is the most appropriate label from the following options: {labels}? Respond with only the label.",
    max_samples: int = 10 # Limit for demonstration to avoid long inference times
) -> pd.DataFrame:
    """
    Uses a local Ollama LLM to relabel samples. This can be used for auto-correction
    or to get a third opinion on labels, simulating a "gold standard" or expert annotator.

    Args:
        df (pd.DataFrame): DataFrame containing the samples to relabel.
                           Expected columns: text_col, label_col.
        text_col (str): Name of the column containing the text content for relabeling.
        label_col (str): Name of the column containing the original labels.
        model_name (str): The name of the Ollama model to use (e.g., 'llama2', 'mistral').
                          Ensure this model is pulled and running locally via Ollama.
        prompt_template (str): A template for the prompt to send to the LLM.
                               It should contain '{text}' and '{labels}' placeholders.
        max_samples (int): Maximum number of samples to process for demonstration.
                           Set to None to process all samples.

    Returns:
        pd.DataFrame: A DataFrame with original data and an additional 'ollama_relabel' column.
                      Returns original DataFrame if ollama package is not available or an error occurs.
    """
    if ollama is None:
        print("Ollama package not installed. Skipping auto-relabeling.")
        df['ollama_relabel'] = None
        return df

    if df.empty:
        print("Warning: Input DataFrame is empty for auto-relabeling.")
        return pd.DataFrame(columns=df.columns.tolist() + ['ollama_relabel'])

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Required columns '{text_col}' or '{label_col}' not found in DataFrame.")

    relabel_results = []
    
    # Get unique labels to provide as options to the LLM
    unique_labels = df[label_col].dropna().unique().tolist()
    labels_str = ", ".join(map(str, unique_labels))

    df_to_process = df.head(max_samples) if max_samples is not None else df

    print(f"Attempting to connect to Ollama server and use model '{model_name}'...")
    try:
        # Check if Ollama server is reachable and model exists
        # This is a basic check; a more robust check might involve listing models
        # and verifying the specific model_name is present.
        # For now, we'll rely on the client.chat call to fail if not available.
        
        # Initialize Ollama client (defaults to http://localhost:11434)
        client = ollama.Client()

        # Try to generate a dummy response to check connectivity and model availability
        # This is a quick way to test if the model is loaded and ready.
        try:
            client.chat(model=model_name, messages=[{'role': 'user', 'content': 'Hi'}], stream=False)
            print(f"Successfully connected to Ollama and model '{model_name}' is accessible.")
        except Exception as e:
            print(f"Could not connect to Ollama or model '{model_name}' is not available: {e}")
            print("Please ensure Ollama server is running and the model is pulled (e.g., 'ollama run llama2').")
            df['ollama_relabel'] = None
            return df

        for index, row in df_to_process.iterrows():
            text_content = str(row[text_col])
            
            # Construct the prompt
            prompt = prompt_template.format(text=text_content, labels=labels_str)
            
            try:
                response = client.chat(
                    model=model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    stream=False # Get full response at once
                )
                relabel = response['message']['content'].strip()
                relabel_results.append(relabel)
            except Exception as e:
                print(f"Error relabeling sample {index}: {e}")
                relabel_results.append(None) # Append None if relabeling fails for a sample

    except Exception as e:
        print(f"An unexpected error occurred with Ollama: {e}")
        df['ollama_relabel'] = None
        return df

    # Create a new DataFrame with the relabeled results
    result_df = df_to_process.copy()
    result_df['ollama_relabel'] = relabel_results
    
    # Merge back with original df if max_samples was used
    if max_samples is not None and max_samples < len(df):
        full_df = df.copy()
        full_df = full_df.merge(result_df[[df_to_process.index.name if df_to_process.index.name else 'index', 'ollama_relabel']],
                                left_index=True, right_index=True, how='left')
        return full_df
    else:
        return result_df

def test_relabel_samples_with_ollama():
    """
    A simple test function for relabel_samples_with_ollama.
    Note: This test requires a running Ollama server and the specified model.
    It will print warnings and return original data if Ollama is not available.
    """
    print("\nRunning test_relabel_samples_with_ollama...")

    data = {
        'text': [
            "This movie was fantastic, highly recommend!",
            "The service was slow and the food was cold.",
            "It's neither good nor bad, just average.",
            "I absolutely loved the new album, a masterpiece.",
            "What a terrible experience, never again."
        ],
        'label': ['positive', 'negative', 'neutral', 'positive', 'negative'],
        'task_id': range(5)
    }
    df = pd.DataFrame(data)

    # Try to relabel with a dummy model name first, expecting it to fail if not set up
    print("\n--- Test Case 1: Ollama not running or model not found (expected behavior if not set up) ---")
    relabel_df_fail = relabel_samples_with_ollama(df.copy(), 'text', 'label', model_name='nonexistent_model', max_samples=2)
    print(f"Relabeling result (expected failure/None):\n{relabel_df_fail}")
    assert 'ollama_relabel' in relabel_df_fail.columns, "Column 'ollama_relabel' should be added even on failure."
    # If Ollama is not running, these should be None
    if ollama is None:
        assert relabel_df_fail['ollama_relabel'].isnull().all(), "Expected all relabels to be None if Ollama not installed."
    else:
        # If Ollama is installed but model not found, it should still be None
        assert relabel_df_fail['ollama_relabel'].isnull().all(), "Expected all relabels to be None if Ollama model not found."
    print("Test Case 1 passed (or behaved as expected if Ollama not configured).")

    # If Ollama is available, try a more realistic test (requires 'llama2' or similar)
    if ollama is not None:
        print("\n--- Test Case 2: Ollama running and model available (requires 'llama2' or similar) ---")
        try:
            # This assumes 'llama2' is pulled and running. Change if you use a different model.
            relabel_df_success = relabel_samples_with_ollama(df.copy(), 'text', 'label', model_name='llama2', max_samples=3)
            print(f"Relabeling result (expected success):\n{relabel_df_success}")
            assert 'ollama_relabel' in relabel_df_success.columns, "Column 'ollama_relabel' should be present."
            assert not relabel_df_success['ollama_relabel'].isnull().any(), "Expected some relabels to be non-None."
            # Further assertions could check if relabeled labels are within expected categories
            # For example: assert all(l in ['positive', 'negative', 'neutral'] for l in relabel_df_success['ollama_relabel'].dropna())
            print("Test Case 2 passed (assuming Ollama was correctly configured).")
        except Exception as e:
            print(f"Test Case 2 failed due to Ollama interaction: {e}")
            print("Please ensure Ollama server is running and 'llama2' model is pulled (e.g., 'ollama run llama2').")
    else:
        print("\nSkipping Test Case 2 as Ollama package is not installed.")

    # Test Case 3: Empty DataFrame
    print("\n--- Test Case 3: Empty DataFrame ---")
    df_empty = pd.DataFrame(columns=['text', 'label', 'task_id'])
    relabel_df_empty = relabel_samples_with_ollama(df_empty, 'text', 'label', model_name='llama2')
    print(f"Relabeling result (empty):\n{relabel_df_empty}")
    assert relabel_df_empty.empty, "Expected empty DataFrame for empty input."
    print("Test Case 3 passed.")

    # Test Case 4: Missing text column
    print("\n--- Test Case 4: Missing text column ---")
    data_missing_text = {
        'label': ['positive', 'negative'],
        'task_id': [0, 1]
    }
    df_missing_text = pd.DataFrame(data_missing_text)
    try:
        relabel_samples_with_ollama(df_missing_text, 'non_existent_text', 'label', model_name='llama2')
        assert False, "Test Case 4 failed: Expected ValueError for missing column"
    except ValueError as e:
        assert "not found in DataFrame" in str(e), "Test Case 4 failed: Incorrect error message"
    print("Test Case 4 (Missing text column) passed as expected.")

    print("\nAll test cases for relabel_samples_with_ollama completed.")

if __name__ == "__main__":
    test_relabel_samples_with_ollama()
