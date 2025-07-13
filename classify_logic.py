import pandas as pd
from proReg import classify_with_regex
from probr import classify_with_bert
from prollm import classify_with_llm
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor


def classify(logs, fast=False):
    """
    Classify a list of log messages based on source and content.
    Args:
        logs (list of tuples): [(source, log_message), ...]
        fast (bool): If True, skip LLM and BERT for faster classification.
    Returns:
        list: List of predicted labels
    """
    with ThreadPoolExecutor() as executor:
        return list(executor.map(lambda log: classify_cached(log[0], log[1], fast), logs))


@lru_cache(maxsize=10000)
def classify_cached(source, log_msg, fast):
    return classify_log(source, log_msg, fast)


def classify_log(source, log_msg, fast=False):
    """
    Classify a single log message.
    Priority:
    - If source == "LegacyCRM" and not fast, use LLM
    - Else, try regex
        - If regex fails and not fast, fall back to BERT
    Returns:
        str: Classification label
    """
    if source == "LegacyCRM" and not fast:
        return classify_with_llm(log_msg)

    label = classify_with_regex(log_msg)
    if not label and not fast:
        label = classify_with_bert(log_msg)
    return label or "Unclassified"


def classify_csv(input_file, output_file="output.csv", fast=False):
    """
    Read a CSV file, classify log messages, and write results.
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        fast (bool): Use fast classification mode
    Returns:
        str: Path to the output CSV file
    """
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        raise Exception(f"Failed to read CSV: {e}")

    if "source" not in df.columns or "log_message" not in df.columns:
        raise ValueError("CSV must contain 'source' and 'log_message' columns.")

    df["target_label"] = classify(list(zip(df["source"], df["log_message"])), fast=fast)
    df.to_csv(output_file, index=False)
    return output_file


if __name__ == '__main__':
    try:
        output_path = classify_csv("test.csv", fast=True)  # Change to fast=False for full model run
        print(f"✅ Classification completed. Output saved to: {output_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
