"""
Module for saving evaluation results to files.

This module provides utility functions for:
- Saving evaluation results to JSON files
- Exporting results to Google Drive (for Colab environments)
"""

import os
import json
import csv


def save_evaluation_results(results, split_name, output_dir="results"):
    """
    Save evaluation results (from evaluate_model) into a JSON file.
    The file will be named as: evaluation_<split_name>.json

    Parameters:
    results: dict
        Dictionary containing evaluation metrics
    split_name: str
        Dataset split name
    output_dir: str
        Directory where the JSON file will be stored
    """

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"evaluation_{split_name}.json")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Evaluation results saved to: {filepath}")


def copy_results_to_drive(output_dir="results"):
    """
    Copy the local results directory to Google Drive.

    This function is intended to be used in Google Colab environments.
    It mounts Google Drive and copies the entire results folder.

    Parameters:
    output_dir: str
        Local directory containing results
    """

    try:
        from google.colab import drive

        drive.mount("/content/drive")

        import shutil

        dest = "/content/drive/MyDrive/nlp-robot-command-parser/results"

        # Remove existing destination to avoid conflicts
        if os.path.exists(dest):
            shutil.rmtree(dest)

        # Copy entire directory
        shutil.copytree(output_dir, dest)

        print(f"Results folder copied to Google Drive: {dest}")

    except ImportError:
        print(
            "Not running in Colab environment - results are saved locally in results/"
        )
