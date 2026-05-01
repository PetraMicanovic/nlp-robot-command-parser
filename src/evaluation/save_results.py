"""
Module for saving evaluation results to files.

This module provides utility functions for:
- Saving evaluation results to JSON files
- Exporting results to Google Drive (for Colab environments)
"""

import os
import json

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


def save_length_analysis(buckets, output_dir="results"):
    """
    Save length analysis results into a JSON file.

    Parameters:
    buckets: dict
        Dictionary containing grouped analysis results.
        Keys usually represent categories, ranges or sequence lengths, while values contain statistics or collected data for each group.
    output_dir: str
        Directory where the JSON file will be stored
    """

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"length_analysis.json")

    serializable = {}
    for key in buckets:
        serializable[str(key)] = buckets[key]

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"Length analysis saved to: {filepath}")


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

        if not os.path.isdir("/content/drive/MyDrive"):
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


def save_asr_results(asr_results, output_dir="results", filename="asr_results.json"):
    """
    Save ASR pipeline results to a JSON file.

    Parameters
    asr_results : list[dict]
        Output of ``run_asr_pipeline`` — each dict has keys: ``command``, ``audio_path``, ``transcript``, ``match``
    output_dir : str
        Directory where the file will be stored
    filename : str
        Output filename
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    n_match = sum(r["match"] for r in asr_results)
    summary = {
        "n_commands": len(asr_results),
        "n_exact_match": n_match,
        "exact_match_rate": (
            round(n_match / len(asr_results), 4) if asr_results else 0.0
        ),
        "results": asr_results,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"ASR results saved to: {filepath}")
    return filepath


def save_audio_to_drive(audio_path, filename=None):
    """
    Copy a single audio file to Google Drive (Colab only).

    Parameters
    audio_path : str
        Local path to the .mp3 file.
    filename : str, optional
        Name to use on Google Drive.  Defaults to the basename of *audio_path*.
    """
    try:
        from google.colab import drive
        import shutil

        if not os.path.isdir("/content/drive/MyDrive"):
            drive.mount("/content/drive")
        dest_dir = "/content/drive/MyDrive/nlp-robot-command-parser/audio"
        os.makedirs(dest_dir, exist_ok=True)

        dest_name = filename if filename else os.path.basename(audio_path)
        dest_path = os.path.join(dest_dir, dest_name)
        shutil.copy2(audio_path, dest_path)
        print(f"Audio file copied to Google Drive: {dest_path}")

    except ImportError:
        print("Not running in Colab — audio file stays at:", audio_path)
