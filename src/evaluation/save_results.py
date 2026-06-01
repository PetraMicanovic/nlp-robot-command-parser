"""
Module for saving evaluation results to files.

This module provides utility functions for:
- Saving evaluation results to JSON files
- Exporting results to Google Drive (for Colab environments)
"""

import os
import json


def get_results_dir(config, model_key="model"):
    """
    Returns the results directory for a given model key from config.

    Uses cfg["results"] mapping:
        "model" -> cfg["results"]["t5_small_dir"]
        "model_t5_base"-> cfg["results"]["t5_base_dir"]
        "model_mbart" -> cfg["results"]["mbart_dir"]
    Falls back to "results/" if the mapping is not found.

    Parameters:
    cfg: dict
        Full config loaded from config.json
    model_key: str
        Key into cfg that identifies the model section

    Returns:
    output_dir: str
    """
    mapping = {
        "model": config.get("results", {}).get("t5_small_dir", "results/t5-small"),
        "model_t5_base": config.get("results", {}).get(
            "t5_base_dir", "results/t5-base"
        ),
        "model_mbart": config.get("results", {}).get("mbart_dir", "results/mbart"),
    }
    return mapping.get(model_key, "results")


def save_evaluation_results(results, split_name, cfg, model_key="model"):
    """
    Save evaluation results (from evaluate_model) into a JSON file. The output directory is resolved automatically from config using get_results_dir(cfg, model_key), so the caller does not need to construct or pass a path manually.
    The file will be named as: evaluation_<split_name>.json

    Parameters:
    results: dict
        Dictionary containing evaluation metrics
    split_name: str
        Dataset split name
    cfg: dict
        Full config loaded from config.json
    model_key: str
        Key into cfg that identifies the model section
        ("model", "model_t5_base", or "model_mbart")
    """
    output_dir = get_results_dir(cfg, model_key)
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"evaluation_{split_name}.json")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Evaluation results saved to: {filepath}")


def save_length_analysis(buckets, cfg, model_key="model"):
    """
    Save length analysis results into a JSON file.

    Parameters:
    buckets: dict
        Dictionary containing grouped analysis results.
        Keys usually represent categories, ranges or sequence lengths, while values contain statistics or collected data for each group.
    cfg: dict
        Full config loaded from config.json
    model_key: str
        Key into cfg that identifies the model section
        ("model", "model_t5_base", or "model_mbart")
    """
    output_dir = os.path.join(get_results_dir(cfg, model_key), "length_analysis")

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "length_analysis.json")

    serializable = {}
    for key in buckets:
        serializable[str(key)] = buckets[key]

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"Length analysis saved to: {filepath}")


def copy_results_to_drive(cfg, model_key="model"):
    """
    Copy the local results directory to Google Drive.

    This function is intended to be used in Google Colab environments.
    It mounts Google Drive and copies the entire results folder.

    Parameters:
    cfg: dict
        Full config loaded from config.json
    model_key: str
        Key into cfg that identifies the model section
        ("model", "model_t5_base", or "model_mbart")
    """

    try:
        from google.colab import drive

        if not os.path.isdir("/content/drive/MyDrive"):
            drive.mount("/content/drive")

        import shutil

        output_dir = get_results_dir(cfg, model_key)

        dest = f"/content/drive/MyDrive/nlp-robot-command-parser/{output_dir}"
        os.makedirs(dest, exist_ok=True)

        # Copy entire directory
        shutil.copytree(output_dir, dest, dirs_exist_ok=True)

        print(f"Results folder copied to Google Drive: {dest}")

    except ImportError:
        print(
            "Not running in Colab environment - results are saved locally in results/"
        )


def save_asr_results(asr_results, cfg, model_key="model", filename="asr_results.json"):
    """
    Save ASR pipeline results to a JSON file.

    Parameters
    asr_results : list[dict]
        Output of ``run_asr_pipeline`` — each dict has keys: ``command``, ``audio_path``, ``transcript``, ``match``
    cfg: dict
        Full config loaded from config.json
    model_key: str
        Key into cfg that identifies the model section
        ("model", "model_t5_base", or "model_mbart")
    filename : str
        Output filename
    """
    output_dir = get_results_dir(cfg, model_key)

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
