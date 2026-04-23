"""
Module for loading and parsing the SCAN dataset.

This module provides:
    - loading raw SCAN files
    - parsing commands and actions
    - returning train/test splits (English or Serbian)
    - loading the HuRIC validation set
"""

import json


def parse_line(line):
    """
    Parses a single line from SCAN dataset.

    Parameters:
    line: str

    Returns:
    command: str
        Natural language command
    actions: str
        Target action sequence
    """
    line = line.strip()

    command = line.split("IN: ")[1].split(" OUT:")[0]
    actions = line.split("OUT: ")[1]

    return command, actions


def load_scan_file(path):
    """
    Loads a SCAN dataset file and parses all lines.

    Parameters:
    path: str
        Path to SCAN file

    Returns:
    data: list of dict
        Each element has:  {
                            "commands": str,
                            "actions": str
                            }
    """
    data = []

    with open(path, "r") as f:
        for line in f:
            command, actions = parse_line(line)
            data.append({"commands": command, "actions": actions})

    return data


def load_scan(split="simple", base_path="data/scan", lang="en"):
    """
    Loads SCAN dataset for a given split and optionally translates it to Serbian.

    This function handles the internal directory structure of the SCAN dataset,
    where each split (e.g. simple, length, add_prim) is stored inside its own
    subfolder. It maps the user-provided split name to the correct folder,
    constructs file paths, and loads both training and test data.

    Parameters:
    split: str
        Type of dataset split
    base_path: str
        Folder where SCAN .txt files are located
    lang: str
        Language of the returned dataset: "en"(default) or "sr"(Serbian)
        When "sr", commands and action tokens are translated using translate_scan.translate_dataset().

    Returns:
    train_data: list of dict
    test_data: list of dict
    """
    
    subfolder_map = {
        # simple
        "simple":                        "simple_split",
        # length
        "length":                        "length_split",
        # add_prim
        "addprim_jump":                  "add_prim_split",
        "addprim_turn_left":             "add_prim_split",
        # template
        "template_around_right":         "template_split",
        "template_jump_around_right":    "template_split",
        "template_opposite_right":       "template_split",
        "template_right":                "template_split",
        # filler
        "filler_num0":                   "filler_split",
        "filler_num1":                   "filler_split",
        "filler_num2":                   "filler_split",
        "filler_num3":                   "filler_split",
        # few_shot (num x rep kombinacije, npr. fewshot_num1_rep1)
        "fewshot_num1_rep1":             "few_shot_split",
        "fewshot_num2_rep1":             "few_shot_split",
        "fewshot_num4_rep1":             "few_shot_split",
        "fewshot_num8_rep1":             "few_shot_split",
        "fewshot_num16_rep1":            "few_shot_split",
        "fewshot_num32_rep1":            "few_shot_split",
        "fewshot_num64_rep1":            "few_shot_split",
        "fewshot_num128_rep1":           "few_shot_split",
        "fewshot_num256_rep1":           "few_shot_split",
        "fewshot_num512_rep1":           "few_shot_split",
        "fewshot_num1024_rep1":          "few_shot_split",
    }

    if split in subfolder_map:
        subfolder = subfolder_map[split]
    else:
        subfolder = split

    train_path = f"{base_path}/{subfolder}/tasks_train_{split}.txt"
    test_path = f"{base_path}/{subfolder}/tasks_test_{split}.txt"

    train_data = load_scan_file(train_path)
    test_data = load_scan_file(test_path)

    if lang == "sr":
        from src.data.translate_scan import translate_dataset

        train_data = translate_dataset(train_data, lang="sr")
        test_data = translate_dataset(test_data, lang="sr")

    return train_data, test_data
