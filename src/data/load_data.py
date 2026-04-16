"""
Module for loading and parsing the SCAN dataset.

This module provides:
    - loading raw SCAN files
    - parsing commands and actions
    - returning train/test splits
"""

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
            data.append({
                "commands": command,
                "actions": actions
            })
    
    return data

def load_scan(split="simple", base_path = "data/SCAN"):
    """
    Loads SCAN dataset.

    Parameters:
    split: str
        Type of dataset split
    base_path: str
        Folder where SCAN dataset is located

    Returns:
    train_data: list
    test_data: list
    """    
    train_path = f"{base_path}/tasks_train_{split}.txt"
    test_path = f"{base_path}/tasks_test_{split}.txt"
    
    train_data = load_scan_file(train_path)
    test_data = load_scan_file(test_path)
    
    return train_data, test_data