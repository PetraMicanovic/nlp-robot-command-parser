"""
Module for preprocessing SCAN dataset for T5 seq2seq training.

This module provides:
    - converting raw list-of-dicts to HuggingFace Dataset
    - tokenizing commands and actions with T5Tokenizer
    - returning tokenized train/test splits ready for Trainer
"""

from transformers import T5Tokenizer
from datasets import Dataset

def to_hf_dataset(data):
    """
    Converts a list of dicts to a HuggingFace Dataset.

    Parameters:
    data: list of dict
        Each element has {"commands": str, "actions": str}

    Returns:
    dataset: datasets.Dataset
    """
    commands = []
    actions = []
    for ex in data:
        commands.append(ex["commands"])
        actions.append(ex["actions"])

    return Dataset.from_dict({"commands": commands, "actions": actions})

def get_tokenizer(model_name):
    """
    Loads a T5 tokenizer.

    Parameters:
    model_name: str
        HuggingFace model name 

    Returns:
    tokenizer: T5Tokenizer 
    """
    return T5Tokenizer.from_pretrained(model_name)

def preprocess(examples, tokenizer, prefix, max_input_len, max_target_len):
    """
    Preprocesses a batch of examples for T5 training.

    Adds a task prefix to each command, tokenizes inputs (commands) and targets (actions), and replaces padding token IDs in labels with -100 (ignored in loss).
    
    Parameters:
    examples: dict
        Batch from HuggingFace Dataset with keys "commands" and "actions"
    tokenizer: T5Tokenizer
        Tokenizer used for encoding text
    prefix: str
        Task-specific prefix (e.g. "translate command to actions: ")
    max_input_len: int
        Maximum length for input sequences
    max_target_len: int
        Maximum length for target sequences

    Returns:
    model_inputs: dict
        Tokenized inputs with labels ready for training
    """
    inputs = []
    for cmd in examples["commands"]:
        inputs.append(prefix + cmd)

    targets = examples["actions"]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_len,
        truncation=True,
        padding="max_length",
    )

    labels = tokenizer(
        text_target=targets,
        max_length=max_target_len,
        truncation=True,
        padding="max_length",
    )

    new_labels = []

    for label in labels["input_ids"]:
        new_label = []

        for tok in label:
            if tok == tokenizer.pad_token_id:
                new_label.append(-100)
            else:
                new_label.append(tok)

        new_labels.append(new_label)

    model_inputs["labels"] = new_labels

    return model_inputs


def tokenize_dataset(dataset: Dataset, tokenizer, prefix, max_input_len, max_target_len):
    """
    Applies preprocessing to the entire dataset using HuggingFace map().

    This function:
    - wraps the preprocess function (to pass additional arguments)
    - applies it in batches for efficiency
    - removes original text columns after tokenization

    Parameters:
    dataset: datasets.Dataset
        Dataset with "commands" and "actions"
    tokenizer: T5Tokenizer
    prefix: str
    max_input_len: int
    max_target_len: int

    Returns:
    tokenized: datasets.Dataset
        Dataset ready for training (input_ids, attention_mask, labels)
    """
    def preprocess_wrapper(examples):
        return preprocess(
            examples,
            tokenizer,
            prefix,
            max_input_len,
            max_target_len
        )

    tokenized = dataset.map(
        preprocess_wrapper,
        batched=True,
        remove_columns=["commands", "actions"], # remove raw text after tokenization
    )

    return tokenized