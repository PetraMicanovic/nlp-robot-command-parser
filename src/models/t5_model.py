"""
Module for the T5 encoder-decoder semantic parsing model.

This module provides:
    - loading a pretrained T5 model
    - predicting action sequences from text commands
    - support for both standard and constrained decoding
"""

import torch
from transformers import T5ForConditionalGeneration


def load_model(model_name, device):
    """
    Loads a pretrained T5 model and moves it to the target device.

    Parameters:
    model_name: str
        HuggingFace model name (e.g. "t5-small", "t5-base")
    device: str
        Target device ("cuda" or "cpu")

    Returns:
    model: T5ForConditionalGeneration
    """
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    n_params = 0
    for p in model.parameters():
      if p.requires_grad:
          n_params += p.numel()

    print(f"Model loaded: {model_name}  |  trainable parameters: {n_params:,}")
    return model


def predict(command,
            model,
            tokenizer,
            prefix,
            max_input_len,
            max_target_len,
            device,
            num_beams,
            bad_word_ids = None):
    """
    Generates an action sequence for a given text command.

    Parameters:
    command: str
        Natural language command (without task prefix)
    model: T5ForConditionalGeneration
    tokenizer: T5Tokenizer
    prefix: str
        Task prefix prepended to the command before tokenization
    max_input_len: int
        Maximum input token length
    max_target_len: int
        Maximum number of new tokens to generate
    device: str
        Device the model is on ("cuda" or "cpu")
    num_beams: int
        Beam size for beam search decoding
    bad_word_ids: list or None
        List of token id sequences that are banned during generation (for constrained decoding).
    Returns:
    actions: str
        Predicted action sequence string
    """
    # Add prefix
    input_text = prefix + command

    # Tokenize input
    encoded = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=max_input_len,
        truncation=True,
    )

    input_ids = encoded["input_ids"].to(device)

    # Prepare generation arguments
    generate_kwargs = {
        "max_new_tokens": max_target_len, 
        "num_beams": num_beams
    }

    if bad_word_ids is not None:
        generate_kwargs["bad_words_ids"] = bad_word_ids

    with torch.no_grad():
        outputs = model.generate(input_ids, **generate_kwargs)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return result