"""
Module for the mBART encoder-decoder semantic parsing model.

This module provides:
    - loading a pretrained mBART model and tokenizer
    - predicting action sequences from text commands
    - support for both standard and constrained decoding

mBART (multilingual BART) is pretrained on 50 languages including Croatian (hr_HR),
which is used as the source language since the input data is in Latin script and
mBART-50 does not include Serbian Latin directly.
"""

import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

MBART_SRC_LANG = "hr_HR"
MBART_TGT_LANG = "en_XX"


def load_mbart_model(model_name, device):
    """
    Loads a pretrained mBART model and moves it to the target device.
    Parameters:
    model_name: str
        HuggingFace model name (e.g. "facebook/mbart-large-50")
    device: str
        Target device ("cuda" or "cpu")

    Returns:
    model: MBartForConditionalGeneration
    tokenizer: MBart50TokenizerFast
    """
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    tokenizer.src_lang = MBART_SRC_LANG

    model = MBartForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)

    n_params = 0
    for p in model.parameters():
        if p.requires_grad:
            n_params += p.numel()
    print(f"mBART model loaded: {model_name}")
    print(f"Trainable parameters: {n_params:,}")

    return model, tokenizer


def predict_mbart(
    command,
    model,
    tokenizer,
    prefix,
    max_input_len,
    max_target_len,
    device,
    num_beams,
    bad_words_ids=None,
):
    """
    Generates an action sequence for a given text command using mBART.

    Parameters:
    command: str
        Natural language command (without task prefix)
    model: MBartForConditionalGeneration
    tokenizer: MBart50TokenizerFast
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
    bad_words_ids: list or None
        List of token id sequences that are banned during generation (for constrained decoding). Must use plural form — as required by HuggingFace generate().

    Returns:
    actions: str
        Predicted action sequence string
    """
    input_text = prefix + command

    encoded = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=max_input_len,
        truncation=True,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # mBART requires forced_bos_token_id to set the target language
    forced_bos_token_id = tokenizer.lang_code_to_id[MBART_TGT_LANG]

    generate_kwargs = {
        "max_new_tokens": max_target_len,
        "num_beams": num_beams,
        "forced_bos_token_id": forced_bos_token_id,
        "attention_mask": attention_mask,
    }

    if bad_words_ids is not None:
        generate_kwargs["bad_words_ids"] = bad_words_ids

    model.eval()
    with torch.no_grad():
        outputs = model.generate(input_ids, **generate_kwargs)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result
