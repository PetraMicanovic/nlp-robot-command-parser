"""
Module for SCAN action vocabulary, constrained decoding support and English -> Serbian translation of SCAN commands and actions.

This module provides:
    - the canonical SCAN action vocabulary (EN and SR)
    - word-by-word translation of SCAN commands (EN -> SR)
    - token-by-token translation of SCAN action sequences (EN -> SR)
    - mapping actions to token IDs for constrained decoding
    - utility for printing dataset statistics
"""

import numpy as np

# Action vocabularies

# English action tokens (original SCAN output vocabulary)
SCAN_ACTIONS_EN = [
    "I_WALK",
    "I_RUN",
    "I_JUMP",
    "I_LOOK",
    "I_TURN_LEFT",
    "I_TURN_RIGHT",
]

# Serbian action tokens (used when lang="sr")
# These are the Serbian equivalents used as output tokens.
SCAN_ACTIONS_SR = [
    "I_HODAJ",
    "I_TRCI",
    "I_SKOCI",
    "I_GLEDAJ",
    "I_OKRENI_LIJEVO",
    "I_OKRENI_DESNO",
]

# Mapping: English action token → Serbian action token
ACTION_EN_TO_SR = {
    "I_WALK":       "I_HODAJ",
    "I_RUN":        "I_TRCI",
    "I_JUMP":       "I_SKOCI",
    "I_LOOK":       "I_GLEDAJ",
    "I_TURN_LEFT":  "I_OKRENI_LIJEVO",
    "I_TURN_RIGHT": "I_OKRENI_DESNO",
}

# Command word translation (EN → SR)
# Complete closed vocabulary — all 13 SCAN command words.
COMMAND_EN_TO_SR = {
    # primitives
    "walk":     "hodaj",
    "run":      "trci",
    "jump":     "skoci",
    "look":     "gledaj",
    # direction primitives
    "turn":     "okreni se",
    "left":     "lijevo",
    "right":    "desno",
    # modifiers that expand into repeated tokens in the output
    "around":   "okolo",      # expands to 4x (TURN+ACTION) in output
    "opposite": "suprotno",   # expands to 2x TURN in output
    # repetition and sequence connectors
    "twice":    "dva puta",
    "thrice":   "tri puta",
    "and":      "i",
    "after":    "nakon",
}

# Translation functions

def translate_command(command, lang = "en"):
    """
    Translates a SCAN command string from English to Serbian (or returns
    it unchanged for lang="en").

    Translation is word-by-word using COMMAND_EN_TO_SR.
    Multi-word Serbian equivalents (e.g. "twice" → "dva puta") are
    inserted as-is into the output string.

    Parameters:
    command: str
        Original English SCAN command
    lang: str
        Target language: "en" or "sr" 

    Returns:
    translated: str
        Translated command string
    """
    if lang == "en":
        return command
    tokens = command.strip().split()
    translated = []
    for tok in tokens:
        if tok in COMMAND_EN_TO_SR:
            translated.append(COMMAND_EN_TO_SR[tok])
        else:
            translated.append(tok)

    return " ".join(translated)


def translate_actions(actions, lang = "en"):
    """
    Translates a SCAN action sequence from English to Serbian
    action tokens (or returns it unchanged for lang="en").

    Translation is token-by-token using ACTION_EN_TO_SR.

    Parameters:
    actions: str
        Original English action sequence
    lang: str
        Target language: "en" or "sr"

    Returns:
    translated: str
        Action sequence with Serbian tokens, e.g. "OKRENI_LIJEVO SKOCI OKRENI_LIJEVO SKOCI"
    """
    if lang == "en":
        return actions
    tokens = actions.strip().split()
    translated = []
    for tok in tokens:
        if tok in ACTION_EN_TO_SR:
            translated.append(ACTION_EN_TO_SR[tok])
        else:
            translated.append(tok)
    return " ".join(translated)


def translate_dataset(data, lang = "sr"):
    """
    Translates an entire dataset split (commands and actions) to Serbian.

    Parameters:
    data: list of dict
        Each element has {"commands": str, "actions": str}
    lang: str
        Target language: "en" (no-op) or "sr"

    Returns:
    translated_data: list of dict
        Each element has {"commands": str, "actions": str} in the target language
    """
    if lang == "en":
        return data
    
    translated_data = []

    for ex in data:
        translated_example =  {
                "commands": translate_command(ex["commands"], lang=lang),
                "actions":  translate_actions(ex["actions"],  lang=lang),
            }
        translated_data.append(translated_example)
    return translated_data

# Constrained decoding helpers

def get_valid_actions(lang = "en"):
    """
    Returns the list of valid action tokens for the given language.

    Parameters:
    lang: str
        "en" or "sr"

    Returns:
    actions: list of str
    """
    if lang == "sr":
        return SCAN_ACTIONS_SR  
    
    return SCAN_ACTIONS_EN


def build_bad_word_ids(tokenizer, valid_actions):
    """
    Builds the list of forbidden token IDs for constrained decoding.
    All tokens that are NOT part of valid actions, whitespace, or EOS
    are forbidden during generation.

    Parameters:
    tokenizer: T5Tokenizer
    valid_actions: list of str
        List of allowed action strings

    Returns:
    bad_word_ids: list of list of int
        Format expected by model.generate(bad_words_ids=...)
    """
    allowed = set()
    for action in valid_actions:
        ids = tokenizer.encode(action, add_special_tokens=False)
        allowed.update(ids)

    space_ids = tokenizer.encode(" ", add_special_tokens=False)
    allowed.update(space_ids)
    allowed.add(tokenizer.eos_token_id)
    allowed.add(tokenizer.pad_token_id)

    bad_word_ids = []
    for i in range(tokenizer.vocab_size):
        if i not in allowed:
            bad_word_ids.append([i])
    return bad_word_ids

# Statistics

def get_action_vocab(data):
    """
    Extracts the unique action tokens from a dataset split.

    Parameters:
    data: list of dict
        Each element has {"commands": str, "actions": str}

    Returns:
    vocab: list of str
        Sorted list of unique action tokens found in the data
    """
    vocab = set()
    for ex in data:
        vocab.update(ex["actions"].split())
    return sorted(vocab)


def print_stats(train_data, test_data):
    """
    Prints basic statistics about the dataset splits.

    Parameters:
    train_data: list of dict
    test_data: list of dict
    """
    cmd_lens = [len(ex["commands"].split()) for ex in train_data]
    act_lens = [len(ex["actions"].split())  for ex in train_data]
    vocab    = get_action_vocab(train_data)

    print("Dataset statistics:")
    print(f"  Train examples : {len(train_data)}")
    print(f"  Test  examples : {len(test_data)}")
    print(f"  Commands  – min/max/avg tokens: "
          f"{min(cmd_lens)} / {max(cmd_lens)} / {np.mean(cmd_lens):.1f}")
    print(f"  Actions   – min/max/avg tokens: "
          f"{min(act_lens)} / {max(act_lens)} / {np.mean(act_lens):.1f}")
    print(f"  Action vocab ({len(vocab)}): {vocab}")