"""
Module for Automatic Speech Recognition(ASR).
This module provides:
    - generating audio from text using gTTS
    - transcribing audio to text using OpenAI Whisper
    - batch pipeline: TTS → audio files → Whisper transcription for N SCAN commands
"""

import whisper
from gtts import gTTS
import os
import re

asr_model = None
# Diacritic characters -> ASCII equivalents for T5 compatibility
_DIACRITIC_MAP = str.maketrans("čćšžđČĆŠŽĐ", "ccszd" "CCSZD")

# Phonetic variants that Whisper frequently produces for Serbian
_SCAN_ALIASES = {
    "ljevo": "lijevo",
    "ljivo": "lijevo",
    "lievo": "lijevo",
    "levo": "lijevo",
    "desno": "desno",
    "skoci": "skoci",
    "hodai": "hodaj",
    "hodati": "hodaj",
    "hoda": "hodaj",
    "hodej": "hodaj",
    "hodei": "hodaj",
    "hode": "hodaj",
    "trcati": "trci",
    "trcite": "trci",
    "trcimo": "trci",
    "trdji": "trci",
    "drci": "trci",
    "skociti": "skoci",
    "skolci": "skoci",
    "hodati": "hodaj",
    "trcati": "trci",
    "gledati": "gledaj",
    "gledajte": "gledaj",
    "gleda": "gledaj",
    "gledaji": "gledaj",
    "okrenite": "okreni",
    "okrenuti": "okreni",
}

_BIGRAM_FIXES: dict = {
    ("dva", "puta"): "dva puta",
    ("tri", "puta"): "tri puta",
    ("2", "puta"): "dva puta",
    ("3", "puta"): "tri puta",
    ("okreni", "se"): "okreni se",
    ("li", "jevo"): "lijevo",
    ("de", "sno"): "desno",
    ("hoda", "i"): "hodaj",
    ("hodaj", "i"): "hodaj i",
    ("hode", "i"): "hodaj",
    ("gleda", "i"): "gledaj",
    ("oko", "lo"): "okolo",
}

# Mapping: SCAN Serbian token -> natural Serbian with diacritics (for gTTS)
_SCAN_TO_NATURAL = {
    "skoci": "skoči",
    "trci": "trči",
    "gledaj": "gledaj",
    "hodaj": "hodaj",
    "lijevo": "lijevo",
    "desno": "desno",
    "okolo": "okolo",
    "suprotno": "suprotno",
    "dva puta": "dva puta",
    "tri puta": "tri puta",
    "okreni se": "okreni se",
    "i": "i",
    "nakon": "nakon",
}


def add_diacritics(command):
    """
    Converts a SCAN Serbian command to natural Serbian with diacritics.
    Used before gTTS so that the audio sounds natural.

    Parameters:
    command: str
        SCAN Serbian command without diacritics, e.g. "skoci lijevo dva puta"

    Returns:
    natural: str
        Command with diacritics, e.g. "skoči lijevo dva puta"
    """
    tokens = command.strip().split()
    result = []
    for tok in tokens:
        result.append(_SCAN_TO_NATURAL.get(tok, tok))
    return " ".join(result)


def text_to_speech(text, filepath, language="sr"):
    """
    Converts text to an audio file using gTTS.

    Parameters:
    text: str
        Text to synthesize
    filepath: str
        Path where the .mp3 file will be saved
    language: str
        Language code

    Returns:
    filepath: str
        Path to the saved audio file
    """
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save(filepath)
    return filepath


def normalize_transcript(text):
    """
    Normalizes ASR output for T5 compatibility:
      1. Removes punctuation
      2. Converts to lowercase
      3. Strips diacritics (T5 tokenizes č,ć,š,đ,ž poorly)
      4. Applies phonetic variant corrections
      5. Stitch Whisper-split multi-word SCAN tokens ("dva puta", "tri puta",
       "okreni se") with a greedy left-to-right bigram pass.

    Parameters:
    text: str
        Raw Whisper transcript

    Returns:
    normalized: str
        Cleaned text ready for T5
    """
    text = re.sub(r"[^\w\s]", "", text).lower().strip()
    text = text.replace("đ", "dj").replace("Đ", "DJ")
    text = text.translate(_DIACRITIC_MAP)

    # Bigram pass
    tokens = text.split()
    stitched = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) in _BIGRAM_FIXES:
            stitched.append(_BIGRAM_FIXES[(tokens[i], tokens[i + 1])])
            i += 2
        else:
            stitched.append(tokens[i])
            i += 1

    # Alias pass
    result = []
    for t in stitched:
        result.append(_SCAN_ALIASES.get(t, t))

    return " ".join(result)


def transcribe(filepath, whisper_model_name="small", language="sr", normalize=True):
    """
    Transcribes an audio file to text using Whisper.
    The model is loaded once and cached in the module-level ``asr_model`` variable so that repeated calls do not reload weights from disk.

    Parameters:
    filepath: str
        Path to the audio file
    whisper_model_name: str
        Whisper model size
    language: str
        Language of the audio
    normalize: bool
        If True and language='sr', strips diacritics and applies phonetic corrections before returning.
        Set to False to observe raw Whisper output and measure T5 degradation on unclean input.

    Returns:
    transcript: str
        Transcribed text (lowercase, stripped)
    """
    global asr_model

    if asr_model is None:
        print(f"Loading Whisper model: {whisper_model_name}")
        asr_model = whisper.load_model(whisper_model_name)

    result = asr_model.transcribe(filepath, language=language)
    raw = result["text"].strip().lower()

    if language == "sr" and normalize:
        return normalize_transcript(raw)
    return raw


def generate_audio_files(commands, audio_dir, language="sr", prefix="cmd"):
    """
    Generates one .mp3 file per command using gTTS and saves them to
    *audio_dir*.

    Parameters
    commands: list[str]
        List of text commands to synthesise (SCAN commands in
        Serbian, e.g. "skoci lijevo dva puta i hodaj desno").
    audio_dir : str
        Directory where .mp3 files will be saved
    language : str
        gTTS language code
    prefix : str
        Filename prefix.  Files are named ``<prefix>_0000.mp3``.

    Returns
    audio_paths : list[str]
        Ordered list of paths to the generated .mp3 files.
    """
    os.makedirs(audio_dir, exist_ok=True)
    audio_paths = []

    for idx, cmd in enumerate(commands):
        filename = f"{prefix}_{idx:04d}.mp3"
        filepath = os.path.join(audio_dir, filename)
        natural_cmd = add_diacritics(cmd)
        text_to_speech(natural_cmd, filepath, language=language)
        audio_paths.append(filepath)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(commands):
            print(f"  Generated {idx + 1}/{len(commands)} audio files ...")

    print(f"All {len(commands)} audio files saved to: {audio_dir}")
    return audio_paths


def transcribe_batch(
    audio_paths, whisper_model_name="small", language="sr", normalize=True
):
    """
    Transcribes a list of audio files using Whisper.

    The Whisper model is loaded once and reused for every file.

    Parameters
    audio_paths : list[str]
        Ordered list of paths to .mp3 files.
    whisper_model_name : str
        Whisper model size
    language : str
        Language hint for Whisper
    normalize: bool
        If True and language='sr', strips diacritics and applies phonetic corrections before returning.
        Set to False to observe raw Whisper output and measure T5 degradation on unclean input.


    Returns
    transcripts : list[str]
        Transcribed text for each audio file (lowercase, stripped).
    """
    global asr_model

    if asr_model is None:
        print(f"Loading Whisper model: {whisper_model_name}")
        asr_model = whisper.load_model(whisper_model_name)

    transcripts = []
    n = len(audio_paths)

    for idx, path in enumerate(audio_paths):
        result = asr_model.transcribe(path, language=language)
        raw = result["text"].strip().lower()
        if language == "sr" and normalize:
            transcript = normalize_transcript(raw)
        else:
            transcript = raw
        transcripts.append(transcript)

        if (idx + 1) % 10 == 0 or (idx + 1) == n:
            print(f"  Transcribed {idx + 1}/{n} files ...")

    return transcripts


def run_asr_pipeline(
    commands,
    audio_dir="audio_commands",
    whisper_model_name="small",
    tts_language="sr",
    asr_language="sr",
    prefix="cmd",
):
    """
    Full TTS -> Whisper pipeline for a list of SCAN commands.

    Steps
    1. Generate one .mp3 per command with gTTS (``tts_language``).
    2. Transcribe every .mp3 with Whisper (``asr_language``).
    3. Return structured results.

    Parameters
    commands : list[str]
        SCAN commands (in Serbian or English) to process
    audio_dir : str
        Directory for .mp3 output files
    whisper_model_name : str
        Whisper model size
    tts_language : str
        gTTS language code for synthesis
    asr_language : str
        Whisper language for transcription
    prefix : str
        Filename prefix for audio files

    Returns
    results : list[dict]
        Each dict has keys:
        - "command"    -- original text command
        - "audio_path" -- path to the generated .mp3 file
        - "transcript" -- Whisper transcription (lowercase)
        - "match"      -- True if transcript == command (exact, lowercased)
    """
    print(f"\n{'='*55}")
    print(f"ASR Pipeline -- {len(commands)} commands")
    print(f"  TTS language   : {tts_language}")
    print(f"  Whisper model  : {whisper_model_name}  |  language: {asr_language}")
    print(f"  Audio folder   : {audio_dir}")
    print(f"{'='*55}\n")

    # 1. TTS
    print("1. Generating audio files with gTTS ...")
    audio_paths = generate_audio_files(
        commands, audio_dir, language=tts_language, prefix=prefix
    )

    # 2. ASR
    print("\n2. Transcribing with Whisper ...")
    transcripts = transcribe_batch(
        audio_paths, whisper_model_name=whisper_model_name, language=asr_language
    )

    # 3. Assemble results
    results = []
    for cmd, path, tr in zip(commands, audio_paths, transcripts):
        results.append(
            {
                "command": cmd,
                "audio_path": path,
                "transcript": tr,
                "match": tr == cmd.lower().strip(),
            }
        )

    n_match = 0
    for r in results:
        if r["match"]:
            n_match += 1
    print(
        f"\nDone. Exact-match transcription accuracy: {n_match}/{len(results)} "
        f"({n_match / len(results):.1%})"
    )

    return results
