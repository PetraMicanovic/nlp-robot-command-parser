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

asr_model = None


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


def transcribe(filepath, whisper_model_name="small", language="sr"):
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

    Returns:
    transcript: str
        Transcribed text (lowercase, stripped)
    """
    global asr_model

    if asr_model is None:
        print(f"Loading Whisper model: {whisper_model_name}")
        asr_model = whisper.load_model(whisper_model_name)

    result = asr_model.transcribe(filepath, language=language)
    return result["text"].strip().lower()


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
        text_to_speech(cmd, filepath, language=language)
        audio_paths.append(filepath)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(commands):
            print(f"  Generated {idx + 1}/{len(commands)} audio files ...")

    print(f"All {len(commands)} audio files saved to: {audio_dir}")
    return audio_paths


def transcribe_batch(audio_paths, whisper_model_name="small", language="sr"):
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
        transcripts.append(result["text"].strip().lower())

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

    # Step 1 -- TTS
    print("Step 1 / 2 -- Generating audio files with gTTS ...")
    audio_paths = generate_audio_files(
        commands, audio_dir, language=tts_language, prefix=prefix
    )

    # Step 2 -- ASR
    print("\nStep 2 / 2 -- Transcribing with Whisper ...")
    transcripts = transcribe_batch(
        audio_paths, whisper_model_name=whisper_model_name, language=asr_language
    )

    # Step 3 -- Assemble results
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

    n_match = sum(r["match"] for r in results)
    print(
        f"\nDone. Exact-match transcription accuracy: {n_match}/{len(results)} "
        f"({n_match / len(results):.1%})"
    )

    return results
