"""
Module for Automatic Speech Recognition(ASR).
This module provides:
    - generating audio from text using gTTS
    - transcribing audio to text using OpenAI Whisper
"""

import whisper
from gtts import gTTS

asr_model = None

def text_to_speech(text, filepath, language = "en"):
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
        path to the saved audio file
    """
    tts = gTTS(text = text, lang = language, slow = False )
    tts.save(filepath)
    return filepath

def transcribe(filepath, whisper_model_name = "base",language = "en"):
    """
    Transcribes an audio file to text using Whisper.

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

    result = asr_model.transcribe(filepath, language = language)
    return result['text'].strip().lower()