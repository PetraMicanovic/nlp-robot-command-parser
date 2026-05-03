"""
End-to-end pipeline: audio -> text -> action sequence.
"""

from src.models.asr import text_to_speech, transcribe
from src.models.t5_model import predict
from src.data.translate_scan import build_bad_word_ids


class RobotCommandPipeline:
    """
    Integrated pipeline connecting ASR and semantic parsing.

    Flow:
        Text -> Audio (gTTS) -> Text (Whisper) -> Actions (T5)

    Parameters:
    model: T5ForConditionalGeneration
    tokenizer: T5Tokenizer
    cfg: dict
        Full config loaded from config.json
    device: str
    """

    def __init__(self, model, tokenizer, cfg, device):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device
        self.model_cfg = cfg["model"]
        self.asr_cfg = cfg["asr"]
        self.bad_word_ids = build_bad_word_ids(tokenizer, cfg["valid_actions"])

    def run(self, command_text, constrained=True):
        """
        Runs the full pipeline for a given text command.

        Steps:
            1. Text -> Audio (gTTS)
            2. Audio -> Text (Whisper)
            3. Text -> Action sequence (T5)

        Parameters:
        command_text: str
            Input command in Serbian
        constrained: bool
            If True applies constrained decoding

        Returns:
        result: dict
            Keys: original_command, asr_transcript, predicted_actions, constrained
        """
        audio_path = text_to_speech(
            command_text,
            filepath=self.asr_cfg["audio_path"],
            language=self.asr_cfg["tts_lang"],
        )

        transcript = transcribe(
            audio_path,
            whisper_model_name=self.asr_cfg["whisper_model"],
            language=self.asr_cfg["language"],
        )

        actions = predict(
            transcript,
            model=self.model,
            tokenizer=self.tokenizer,
            prefix=self.model_cfg["prefix"],
            max_input_len=self.model_cfg["max_input_len"],
            max_target_len=self.model_cfg["max_target_len"],
            device=self.device,
            num_beams=self.model_cfg["num_beams"],
            bad_word_ids=self.bad_word_ids if constrained else None,
        )

        return {
            "original_command": command_text,
            "asr_transcript": transcript,
            "predicted_actions": actions,
            "constrained": constrained,
        }

    def run_from_text(self, command_text, constrained=True):
        """
        Parses a command directly from text, skipping the ASR step.
        Useful for testing the semantic parser in isolation.

        Parameters:
        command_text: str
            Input command in Serbian
        constrained: bool
            If True, applies constrained decoding

        Returns:
        result: dict
            Keys: original_command, asr_transcript (None), predicted_actions, constrained
        """
        actions = predict(
            command_text,
            model=self.model,
            tokenizer=self.tokenizer,
            prefix=self.model_cfg["prefix"],
            max_input_len=self.model_cfg["max_input_len"],
            max_target_len=self.model_cfg["max_target_len"],
            device=self.device,
            num_beams=self.model_cfg["num_beams"],
            bad_word_ids=self.bad_word_ids if constrained else None,
        )

        return {
            "original_command": command_text,
            "asr_transcript": None,
            "predicted_actions": actions,
            "constrained": constrained,
        }

    def run_from_audio(self, audio_path, gold_actions=None, normalize=True):
        """
        Runs the pipeline from an existing audio file, skipping the TTS step.
        Used for end-to-end evaluation on pre-generated audio files.

        Steps:
            1. Audio -> Text (Whisper)
            2. Text -> Action sequence (T5)

        Parameters:
        audio_path: str
            Path to an existing .mp3 audio file
        gold_actions: str or None
            Expected action sequence for accuracy computation
        normalize: bool
            If True, applies transcript normalization before T5

        Returns:
        result: dict
            Keys: audio_path, asr_transcript, predicted_actions, normalize,
                  gold_actions (if provided), correct (if provided)
        """
        transcript = transcribe(
            audio_path,
            whisper_model_name=self.asr_cfg["whisper_model"],
            language=self.asr_cfg["language"],
            normalize=normalize,
        )

        actions = predict(
            transcript,
            model=self.model,
            tokenizer=self.tokenizer,
            prefix=self.model_cfg["prefix"],
            max_input_len=self.model_cfg["max_input_len"],
            max_target_len=self.model_cfg["max_target_len"],
            device=self.device,
            num_beams=self.model_cfg["num_beams"],
        )

        result = {
            "audio_path": audio_path,
            "asr_transcript": transcript,
            "predicted_actions": actions,
            "normalize": normalize,
        }

        if gold_actions is not None:
            result["gold_actions"] = gold_actions
            result["correct"] = actions.strip() == gold_actions.strip()

        return result

    @staticmethod
    def print_result(result):
        """
        Prints a formatted pipeline result.

        Parameters:
        result: dict
            Output of run() or run_from_text()
        """
        mode = "constrained" if result["constrained"] else "unconstrained"
        print(f"Command: {result['original_command']}")
        if result["asr_transcript"] is not None:
            print(f"Transcript: {result['asr_transcript']}")
        print(f"Actions ({mode}): {result['predicted_actions']}")
        print()
