# NLP Robot Command Parser

This repository contains an end-to-end system for converting spoken or textual commands into structured action sequences for robot control.
The project combines:
- **Automatic Speech Recognition (ASR)** using Whisper
- **Semantic parsing** using a Transformer-based seq2seq models: T5-small, T5-base and mBART-large-50.

The system maps natural language commands into executable action sequences, with a focus on compositional generalization and robustness to different input forms.

---
## Environment Notes

- Developed and tested using **Python 3.9+**
- Experiments are conducted in **Jupyter Notebook / Google Colab**
- GPU (CUDA) is recommended for training, but not required
- Whisper requires **ffmpeg** for audio processing
- It is recommended to use a virtual environment (venv or conda) when running locally

---

## Datasets

- **SCAN** - [Simplified versions of the CommAI Navigation tasks](https://github.com/brendenlake/SCAN)

Synthetic dataset of command-actions pairs used for training and primary evaluation. Commands are additionally translated into Serbian using rule-based word-by-word mapping.

- **HuRIC** - [Human Robot Interaction Corpus](https://github.com/crux82/huric)

Real-world human-robot interaction dataset used for additional evaluation and generalization testing. For this project, a subset(~18 samples) is manually selected, analyzed and adapted.

- **Live voice demo (own recordings)**

A small, manually recorded set of 7 commands, spoken and recorded by hand rather than synthesized, used to sanity-check the full pipeline (Audio → Whisper → model) against real human speech instead of gTTS audio.
  - Recordings: `data/audio/my_voice_demo/*.mp3`
  - Expected commands + actions: `data/my_voice_commands.json` (same format as the HuRIC json — English action tokens, translated to Serbian at evaluation time for t5-small/t5-base/mBART, since all three were trained with `cfg['data']['lang'] == 'sr'`)
  - Run from the "Live voice demo" section added to each of the three model notebooks. Results are saved to `results/<model>/evaluation_voice_demo.json` and visualized in `comparision.ipynb`.

---

## Requirements

### Local
```bash
pip install -r requirements.txt
```

### Google Colab

The first few cells in each notebook handle the setup automatically:

```python
!git clone https://github.com/PetraMicanovic/nlp-robot-command-parser.git
%cd nlp-robot-command-parser
```
```python
!git clone https://github.com/brendenlake/SCAN.git data/scan
```
```python
!pip install -q -r requirements.txt
```

Just run these cells in order at the start of each notebook before anything else.

---

## Structure 
```bash
.
├── data/
│   ├── scan/
│   ├── audio/
│   │   └── my_voice_demo/
│   ├── my_voice_commands.json
│   └── sr_huric_scan_generalization_subset_18.json
│
├── notebooks/
│   ├── t5_small.ipynb
│   ├── t5_base.ipynb
│   ├── mbart.ipynb
│   └── comparision.ipynb
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── preprocess.py
│   │   └── translate_scan.py
│   │
│   ├── models/
│   │   ├── t5_model.py
│   │   ├── mbart_model.py
│   │   └── asr.py
│   │
│   ├── training/
│   │   └── trainer.py
│   │
│   └── evaluation/
│   │   ├── evaluation.py
│   │   └── save_results.py
│   │
│   └── pipeline.py
│
├── results/
│
├── .gitignore
├── config.json
├── LICENSE
├── README.md
└── requirements.txt
```
---
## Notes
- `t5_small.ipynb`, `t5_base.ipynb` and `mbart.ipynb` must all be run before `comparision.ipynb`, as it depends on their saved results.
- The system operates in **Serbian** — SCAN commands are translated rule-by-rule from English and Whisper is configured with `language="sr"`
- SCAN translations are implemented via simple rule-based mappings (no multilingual models)
- Whisper output is normalized before being passed to T5 (diacritics stripped, some phonetic aliases fixed) because T5 was trained on ASCII-only tokens
- HuRIC dataset is not used for training, only for evaluation
- mBART-50 has no dedicated language code for Serbian. Croatian (hr_HR) is used as the source-language tag for tokenization instead, since Serbian and Croatian are mutually intelligible standard languages and Croatian is the closest code available among mBART-50's 50 supported languages. T5-small and T5-base are unaffected, as their tokenizers have no concept of a source-language tag.
- Each of `t5_small.ipynb`, `t5_base.ipynb` and `mbart.ipynb` includes a "Live voice demo" section that evaluates the trained model on the 7 hand-recorded commands, both with and without transcript normalization, mirroring the main 100-sample pipeline evaluation.

---
## License
MIT
---
