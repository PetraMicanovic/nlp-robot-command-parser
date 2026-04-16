# NLP Robot Command Parser

This repository contains an end-to-end system for converting spoken or textual commands into structured action sequences for robot control. 
The project combines:
- **Automatic Speech Recognition (ASR)** using Whisper
- **Semantic parsing** using a Transformer-based seq2seq model T5

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

- **SCAN** [Simplified versions of the CommAI Navigation tasks](https://github.com/brendenlake/SCAN)

Synthetic dataset of command-actions pairs used for training and primary evaluation. Additionally translated into Serbian using rule-based mapping of commands.
- **HuRIC** [Human Robot Interaction Corpus](https://github.com/crux82/huric)

Real-world human-robot interaction dataset used for additional evaluation and generalization testing. For this project, a subset(~25-50 samples) is manually selected, analyzed and adapted.

---
## Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
```
---
## Structure 
```bash
.
├── data/
│   ├── scan/
│   └── huric/
│
├── notebooks/
│   └── main.ipynb
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── preprocess.py
│   │   └── translate_scan.py
│   │
│   ├── models/
│   │   ├── t5_model.py
│   │   └── asr.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   └── pipeline.py
│   │
│   └── evaluation/
│       └── evaluation.py
│
├── outputs/
│   ├── predictions/
│   └── models/
│
├── .gitignore
├── config.json
├── LICENSE
├── README.md
└── requirements.txt
```
---
## Notes
- The system uses constrained decoding to ensure valid action sequences
- SCAN translations are implemented via simple rule-based mappings (no multilingual models)
- HuRIC dataset is not used for training, only for evaluation
- Audio inputs are processed through Whisper before being passed to the seq2seq model

---
## License
MIT
---
