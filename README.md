# CEDSD: cross model emotion-driven deepfake speech detection

[中文说明](README.zh-CN.md)

This repository contains the lightweight research code for detecting forged speech with multimodal affective cues. The system combines acoustic features, emotion representations, ASR-derived text semantics, inverse attention, and FAPI-style language-model consistency scoring.

The paper is not yet submitted or published. Keep the repository private until the authors decide to release the code.

## Architecture

- `main.py`: command-line entry point for training, evaluation, prediction, and inverse-attention inspection.
- `config.py`: training, cache, uncertainty weighting, FAPI, and interpretability defaults.
- `Dataset_all.py`: unified multimodal dataset, waveform loading, MFCC/F0 extraction, and cache placeholders.
- `Model_all.py`: multimodal detector wiring spectrum, emotion, text, inverse-attention, IACA, and FAPI components.
- `Audio_united/`: waveform/emotion modules, including MFCASTDA and emotion2vec wrappers.
- `Text_encoder/`: BERT text encoder and FunASR wrappers/download helpers.
- `spectrum/`: MFCC/F0 acoustic encoders.
- `offline_cache.py`: ASR, FAPI, emotion2vec, and acoustic cache helpers.
- `prewarm_*.py`: offline cache prewarm scripts.
- `scripts/README.md`: compatibility command list for helper scripts.

## Installation

Python 3.10 or newer is recommended.

```bash
cd Program_lightweight_for_github
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Heavy optional dependencies include `torch`, `torchaudio`, `funasr`, `modelscope`, `torchcrepe`, `laion-clap`, and Hugging Face `transformers`. Install CUDA-specific PyTorch wheels according to your local GPU/runtime.

## Data Layout

Expected CSV files contain at least:

- `file`: audio filename or relative path under the split root.
- `label`: `bonafide`/`real` for genuine speech or `spoof`/`fake` for forged speech.
- Emotion score columns: `angry`, `disgusted`, `fearful`, `happy`, `neutral`, `other`, `sad`, `surprised`, `unknown`.

Default relative layout:

```text
data/
  train/train.csv
  train/<audio files>
  val/val.csv
  val/<audio files>
  test/test.csv
  test/<audio files>
```

All paths can be overridden with CLI flags or environment variables such as `FAPI_TRAIN_CSV`, `FAPI_TRAIN_ROOT`, `FAPI_MODEL_PATH`, `FAPI_ASR_MODEL_DIR`, and `FAPI_CLAP_CKPT`.

## Commands

Show the main CLI:

```bash
python main.py --help
```

Train:

```bash
python main.py --mode train ^
  --train-csv data/train/train.csv --train-root data/train ^
  --val-csv data/val/val.csv --val-root data/val ^
  --output-dir outputs_fapi_viz
```

Evaluate:

```bash
python main.py --mode test ^
  --test-csv data/test/test.csv --test-root data/test ^
  --model-path outputs_fapi_viz/run_default/best_model.pth
```

Predict one or more files:

```bash
python main.py --mode predict ^
  --model-path outputs_fapi_viz/run_default/best_model.pth ^
  --audio path\to\sample.wav
```

Prewarm caches:

```bash
python prewarm_offline_cache.py --mode all
python prewarm_e2v_cache.py
python prewarm_acoustic_cache.py
```

Download local ASR assets:

```bash
python Text_encoder/FunASR/sense_download.py --help
python Text_encoder/FunASR/nano_download.py --help
```

## Outputs

Training and evaluation outputs are written under `outputs_fapi_viz/` by default. Typical generated files include logs, best checkpoints, convergence summaries, interpretability reports, and cached features. These files are intentionally ignored by `.gitignore`.

## Citation

Citation information will be added after the paper is submitted or accepted.

```bibtex
@misc{fapi_speech_forgery_2026,
  title = {Emotion-Aware Speech Forgery Detection},
  author = {Project Authors},
  year = {2026},
  note = {Citation to be updated after publication}
}
```

## License And Responsible Use

Code is released under the MIT License after the repository is made public. Datasets, pretrained checkpoints, third-party model files, local caches, and generated reports are not included unless separately licensed.

Use this software responsibly. Speech forgery detection can be dual-use, and results should not be used for unsupported claims about individuals or for privacy-invasive monitoring.

## Recreating the environment

Run the following in the project root:

```bash
pip install -r requirements.txt
# (If CLAP functionality is needed)
pip install laion-clap
```

Note: For CUDA-enabled `torch` / `torchaudio` wheels, choose the appropriate prebuilt packages for your CUDA/runtime or follow the official PyTorch installation instructions.
