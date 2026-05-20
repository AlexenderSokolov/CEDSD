# Script Entry Points

The project keeps the original helper scripts at the repository root for backward compatibility. Use these commands from `Program_lightweight_for_github/`:

| Task | Command |
| --- | --- |
| Main train/test/predict CLI | `python main.py --help` |
| Prewarm all offline caches | `python prewarm_offline_cache.py --help` |
| Prewarm emotion2vec cache | `python prewarm_e2v_cache.py --help` |
| Prewarm acoustic cache | `python prewarm_acoustic_cache.py --help` |
| Inspect feature cache coverage | `python check_cache.py --help` |
| Inspect audio cache coverage | `python check_audio_cache.py --help` |
| Add emotion2vec score columns | `python e2v_labels.py --help` |
| Download SenseVoiceSmall assets | `python Text_encoder/FunASR/sense_download.py --help` |
| Download FunASR Nano assets | `python Text_encoder/FunASR/nano_download.py --help` |

Future releases may move wrappers into this directory, but the current cleanup preserves existing paths to avoid breaking collaborators' commands.
