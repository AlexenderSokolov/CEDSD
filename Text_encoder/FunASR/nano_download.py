import argparse
import os
from pathlib import Path

from modelscope import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download optional FunASR Nano and VAD model assets.")
    parser.add_argument(
        "--nano-dir",
        default=os.environ.get(
            "FAPI_FUNASR_NANO_DIR",
            str(Path(__file__).resolve().parent / "local_models" / "FunAsr-Nano"),
        ),
        help="Local directory for Fun-ASR-Nano-2512.",
    )
    parser.add_argument(
        "--vad-dir",
        default=os.environ.get(
            "FAPI_FUNASR_VAD_DIR",
            str(Path(__file__).resolve().parent / "local_models" / "fsmn_vad"),
        ),
        help="Local directory for the FSMN VAD model.",
    )
    args = parser.parse_args()

    print(f"Downloading Fun-ASR-Nano-2512 to {args.nano_dir}...")
    snapshot_download("FunAudioLLM/Fun-ASR-Nano-2512", local_dir=args.nano_dir)

    print(f"Downloading FSMN VAD model to {args.vad_dir}...")
    snapshot_download("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", local_dir=args.vad_dir)

    print("Model downloads completed.")


if __name__ == "__main__":
    main()
