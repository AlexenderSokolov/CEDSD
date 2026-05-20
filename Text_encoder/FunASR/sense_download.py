import argparse
import os
from pathlib import Path

from modelscope import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download optional SenseVoiceSmall and VAD model assets.")
    parser.add_argument(
        "--sense-dir",
        default=os.environ.get(
            "FAPI_ASR_MODEL_DIR",
            str(Path(__file__).resolve().parent / "local_models" / "SenseVoiceSmall"),
        ),
        help="Local directory for SenseVoiceSmall.",
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

    print(f"Downloading SenseVoiceSmall to {args.sense_dir}...")
    snapshot_download("iic/SenseVoiceSmall", local_dir=args.sense_dir)

    print(f"Downloading FSMN VAD model to {args.vad_dir}...")
    snapshot_download("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", local_dir=args.vad_dir)

    print("Model downloads completed.")


if __name__ == "__main__":
    main()
