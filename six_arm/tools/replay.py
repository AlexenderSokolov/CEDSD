"""Local entry to the preserved experiment functions; private artifacts required.

This replaces the desktop/native-run adapter, not the scientific implementation.
No stored clocks, experiment states, data, or weights are created by --help.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "uica_exec/src"))

ARMS = ("acoustic", "ae", "pooljoint", "ca", "linear", "log")
STAGES = ("audit", "prepare", "engineering", "train", "freeze", "evaluate",
          "statistics", "donor", "components", "analyze", "summarize")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Private configuration, not the example template")
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=(17, 29))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if "/path/to/" in json.dumps(config):
        parser.error("Replace all template paths with authorized local artifacts.")
    started = config.get("full", {}).get("started_unix")
    if not isinstance(started, (int, float)) or isinstance(started, bool) or not math.isfinite(started) or started <= 0:
        parser.error("full.started_unix must be the fixed T0 of the intended experiment; never reset a resumed run.")
    if config["full"].get("output_storage_copy"):
        parser.error("Server storage migration is not included; resolve output_root and feature_store explicitly.")
    if args.stage == "train" and (args.arm is None or args.seed is None):
        parser.error("train requires --arm and --seed.")
    if args.stage != "train" and (args.arm is not None or args.seed is not None or args.resume):
        parser.error("--arm, --seed and --resume apply only to train.")
    final_stage = args.stage in {"freeze", "evaluate", "statistics", "donor", "components", "analyze", "summarize"}
    state_path = Path(config["full"]["output_root"]) / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    # Existing run state owns its clock, even if a relocated config omits it.
    has_closeout = bool(state.get("closeout_clock") or state.get("closeout_contract")
                        or config["full"].get("closeout_clock"))
    if final_stage or has_closeout:
        if not state.get("closeout_clock") or not state.get("closeout_contract"):
            parser.error("state.json must contain the frozen closeout clock and contract; the example does not supply them.")
        configured_clock = config["full"].get("closeout_clock")
        if configured_clock and configured_clock != state["closeout_clock"]:
            parser.error("Configuration closeout clock differs from the existing run state.")
    if has_closeout and not final_stage:
        if args.stage != "train" or (args.arm, args.seed, args.resume) != ("log", 29, True):
            parser.error("The historical closeout permits only its original Log29 resume or final stages.")
    if final_stage or has_closeout:
        # Validate before Budget creates or updates any on-disk state.
        from types import SimpleNamespace
        from uica.coling_evaluation import _clock_contract
        _clock_contract(config, SimpleNamespace(state=state))
    config["task"] = {"stage": args.stage, "scale": "full", "resume": args.resume}
    if args.arm is not None:
        config["task"].update(arm=args.arm, seed=args.seed)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config["full"].get("physical_gpu", 0))
    from uica.full_runtime import Budget, execution_lock
    with execution_lock(config):
        budget = Budget(config)
        if final_stage or has_closeout:
            from uica.coling_evaluation import _clock_contract
            _, _, deadlines = _clock_contract(config, budget)
            inference_stage = args.stage in {"train", "freeze", "evaluate", "donor", "components"}
            budget.state["deadlines"] = {
                "train_stop": deadlines["train_stop"],
                "delivery": min(deadlines["delivery"], deadlines["results_lock"]) if inference_stage else deadlines["delivery"],
            }
            budget.save()
        budget.guard(training=not final_stage)
        if args.stage == "audit":
            from uica.full_data import audit
            audit(config, budget)
        elif args.stage == "prepare":
            from uica.full_prepare import prepare
            with budget.gpu("full_cache_prepare"):
                prepare(config, budget)
        elif args.stage == "engineering":
            from uica.full_runtime import engineering
            engineering(config, budget)
        elif args.stage == "train":
            from uica.full_training import train
            train(config, budget, args.arm, args.seed, "full")
        elif args.stage == "freeze":
            from uica import coling_diagnostics, coling_evaluation
            coling_diagnostics.prepare(config, budget)
            budget.state["closeout_contract"]["diagnostic_manifest_path"] = "coling_diagnostic_manifest.json"
            budget.save()
            coling_evaluation.freeze(config, budget)
        elif args.stage in {"evaluate", "statistics", "summarize"}:
            from uica import coling_evaluation
            getattr(coling_evaluation, args.stage)(config, budget)
        elif args.stage in {"donor", "components"}:
            from uica import coling_diagnostics
            with budget.gpu("coling_" + args.stage):
                getattr(coling_diagnostics, args.stage)(config, budget)
        elif args.stage == "analyze":
            from uica.coling_diagnostics import saved_prediction_analysis
            saved_prediction_analysis(config, budget)


if __name__ == "__main__":
    main()
