"""Publication entry checks that never start an experiment or load a model."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools/replay.py"
EXAMPLE = ROOT / "uica_exec/configs/six_arm.example.json"


class ReplayEntryTests(unittest.TestCase):
    def test_example_is_rejected_before_any_experiment(self):
        result = subprocess.run(
            [sys.executable, str(RUNNER), "--config", str(EXAMPLE), "--stage", "audit"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Replace all template paths", result.stderr)

    def test_existing_state_clock_cannot_be_bypassed_by_omitting_config_clock(self):
        root = Path(tempfile.mkdtemp(prefix="cedsd-replay-entry-"))
        config = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        config = json.loads(json.dumps(config).replace("/path/to/", root.as_posix() + "/"))
        config["full"]["started_unix"] = 1
        output = Path(config["full"]["output_root"])
        output.mkdir(parents=True)
        state = output / "state.json"
        state.write_text(json.dumps({
            "closeout_clock": {"schema": "existing-clock"},
            "closeout_contract": {"seeds": [17, 29]},
        }), encoding="utf-8")
        private_config = root / "settings.json"
        private_config.write_text(json.dumps(config), encoding="utf-8")
        before = state.read_bytes()
        result = subprocess.run(
            [sys.executable, str(RUNNER), "--config", str(private_config),
             "--stage", "train", "--arm", "acoustic", "--seed", "17"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("only its original Log29 resume", result.stderr)
        self.assertEqual(state.read_bytes(), before)
        print("RETAINED_REPLAY_ENTRY_FIXTURE", root)


if __name__ == "__main__":
    unittest.main()
