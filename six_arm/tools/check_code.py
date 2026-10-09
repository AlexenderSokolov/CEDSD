"""Check the extracted source and run the existing CPU-only verification."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "uica_exec" / "src"


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(SOURCE))
    files = sorted((ROOT / "uica_exec").rglob("*.py")) + sorted((ROOT / "tools").glob("*.py"))
    modules = {p.stem for p in (SOURCE / "uica").glob("*.py")}
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        if path.parent != SOURCE / "uica":
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                if node.module.split(".")[0] not in modules:
                    raise AssertionError(f"Missing internal dependency: {path.name}: {node.module}")
    config = json.loads((ROOT / "uica_exec/configs/six_arm.example.json").read_text(encoding="utf-8"))
    assert config["training"]["seeds"] == [17, 29]
    assert config["full"]["started_unix"] is None
    import torch
    torch.set_num_threads(1)
    from uica.closeout_model import engineering_operator_checks
    report = engineering_operator_checks("cpu")
    assert report["status"] == "passed"
    print(f"Source parse/import closure: {len(files)} files", flush=True)
    print(f"Six-arm engineering checks: {len(report['checks'])} passed", flush=True)
    suite = unittest.defaultTestLoader.discover(str(ROOT / "uica_exec/tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1
    print(json.dumps({"scientific_checks": len(report["checks"]), "unit_tests": result.testsRun,
                      "scope": "CPU engineering and synthetic fixtures; no audio, pretrained model or GPU training"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
