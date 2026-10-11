"""Check source syntax, clean notebooks, provenance, and reference-result coverage."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    scripts = sorted(ROOT.rglob("*.py"))
    notebooks = sorted(ROOT.rglob("*.ipynb"))
    scripts = [p for p in scripts if not any(part in {".venv", "venv", "results"} for part in p.relative_to(ROOT).parts)]
    notebooks = [p for p in notebooks if not any(part in {".venv", "venv", "results"} for part in p.relative_to(ROOT).parts)]
    for path in scripts:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for path in notebooks:
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["nbformat"] == 4, path
        for index, cell in enumerate(document["cells"]):
            if cell["cell_type"] != "code":
                continue
            assert cell.get("outputs", []) == [], (path, index, "output was not cleared")
            assert cell.get("execution_count") is None, (path, index, "execution count was not cleared")
            code = "".join(cell.get("source", []))
            ast.parse(code, filename=f"{path}:cell{index}")
            assert "/home/lanran/" not in code, (path, index, "machine-specific path")
    print(f"Syntax and clean-notebook checks passed: {len(scripts)} scripts, {len(notebooks)} notebooks")

    manifest = json.loads((ROOT / "docs" / "source_manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = ROOT / entry["target"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["prepared_sha256"], path
    print(f"Provenance checks passed: {len(manifest['files'])} files")

    reference = ROOT / "reference_results"
    for suffix in ("", "_student_t_xi"):
        paths = list((reference / "convergence").rglob(f"suboptimal_known_opt_gap_results{suffix}.csv"))
        assert len(paths) == 1, paths
        data = pd.read_csv(paths[0])
        assert len(data) == 1200, paths[0]
        assert set(data["dimension"]) == {5, 25, 100}, paths[0]
        assert set(data["M"]) == {2, 5, 10, 20, 50, 100, 250, 500}, paths[0]
        assert (data.groupby(["dimension", "M"]).size() == 50).all(), paths[0]

    for dimension in (1, 5, 10, 50):
        paths = list((reference / "regression").rglob(f"dim_sweep_d{dimension}.csv"))
        assert len(paths) == 1, paths
        data = pd.read_csv(paths[0])
        assert set(data["n"]) == {10, 20, 50, 75, 100}, paths[0]
        assert (data.groupby("n").size() == 50).all(), paths[0]

    for scale in range(5, 21):
        paths = list((reference / "regression").rglob(f"cont_scale_cs{scale}.csv"))
        assert len(paths) == 1, paths
        data = pd.read_csv(paths[0])
        assert len(data) == 50 and set(data["contamination_scale"]) == {scale}, paths[0]

    for distribution in ("gaussian", "chisquare", "lognormal", "pareto"):
        paths = list((reference / "newsvendor" / distribution).rglob("NV_3D_selected_trials.csv"))
        assert len(paths) == 1, paths
        data = pd.read_csv(paths[0])
        assert set(data["size"]) == {10, 20, 30, 50, 75, 100, 150}, paths[0]
        assert (data.groupby("size").size() == 50).all(), paths[0]
    print("Reference-result coverage checks passed for paper Figures 1–5")


if __name__ == "__main__":
    main()
