"""Build one harness tree per classical config: OUT/trees/<config>/{chain.json, modules/}.
  python make_trees.py OUT"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
out = Path(sys.argv[1]) / "trees"
cfg = json.loads((HERE / "configs.json").read_text())
chains = {"empty": []}
for fam in ("smoothers", "align2d", "ensemble"):
    chains.update(cfg[fam])
for name, mods in chains.items():
    t = out / name
    if t.exists():
        shutil.rmtree(t)
    shutil.copytree(HERE / "modules", t / "modules", ignore=shutil.ignore_patterns("__pycache__"))
    (t / "chain.json").write_text(json.dumps({"modules": mods}, indent=1))
print(f"{len(chains)} trees -> {out}")
