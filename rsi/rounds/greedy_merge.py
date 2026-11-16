"""Greedy merge of the Loop A patches of one round -> the teacher harness H_k*.

  python rounds/greedy_merge.py --exp loopA_runs/<R> --datasets rounds/<R>/datasets.json --base-tree TREE --out rounds/<R>/teacher_tree

1. The accepted trees (state.json "accepted"; archive/<name>/tree) are ranked by their val S (gate/<name>/score.json).
2. current = the best one. For each next tree, its modules that current does not already have (same file content) are
   appended in their own order (renamed <agent>__<module> on a name clash) and the candidate is replayed on DEV. It is
   kept only if it beats current on dev (CI90 lower > 0, S >= --dev-min-gain, no column worse than 2 %). The greedy
   search never looks at val, so val is not worn out by the merge.
3. Final: the merged tree and the best single tree are each scored ONCE on val against the round's base node; the one
   with the larger S becomes the teacher (merge.json records both).
Agent code only runs in the sandbox (gate_service.replay).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import shutil
import sys
from pathlib import Path

S = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S)); sys.path.insert(0, str(S / "loopA"))
import gate_service as G  # noqa: E402
from eval.score import compare, per_sequence  # noqa: E402


def mods(tree: Path):
    spec = json.loads((tree / "chain.json").read_text()) if (tree / "chain.json").is_file() else {"modules": []}
    out = []
    for m in spec.get("modules", []):
        f = tree / "modules" / f"{m['name']}.py"
        h = hashlib.sha256(f.read_bytes() + json.dumps(m.get("params", {}), sort_keys=True).encode()).hexdigest()[:12]
        out.append((m, f, h))
    return out


def write_tree(dst: Path, entries):
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "modules").mkdir(parents=True)
    chain = []
    for m, f, _ in entries:
        shutil.copy2(f, dst / "modules" / f"{m['name']}.py")
        # a module may import a helper next to it: copy its siblings that are not modules of the chain
        for side in f.parent.glob("*.py"):
            if not (dst / "modules" / side.name).exists():
                shutil.copy2(side, dst / "modules" / side.name)
        chain.append(m)
    (dst / "chain.json").write_text(json.dumps({"modules": chain}, indent=1))
    return dst


def replay_all(tree, cfg, split, out_root, jobs):
    """replay a candidate chain on every dataset of `split`. A crash or a degraded output (a model that could not load or
    run, e.g. CUDA OOM next to another GPU job) is redone once with a third of the
    workers after 30 s; only a second failure counts as the chain's own crash."""
    from revalidate import DEGRADED, replay as full_replay
    outs = {}
    for ds, d in cfg["datasets"].items():
        out = out_root / f"{split}_{ds}"
        for attempt, j in enumerate((jobs, max(4, jobs // 3))):
            if out.exists():
                shutil.rmtree(out)
            ok, log = full_replay(tree, Path(d[f"{split}_view"]), Path(d[f"{split}_cache"]), out, j)
            if ok and not DEGRADED.search(log):
                break
            time.sleep(30)
        else:
            return None, log[-4000:]
        outs[ds] = str(out)
    return outs, ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, type=Path)
    ap.add_argument("--datasets", required=True, type=Path)
    ap.add_argument("--base-tree", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dev-min-gain", type=float, default=0.001)
    ap.add_argument("--jobs", type=int, default=24)
    a = ap.parse_args()
    __import__("os").umask(0o022)          # new trees / prediction dirs must be readable inside the sandbox
    exp, work = a.exp.resolve(), a.out.resolve().parent / "merge_work"
    work.mkdir(parents=True, exist_ok=True)
    cfg = dict(datasets=json.loads(a.datasets.read_text()), min_gain=a.dev_min_gain, max_regress=0.02, n_boot=500)
    acc = json.loads((exp / "state.json").read_text()).get("accepted", [])
    rec = dict(accepted=len(acc), steps=[])
    if not acc:
        shutil.copytree(a.base_tree, a.out, dirs_exist_ok=True)
        rec["teacher"] = "base"; (a.out.parent / "merge.json").write_text(json.dumps(rec, indent=1)); print("no accepted patch"); return
    for x in acc:
        sc = exp / "gate" / x["name"] / "score.json"
        x["val_S"] = json.loads(sc.read_text())["S"] if sc.is_file() else x.get("S", 0.0)
    acc.sort(key=lambda x: -x["val_S"])
    base_mods = {h for _, _, h in mods(a.base_tree)}
    best_tree = exp / "archive" / acc[0]["name"] / "tree"
    cur = mods(best_tree)
    cur_dev, tail = replay_all(best_tree, cfg, "dev", work / "cur0", a.jobs)
    if cur_dev is None:
        raise SystemExit(f"best tree does not replay on dev: {tail[-500:]}")
    for i, x in enumerate(acc[1:], 1):
        have = {h for _, _, h in cur}
        add = [(m, f, h) for m, f, h in mods(exp / "archive" / x["name"] / "tree") if h not in have and h not in base_mods]
        if not add:
            rec["steps"].append(dict(tree=x["name"], kept=False, why="nothing new")); continue
        names = {m["name"] for m, _, _ in cur}
        add = [(dict(m, name=f"{x['agent']}__{m['name']}") if m["name"] in names else m, f, h) for m, f, h in add]
        cand = write_tree(work / f"cand{i}", cur + add)
        cand_dev, tail = replay_all(cand, cfg, "dev", work / f"cand{i}_pred", a.jobs)
        if cand_dev is None:
            rec["steps"].append(dict(tree=x["name"], kept=False, why=f"crash: {tail[-200:]}")); continue
        j = G.judge(cfg, "dev", cur_dev, cand_dev)
        rec["steps"].append(dict(tree=x["name"], modules=[m["name"] for m, _, _ in add], dev_S=j["S"], dev_lo=j["S_lo"], kept=j["passed"]))
        print(f"{x['name']}: +{[m['name'] for m, _, _ in add]} dev S {100 * j['S']:+.2f}% lo {100 * j['S_lo']:+.2f}% -> "
              f"{'KEEP' if j['passed'] else 'drop'}", flush=True)
        if j["passed"]:
            cur, cur_dev = mods(cand), cand_dev
    merged = write_tree(work / "merged", cur)
    # one val look each: merged vs the best single tree, both against the round's base node
    base_val = {ds: d["base_val_pred"] for ds, d in cfg["datasets"].items()}
    final = {}
    for name, tree in (("merged", merged), ("best_single", best_tree)):
        outs, tail = replay_all(tree, cfg, "val", work / f"{name}_val", a.jobs)
        if outs is None:
            final[name] = dict(S=-1.0, why=tail[-200:]); continue
        jv = G.judge(dict(cfg, min_gain=0.01), "val", base_val, outs)
        final[name] = dict(S=jv["S"], lo=jv["S_lo"], worst=jv["worst"], regressed=jv.get("regressed", {}))
    pick = "merged" if final["merged"]["S"] >= final["best_single"]["S"] else "best_single"
    shutil.copytree(merged if pick == "merged" else best_tree, a.out, dirs_exist_ok=True)
    rec.update(final=final, teacher=pick, merged_modules=[m["name"] for m, _, _ in cur])
    (a.out.parent / "merge.json").write_text(json.dumps(rec, indent=1, default=float))
    print(f"teacher = {pick}: merged S {100 * final['merged']['S']:+.2f}% vs best single {100 * final['best_single']['S']:+.2f}% "
          f"(val, vs base); modules {[m['name'] for m, _, _ in cur]}")


if __name__ == "__main__":
    main()
