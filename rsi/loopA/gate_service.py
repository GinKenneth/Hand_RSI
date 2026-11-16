"""gate service (root): scores Loop A submissions on the hidden val split with the composite S.

Per submission (spool/<agent>/req_k/{tree/, request.json, READY}):
  1. dedup: an identical tree (any agent) is not evaluated again; not charged
  2. dev pre-check (not charged): the tree is replayed on dev in the sandbox and must beat the agent's incumbent on dev
     (S > 0 with CI90 lower bound > 0); this keeps the val split from being probed by every idea
  3. charged: replay on every val dataset in the sandbox (inputs: raw captures + videos only; the val GT is never
     mounted), scored as root against the val GT, compared with the agent's incumbent on val
  4. accept (GA): S CI90 lower > 0 and S >= min_gain (averaged over datasets), and no metric worse than max_regress on
     any dataset. Accepted trees go to archive/<agent>_req_k and become that agent's incumbent.
The verdict sent back: accepted, the reason, a coarse S (0.1 %), budget left. Every verdict (pre-check failures
included) goes to the cross-round bucket (rounds/bucket.py) with the note, the diff against the agent's incumbent and the
DEV per-column deltas; val per-column numbers are never written there.
0. file whitelist: a tree may hold only .py / .json / .md files, at most 1 MB in total (no weights or
   data smuggled in from downloaded code); otherwise rejected, not charged.

  gate_service.py --exp EXPDIR   (loops until EXPDIR/STOP exists)
config.json (written by driver.py): datasets {name: {dev_view, dev_cache, val_view, val_cache}}, budget_per_agent,
min_gain, max_regress, precheck, dedup, jobs, submit_cutoff_min.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

S = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S)); sys.path.insert(0, str(S / "loopA"))
sys.path.insert(0, str(S / "rounds"))
import rsi_paths  # noqa: E402
import bucket  # noqa: E402
import sandbox  # noqa: E402
from eval.score import combine, compare, per_sequence  # noqa: E402

UID = rsi_paths.SANDBOX_UID
PY = rsi_paths.PY
LOCK = threading.Lock()
REPLAY_TIMEOUT = 2 * 3600
ALLOWED = (".py", ".json", ".md")
MAX_TREE_BYTES = 1 << 20


def bad_files(tree: Path) -> str:
    files = [p for p in tree.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    bad = [str(p.relative_to(tree)) for p in files if p.suffix not in ALLOWED]
    size = sum(p.stat().st_size for p in files)
    if bad:
        return f"files other than .py/.json/.md in the tree: {bad[:10]}"
    if size > MAX_TREE_BYTES:
        return f"tree is {size / 1e6:.2f} MB; the limit is 1 MB (code only, no weights or data)"
    return ""


def incumbent_tree(exp: Path, cfg: dict, a: dict) -> Path | None:
    name = a.get("incumbent_name", "base")
    t = exp / "archive" / name / "tree"
    return t if name != "base" and t.is_dir() else (Path(cfg["base_tree"]) if cfg.get("base_tree") else None)


def to_bucket(exp: Path, cfg: dict, agent: str, req: Path, tree: Path, note: str, verdict: str, a: dict,
              dev_judge: dict | None = None, s_coarse=None, reason: str = ""):
    try:
        mods = json.loads((tree / "chain.json").read_text()).get("modules", []) if (tree / "chain.json").is_file() else []
        rel = None
        if dev_judge:
            per = dev_judge["per"]
            rel = {m: round(sum(c["rel"][m] for c in per.values()) / len(per), 4) for m in next(iter(per.values()))["rel"]}
        bucket.add("submission", cfg.get("round", exp.name), f"{exp.name}_{agent}_{req.name}", note[:2000],
                   dict(agent=agent, verdict=verdict, S_coarse=s_coarse, dev_rel=rel,
                        dev_S=None if not dev_judge else round(100 * dev_judge["S"], 2), reason=reason[:300],
                        modules=[m["name"] for m in mods if m.get("enabled", True)]),
                   files={"note.md": note, "diff.patch": bucket.tree_diff(incumbent_tree(exp, cfg, a), tree)})
    except Exception as e:                                              # the bucket must never block a verdict
        print(f"bucket write failed: {e}", flush=True)


def fingerprint(tree: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(tree.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(tree)).encode()); h.update(p.read_bytes())
    return h.hexdigest()[:16]


def replay(tree: Path, view: Path, cache: Path, out: Path, jobs: int) -> tuple[bool, str]:
    """Sandboxed harness.run: sees the tree, the raw captures and the videos/segments of the view, never its gt/."""
    ok, tail, _ = _replay(tree, view, cache, out, jobs)
    return ok, tail


def _replay(tree: Path, view: Path, cache: Path, out: Path, jobs: int) -> tuple[bool, str, int]:
    """replay() plus the number of WiLoR failure lines in the run's output"""
    out.mkdir(parents=True, exist_ok=True); os.chown(out, UID, UID)
    ro = [str(tree), str(cache), str(view / "segments"), str(view / "videos")]
    ro = list(dict.fromkeys(ro))
    r = sandbox.run([PY, "-m", "harness.run", "--tree", str(tree), "--cache", str(cache), "--view", str(view), "--out", str(out),
                     "--jobs", str(jobs)], ro=ro, rw=[str(out)], cwd=str(S),
                    env=dict(PYTHONPATH=str(S),   # the gate's cards during Loop A (driver: GATE_CARDS)
                             **({"CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"]} if os.environ.get("CUDA_VISIBLE_DEVICES") else {})),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    timeout=REPLAY_TIMEOUT, net=False)      # no network: a module cannot fetch outside data at run time
    out_txt = r.stdout or ""
    return r.returncode == 0, out_txt[-4000:], sum(out_txt.count(m) for m in WILOR_FAIL)


# WiLoR can run out of GPU memory on shared cards and then skip its evidence silently ("crop evidence skipped").
# A replay with more than WILOR_MAX_FAIL of its segments failing is retried (WILOR_TRIES in all); still degraded = not scored, not charged.
WILOR_FAIL = ("[wilor] inference failed", "[wilor] could not load", "[wilor] frames unavailable")
WILOR_MAX_FAIL, WILOR_TRIES = 0.01, 3


def n_segments(view: Path) -> int:
    return sum(len(json.loads(p.read_text())["segments"]) for p in (view / "segments").glob("*.json"))


def replay_checked(tree: Path, view: Path, cache: Path, out: Path, jobs: int, log=None, tag="") -> tuple[bool, str, bool]:
    """replay() with the WiLoR coverage check; returns (ok, tail, degraded)"""
    nseg = max(n_segments(view), 1)
    for k in range(WILOR_TRIES):
        ok, tail, nfail = _replay(tree, view, cache, out, jobs)
        if not ok or nfail <= WILOR_MAX_FAIL * nseg:
            return ok, tail, False
        if log is not None:
            log.write(f"{time.strftime('%T')} {tag} WiLoR failed on {nfail}/{nseg} segments (try {k + 1}/{WILOR_TRIES})\n"); log.flush()
        time.sleep(60)
    return ok, tail, True


def judge(cfg: dict, split: str, inc: dict, cand_dirs: dict) -> dict:
    """compare candidate vs incumbent on every dataset of `split` ('dev' or 'val'); combined verdict."""
    per = {}
    for ds, d in cfg["datasets"].items():
        gt = Path(d[f"{split}_view"]) / "gt"
        per[ds] = compare(per_sequence(gt, Path(inc[ds])), per_sequence(gt, Path(cand_dirs[ds])), n_boot=cfg.get("n_boot", 1000),
                          min_gain=cfg["min_gain"], max_regress=cfg["max_regress"])
    g = combine(per)          # S on HOT3D only (eval.score.S_DATASETS), regressions on every dataset
    return dict(S=g["S"], S_lo=g["S_lo"], worst=g["worst"], regressed=g["regressed"], s_datasets=g["s_datasets"], per=per,
                passed=bool(g["S_lo"] > 0 and g["S"] >= cfg["min_gain"] and not g["regressed"]))


def process(exp: Path, cfg: dict, agent: str, req: Path, log) -> dict:
    state_p = exp / "state.json"
    gdir = exp / "gate" / f"{agent}_{req.name}"
    gdir.mkdir(parents=True, exist_ok=True)
    tree = gdir / "tree"
    if not tree.exists():
        shutil.copytree(req / "tree", tree, ignore=shutil.ignore_patterns("__pycache__"))
    note = json.loads((req / "request.json").read_text()).get("note", "")
    bad = bad_files(tree)
    if bad:
        a0 = json.loads(state_p.read_text())["agents"][agent]
        to_bucket(exp, cfg, agent, req, tree, note, "bad-files", a0, reason=bad)
        return dict(accepted=False, reason=f"{bad}; not evaluated, not charged", budget_left=cfg["budget_per_agent"] - a0["used"])
    fp = fingerprint(tree)
    with LOCK:
        st = json.loads(state_p.read_text()); a = st["agents"][agent]
        fps = st.setdefault("fingerprints", {})
        if cfg.get("dedup", True) and fp in fps:
            log.write(f"{time.strftime('%T')} {agent} {req.name} DUPLICATE of {fps[fp]} (not charged)\n"); log.flush()
            return dict(accepted=False, reason=f"identical to {fps[fp]}; not evaluated, not charged", budget_left=cfg["budget_per_agent"] - a["used"])
        fps[fp] = f"{agent}_{req.name}"; state_p.write_text(json.dumps(st, indent=1))
    dl = exp / "DEADLINE"
    if dl.is_file() and time.time() > float(dl.read_text()) - cfg.get("submit_cutoff_min", 45) * 60:
        return dict(accepted=False, reason="submissions are closed for this round; not charged", budget_left=cfg["budget_per_agent"] - a["used"])
    if cfg.get("precheck", True):                                   # dev pre-check (not charged)
        dev_out = {ds: gdir / f"dev_{ds}" for ds in cfg["datasets"]}
        for ds, d in cfg["datasets"].items():
            ok, tail, degraded = replay_checked(tree, Path(d["dev_view"]), Path(d["dev_cache"]), dev_out[ds], cfg.get("jobs", 16),
                                                log, f"{agent} {req.name} dev {ds}")
            if ok and degraded:
                return dict(accepted=False, reason=f"WiLoR ran out of GPU memory on more than {100 * WILOR_MAX_FAIL:.0f}% of the dev "
                            f"({ds}) segments in {WILOR_TRIES} tries (shared cards); not scored, not charged: submit again",
                            budget_left=cfg["budget_per_agent"] - a["used"])
            if not ok:
                (gdir / "precheck_crash.txt").write_text(tail)
                return dict(accepted=False, reason=f"crashed on dev ({ds}) during the pre-check: {tail[-600:]}; not charged",
                            budget_left=cfg["budget_per_agent"] - a["used"])
        pre = judge(cfg, "dev", a["incumbent_dev"], dev_out)
        (gdir / "precheck.json").write_text(json.dumps(pre, indent=1, default=float))
        if not (pre["S_lo"] > 0 and pre["S"] > 0):
            to_bucket(exp, cfg, agent, req, tree, note, "dev-precheck-fail", a, dev_judge=pre)
            log.write(f"{time.strftime('%T')} {agent} {req.name} PRECHECK-FAIL dev S={100 * pre['S']:+.2f}% (not charged)\n"); log.flush()
            return dict(accepted=False, reason=f"dev pre-check: not better than your incumbent on dev (S={100 * pre['S']:+.2f}%, "
                        f"CI90 lower {100 * pre['S_lo']:+.2f}%); not sent to the hidden split, not charged",
                        budget_left=cfg["budget_per_agent"] - a["used"])
    with LOCK:
        st = json.loads(state_p.read_text()); a = st["agents"][agent]
        if a["used"] >= cfg["budget_per_agent"]:
            return dict(accepted=False, reason="gate budget exhausted", budget_left=0)
        a["used"] += 1; state_p.write_text(json.dumps(st, indent=1)); a = dict(a)
    val_out = {ds: gdir / f"val_{ds}" for ds in cfg["datasets"]}
    for ds, d in cfg["datasets"].items():
        ok, tail, degraded = replay_checked(tree, Path(d["val_view"]), Path(d["val_cache"]), val_out[ds], cfg.get("jobs", 16),
                                            log, f"{agent} {req.name} val {ds}")
        if ok and degraded:
            with LOCK:                                              # refund: the hidden split was not scored
                st = json.loads(state_p.read_text()); st["agents"][agent]["used"] -= 1; state_p.write_text(json.dumps(st, indent=1))
                left = cfg["budget_per_agent"] - st["agents"][agent]["used"]
            return dict(accepted=False, reason=f"WiLoR ran out of GPU memory on the hidden split ({ds}) in {WILOR_TRIES} tries "
                        "(shared cards); not scored, not charged: submit again", budget_left=left)
        if not ok:
            return dict(accepted=False, reason=f"crashed on the hidden split ({ds}); charged. Run `lt replay` on all of dev first.",
                        budget_left=cfg["budget_per_agent"] - a["used"])
    v = judge(cfg, "val", a["incumbent_val"], val_out)
    (gdir / "score.json").write_text(json.dumps(v, indent=1, default=float))
    accept = v["passed"]
    reason = ("accepted: significant composite gain, no metric regressed" if accept else
              "not significantly better than your incumbent" if v["S_lo"] <= 0 else
              f"gain below the minimum ({100 * cfg['min_gain']:.1f}%)" if v["S"] < cfg["min_gain"] else
              f"a metric regressed significantly on some dataset: {v['regressed']}")
    name = f"{agent}_{req.name}"
    with LOCK:
        st = json.loads(state_p.read_text())
        if accept:
            node = exp / "archive" / name
            shutil.copytree(tree, node / "tree", dirs_exist_ok=True)
            st["agents"][agent].update(incumbent_name=name, incumbent_val={ds: str(p) for ds, p in val_out.items()},
                                       incumbent_dev={ds: str(gdir / f"dev_{ds}") for ds in cfg["datasets"]})
            st.setdefault("accepted", []).append(dict(name=name, agent=agent, S=v["S"], S_lo=v["S_lo"], note=note, time=time.strftime("%F %T")))
        state_p.write_text(json.dumps(st, indent=1))
        to_bucket(exp, cfg, agent, req, tree, note, "ACCEPT" if accept else "reject", a,
                  dev_judge=pre if cfg.get("precheck", True) else None, s_coarse=round(100 * v["S"], 1), reason=reason)
        log.write(f"{time.strftime('%T')} {agent} {req.name} accept={accept} S={100 * v['S']:+.2f}% lo={100 * v['S_lo']:+.2f}% "
                  f"worst={100 * v['worst']:+.2f}%\n"); log.flush()
    return dict(accepted=accept, reason=reason, coarse_val_S_percent=round(100 * v["S"], 1), budget_left=cfg["budget_per_agent"] - a["used"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, type=Path)
    a = ap.parse_args()
    exp = a.exp
    cfg = json.loads((exp / "config.json").read_text())
    log = open(exp / "gate.log", "a")
    busy = set()

    def handle(agent, req):
        try:
            v = process(exp, cfg, agent, req, log)
        except Exception as e:                                    # never leave an agent waiting forever
            v = dict(accepted=False, reason=f"gate error: {type(e).__name__}: {e}"[:500])
        (req / "verdict.json").write_text(json.dumps(v, indent=1)); os.chown(req / "verdict.json", UID, UID)
        busy.discard(str(req))

    while not (exp / "STOP").exists():
        # first come, first served (by submit time, not by path order)
        ready = [sp for sp in (exp / "spool").glob("*/req_*") if (sp / "READY").is_file()]
        for sp in sorted(ready, key=lambda p: (p / "READY").stat().st_mtime):
            if (sp / "READY").is_file() and not (sp / "verdict.json").is_file() and str(sp) not in busy \
                    and len(busy) < cfg.get("gate_parallel", 3):
                busy.add(str(sp))
                threading.Thread(target=handle, args=(sp.parent.name, sp), daemon=True).start()
        time.sleep(10)


if __name__ == "__main__":
    main()
