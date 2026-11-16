"""Re-validate the teacher harness H_k* on θ_k before gate GB, DEV only.

H_k* was tuned on θ_{k-1}'s outputs; θ_k has absorbed part of it, so a module may now do its correction twice. Backward
elimination on θ_k's dev captures (every dataset), with the gate's composite S and paired bootstrap:
  1. for every module m still in the chain: replay the chain without m (sandboxed, no network);
     compare "with m" vs "without m": m is USEFUL iff that S has CI90 lower bound > 0 (summed over datasets as the
     gate does);
  2. if every module is useful, stop; otherwise drop the least useful one (smallest S of with-vs-without; a negative S
     means harmful) and repeat with the shorter chain;
  3. a module whose removal crashes the chain (a later module needs it) counts as needed;
  4. guard: the pruned chain must not be worse than the full H_k* on dev (S >= 0), else the full H_k* is kept.
The result (possibly the empty chain, i.e. N_k = F_k) is the harness of the round's node; val is not looked at.

  python rounds/revalidate.py --tree rounds/R1/teacher_tree --datasets rounds/R1/datasets.json --node R1_theta \
      --out rounds/R1/node_tree
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

S = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(S), str(S / 'loopA')]
import gate_service as G  # noqa: E402
import multiprocessing as mp  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import numpy as np  # noqa: E402
from eval.score import combine, compare, per_sequence, pooled  # noqa: E402

# worst-recording guard: pooled S can hide a module that blows up a single recording (e.g. a dev contribution of +0.02 %
# with one recording's EPE2D going from 8 to 38 px). A module that makes any dev recording's EPE2D-p, CT-p or
# GO-p more than BLOW_RATIO times worse, without a clearly significant pooled gain (CI90 lower bound > BLOW_KEEP_LO), is
# dropped from the final chain whatever the pick.
BLOW_COLS, BLOW_RATIO, BLOW_KEEP_LO = ('EPE2D-p', 'CT-p', 'GO-p'), 2.0, 0.005


def blowup(with_seq: dict, wo_seq: dict) -> tuple[float, str]:
    """largest per-recording ratio (chain with m / chain without m) of BLOW_COLS over every dev recording of every dataset"""
    worst, where = 1.0, ''
    for ds, accs in with_seq.items():
        for name, acc in accs.items():
            if name not in wo_seq.get(ds, {}):
                continue
            a, b = pooled([acc]), pooled([wo_seq[ds][name]])
            for c in BLOW_COLS:
                if np.isfinite(a[c]) and np.isfinite(b[c]) and b[c] > 1e-9 and a[c] / b[c] > worst:
                    worst, where = float(a[c] / b[c]), f'{ds}/{name}/{c}'
    return worst, where


def chain(tree: Path) -> list[dict]:
    p = tree / 'chain.json'
    return [m for m in json.loads(p.read_text()).get('modules', []) if m.get('enabled', True)] if p.is_file() else []


def write(src: Path, dst: Path, mods: list[dict]) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__'))
    (dst / 'chain.json').write_text(json.dumps({'modules': mods}, indent=1))
    return dst


def n_segments(view: str) -> int:
    return sum(len(json.loads(p.read_text())['segments']) for p in (Path(view) / 'segments').glob('*.json'))


def complete(out: Path, n: int) -> bool:
    """a replay output that can be reused: DONE marker, or (from an interrupted earlier run) every segment
    present and the newest files readable (a replay killed mid-write leaves its last files partial)"""
    if not out.is_dir():
        return False
    fs = list(out.glob('*.npz'))
    if len(fs) != n:
        return False
    if (out / 'DONE').is_file():
        return True
    for f in sorted(fs, key=lambda f: f.stat().st_mtime)[-64:]:
        try:
            with np.load(f) as z:
                [z[k] for k in z.files]
        except Exception:
            return False
    return True


# a module that falls back silently when a model cannot load or run (e.g. CUDA out of memory on a crowded card) changes
# the chain's output without a crash (under heavy contention many leave-one-out scores can move). Such a replay is redone alone, without contention, and the alone result is the one used.
DEGRADED = re.compile(r'could not load|inference failed|module disabled|out of memory|CUDA error|CUBLAS_STATUS', re.I)


def replay(tree: Path, view: Path, cache: Path, out: Path, jobs: int) -> tuple[bool, str]:
    """gate_service.replay (sandboxed harness.run, no network), but returning the whole output"""
    out.mkdir(parents=True, exist_ok=True); os.chown(out, G.UID, G.UID)
    ro = list(dict.fromkeys([str(tree), str(cache), str(view / 'segments'), str(view / 'videos')]))
    r = G.sandbox.run([G.PY, '-m', 'harness.run', '--tree', str(tree), '--cache', str(cache), '--view', str(view), '--out',
                       str(out), '--jobs', str(jobs)], ro=ro, rw=[str(out)], cwd=str(S), env=dict(PYTHONPATH=str(S)),
                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=G.REPLAY_TIMEOUT, net=False)
    return r.returncode == 0, r.stdout or ''


def _task(job):
    """one (chain, dataset) task in its own process: replay the chain on that dev view (reusing a complete earlier
    output), score it per sequence; scores are pickled in the work dir so a restarted run does not redo them.
    strict (under parallelism): a replay whose output shows a degraded module counts as failed (it is redone alone)."""
    tag, ds, cfg, work, jobs, strict = job
    d = cfg['datasets'][ds]
    sc_file = work / f'score_{tag}_{ds}.pkl'
    if sc_file.is_file():
        with open(sc_file, 'rb') as f:
            return tag, ds, pickle.load(f)
    out = work / f'pred_{tag}_{ds}'
    if not complete(out, cfg['nseg'][ds]):
        if out.exists():
            shutil.rmtree(out)
        ok, log = replay(work / f'tree_{tag}', Path(d['dev_view']), Path(d['dev_cache']), out, jobs)
        if not ok or (strict and DEGRADED.search(log)):
            (work / f'degraded_{tag}_{ds}.log').write_text(log[-20000:])
            shutil.rmtree(out, ignore_errors=True)
            return tag, ds, None
        (out / 'DONE').touch()
    sc = per_sequence(Path(d['dev_view']) / 'gt', out)
    tmp = sc_file.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(sc, f)
    tmp.rename(sc_file)
    return tag, ds, sc


def gpu_mib():
    try:
        q = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=30).stdout
        rows = [[int(x) for x in l.split(',')] for l in q.strip().splitlines()]
        return sum(r[0] for r in rows), sum(r[1] for r in rows)
    except Exception:
        return None, None


class GpuPeak(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.peak = 0; self.stop = False

    def run(self):
        while not self.stop:
            u, _ = gpu_mib()
            self.peak = max(self.peak, u or 0); time.sleep(1.0)


def run_chains(chains: list, a, cfg: dict, work: Path, parallel: int) -> dict:
    """{tag: {ds: per-sequence scores} or None (the chain crashes)} for [(tag, mods)], (chain, dataset) tasks at once.
    A crash or degraded output under parallelism is redone alone before it counts."""
    for tag, mods in chains:
        write(a.tree, work / f'tree_{tag}', mods)
    jobs = [(tag, ds, cfg, work, a.jobs, True) for tag, _ in chains for ds in cfg['datasets']]
    with ProcessPoolExecutor(max(1, min(parallel, len(jobs))), mp_context=mp.get_context('fork')) as ex:
        res = list(ex.map(_task, jobs))
    redo = [r for r in res if r[2] is None]
    if redo:
        print(f'[revalidate] {len(redo)} task(s) crashed or degraded under parallelism: redone alone', flush=True)
    res = [r if r[2] is not None else _task((r[0], r[1], cfg, work, a.jobs, False)) for r in res]
    out = {}
    for tag, ds, sc in res:
        out.setdefault(tag, {})[ds] = sc
    return {tag: (None if any(v is None for v in d.values()) else d) for tag, d in out.items()}


def judge_cached(cfg: dict, inc_seq: dict, cand_seq: dict) -> dict:
    """gate_service.judge on precomputed per-sequence scores (same compare / combine, same arguments)"""
    per = {ds: compare(inc_seq[ds], cand_seq[ds], n_boot=cfg.get('n_boot', 1000), min_gain=cfg['min_gain'],
                       max_regress=cfg['max_regress']) for ds in cfg['datasets']}
    g = combine(per)
    return dict(S=g['S'], S_lo=g['S_lo'], worst=g['worst'], regressed=g['regressed'], per=per)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tree', required=True, type=Path)
    ap.add_argument('--datasets', required=True, type=Path)
    ap.add_argument('--node', required=True, help='capture node of θ_k (captures/<node>/<ds>_dev)')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--jobs', type=int, default=8, help='harness workers per replay')
    ap.add_argument('--one-shot', action='store_true', help='one leave-one-out pass; drop every weak module at once')
    ap.add_argument('--parallel', type=int, default=40,
                    help='(chain, dataset) tasks at once, each its own process (replay + per-sequence scoring). '
                         '(one at a time, the leave-one-outs took ~3 min each with the machine mostly idle); a crash '
                         'under parallelism is retried alone')
    a = ap.parse_args()
    os.umask(0o022)
    work = a.out.parent / 'revalidate_work'
    work.mkdir(parents=True, exist_ok=True)
    dsets = json.loads(a.datasets.read_text())
    cfg = dict(datasets={ds: dict(dev_view=d['dev_view'], dev_cache=str(S / 'captures' / a.node / f'{ds}_dev'))
                         for ds, d in dsets.items()}, min_gain=0.0, max_regress=0.02, n_boot=500)
    cfg['nseg'] = {ds: n_segments(d['dev_view']) for ds, d in cfg['datasets'].items()}
    # a chain is its modules WITH their params: the merge can keep one module twice with different params; keyed by
    # names only, removing either copy would look like the same chain
    key = lambda ms: json.dumps([[m['name'], m.get('params', {})] for m in ms], sort_keys=True)      # noqa: E731
    seq = {}                                   # module-name list -> per-sequence dev scores (each chain scored once)

    full = chain(a.tree)
    rec = dict(tree=str(a.tree), node=a.node, full=[m['name'] for m in full], steps=[])
    cur = list(full)
    # GPU budget: one probe replay of the full chain on the smallest dev view while sampling GPU memory (always run:
    # the chain's own outputs may be reused from an earlier run); if the chain uses the GPU, cap the parallel tasks so
    # that their estimated peak stays within 60 % of the free memory
    ds_probe = min(cfg['nseg'], key=cfg['nseg'].get)
    probe_tree = write(a.tree, work / 'tree_gpu_probe', cur)
    probe_out = work / f"gpu_probe_{time.strftime('%H%M%S')}_{ds_probe}"
    used0, total = gpu_mib()
    mon = GpuPeak(); mon.start()
    replay(probe_tree, Path(cfg['datasets'][ds_probe]['dev_view']), Path(cfg['datasets'][ds_probe]['dev_cache']), probe_out, a.jobs)
    mon.stop = True; mon.join()
    parallel, per_task = a.parallel, None
    if used0 is not None:
        per_task = mon.peak - used0
        if per_task > 1024:
            parallel = max(1, min(a.parallel, int(0.6 * (total - used0) / per_task)))
    rec['gpu'] = dict(probe=str(probe_out), per_task_mib=per_task, parallel=parallel)
    print(f'[revalidate] GPU probe: ~{per_task} MiB per task ({a.jobs} workers) -> {parallel} tasks at once', flush=True)
    seq[key(cur)] = run_chains([('full', cur)], a, cfg, work, parallel)['full']
    if seq[key(cur)] is None:
        raise SystemExit('the teacher chain does not run on θ_k dev captures')
    seq_full = seq[key(cur)]
    path = []                                       # every chain the elimination went through (final pick below)
    guard = {}                                      # id(module) -> worst-recording blow-up record (BLOW_*)
    while cur:
        path.append(list(cur))
        todo, seen = [], set()
        for i, m in enumerate(cur):                     # tag by position: module names need not be unique
            k = key(cur[:i] + cur[i + 1:])
            if k not in seq and k not in seen:
                todo.append((f'wo_{len(cur)}_{i:02d}_{m["name"]}', cur[:i] + cur[i + 1:])); seen.add(k)
        got = run_chains(todo, a, cfg, work, parallel) if todo else {}
        for tag, mods in todo:
            seq[key(mods)] = got[tag]
        scores = []
        for i, m in enumerate(cur):
            k = key(cur[:i] + cur[i + 1:])
            if seq[k] is None:
                scores.append((m, None)); continue
            j = judge_cached(cfg, seq[k], seq[key(cur)])     # with m vs without m
            j['blowup'] = blowup(seq[key(cur)], seq[k])
            scores.append((m, j))
        step = [dict(module=m['name'], needed_to_run=j is None, S_with_vs_without=None if j is None else j['S'],
                     S_lo=None if j is None else j['S_lo'], regressed=None if j is None else j['regressed'],
                     blowup=None if j is None else j['blowup']) for m, j in scores]
        for m, j in scores:
            if j is not None and j['blowup'][0] > BLOW_RATIO and j['S_lo'] <= BLOW_KEEP_LO:
                guard[id(m)] = dict(module=m['name'], ratio=j['blowup'][0], where=j['blowup'][1], S_lo=j['S_lo'])
        rec['steps'].append(step)
        # weak: no significant gain, OR the module causes a significant regression on some column / dataset (the GA rule;
        # a module kept for S alone, e.g. +1.5 % on dev with FAcc / F1 down, can make GB fail)
        weak = [(m, j) for m, j in scores if j is not None and (j['S_lo'] <= 0 or j['regressed'] or id(m) in guard)]
        print(json.dumps(step), flush=True)
        # this pass's replay arrays are no longer read (their scores are cached): move them to the archive in the
        # background (~35-50 GB per pass; nothing is deleted, rounds/archive_arrays.py)
        roots = [str(work / f'pred_{tag}_{ds}') for tag, _ in todo for ds in cfg['datasets'] if (work / f'pred_{tag}_{ds}').is_dir()]
        if roots:
            subprocess.Popen([sys.executable, str(S / 'rounds/archive_arrays.py'), '--name',
                              f'{a.out.parent.name}_revalidate_pass{len(cur)}_{time.strftime("%H%M%S")}',
                              *sum((['--root', r] for r in roots), [])], cwd=str(S), stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
        if not weak:
            break
        if a.one_shot:      # drop every weak module of this first pass at once: one pass instead of ~10
            cur = [x for x in cur if all(x is not w for w, _ in weak)]
            if key(cur) not in seq:     # replay the pruned chain itself: the final pick below only judges scored chains
                seq[key(cur)] = run_chains([('oneshot', cur)], a, cfg, work, parallel)['oneshot']
            break
        m, _ = min(weak, key=lambda x: x[1]['S'])
        cur = [x for x in cur if x is not m]           # its leave-one-out scores are those of the new chain
    # final pick: every chain on the elimination path, plus the empty chain (θ_k alone), is judged on dev
    # against the round's base N_{k-1} with the GB rule; the pick is the highest S among chains without a significant
    # regression, else the highest S. (An S-only guard against the full teacher would put back the modules the
    # regression rule removed.)
    path.append(list(cur))
    if key([]) not in seq:
        seq[key([])] = run_chains([('empty', [])], a, cfg, work, parallel)['empty']
    base_seq = {ds: per_sequence(Path(d['dev_view']) / 'gt', Path(dsets[ds]['base_dev_pred'])) for ds, d in cfg['datasets'].items()}
    cands = []
    for ch in path + [[]]:
        if seq.get(key(ch)) is None or any(key(ch) == key(c[0]) for c in cands):
            continue
        j = judge_cached(cfg, base_seq, seq[key(ch)])
        cands.append((ch, j))
    rec['path'] = [dict(n=len(ch), modules=[m['name'] for m in ch], S_vs_base=j['S'], S_lo=j['S_lo'], regressed=j['regressed'])
                   for ch, j in cands]
    # a column that regresses vs base in EVERY candidate, the empty chain included, is the network's (θ_k vs N_(k-1)),
    # not the harness's: it no longer decides between chains (otherwise a drop shared by every chain sends the pick back
    # to the highest S, i.e. the full chain with the modules the pass found weak)
    regs = [{(ds, c) for ds, cols in j['regressed'].items() for c in cols} for _, j in cands]
    common = set.intersection(*regs) if regs else set()
    rec['common_regressions'] = sorted(f'{ds}:{c}' for ds, c in common)
    clean = [(ch, j) for (ch, j), r in zip(cands, regs) if not (r - common)]
    pick, why = (max(clean, key=lambda c: c[1]['S']), 'highest dev S vs base among chains without a harness-made significant '
                 'regression' + (f' (ignored, in every chain: {rec["common_regressions"]})' if common else '')) \
        if clean else (max(cands, key=lambda c: c[1]['S']), 'every chain regresses vs base on dev: highest dev S')
    cur = list(pick[0]); rec['pick'] = dict(n=len(cur), why=why, S_vs_base=pick[1]['S'], S_lo=pick[1]['S_lo'], regressed=pick[1]['regressed'])
    hit = [guard[id(m)] for m in cur if id(m) in guard]
    if hit:                                         # worst-recording guard: out of the final chain whatever the pick
        cur = [m for m in cur if id(m) not in guard]
        if key(cur) not in seq:
            seq[key(cur)] = run_chains([('guarded', cur)], a, cfg, work, parallel)['guarded']
        rec['guard_dropped'] = hit
        print(json.dumps(dict(guard_dropped=hit), default=float), flush=True)
    print(json.dumps(rec['pick'], default=float), flush=True)
    write(a.tree, a.out, cur)
    rec['kept'] = [m['name'] for m in cur]
    rec['dropped'] = [m['name'] for m in full if m not in cur]
    (a.out.parent / 'revalidate.json').write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps(dict(kept=rec['kept'], dropped=rec['dropped'])))


if __name__ == '__main__':
    main()
