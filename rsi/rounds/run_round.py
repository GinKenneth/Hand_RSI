"""One round of the self-improvement loop (README.md), resumable stage by stage (a stage that left its DONE marker
is skipped on a restart).

  python rounds/run_round.py --round R1 --gpus 0,1,2,3,4,5,6,7 [--loopa-hours 4] [--agents roster|smoke]
                             [--stages capture,base,loopA,teacher,loopB,gate,sealed,loopC] [--datasets hot3d,arctic]

State: rounds/<R>/round.json (base node, teacher, θ_k, verdicts). Base node of R1 = 00 (release ACE, empty harness);
of R_k = the last ACCEPTED node (rounds/ARCHIVE.json).
  capture   θ_{k-1} raw outputs on dev / val / train / test        captures/<node>/<view>
  base      H_{k-1} on θ_{k-1}'s dev / val captures                preds/<R>/base_<view>
  loopA     loopA/driver.py (gate = val, composite S)               loopA_runs/<R>
  teacher   H_k* = greedy merge of the accepted patches (dev), best of merged / best single on val   rounds/<R>/teacher_tree
  loopB     teacher targets on train -> train θ_k -> merge          loopB/runs/<R>, models/<R>_theta
  revalidate  H_k* re-checked module by module on θ_k's DEV captures (backward elimination, rounds/revalidate.py):
            modules that no longer help θ_k (or now hurt it) are dropped -> rounds/<R>/node_tree
  gate      GB: N_k = (θ_k, node_tree) vs N_{k-1} on val (all datasets); F_k (θ_k, empty harness) vs F_{k-1} recorded;
            FPS of N_k (rounds/fps.py)
  sealed    test scores of N_k and F_k on every dataset's test view, HOI4D zero-shot included when its view exists
            (recorded only, never read by any decision)
  loopC     the research lead: dossier -> plan for the next round (loopC/run_loopC.py); applied at the next round's start
Loop C's plan (rounds/PLAN_LATEST.json) is applied at the start of a round: idea cards, lessons and focus go to Loop A,
whitelisted Loop B parameters to Loop B (rounds/LOOPB_PARAMS.json). Every stage writes to the cross-round bucket.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

S = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S)); sys.path.insert(0, str(S / 'rounds')); sys.path.insert(0, str(S / 'loopA'))
import bucket  # noqa: E402
import rsi_paths  # noqa: E402
PY = rsi_paths.PY
TORCHRUN = rsi_paths.TORCHRUN
EMPTY = S / 'harness/trees/empty'
STATUS = S / 'logs' / 'rounds.md'
STATE = Path(os.environ.get('RSI_STATE_DIR', S / 'rounds'))   # ARCHIVE / PLAN_LATEST / LOOPB_PARAMS (smoke: elsewhere)


def say(msg):
    line = f"- {time.strftime('%m-%d %H:%M')} {msg}"
    print(line, flush=True)
    with open(STATUS, 'a') as f:
        f.write(line + '\n')


def run(cmd, log, env=None, check=True):
    with open(log, 'a') as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONPATH=str(S), **(env or {})), cwd=str(S))
    if check and r.returncode:
        raise SystemExit(f'FAILED: {" ".join(map(str, cmd))[:200]} (see {log})')
    return r.returncode


def views(datasets, split):
    return [f'{ds}_{split}' for ds in datasets]


CAPTURE_ALIAS = {'hoi4d_testpseudo': 'hoi4d_test'}      # same segments and videos, other GT: one capture serves both


def free_gpus(gpus, min_free_mib):
    """the given cards that have at least min_free_mib free right now (the host may be shared: other users' jobs come and go;
    a capture shard once died of CUDA OOM next to a 62 GB job)"""
    try:
        q = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=60).stdout
        free = {l.split(',')[0].strip(): int(l.split(',')[1]) for l in q.strip().splitlines()}
    except Exception:                                               # noqa: BLE001
        return list(gpus)
    return [g for g in gpus if free.get(g, 0) >= min_free_mib]


def capture(node, ckpt, view, gpus, log, attempts=6):   # other users' jobs may grab memory
    """parallel shards over the cards that have room; a capture with every segment present is skipped. A failed shard
    (e.g. CUDA OOM from another user's job) is retried on the cards free at that moment; done segments are kept."""
    view = CAPTURE_ALIAS.get(view, view)
    out = S / 'captures' / node / view
    n_seg = sum(len(json.loads(p.read_text())['segments']) for p in (S / 'data/views' / view / 'segments').glob('*.json'))
    rc = []
    for attempt in range(attempts):
        if out.is_dir() and len(list(out.glob('*.npz'))) >= n_seg:
            return out
        use = free_gpus(gpus, 20000)
        if not use:
            say(f'capture {node}/{view}: no card with 20 GB free, waiting'); time.sleep(300); continue
        procs = [subprocess.Popen(['bash', str(S / 'capture/capture.sh'), node, ckpt, view, g, f'{i}/{len(use)}'],
                                  stdout=open(log, 'a'), stderr=subprocess.STDOUT) for i, g in enumerate(use)]
        rc = [p.wait() for p in procs]
        if any(rc):
            say(f'capture {node}/{view}: shard failure on cards {use} (rc {rc}), attempt {attempt + 1}/{attempts}')
            time.sleep(60)
    if not (out.is_dir() and len(list(out.glob('*.npz'))) >= n_seg):
        raise SystemExit(f'FAILED capture {node}/{view}: rc {rc}, {len(list(out.glob("*.npz")))}/{n_seg}')
    return out


def numa_of(gpus):
    """card -> NUMA node (from the PCI bus id); {} if unknown"""
    try:
        q = subprocess.run(['nvidia-smi', '--query-gpu=index,pci.bus_id', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=60).stdout
        out = {}
        for l in q.strip().splitlines():
            i, bus = (x.strip() for x in l.split(','))
            out[i] = int(Path(f'/sys/bus/pci/devices/{bus[-12:].lower()}/numa_node').read_text())
        return {g: out[g] for g in gpus if g in out}
    except Exception:                                               # noqa: BLE001
        return {}


def train_gpus(gpus):
    """cards for a torchrun job: those with 25 GB free, as 8 / 4 / 2 / 1 of them (the 8 windows per step split evenly).
    The cards are taken round-robin over the NUMA nodes: on our host NCCL with its default Simple protocol dies
    with 'illegal memory access' (Xid 13 Out Of Range Address) whenever every rank sits on one NUMA node (0-3, 4-7, 0,1,
    0,1,2), while any set spanning both nodes (8, 1-4, 2-5, 0,1,4,5, 0,4) runs."""
    use = free_gpus(gpus, 25000)
    numa = numa_of(use)
    by = {}
    for g in use:
        by.setdefault(numa.get(g, 0), []).append(g)
    order, queues = [], [list(q) for q in by.values()]
    while any(queues):
        for q in queues:
            if q:
                order.append(q.pop(0))
    for n in (8, 4, 2, 1):
        if len(order) >= n:
            return sorted(order[:n], key=int)
    return []


def nccl_env(use):
    """NCCL_PROTO=LL when every card of a multi-card job is on one NUMA node (the Simple protocol crashes there on
    our host; LL was tested to run on cards 0-3); nothing otherwise, so 8-card runs stay as they were"""
    numa = numa_of(use)
    if len(use) > 1 and len(numa) == len(use) and len(set(numa.values())) == 1:
        return ['NCCL_PROTO=LL']
    return []


def finished(args_after_module):
    """True when the run already saved its last checkpoint and logged its last step (a rank that dies at exit after
    saving must not trigger a full retrain).
    train_log.jsonl restarts with every attempt, so a stale checkpoint of an earlier attempt does not count."""
    a = list(args_after_module)
    try:
        out, steps = Path(a[a.index('--out') + 1]), int(a[a.index('--steps') + 1])
        last = json.loads((out / 'train_log.jsonl').read_text().splitlines()[-1])
    except (ValueError, IndexError, OSError, json.JSONDecodeError):
        return False
    return (out / f'step{steps}.pt').is_file() and int(last.get('step', 0)) >= steps


def run_torch(args_after_module, gpus, log, port, attempts=3):
    """torchrun loopB.train on the cards that have room, retried (a crash restarts the run; the train script has no resume)"""
    for attempt in range(attempts):
        use = train_gpus(gpus)
        if not use:
            say(f'torchrun: no card with 25 GB free, waiting'); time.sleep(300); continue
        rc = run(['env', *nccl_env(use), f'CUDA_VISIBLE_DEVICES={",".join(use)}', TORCHRUN, '--nproc_per_node', str(len(use)),
                  '--master_port', str(port), '-m', 'loopB.train', *args_after_module], log, check=False)
        if rc == 0:
            return use
        if finished(args_after_module):
            say(f'torchrun exited with rc {rc} on cards {use} after saving its final checkpoint: run accepted')
            return use
        say(f'torchrun failed on cards {use} (rc {rc}), attempt {attempt + 1}/{attempts}')
        time.sleep(60)
    raise SystemExit('FAILED: loopB.train after retries (see run.log)')


def one_gpu(gpus):
    use = free_gpus(gpus, 25000)
    return (use or list(gpus))[0]


def gpu_free(gpus):
    """{card: free MiB} for the given cards right now ({} if nvidia-smi fails)"""
    try:
        q = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=60).stdout
        free = {l.split(',')[0].strip(): int(l.split(',')[1]) for l in q.strip().splitlines()}
        return {g: free[g] for g in gpus if g in free}
    except Exception:                                               # noqa: BLE001
        return {}


def best_gpu(gpus, need_mib=0, wait_s=0):
    """the card with the most free memory; with need_mib, wait up to wait_s for one that has that much"""
    t0 = time.time()
    while True:
        free = gpu_free(gpus)
        if not free:
            return list(gpus)[0]
        g = max(free, key=free.get)
        if free[g] >= need_mib or time.time() - t0 >= wait_s:
            return g
        time.sleep(60)


def merge_retry(student, out_dir, gpus, log, attempts=2, need_mib=50000):
    """loopB.merge --verify on the card with the most free memory (waiting up to 15 min for one with need_mib free: the
    verify holds the adapter model and the merged one, ~40 GB), retried (a verify can hit CUDA OOM next to other users' jobs). If every verified attempt fails, one last
    merge without --verify keeps the round going and is logged loudly."""
    for i in range(attempts + 1):
        verify = i < attempts
        g = best_gpu(gpus, need_mib, 600)
        rc = run([PY, '-m', 'loopB.merge', '--student', str(student), '--out_dir', str(out_dir), '--gpu', '0',
                  *(['--verify'] if verify else [])], log, env=dict(CUDA_VISIBLE_DEVICES=g), check=False)
        if rc == 0:
            if not verify:
                say(f'merge {Path(out_dir).name}: merged WITHOUT --verify on card {g} after {attempts} verified attempts failed')
            return
        say(f'merge {Path(out_dir).name}: failed on card {g} (rc {rc}), attempt {i + 1}/{attempts + 1}')
        time.sleep(60)
    raise SystemExit(f'FAILED: loopB.merge of {student} after {attempts + 1} attempts (see {log})')


def sealed_views(dsets, test_split):
    """(dataset, view) pairs scored in the sealed stage: every dataset's test view, plus HOI4D zero-shot (official
    annotations = hoi4d_test; with WiLoR pseudo-GT = hoi4d_testpseudo, appendix) when its views exist"""
    out = [(ds, f'{ds}_{test_split}') for ds in dsets]
    if test_split == 'test':
        out += [('hoi4d', v) for v in ('hoi4d_test', 'hoi4d_testpseudo') if (S / 'data/views' / v).is_dir()]
    return out


def cleanup_round(prev: str, dry=False) -> float:
    """Delete the bulky per-segment arrays (.npz / .npy) of a finished round; returns GB freed. Our data disk is
    shared and Loop A writes tens of GB per hour (gate predictions per submission, every agent's lt replays).
    Kept: trees, notes, transcripts, scores and all .json, the bucket, models, captures, the sealed test predictions
    (sealed/<prev>/) and the N_k / F_k val predictions (preds/<prev>/Nk_*, Fk_*). Called at the next round's base
    stage, i.e. after Loop C of <prev> (rsi_loop waits for it), so nothing deleted is still read."""
    roots = [S / 'loopA_runs' / prev, S / 'loopB' / 'teacher_pred' / prev,
             S / 'rounds' / prev / 'merge_work', S / 'rounds' / prev / 'revalidate_work', S / 'rounds' / prev / 'loopC']
    roots += [q for q in (S / 'preds' / prev).glob('*') if q.is_dir() and not q.name.startswith(('Nk_', 'Fk_'))]
    files = [f for r in roots if r.exists() for f in r.rglob('*') if f.suffix in ('.npz', '.npy') and f.is_file()]
    files += list((S / 'loopB' / 'cache').glob(f'*/teacher_{prev}.npz'))
    n = 0
    for f in files:
        try:
            n += f.stat().st_size
            if not dry:
                f.unlink()
        except OSError:
            pass
    return n / 1e9


def reset_for_loopB_retry(R: str) -> int:
    """Set round R back to just before its Loop B, keeping its Loop A, teacher, base predictions and C1 / C2 / Fprev
    controls (Loop C may ask for this once per round, loopC/run_loopC.py check_action). Everything that depends on θ_k is RENAMED with suffix _v<n>, nothing is deleted; the round's GB-fail entry
    (harness only) leaves ARCHIVE.json; Loop C's validated Loop B parameters become LOOPB_PARAMS.json. Returns n."""
    rd = S / 'rounds' / R
    n = len(list(rd.glob('round_v*.json'))) + 1
    sfx = f'_v{n}'
    plan = rd / 'loopC' / 'plan_validated' / 'loopB_params.json'
    params = json.loads(plan.read_text()) if plan.is_file() else None
    moves = [(S / 'models' / f'{R}_theta', S / 'models' / f'{R}_theta{sfx}'),
             (S / 'loopB/runs' / R, S / 'loopB/runs' / f'{R}{sfx}'),
             (S / 'captures' / f'{R}_theta', S / 'captures' / f'{R}_theta{sfx}'),
             (S / 'sealed' / R, S / 'sealed' / f'{R}{sfx}'),
             (S / 'sealed' / f'{R}.json', S / 'sealed' / f'{R}{sfx}.json')]
    moves += [(rd / x, rd / f'{Path(x).stem}{sfx}{Path(x).suffix}') for x in ('node_tree', 'revalidate.json', 'revalidate_work',
                                                                             'loopC', 'fps.json')]
    for top in ('models', 'captures'):                 # Loop B checkpoint candidates (P['ckpt_select'])
        moves += [(p, p.with_name(p.name + sfx)) for p in (S / top).glob(f'{R}_ck[0-9]*') if '_v' not in p.name]
    for src, dst in moves:
        if src.exists():
            src.rename(dst)
    pv = S / 'preds' / f'{R}{sfx}'
    for q in list((S / 'preds' / R).glob('Nk_*')) + list((S / 'preds' / R).glob('Fk_*')):
        pv.mkdir(parents=True, exist_ok=True); q.rename(pv / q.name)
    stp = rd / 'round.json'
    st = json.loads(stp.read_text())
    (rd / f'round{sfx}.json').write_text(json.dumps(st, indent=1, default=float))
    for k in ('verdict', 'revalidate', 'sealed', 'fps', 'controls', 'next_base', 'loopB_cards'):
        st.pop(k, None)
    stp.write_text(json.dumps(st, indent=1, default=float))
    ap_ = STATE / 'ARCHIVE.json'
    if ap_.is_file():
        arch = json.loads(ap_.read_text())
        ap_.write_text(json.dumps([e for e in arch if not (e.get('round') == R and 'harness only' in e.get('kind', ''))], indent=1))
    if params is not None:
        (STATE / 'LOOPB_PARAMS.json').write_text(json.dumps(params, indent=1))
    pl = STATE / 'PLAN_LATEST.json'                    # the plan of this round now lives under loopC_v<n>
    if pl.is_file():
        d = json.loads(pl.read_text())
        if d.get('round') == R:
            d['dir'] = d['dir'].replace(f'/rounds/{R}/loopC/', f'/rounds/{R}/loopC{sfx}/'); pl.write_text(json.dumps(d, indent=1))
    say(f'[{R}] Loop B redo #{n} (Loop C decision): θ_k-dependent results renamed *{sfx}, Loop B params {params}')
    bucket.add('loopB_retry', R, f'{R}_loopB_retry{n}', f'{R}: Loop B redone (#{n}) with {params}; earlier results kept as *{sfx}',
               dict(params=params, n=n))
    return n


def harness_preds(tree, cache, view, out, log, jobs=24):
    """harness.run over one view, in the sandbox without network, like every other replay of agent-written code (gate,
    Loop B targets, revalidate, FPS): it sees the tree, the captures and the view's segments / videos, never its gt/."""
    if out.is_dir() and (out / 'DONE').is_file():
        return out
    import sandbox
    vd = S / 'data/views' / view
    out.mkdir(parents=True, exist_ok=True)
    for q in [out, *out.iterdir()]:                     # a partial dir from an earlier (root) run must be writable
        os.chown(q, sandbox.UID, sandbox.GID)
    with open(log, 'a') as fh:
        r = sandbox.run([PY, '-m', 'harness.run', '--tree', str(tree), '--cache', str(cache), '--view', str(vd),
                         '--out', str(out), '--jobs', str(jobs)],
                        ro=list(dict.fromkeys([str(tree), str(cache), str(vd / 'segments'), str(vd / 'videos')])), rw=[str(out)],
                        cwd=str(S), env=dict(PYTHONPATH=str(S)), stdout=fh, stderr=subprocess.STDOUT, net=False)
    if r.returncode:
        raise SystemExit(f'FAILED: sandboxed harness replay of {tree} on {view} (rc {r.returncode}, see {log})')
    (out / 'DONE').touch()
    return out


def c1_vs_prev(R, bnode, c1node, dsets, log):
    """Is C1_k (θ_(k-1) + GT only) ahead of θ_(k-1)? Both networks alone on DEV (never val / test), the
    gate's rule (eval.score: S on HOT3D, blocking regressions on HOT3D; detection: GT holes ignored). Ahead = the
    composite S significantly above 0 (CI90 lower bound > 0); a single regressed column is reported but does not decide
    (GT-only fine-tuning tends to lose some smoothness, which the harness restores). Not ahead: Loop B builds the teacher on θ_(k-1) (F_(k-1) + H_k) and the
    student starts from θ_(k-1)."""
    from eval.score import combine, compare, per_sequence
    per = {}
    for ds in dsets:
        v = f'{ds}_dev'; gt = S / 'data/views' / v / 'gt'
        fp = harness_preds(EMPTY, S / 'captures' / bnode / v, v, S / 'preds' / R / f'Fprev_{v}', log)
        c1 = harness_preds(EMPTY, S / 'captures' / c1node / v, v, S / 'preds' / R / f'C1k_{v}', log)
        per[ds] = compare(per_sequence(gt, fp), per_sequence(gt, c1), n_boot=500)
    g = combine(per)
    return dict(better=bool(g['S_lo'] > 0), S=g['S'], S_lo=g['S_lo'], regressed=g['regressed'],
                rel={ds: q['rel'] for ds, q in per.items()})


def pick_ckpt(R, run_dir, sel, bnode, bckpt, dsets, gpus, log, split='dev'):
    """Loop B trains P['steps'] (2000) steps and keeps the best of the checkpoints P['ckpt_select'] on
    `split` (P['ckpt_select_split']: val, because dev FAcc can miss border false positives that show on val;
    the GB gate then looks at the same val, so its verdict is optimistic and the sealed test stays the unbiased check;
    never test). Each candidate is merged (models/<R>_ck<step>) and captured on the `split` views; its network
    alone (empty harness) is compared with F_(k-1) by the gate's rule (eval.score: S on HOT3D, blocking regressions on
    HOT3D). Chosen: the best S among the candidates without a blocking regression, else the best S.
    Returns (step, {step: summary})."""
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    from eval.score import combine, compare, per_sequence
    nodes = {s: f'{R}_ck{s}' for s in sel}
    for s in sel:
        md = S / 'models' / nodes[s]
        if not (md / 'MERGED_OK').is_file():
            merge_retry(str(run_dir / f'step{s}.pt'), str(md), gpus, log)
            (md / 'MERGED_OK').touch()
    # capture/run_ace.py is CPU-bound: 2 captures at a time (on a shared host other jobs may hold ~35 GB of every card;
    # a capture shard takes ~15 GB), 10 threads per process (as rounds/c1k.py)
    keep = {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
    os.environ.update({k: '10' for k in keep})
    try:
        jobs = [(bnode, bckpt, f'{ds}_{split}') for ds in dsets] + [(nodes[s], str(S / 'models' / nodes[s]), f'{ds}_{split}')
                                                                    for s in sel for ds in dsets]
        with ThreadPoolExecutor(2) as ex:
            list(ex.map(lambda j: capture(*j, gpus, log), jobs))
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    gt = {ds: S / 'data/views' / f'{ds}_{split}' / 'gt' for ds in dsets}
    ref = {ds: per_sequence(gt[ds], harness_preds(EMPTY, S / 'captures' / bnode / f'{ds}_{split}', f'{ds}_{split}',
                                                  S / 'preds' / R / f'Fprev_{ds}_{split}', log)) for ds in dsets}
    res = {}
    for s in sel:
        per = {ds: compare(ref[ds], per_sequence(gt[ds], harness_preds(EMPTY, S / 'captures' / nodes[s] / f'{ds}_{split}',
                                                                       f'{ds}_{split}', S / 'preds' / R / f'ck{s}_{ds}_{split}', log)),
                           n_boot=500) for ds in dsets}
        g = combine(per)
        res[s] = dict(S=g['S'], S_lo=g['S_lo'], regressed=g['regressed'], regressed_info=g['regressed_info'],
                      rel={ds: p['rel'] for ds, p in per.items()})
        say(f"[{R}] Loop B checkpoint step {s}: {split} F vs F_(k-1) S {100 * g['S']:+.2f}% (lo {100 * g['S_lo']:+.2f}%), "
            f"blocking regressions {g['regressed'] or 'none'}")
    ok = [s for s in sel if not res[s]['regressed']] or sel
    # S caps every column at +-5 %, so strong candidates can tie at +5 %: ties go to the larger mean uncapped HOT3D gain
    raw = lambda s: float(np.mean(list(res[s]['rel'].get('hot3d', next(iter(res[s]['rel'].values()))).values())))  # noqa: E731
    best = max(ok, key=lambda s: (round(res[s]['S'], 4), raw(s)))
    say(f'[{R}] Loop B checkpoint chosen on {split}: step {best}' + ('' if best in ok and not res[best]['regressed']
                                                               else ' (every candidate had a blocking regression)'))
    return best, res


def main():
    # temporary files of this round's own processes (captures, training, merges) go to rsi_paths.TMP (a small root disk
    # with /tmp on it filled up twice; the sandbox passes its own environment)
    tmpd = Path(rsi_paths.TMP); tmpd.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('TMPDIR', str(tmpd))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--round', required=True)
    ap.add_argument('--gpus', required=True)
    ap.add_argument('--datasets', default='hot3d,arctic')
    ap.add_argument('--stages', default='capture,base,loopA,teacher,loopB,revalidate,gate,sealed,controls,loopC')
    ap.add_argument('--loopc-minutes', type=int, default=60)
    ap.add_argument('--integrate-minutes', type=int, default=60, help='merge agent session (stage integrate)')
    ap.add_argument('--retry-loopB', action='store_true', help='first set the round back to before its Loop B '
                    '(reset_for_loopB_retry; Loop C decision), then run --stages')
    ap.add_argument('--c1', action='store_true', help='also run control C1 (GT-only fine-tuning of θ0, same data / steps)')
    ap.add_argument('--agents', default='roster')
    ap.add_argument('--loopa-hours', type=float, default=4)
    ap.add_argument('--budget', type=int, default=10)
    ap.add_argument('--submit-cutoff-min', type=int, default=60, help='Loop A: the gate closes this long before the end, '
                    'so the queue drains before the agents stop')
    ap.add_argument('--steps', type=int, default=None, help='override (smoke); default: rounds/LOOPB_PARAMS.json')
    ap.add_argument('--w-teacher', type=float, default=None, help='override (C1 control = 0)')
    ap.add_argument('--train-seqs', default='', help='Loop B subset (smoke tests)')
    ap.add_argument('--train-split', default='train', help="split name of Loop B's training views (smoke: smoketrain)")
    ap.add_argument('--test-split', default='test', help='split name of the sealed views (smoke: smoketest)')
    a = ap.parse_args()
    __import__("os").umask(0o022)          # new trees / prediction dirs must be readable inside the sandbox
    STATE.mkdir(parents=True, exist_ok=True)
    bucket._ensure()                       # exists before any agent mounts it
    R, gpus, dsets, stages = a.round, a.gpus.split(','), a.datasets.split(','), a.stages.split(',')
    rd = S / 'rounds' / R; rd.mkdir(parents=True, exist_ok=True); log = rd / 'run.log'
    if a.retry_loopB:                      # before round.json is read: the reset edits it
        reset_for_loopB_retry(R)
    stp = rd / 'round.json'
    st = json.loads(stp.read_text()) if stp.is_file() else {}
    if 'base' not in st:
        arch = STATE / 'ARCHIVE.json'
        last = json.loads(arch.read_text())[-1] if arch.is_file() else dict(node='00', ckpt='release', tree=str(EMPTY))
        st['base'] = last; stp.write_text(json.dumps(st, indent=1))
    base = st['base']; bnode, bckpt, btree = base['node'], base['ckpt'], Path(base['tree'])
    say(f'[{R}] start; base node {bnode} (ckpt {bckpt}, tree {btree.name}); stages {stages}')
    if 'plan_in' not in st and any(x in stages for x in ('base', 'loopA', 'teacher', 'loopB')):
        # Loop C's latest plan (from an earlier round) takes effect now (not in a capture-only call, which can run
        # while the previous round's Loop C is still writing it)
        pl = STATE / 'PLAN_LATEST.json'
        plan = json.loads(pl.read_text()) if pl.is_file() else None
        st['plan_in'] = plan['dir'] if plan and plan.get('round') != R else None
        if st['plan_in']:
            shutil.copy(Path(st['plan_in']) / 'loopB_params.json', STATE / 'LOOPB_PARAMS.json')
            rec = json.loads((Path(st['plan_in']) / 'record.json').read_text())
            bucket.add('plan_applied', R, f'{R}_plan_applied', f"Loop C plan of {plan['round']} in force: {rec['ideas']} idea "
                       f"cards, focus for {sorted(rec['focus'])}, Loop B changes {rec['loopB_changes']}",
                       dict(loopB_changes=rec['loopB_changes'], focus=rec['focus']))
        stp.write_text(json.dumps(st, indent=1))
    from loopC.run_loopC import loopb_params
    P = loopb_params()
    if a.steps is not None:
        P['steps'] = a.steps
    if a.w_teacher is not None:
        P['w_teacher'] = a.w_teacher
    st['loopB_params'] = P; stp.write_text(json.dumps(st, indent=1))

    if 'capture' in stages:
        for sp in ('dev', 'val', a.train_split, a.test_split):
            for v in views(dsets, sp):
                capture(bnode, bckpt, v, gpus, log)
        say(f'[{R}] captures of {bnode} ready')
    cap = lambda node, v: S / 'captures' / node / CAPTURE_ALIAS.get(v, v)           # noqa: E731

    # keep everything the process produced: the previous round's per-segment arrays are MOVED (not deleted) to the archive by rounds/archive_arrays.py, in the background; kept local: sealed/, preds/<prev>/Nk_*,
    # Fk_*, every tree / json / note, models and captures. (cleanup_round, which deleted, is no longer called.)
    if 'base' in stages and not st.get('archived_prev'):
        import re
        m = re.fullmatch(r'(.*?)(\d+)', R)
        prev = f'{m.group(1)}{int(m.group(2)) - 1}' if m and int(m.group(2)) > 1 else None
        if prev and (S / 'rounds' / prev / 'round.json').is_file() and (S / 'loopA_runs' / prev / 'STOP').is_file():
            roots = [f'loopA_runs/{prev}', f'loopB/teacher_pred/{prev}', f'rounds/{prev}/merge_work',
                     f'rounds/{prev}/revalidate_work', f'rounds/{prev}/loopC', f'preds/{prev}']
            cmd = [PY, str(S / 'rounds/archive_arrays.py'), '--name', f'{prev}_rest', '--glob', f'loopB/cache/*/teacher_{prev}.npz',
                   '--keep', f'preds/{prev}/Nk_', '--keep', f'preds/{prev}/Fk_'] + sum((['--root', r] for r in roots), [])
            subprocess.Popen(cmd, cwd=str(S), stdout=open(log, 'a'), stderr=subprocess.STDOUT, start_new_session=True)
            say(f'[{R}] archiving the per-segment arrays of {prev} to the archive (background)')
            st['archived_prev'] = prev; stp.write_text(json.dumps(st, indent=1, default=float))
    if 'base' in stages:
        for sp in ('dev', 'val'):
            for v in views(dsets, sp):
                harness_preds(btree, cap(bnode, v), v, S / 'preds' / R / f'base_{v}', log)
        say(f'[{R}] base predictions (N_{{k-1}}) on dev / val ready')

    # teacher-on-C1 Loop B: C1_k = θ_(k-1) + GT-only LoRA, trained and captured in the background (rounds/c1k.py); the
    # Loop B stage waits for it. Started when Loop A is over (during Loop A the agents' experiments can fill every card),
    # so it runs next to the teacher / integrate stages.
    c1k_node = f'{R}_C1k'; c1k_m = S / 'models' / c1k_node

    def start_c1k():
        if P.get('teacher_on_c1') and 'loopB' in stages and not (S / 'models' / f'{R}_theta' / 'MERGED_OK').is_file() \
                and not (c1k_m / 'CAPTURED_OK').is_file() and not (rd / 'c1k.pid').is_file():
            pr = subprocess.Popen([PY, str(S / 'rounds/c1k.py'), '--round', R, '--base-ckpt', bckpt, '--gpus', ','.join(gpus),
                                   '--datasets', ','.join(dsets), '--train-split', a.train_split], cwd=str(S),
                                  stdout=open(rd / 'c1k.out', 'a'), stderr=subprocess.STDOUT, start_new_session=True,
                                  env=dict(os.environ, PYTHONPATH=str(S)))
            (rd / 'c1k.pid').write_text(str(pr.pid))
            say(f'[{R}] C1_k launched in the background (pid {pr.pid}): θ_(k-1) + GT only, then raw outputs for the teacher')

    if 'loopA' in stages:
        dj = {ds: dict(dev_view=str(S / f'data/views/{ds}_dev'), dev_cache=str(cap(bnode, f'{ds}_dev')),
                       val_view=str(S / f'data/views/{ds}_val'), val_cache=str(cap(bnode, f'{ds}_val')),
                       base_dev_pred=str(S / 'preds' / R / f'base_{ds}_dev'), base_val_pred=str(S / 'preds' / R / f'base_{ds}_val'))
              for ds in dsets}
        (rd / 'datasets.json').write_text(json.dumps(dj, indent=1))
        import roster
        agents = roster.SMOKE if a.agents == 'smoke' else roster.AGENTS if a.agents == 'roster' else a.agents
        exp = S / 'loopA_runs' / R
        if not (exp / 'STOP').is_file():
            say(f'[{R}] Loop A: {len(agents.split(","))} agents, {a.loopa_hours} h, budget {a.budget}')
            run([PY, str(S / 'loopA/driver.py'), '--exp', str(exp), '--agents', agents, '--datasets', str(rd / 'datasets.json'),
                 '--base-tree', str(btree), '--hours', str(a.loopa_hours), '--budget', str(a.budget),
                 '--submit-cutoff-min', str(a.submit_cutoff_min)]
                + (['--plan', st['plan_in']] if st.get('plan_in') else []), log)
        say(f'[{R}] Loop A done: {len(json.loads((exp / "state.json").read_text()).get("accepted", []))} accepted submissions')
        if (exp / 'STOP').is_file() and not st.get('loopA_archived'):
            # the agents are gone; the merge replays every tree itself and reads only gate/<req>/score.json, Loop C reads
            # notes / trees / the bucket. Loop A's per-segment arrays (~250 GB a round) are MOVED to the archive (nothing
            # is deleted), in the background
            subprocess.Popen([PY, str(S / 'rounds/archive_arrays.py'), '--name', f'{R}_loopA', '--root', f'loopA_runs/{R}'],
                             cwd=str(S), stdout=open(log, 'a'), stderr=subprocess.STDOUT, start_new_session=True)
            say(f'[{R}] archiving Loop A arrays to the archive (background)')
            st['loopA_archived'] = True; stp.write_text(json.dumps(st, indent=1, default=float))
        if not st.get('bucket_agent_logs'):        # each agent's reasoning record: notes + session transcripts
            for ag in sorted((exp / 'agents').glob('*')):
                files = {'notes.md': ag / 'notes.md'}
                if (ag / 'lit_log.jsonl').is_file():
                    files['lit_log.jsonl'] = ag / 'lit_log.jsonl'
                for t in sorted((exp / 'logs').glob(f'{ag.name}_session*.jsonl')):
                    files[t.name + '.gz'] = t
                note = (ag / 'notes.md').read_text(errors='replace') if (ag / 'notes.md').is_file() else ''
                bucket.add('agent_log', R, f'{R}_{ag.name}_log', f'{ag.name}: notes ({len(note)} chars) and '
                           f'{len(files) - 1} session transcripts', files=files)
            st['bucket_agent_logs'] = True; stp.write_text(json.dumps(st, indent=1))

    start_c1k()                                       # after Loop A (see above)
    if 'teacher' in stages and not (rd / 'teacher_tree').is_dir():
        exp = S / 'loopA_runs' / R                     # greedy merge of the accepted patches (dev), one val look at the end
        run([PY, str(S / 'rounds/greedy_merge.py'), '--exp', str(exp), '--datasets', str(rd / 'datasets.json'),
             '--base-tree', str(btree), '--out', str(rd / 'teacher_tree')], log)
        m = json.loads((rd / 'merge.json').read_text())
        st['teacher'] = dict(pick=m.get('teacher'), final=m.get('final'), modules=m.get('merged_modules')); stp.write_text(json.dumps(st, indent=1, default=float))
        say(f"[{R}] teacher H_k* = {m.get('teacher')} ({m.get('accepted')} accepted patches; modules {m.get('merged_modules')})")
        steps_txt = '; '.join(f"+{x.get('tree')} {x.get('modules', '')} dev S {x.get('dev_S')} -> {'kept' if x.get('kept') else 'dropped'}"
                              f"{' (' + x['why'] + ')' if x.get('why') else ''}" for x in m.get('steps', []))
        bucket.add('merge', R, f'{R}_merge', f"teacher = {m.get('teacher')} (merged vs best single, one val look); "
                   f"modules {m.get('merged_modules')}; greedy steps on dev: {steps_txt or 'none'}",
                   dict(teacher=m.get('teacher'), modules=m.get('merged_modules'), accepted=m.get('accepted')),
                   files={'teacher_tree.patch': bucket.tree_diff(btree, rd / 'teacher_tree')})
    tree_k = rd / 'teacher_tree'

    # merge agent (rounds/integrate.py): an Opus session integrates the accepted patches on DEV; its chain
    # replaces the greedy teacher only if it beats it on dev. Never after Loop B (the network was distilled from the teacher)
    if 'integrate' in stages and tree_k.is_dir() and not (rd / 'integrate' / 'result.json').is_file() \
            and not (S / 'models' / f'{R}_theta' / 'MERGED_OK').is_file():
        run([PY, str(S / 'rounds/integrate.py'), '--round', R, '--minutes', str(a.integrate_minutes)], log, check=False)
        res = json.loads((rd / 'integrate' / 'result.json').read_text()) if (rd / 'integrate' / 'result.json').is_file() else {}
        if res.get('chosen') == 'integrated':
            m = json.loads((rd / 'merge.json').read_text())
            st['teacher'] = dict(pick='integrated', final=m.get('final'),
                                 modules=[x['name'] for x in json.loads((tree_k / 'chain.json').read_text())['modules']])
            stp.write_text(json.dumps(st, indent=1, default=float))
        dv = res.get('dev_vs_greedy') or {}
        say(f"[{R}] merge agent: teacher = {res.get('chosen', 'greedy (agent failed)')}"
            + (f"; dev vs greedy S {100 * dv['S']:+.2f}% (lo {100 * dv['S_lo']:+.2f}%), regressed {dv.get('regressed')}" if dv else '')
            + (f"; {res['why']}" if res.get('why') else ''))

    new = f'{R}_theta'; mdir = S / 'models' / new
    if 'loopB' in stages and not (mdir / 'MERGED_OK').is_file():
        for v in views(dsets, a.train_split) + (list(P.get('extra_train_views', [])) if P.get('teacher_on_c1') else []):
            # VAE latents + per-frame GT, model independent, once per sequence
            todo = [p for p in (S / 'data/views' / v / 'segments').glob('*.json')
                    if not ((S / 'loopB/cache' / p.stem / 'latents.pt').is_file() and (S / 'loopB/cache' / p.stem / 'gt3d.npz').is_file())]
            if todo:
                say(f'[{R}] Loop B prep: {len(todo)} {v} sequences to encode')
                for attempt in range(3):                # prep skips cached sequences, so a retry only redoes the failures
                    use = free_gpus(gpus, 20000) or [one_gpu(gpus)]
                    procs = [subprocess.Popen([PY, '-m', 'loopB.prep', '--view', str(S / 'data/views' / v), '--gpu', g,
                                               '--shard', f'{i}/{len(use)}'], stdout=open(log, 'a'), stderr=subprocess.STDOUT,
                                              cwd=str(S), env=dict(os.environ, PYTHONPATH=str(S))) for i, g in enumerate(use)]
                    if not any(p.wait() for p in procs):
                        break
                    say(f'[{R}] Loop B prep: a shard failed on {use}, attempt {attempt + 1}/3'); time.sleep(60)
                else:
                    raise SystemExit(f'FAILED Loop B prep on {v} (see {log})')
        extra_views = list(P.get('extra_train_views', [])) if P.get('teacher_on_c1') else []
        if P.get('teacher_on_c1'):
            # the teacher must be ahead of the student, so it is built on C1_k
            # (θ_(k-1) + GT only): H_k pruned on C1_k's dev outputs (one pass), then C1_k + H_k' labels every training
            # view; the student starts from C1_k
            if not (c1k_m / 'CAPTURED_OK').is_file():
                pidf = rd / 'c1k.pid'
                pid = int(pidf.read_text()) if pidf.is_file() else None
                while pid and Path(f'/proc/{pid}').is_dir() and not (c1k_m / 'CAPTURED_OK').is_file():
                    time.sleep(60)
                if not (c1k_m / 'CAPTURED_OK').is_file():           # not launched or it died: run it here
                    run([PY, str(S / 'rounds/c1k.py'), '--round', R, '--base-ckpt', bckpt, '--gpus', ','.join(gpus),
                         '--datasets', ','.join(dsets), '--train-split', a.train_split], log)
            if 'c1_check' not in st:
                st['c1_check'] = c1_vs_prev(R, bnode, c1k_node, dsets, log); stp.write_text(json.dumps(st, indent=1, default=float))
                ck = st['c1_check']
                say(f"[{R}] C1_k vs θ_(k-1) on dev (networks alone): S {100 * ck['S']:+.2f}% (lo {100 * ck['S_lo']:+.2f}%), "
                    f"blocking {ck['regressed'] or 'none'} -> teacher on " + ('C1_k' if ck['better'] else 'θ_(k-1) (C1_k not ahead)'))
        use_c1 = bool(P.get('teacher_on_c1')) and st.get('c1_check', {}).get('better', True)
        if use_c1:
            tdir = rd / 'teacher_c1'; ttree = tdir / 'node_tree'
            if not ttree.is_dir():
                tdir.mkdir(parents=True, exist_ok=True)
                shutil.copy(rd / 'datasets.json', tdir / 'datasets.json')
                run([PY, str(S / 'rounds/revalidate.py'), '--tree', str(tree_k), '--datasets', str(tdir / 'datasets.json'),
                     '--node', c1k_node, '--out', str(ttree), '--one-shot'], log)
                say(f"[{R}] teacher on C1_k: H_k pruned on C1_k dev -> {json.loads((tdir / 'revalidate.json').read_text())['kept']}")
            for v in views(dsets, a.train_split) + extra_views:
                run([PY, '-m', 'loopB.targets', '--tree', str(ttree), '--cache', str(cap(c1k_node, v)), '--view',
                     str(S / 'data/views' / v), '--name', R], log)
            ck_args = ['--base-ckpt', str(c1k_m / 'ace_ego_hand_k.pt'), '--base-opt', str(c1k_m / 'ace_ego_hand_k.yml')]
        elif P.get('teacher_on_c1'):
            # C1_k not ahead of θ_(k-1): teacher = θ_(k-1) + H_k (H_k was validated on θ_(k-1) already),
            # pseudo-labels on every training view incl. the extra no-GT views; the student starts from θ_(k-1)
            for v in views(dsets, a.train_split) + extra_views:
                capture(bnode, bckpt, v, gpus, log)
                run([PY, '-m', 'loopB.targets', '--tree', str(tree_k), '--cache', str(cap(bnode, v)), '--view',
                     str(S / 'data/views' / v), '--name', R], log)
            ck_args = [] if bckpt == 'release' else ['--base-ckpt', f'{bckpt}/ace_ego_hand_k.pt', '--base-opt', f'{bckpt}/ace_ego_hand_k.yml']
        else:
            for v in views(dsets, a.train_split):
                run([PY, '-m', 'loopB.targets', '--tree', str(tree_k), '--cache', str(cap(bnode, v)), '--view', str(S / 'data/views' / v),
                     '--name', R], log)
            ck_args = [] if bckpt == 'release' else ['--base-ckpt', f'{bckpt}/ace_ego_hand_k.pt', '--base-opt', f'{bckpt}/ace_ego_hand_k.yml']
        run_dir = S / 'loopB/runs' / R
        steps = int(P['steps'])
        if not (run_dir / f'step{steps}.pt').is_file():
            say(f'[{R}] Loop B: train {new} on {len(gpus)} cards, params {P}')
            extra = ['--seqs', a.train_seqs] if a.train_seqs else []
            lists = ','.join([str(S / 'data/splits' / f'{ds}_{a.train_split}.txt') for ds in dsets] +
                             [str(S / 'data/splits' / f'{v}.txt') for v in extra_views])
            mix = ','.join(f'{d}={x}' for d, x in P['mix'].items() if d in dsets)
            used = run_torch(['--teacher', R, '--out', str(run_dir), '--steps', str(steps),
                              '--save_every', str(P.get('save_every', 500)), *P.get('loopb_flags', []),
                              '--w_teacher', str(P['w_teacher']), '--w_gt', str(P['w_gt']), '--w_exists', str(P['w_exists']),
                              '--w_rot', str(P['w_rot']), '--lr', str(P['lr']), '--rank', str(P['rank']), '--train-list', lists,
                              '--mix', mix, '--teacher_terms', P.get('teacher_terms', 'articulation'), *ck_args, *extra], gpus, log, 29641)
            st['loopB_cards'] = used; stp.write_text(json.dumps(st, indent=1, default=float))
        best = steps
        if P.get('ckpt_select'):
            if 'loopB_ckpt' not in st:
                split = P.get('ckpt_select_split', 'dev')
                best, cres = pick_ckpt(R, run_dir, [int(x) for x in P['ckpt_select']], bnode, bckpt, dsets, gpus, log, split)
                st['loopB_ckpt'] = dict(chosen=best, split=split, candidates=cres); stp.write_text(json.dumps(st, indent=1, default=float))
            best = int(st['loopB_ckpt']['chosen'])
        # as in the C1 check, a network that is not ahead is not used: if no checkpoint is significantly ahead
        # of θ_(k-1) (CI90 lower bound of S > 0), θ_k := θ_(k-1) (hard links) and the round's harness is H_k* as validated
        # on θ_(k-1) in the teacher stage, so the revalidation is skipped
        cands = (st.get('loopB_ckpt') or {}).get('candidates') or {}
        if cands and bckpt != 'release' and not any(float(c.get('S_lo', -1)) > 0 for c in cands.values()):
            mdir.mkdir(parents=True, exist_ok=True)
            subprocess.run(['cp', '-aln', f'{bckpt}/.', str(mdir)], check=True)
            for vd in (S / 'captures' / bnode).iterdir():
                dst = S / 'captures' / new / vd.name
                if vd.is_dir() and not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True); subprocess.run(['cp', '-al', str(vd), str(dst)], check=True)
            if not (rd / 'node_tree').is_dir():
                shutil.copytree(tree_k, rd / 'node_tree')
            mods = [m['name'] for m in json.loads((rd / 'node_tree' / 'chain.json').read_text())['modules']]
            note = 'θ_k := θ_(k-1): no Loop B checkpoint significantly ahead on ' + str(st['loopB_ckpt'].get('split', 'dev'))
            st['loopB_final'] = dict(kept_prev=True, model=f'{bckpt} (hard-linked)', note=note)
            st['revalidate'] = dict(kept=mods, dropped=[], note='skipped: H_k* was validated on θ_(k-1) in the teacher stage')
            (rd / 'revalidate.json').write_text(json.dumps(dict(kept=mods, dropped=[], note=st['revalidate']['note']), indent=1))
            stp.write_text(json.dumps(st, indent=1, default=float))
            (mdir / 'MERGED_OK').touch()
            say(f'[{R}] Loop B: no checkpoint ahead of θ_(k-1) -> {new} = θ_(k-1) ({bnode}), harness = H_k* (revalidation skipped)')
            return_after_loopB = True
        else:
            return_after_loopB = False
        if not return_after_loopB:
            merge_retry(str(run_dir / f'step{best}.pt'), str(mdir), gpus, log)
        if best != steps or P.get('ckpt_select'):
            for ds in dsets:                          # the chosen checkpoint's outputs are θ_k's: no second capture
                for sp in ('dev', 'val'):
                    src, dst = S / 'captures' / f'{R}_ck{best}' / f'{ds}_{sp}', S / 'captures' / new / f'{ds}_{sp}'
                    if not src.is_dir() or dst.exists():
                        continue
                    dst.parent.mkdir(parents=True, exist_ok=True); dst.symlink_to(src)
        (mdir / 'MERGED_OK').touch()
        if (st.get('loopB_final') or {}).get('kept_prev'):
            say(f'[{R}] Loop B done: {new} = θ_(k-1) (kept: no checkpoint ahead; the best candidate was step {best} of {steps})')
        else:
            say(f'[{R}] Loop B done: {new} = step {best} of {steps}, merged and verified')
        tl = (run_dir / 'train_log.jsonl').read_text().splitlines() if (run_dir / 'train_log.jsonl').is_file() else []
        bucket.add('loopB', R, f'{R}_loopB', f'trained {new}: {P}; last log {tl[-1] if tl else "-"}', dict(params=P),
                   files={'train_log.jsonl': run_dir / 'train_log.jsonl'})

    node_tree = rd / 'node_tree'
    if 'revalidate' in stages and not node_tree.is_dir() and (mdir / 'MERGED_OK').is_file():
        for ds in dsets:
            capture(new, str(mdir), f'{ds}_dev', gpus, log)
        run([PY, str(S / 'rounds/revalidate.py'), '--tree', str(tree_k), '--datasets', str(rd / 'datasets.json'),
             '--node', new, '--out', str(node_tree), *(['--one-shot'] if os.environ.get('RSI_REVALIDATE_ONESHOT') else [])], log)
        rv = json.loads((rd / 'revalidate.json').read_text())
        st['revalidate'] = dict(kept=rv['kept'], dropped=rv['dropped']); stp.write_text(json.dumps(st, indent=1, default=float))
        say(f"[{R}] H_k* re-validated on θ_k (dev): kept {rv['kept']}, dropped {rv['dropped']}")
        bucket.add('revalidate', R, f'{R}_revalidate', f"H_k* re-checked on θ_k's dev outputs: kept {rv['kept']}, dropped "
                   f"{rv['dropped']} (a module is kept only if removing it makes dev significantly worse)",
                   dict(kept=rv['kept'], dropped=rv['dropped'], steps=rv['steps']))
    tree_n = node_tree if node_tree.is_dir() else tree_k

    if 'gate' in stages and 'verdict' not in st:
        from eval.score import compare, per_sequence
        res = {}
        for ds in dsets:
            v = f'{ds}_val'
            capture(new, str(mdir), v, gpus, log)
            gt = S / 'data/views' / v / 'gt'
            nk = harness_preds(tree_n, cap(new, v), v, S / 'preds' / R / f'Nk_{v}', log)
            fk = harness_preds(EMPTY, cap(new, v), v, S / 'preds' / R / f'Fk_{v}', log)
            fprev = harness_preds(EMPTY, cap(bnode, v), v, S / 'preds' / R / f'Fprev_{v}', log)
            res[ds] = dict(system=compare(per_sequence(gt, S / 'preds' / R / f'base_{v}'), per_sequence(gt, nk)),
                           network=compare(per_sequence(gt, fprev), per_sequence(gt, fk)))
        from eval.score import combine           # S and blocking regressions on HOT3D only
        g = combine({ds: r['system'] for ds, r in res.items()})
        Ssys, lo, worst, regressed = g['S'], g['S_lo'], g['worst'], g['regressed']
        ok = lo > 0 and Ssys >= 0.01 and not regressed               # significant-regression rule (eval/score.py)
        # manual pass for the rounds listed in RSI_GB_DETECTION_PASS (an authors' decision). A θ_k may trade HOT3D
        # detection (border FPs: off-screen hands predicted ~10 % too far, pulled into the frame by perspective) for the
        # pose / smoothness columns. Only when S clears the bar and every blocking regression is a detection column.
        override = None
        if not ok and R in os.environ.get('RSI_GB_DETECTION_PASS', '').split(',') and lo > 0 and Ssys >= 0.01 \
                and regressed and all(set(c) <= {'FAcc', 'Recall', 'F1'} for c in regressed.values()):
            ok, override = True, f'manual pass: detection regression {regressed} accepted for {R}'
        # rounds listed in RSI_GB_FORCE_ACCEPT are accepted by the authors' decision (used once, for R1; results/README.md):
        # the val comparison is computed and recorded, the round is accepted whatever it says. Only a fault brake: a system worse than its
        # base overall (S < 0) is not passed on, since that means a broken model rather than a trade-off.
        if not ok and R in os.environ.get('RSI_GB_FORCE_ACCEPT', '').split(',') and Ssys >= 0:
            ok, override = True, f'{R} accepted without the gate (authors\' decision; val recorded only; regressed {regressed or "none"})'
        st['verdict'] = dict(accepted=ok, S_system=Ssys, S_lo=lo, worst=worst, regressed=regressed, override=override,
                             detail=res); stp.write_text(json.dumps(st, indent=1, default=float))
        say(f'[{R}] GB {"ACCEPTED" if ok else "rejected"}: N_k vs N_{{k-1}} S {100 * Ssys:+.2f}% (lo {100 * lo:+.2f}%, worst '
            f'{100 * worst:+.2f}%); F_k vs F_{{k-1}} S ' +
            ', '.join(f'{ds} {100 * r["network"]["S"]:+.2f}%' for ds, r in res.items()) + (f' [{override}]' if override else ''))
        arch = STATE / 'ARCHIVE.json'
        hist = json.loads(arch.read_text()) if arch.is_file() else [dict(node='00', ckpt='release', tree=str(EMPTY))]
        nxt = f'{new} with node_tree'
        if ok:
            hist.append(dict(node=new, ckpt=str(mdir), tree=str(tree_n), round=R, **({'override': override} if override else {})))
            arch.write_text(json.dumps(hist, indent=1))
        else:
            # when GB fails, θ_k is discarded but the round's teacher harness H_k* is kept, provided its
            # one val look in the teacher stage (H_k* on θ_{k-1} vs N_{k-1}) clears the GA bar
            m = json.loads((rd / 'merge.json').read_text()) if (rd / 'merge.json').is_file() else {}
            fin = (m.get('final') or {}).get(m.get('teacher') or '', {})
            if m.get('teacher') in ('merged', 'best_single', 'integrated') and fin.get('S', 0) >= 0.003 and fin.get('lo', -1) > 0 \
                    and not fin.get('regressed'):
                hist.append(dict(node=bnode, ckpt=bckpt, tree=str(tree_k), round=R,
                                 kind='harness only: GB failed, θ_k discarded, H_k* kept'))
                arch.write_text(json.dumps(hist, indent=1))
                nxt = f'{bnode} (θ_k-1) with H_k*'
            else:
                nxt = f'{bnode} with H_k-1 (H_k* did not clear the GA bar on val)'
        st['next_base'] = nxt; stp.write_text(json.dumps(st, indent=1, default=float))
        bucket.add('gate_B', R, f'{R}_GB', f"round {R}: N_k = (θ_k, H_k*) vs N_k-1 -> {'ACCEPTED' if ok else 'rejected'}; "
                   f"next round starts from {nxt}" + (f" ({override}; from the next round on, detection must not regress vs this N_k)"
                   if override else ''), dict(accepted=ok, next_base=nxt, override=override))

    if 'sealed' in stages and 'sealed' not in st:
        from eval.score import per_sequence, pooled
        sealed = {}
        for ds, v in sealed_views(dsets, a.test_split):
            capture(new, str(mdir), v, gpus, log)
            gt = S / 'data/views' / v / 'gt'
            for name, tree in (('N_k', tree_n), ('F_k', EMPTY)):
                out = harness_preds(tree, cap(new, v), v, S / 'sealed' / R / f'{name}_{v}', log)
                sealed[f'{name}/{v}'] = pooled(per_sequence(gt, out).values())
        (S / 'sealed').mkdir(exist_ok=True)
        (S / 'sealed' / f'{R}.json').write_text(json.dumps(sealed, indent=1, default=float))
        st['sealed'] = 'recorded (not read by any decision)'; stp.write_text(json.dumps(st, indent=1))
        say(f'[{R}] sealed test scores recorded')
    if 'sealed' in stages and 'fps' not in st and (mdir / 'MERGED_OK').is_file():
        fv = 'arctic_test' if (S / 'captures' / new / 'arctic_test').is_dir() else f'{dsets[0]}_{a.test_split}'
        out = rd / 'fps.json'
        run([PY, str(S / 'rounds/fps.py'), '--ckpt', str(mdir), '--tree', str(tree_n), '--view', fv, '--node', new,
             '--gpu', one_gpu(gpus), '--out', str(out)], log, check=False)
        if out.is_file():
            st['fps'] = json.loads(out.read_text()); stp.write_text(json.dumps(st, indent=1, default=float))
            say(f"[{R}] FPS (N_k, {st['fps']['gpu']}): network {st['fps']['fps_network']:.1f}, with harness {st['fps']['fps_total']:.1f}")

    if 'controls' in stages and 'controls' not in st:
        # ablations, recorded only: C2 = H_k* on θ0's raw outputs (no distillation), every
        # round; C1 = θ0 fine-tuned on GT only (no teacher, same data / steps / LoRA), with --c1
        from eval.score import compare, per_sequence, pooled
        ctl, sealed = {}, json.loads((S / 'sealed' / f'{R}.json').read_text()) if (S / 'sealed' / f'{R}.json').is_file() else {}
        for ds in dsets:
            v = f'{ds}_val'; gt = S / 'data/views' / v / 'gt'
            p00 = harness_preds(EMPTY, cap('00', v), v, S / 'preds' / '00' / v, log)
            c2 = harness_preds(tree_k, cap('00', v), v, S / 'preds' / R / f'C2_{v}', log)
            nk = S / 'preds' / R / f'Nk_{v}'
            ctl[ds] = dict(C2_vs_00=compare(per_sequence(gt, p00), per_sequence(gt, c2), n_boot=500),
                           Nk_vs_C2=compare(per_sequence(gt, c2), per_sequence(gt, nk), n_boot=500) if nk.is_dir() else None)
        for ds, v in sealed_views(dsets, a.test_split):
            out = harness_preds(tree_k, cap('00', v), v, S / 'sealed' / R / f'C2_{v}', log)
            sealed[f'C2/{v}'] = pooled(per_sequence(S / 'data/views' / v / 'gt', out).values())
        if a.c1:
            c1m, c1run = S / 'models' / f'{R}_C1', S / 'loopB/runs' / f'{R}_C1'
            if not (c1m / 'MERGED_OK').is_file():
                lists = ','.join(str(S / 'data/splits' / f'{ds}_{a.train_split}.txt') for ds in dsets)
                mix = ','.join(f'{d}={x}' for d, x in P['mix'].items() if d in dsets)
                say(f'[{R}] control C1: θ0 + GT-only LoRA, {P["steps"]} steps')
                run_torch(['--out', str(c1run), '--steps', str(P['steps']), '--w_teacher', '0', '--w_gt', str(P['w_gt']),
                           '--w_exists', str(P['w_exists']), '--w_rot', str(P['w_rot']), '--lr', str(P['lr']), '--rank',
                           str(P['rank']), '--train-list', lists, '--mix', mix], gpus, log, 29642)
                merge_retry(str(c1run / f"step{P['steps']}.pt"), str(c1m), gpus, log)
                (c1m / 'MERGED_OK').touch()
            for ds in dsets:
                v = f'{ds}_val'; gt = S / 'data/views' / v / 'gt'
                capture(f'{R}_C1', str(c1m), v, gpus, log)
                c1 = harness_preds(EMPTY, cap(f'{R}_C1', v), v, S / 'preds' / R / f'C1_{v}', log)
                fk = S / 'preds' / R / f'Fk_{v}'
                ctl[ds]['Fk_vs_C1'] = compare(per_sequence(gt, c1), per_sequence(gt, fk), n_boot=500) if fk.is_dir() else None
            for ds, v in sealed_views(dsets, a.test_split):
                capture(f'{R}_C1', str(c1m), v, gpus, log)
                out = harness_preds(EMPTY, cap(f'{R}_C1', v), v, S / 'sealed' / R / f'C1_{v}', log)
                sealed[f'C1/{v}'] = pooled(per_sequence(S / 'data/views' / v / 'gt', out).values())
        (S / 'sealed').mkdir(exist_ok=True)
        (S / 'sealed' / f'{R}.json').write_text(json.dumps(sealed, indent=1, default=float))
        st['controls'] = ctl; stp.write_text(json.dumps(st, indent=1, default=float))
        say(f'[{R}] controls recorded (val): ' + '; '.join(
            f"{ds}: C2 vs 00 S {100 * c['C2_vs_00']['S']:+.2f}%" + (f", N_k vs C2 S {100 * c['Nk_vs_C2']['S']:+.2f}%" if c.get('Nk_vs_C2') else '')
            + (f", F_k vs C1 S {100 * c['Fk_vs_C1']['S']:+.2f}%" if c.get('Fk_vs_C1') else '') for ds, c in ctl.items()))

    if 'loopC' in stages and not (rd / 'loopC' / 'plan_validated' / 'record.json').is_file():
        say(f'[{R}] Loop C: research lead, {a.loopc_minutes} min')
        run([PY, str(S / 'loopC/run_loopC.py'), '--round', R, '--minutes', str(a.loopc_minutes)], log)
        rec = json.loads((rd / 'loopC' / 'plan_validated' / 'record.json').read_text())
        say(f"[{R}] Loop C plan: {rec['ideas']} idea cards, focus for {len(rec['focus'])} agents, Loop B changes "
            f"{rec['loopB_changes']}, {len(rec['proposals'])} proposals for the humans (rounds/{R}/loopC/plan_validated/proposals.md)")


if __name__ == '__main__':
    main()
