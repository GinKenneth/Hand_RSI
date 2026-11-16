"""Merge agent: after the greedy merge, one Opus session integrates the round's accepted Loop A patches
into a single chain, on DEV only (rounds/brief_integrate.md; sandbox; no val/test mounted). The greedy merge stacks
near-duplicates, leaves out patches that crashed and never re-tunes; the agent removes duplicates, resolves conflicts,
adds missing ideas and re-tunes. Its chain replaces the greedy teacher only if it beats it on dev with the gate's rule
(CI90 lower bound of S > 0, no significant regression on any column / dataset). The chosen chain's val S vs the round's
base is recorded in merge.json final['integrated'] (it gates a Loop B redo, loopC/run_loopC.py check_action).

  python rounds/integrate.py --round R2 [--minutes 60]
Outputs: rounds/<R>/integrate/{ws/, session_*.jsonl, dev_*/, val_*/, result.json}; if chosen, rounds/<R>/teacher_tree is
the agent's chain and the greedy one is kept as teacher_tree_greedy.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

S = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(S), str(S / 'rounds'), str(S / 'loopA'), str(S / 'loopC')]
import bucket  # noqa: E402
import driver  # noqa: E402
import rsi_paths  # noqa: E402
import gate_service as G  # noqa: E402
from loopC import run_loopC as LC  # noqa: E402
from revalidate import DEGRADED, replay  # noqa: E402

PROMPT = ('Read TASK.md in the current directory and follow it. You have about {minutes} minutes; keep ./tree runnable '
          'at all times and write ./integrate_notes.md before the time runs out.')


def say(msg):
    with open(S / 'logs' / 'rounds.md', 'a') as f:
        f.write(f"- {time.strftime('%m-%d %H:%M')} {msg}\n")


def replay_checked(tree: Path, view: str, cache: str, out: Path, jobs: int) -> bool:
    """sandboxed replay; a crash or a degraded output (a model that could not load / run) is redone once with fewer
    workers (e.g. CUDA OOM next to another GPU job)"""
    for attempt, j in enumerate((jobs, max(4, jobs // 3))):
        if out.exists():
            shutil.rmtree(out)
        ok, log = replay(tree, Path(view), Path(cache), out, j)
        if ok and not DEGRADED.search(log):
            (out / 'DONE').touch()
            return True
        (out.parent / f'{out.name}.attempt{attempt}.log').write_text(log[-20000:])
        time.sleep(30)
    return False


def setup(R: str, rd: Path, exp: Path, ws: Path, minutes: int) -> int:
    if ws.exists():
        shutil.rmtree(ws)
    for sub in ('home', 'spool', 'accepted', 'notes'):
        (ws / sub).mkdir(parents=True, exist_ok=True)
    shutil.copytree(rd / 'teacher_tree', ws / 'tree', ignore=shutil.ignore_patterns('__pycache__'))
    st = json.loads((exp / 'state.json').read_text())
    names = [x['name'] if isinstance(x, dict) else str(x) for x in st.get('accepted', [])]
    for name in names:
        src = exp / 'archive' / name / 'tree'
        if not src.is_dir():
            continue
        dst = ws / 'accepted' / name
        shutil.copytree(src, dst / 'tree', ignore=shutil.ignore_patterns('__pycache__'))
        agent, req = name.split('_', 1)
        rq = exp / 'spool' / agent / req / 'request.json'
        if rq.is_file():
            (dst / 'note.md').write_text(str(json.loads(rq.read_text()).get('note', ''))[:20000])
        pre = exp / 'gate' / name / 'precheck.json'                  # dev only (score.json is val: not given)
        if pre.is_file():
            (dst / 'dev.json').write_text(pre.read_text())
    for ag in sorted((exp / 'agents').glob('a*')):
        if (ag / 'notes.md').is_file():
            shutil.copy(ag / 'notes.md', ws / 'notes' / f'{ag.name}_notes.md')
    m = json.loads((rd / 'merge.json').read_text()) if (rd / 'merge.json').is_file() else {}
    lines = ['# Greedy merge steps (DEV; S = composite vs the chain before the step)', '']
    for x in m.get('steps', []):
        dev = '' if x.get('dev_S') is None else f"dev S {100 * x['dev_S']:+.2f}% (CI90 lower {100 * x.get('dev_lo', 0):+.2f}%)"
        lines.append(f"- {x.get('tree')}: +{x.get('modules', '')} {dev} -> {'kept' if x.get('kept') else 'dropped'} "
                     f"{x.get('why', '')[:160]}")
    (ws / 'merge_steps.md').write_text('\n'.join(lines) + '\n')
    (ws / 'TASK.md').write_text((S / 'rounds' / 'brief_integrate.md').read_text().replace('{round}', R).replace('{root}', str(S))
                                .replace('{n_accepted}', str(len(names))).replace('{minutes}', str(minutes)))
    (ws / 'CLAUDE.md').write_text("# Workspace\n\nYour task description is in TASK.md in this directory. Read it first and follow it.\n")
    driver.chown_tree(ws)
    return len(names)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--round', required=True)
    ap.add_argument('--minutes', type=int, default=60)
    ap.add_argument('--jobs', type=int, default=24)
    ap.add_argument('--skip-session', action='store_true', help='judge the chain already in <work>/ws/tree (tests, recovery)')
    a = ap.parse_args()
    R = a.round; rd = S / 'rounds' / R; exp = S / 'loopA_runs' / R; work = rd / 'integrate'
    if (work / 'result.json').is_file():
        print((work / 'result.json').read_text()); return
    work.mkdir(parents=True, exist_ok=True)
    dsets = json.loads((rd / 'datasets.json').read_text())
    cfg = dict(datasets={ds: dict(dev_view=d['dev_view'], dev_cache=d['dev_cache'], base_dev_pred=d['base_dev_pred'],
                                  val_view=d['val_view'], val_cache=d['val_cache'])
                         for ds, d in dsets.items()}, min_gain=0.0, max_regress=0.02, n_boot=1000)
    ws = work / 'ws'
    if a.skip_session:
        log = None
    else:
        n_acc = setup(R, rd, exp, ws, a.minutes)
        say(f'[{R}] merge agent: {n_acc} accepted patches, {a.minutes} min (Opus, dev only)')
        ro_extra = [str(bucket.BUCKET), str(S / 'harness'), str(S / 'loopA' / 'brief.md'), rsi_paths.WILOR_ROOT]
        log = LC.run_session(R, ws, cfg, a.minutes, ro_extra=ro_extra, prompt=PROMPT, done_glob='integrate_notes.md')
    res = dict(session_log=str(log), chosen='greedy')
    tree_i = work / 'tree_integrated'
    if tree_i.exists():
        shutil.rmtree(tree_i)
    shutil.copytree(ws / 'tree', tree_i, ignore=shutil.ignore_patterns('__pycache__'))
    if (ws / 'integrate_notes.md').is_file():
        shutil.copy(ws / 'integrate_notes.md', work / 'integrate_notes.md')
    bad = G.bad_files(tree_i)
    if bad:
        res['why'] = f'chain rejected: {bad}'
    elif G.fingerprint(tree_i) == G.fingerprint(rd / 'teacher_tree'):
        res['why'] = 'the agent left the greedy chain unchanged'
    else:
        inc, cand = {}, {}
        for ds, d in cfg['datasets'].items():
            inc[ds], cand[ds] = work / f'dev_greedy_{ds}', work / f'dev_integrated_{ds}'
            if not (replay_checked(rd / 'teacher_tree', d['dev_view'], d['dev_cache'], inc[ds], a.jobs)
                    and replay_checked(tree_i, d['dev_view'], d['dev_cache'], cand[ds], a.jobs)):
                res['why'] = f'replay failed or degraded on {ds} dev (see {work})'
                break
        else:
            j = G.judge(cfg, 'dev', {k: str(v) for k, v in inc.items()}, {k: str(v) for k, v in cand.items()})
            res['dev_vs_greedy'] = dict(S=j['S'], S_lo=j['S_lo'], worst=j['worst'], regressed=j['regressed'], passed=j['passed'],
                                        per={ds: dict(S=p['S'], rel=p['rel']) for ds, p in j['per'].items()})
            if j['passed']:
                res['chosen'] = 'integrated'
                (rd / 'teacher_tree').rename(rd / 'teacher_tree_greedy')
                shutil.copytree(tree_i, rd / 'teacher_tree')
                # one val look of the chosen teacher vs the round's base (recorded; it gates a Loop B redo)
                val = {}
                for ds, d in cfg['datasets'].items():
                    val[ds] = work / f'val_integrated_{ds}'
                    replay_checked(rd / 'teacher_tree', d['val_view'], d['val_cache'], val[ds], a.jobs)
                base_val = {ds: d['base_val_pred'] for ds, d in dsets.items()}
                jv = G.judge(cfg, 'val', base_val, {k: str(v) for k, v in val.items()})
                m = json.loads((rd / 'merge.json').read_text())
                m.setdefault('final', {})['integrated'] = dict(S=jv['S'], lo=jv['S_lo'], worst=jv['worst'], regressed=jv['regressed'])
                m['teacher_greedy'] = m.get('teacher'); m['teacher'] = 'integrated'
                (rd / 'merge.json').write_text(json.dumps(m, indent=1, default=float))
                res['val_vs_base'] = m['final']['integrated']
            else:
                res['why'] = 'not better than the greedy chain on dev'
    (work / 'result.json').write_text(json.dumps(res, indent=1, default=float))
    files = {'integrate_notes.md': work / 'integrate_notes.md'} if (work / 'integrate_notes.md').is_file() else {}
    files['session.jsonl.gz'] = Path(log) if log else None
    bucket.add('integrate', R, f'{R}_integrate', f"merge agent: teacher = {res['chosen']}; dev vs greedy "
               f"{json.dumps({k: res.get('dev_vs_greedy', {}).get(k) for k in ('S', 'S_lo', 'regressed')}, default=float)}"
               f"{'; ' + res['why'] if res.get('why') else ''}",
               dict(chosen=res['chosen'], dev_vs_greedy=res.get('dev_vs_greedy')), files={k: v for k, v in files.items() if v})
    print(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
