"""Loop C, the research lead. It runs after a round's GB / sealed stages, in
parallel with the next round's capture.

  1. dossier   rounds/<R>/loopC/ws/dossier/: round.json, merge.json, Loop A notes + accepted trees + gate log, Loop B
               logs, teacher-vs-base DEV diagnosis per dataset, history of earlier rounds and plans, current config
  2. session   one Claude session (Opus, effort xhigh) in the sandbox, --minutes long, with network for lt lit / lt code
  3. validate  plan/ -> plan_validated/: whitelist and ranges, at most 2 Loop B changes, val numbers scrubbed from
               every file the agents will see
  4. record    bucket events (loopC_plan: public files; loopC_session: transcript + literature log), the fetched papers
               and code become library/<R>/ (read-only for the next Loop A), rounds/PLAN_LATEST.json

  python loopC/run_loopC.py --round R1 [--minutes 60]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

S = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(S), str(S / 'loopA'), str(S / 'rounds')]
import bucket  # noqa: E402
import driver  # noqa: E402
import gate_service as G  # noqa: E402
import roster  # noqa: E402
import sandbox  # noqa: E402

import rsi_paths  # noqa: E402
PY = rsi_paths.PY
DEFAULT_LOOPB = dict(steps=2000, w_teacher=1.0, w_gt=1.0, w_exists=0.5, w_rot=0.25, lr=1e-4, rank=32,
                     mix=dict(hot3d=0.57, arctic=0.43),
                     teacher_terms='articulation')   # the teacher no longer supervises placement
RANGE = dict(steps=(1000, 4000), w_teacher=(0.25, 2.0), w_gt=(0.5, 2.0), w_exists=(0.25, 1.0), w_rot=(0.1, 0.5),
             lr=(5e-5, 2e-4))
STATE = Path(os.environ.get('RSI_STATE_DIR', S / 'rounds'))
PARAMS_FILE = STATE / 'LOOPB_PARAMS.json'          # the Loop B parameters in force (DEFAULT_LOOPB + applied changes)
VAL_LINE = re.compile(r'\bval(idation|_| split| set)?\b.*\d|\d.*\bval(idation)?\b', re.I)
AGENT_NAMES = [x.split(':', 1)[0] for x in roster.AGENTS.split(',')]


def rounds_sorted():
    return sorted((p for p in (S / 'rounds').glob('R*') if p.is_dir() and p.name[1:].isdigit()), key=lambda p: int(p.name[1:]))


def loopb_params() -> dict:
    return json.loads(PARAMS_FILE.read_text()) if PARAMS_FILE.is_file() else json.loads(json.dumps(DEFAULT_LOOPB))


def scrub(text: str) -> tuple[str, int]:
    out, n = [], 0
    for line in text.splitlines():
        if VAL_LINE.search(line):
            out.append('[line removed: it quoted validation numbers]'); n += 1
        else:
            out.append(line)
    return '\n'.join(out) + '\n', n


def sh(cmd, **kw):
    return subprocess.run(cmd, text=True, capture_output=True, env=dict(os.environ, PYTHONPATH=str(S)), cwd=str(S), **kw)


# ------------------------------------------------------------------------------------------------ 1. dossier
def build_dossier(R: str, ws: Path, cfg: dict) -> Path:
    rd, exp = S / 'rounds' / R, S / 'loopA_runs' / R
    d = ws / 'dossier'
    if d.exists():
        shutil.rmtree(d)
    (d / 'loopA' / 'accepted').mkdir(parents=True); (d / 'loopB').mkdir()
    for f in ('round.json', 'merge.json', 'datasets.json', *sorted(x.name for x in rd.glob('round_v*.json'))):
        if (rd / f).is_file():
            shutil.copy(rd / f, d / f)
    for ag in sorted((exp / 'agents').glob('*')) if (exp / 'agents').is_dir() else []:
        if (ag / 'notes.md').is_file():
            shutil.copy(ag / 'notes.md', d / 'loopA' / f'{ag.name}_notes.md')
    for t in sorted((exp / 'archive').glob('*/tree')) if (exp / 'archive').is_dir() else []:
        shutil.copytree(t, d / 'loopA' / 'accepted' / t.parent.name, ignore=shutil.ignore_patterns('__pycache__'))
    for f in ('gate.log', 'ledger.json', 'state.json'):
        if (exp / f).is_file():
            shutil.copy(exp / f, d / 'loopA' / f)
    for f in ('train_log.jsonl', 'args.json'):
        if (S / 'loopB/runs' / R / f).is_file():
            shutil.copy(S / 'loopB/runs' / R / f, d / 'loopB' / f)
    # teacher vs base on DEV, per dataset (the teacher replays in the sandbox like any agent code)
    tree = rd / 'teacher_tree'
    for ds, c in cfg['datasets'].items():
        out = rd / 'loopC' / f'teacher_dev_{ds}'
        if tree.is_dir() and not (out / 'DONE').is_file():
            ok, tail = G.replay(tree, Path(c['dev_view']), Path(c['dev_cache']), out, 24)
            if ok:
                (out / 'DONE').touch()
        if (out / 'DONE').is_file():
            r = sh([PY, '-m', 'eval.diagnose', '--gt', str(Path(c['dev_view']) / 'gt'), '--pred', str(out), '--vs',
                    c['base_dev_pred'], '--n', '15'])
            (d / f'diagnose_{ds}.txt').write_text(f'# teacher H_k* vs the round base N_k-1, DEV, {ds}\n' + r.stdout + r.stderr)
    # history: earlier rounds' verdicts and Loop C plans with their outcome
    hist = ['# History of the run (earlier rounds; val numbers pooled per dataset)\n']
    for p in rounds_sorted():
        st = json.loads((p / 'round.json').read_text()) if (p / 'round.json').is_file() else {}
        v = st.get('verdict', {})
        hist.append(f"## {p.name}: base {st.get('base', {}).get('node')}, teacher {st.get('teacher', {}).get('pick')} "
                    f"{st.get('teacher', {}).get('modules')}; GB {'ACCEPTED' if v.get('accepted') else 'rejected' if v else 'n/a'} "
                    f"S_system {v.get('S_system')}; Loop B params {st.get('loopB_params')}; plan in force: {st.get('plan_in')}")
        for old in sorted(p.glob('round_v*.json')):        # earlier Loop B attempts of this round (Loop B redo)
            o = json.loads(old.read_text()); ov = o.get('verdict', {})
            ctl = {ds: {k: round(100 * x['S'], 2) for k, x in c.items() if isinstance(x, dict) and 'S' in x}
                   for ds, c in (o.get('controls') or {}).items()}
            hist.append(f"   earlier Loop B attempt {old.stem.split('_')[-1]} of {p.name} (superseded): GB "
                        f"{'ACCEPTED' if ov.get('accepted') else 'rejected' if ov else 'n/a'} S_system {ov.get('S_system')} "
                        f"worst {ov.get('worst')} regressed {ov.get('regressed')}; Loop B params {o.get('loopB_params')}; "
                        f"controls S % {ctl}; revalidate kept {(o.get('revalidate') or {}).get('kept')}")
        for lcd in [p / 'loopC', *sorted(p.glob('loopC_v*'))]:
            pv = lcd / 'plan_validated' / 'record.json'
            if pv.is_file():
                hist.append(f"   Loop C plan written after {p.name} ({lcd.name}): {pv.read_text()[:1500]}")
    (d / 'history.md').write_text('\n'.join(hist) + '\n')
    (d / 'current_config.json').write_text(json.dumps(dict(
        loopB_params_in_force=loopb_params(), whitelist=dict(RANGE, rank=[32, 64], mix='each share within +-50% of 0.57/0.43', teacher_terms=['articulation', 'all', 'none']),
        agents=roster.AGENTS, focus_tags=driver.FOCUS3D), indent=1))
    subprocess.run(['chmod', '-R', 'go-w,o+rX', str(d)], check=True)
    return d


# ------------------------------------------------------------------------------------------------ 2. session
def run_session(R: str, ws: Path, cfg: dict, minutes: int, ro_extra: list | None = None, prompt: str | None = None,
                done_glob: str = 'plan/*') -> Path:
    """one Opus session in the sandbox with route failover; ro_extra / prompt replace Loop C's read-only mounts and
    prompt (rounds/integrate.py, the merge agent, sees dev material only)"""
    opts = driver.parse_agent(f'lc:{roster._O}')[1]
    usable = [o for o in opts if driver.probe(*o) == 'ok']
    if not usable:
        raise SystemExit(f'Loop C: no usable model among {opts}')
    # failover as in loopA/driver.py (the first request of a session can fail with a transient 403 /
    # authentication_failed while the same route answers a minute later): an API error at the start of a session moves
    # to the next route, at most twice round the list
    t_end = time.time() + minutes * 60
    for attempt in range(2 * len(usable)):
        prov, model = usable[attempt % len(usable)]
        log = _session(R, ws, cfg, prov, model, max(300, int(t_end - time.time())), ro_extra=ro_extra, prompt=prompt)
        tail = log.read_text(errors='ignore')[-6000:]
        bad = any(p in tail for p in ('"is_api_error_message":true', 'authentication_failed', '"terminal_reason":"api_error"',
                                      *driver.TRANSIENT_PAT, *driver.QUOTA_PAT))
        if not bad or any(ws.glob(done_glob)) or time.time() > t_end - 300:
            return log
        print(f'Loop C: API error on {prov}:{model}; trying the next route', flush=True)
        time.sleep(30)
    return log


def _session(R: str, ws: Path, cfg: dict, prov: str, model: str, seconds: int, ro_extra: list | None = None,
             prompt: str | None = None) -> Path:
    base_url, key = driver.EP[prov]
    env = dict(HOME=str(ws / 'home'), CLAUDE_CONFIG_DIR=str(ws / 'home' / '.claude'), API_TIMEOUT_MS='600000',
               DISABLE_TELEMETRY='1', CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1', DISABLE_AUTOUPDATER='1',
               LOOPA_OUTPUT_CAP='40000', BASH_DEFAULT_TIMEOUT_MS='3600000', BASH_MAX_TIMEOUT_MS='7200000',
               LOOPA_WS=str(ws), LOOPA_SPOOL=str(ws / 'spool'), LOOPA_JOBS='24', PYTHONPATH=str(S),
               LOOPA_DATASETS=json.dumps({ds: dict(dev_view=c['dev_view'], dev_cache=c['dev_cache'], parent_dev=c['base_dev_pred'])
                                          for ds, c in cfg['datasets'].items()}),
               RSI_BUCKET=str(bucket.BUCKET), WILOR_MINI_ROOT=rsi_paths.WILOR_ROOT,
               ANTHROPIC_BASE_URL=base_url, ANTHROPIC_AUTH_TOKEN=key, ANTHROPIC_MODEL=model, ANTHROPIC_SMALL_FAST_MODEL=model)
    ro = [p for c in cfg['datasets'].values() for p in (c['dev_view'], c['dev_cache'], c['base_dev_pred'])]
    # no document that quotes test numbers: Loop C sees none
    ro += ro_extra if ro_extra is not None else [str(bucket.BUCKET), str(S / 'harness'), str(S / 'loopB'), str(S / 'rounds'),
                                                  str(S / 'loopA' / 'brief.md'), rsi_paths.WILOR_ROOT]
    ro = [p for p in dict.fromkeys(ro) if Path(p).exists()]
    prompt = prompt.format(minutes=seconds // 60) if prompt else (
        f'Read TASK.md in the current directory and follow it. You have about {seconds // 60} minutes; '
        'write ./plan/ files early and refine them.')
    log = ws.parent / f'session_{int(time.time())}.jsonl'
    with open(log, 'w') as fh:
        try:
            sandbox.run([rsi_paths.CLAUDE_BIN, '-p', prompt, '--output-format', 'stream-json', '--verbose',
                         '--dangerously-skip-permissions', '--effort', 'xhigh'], ro=ro, rw=[str(ws)], env=env, cwd=str(ws),
                        stdout=fh, stderr=subprocess.STDOUT, timeout=seconds)
        except subprocess.TimeoutExpired:
            fh.write(json.dumps(dict(type='driver', event='timeout')) + '\n')
    return log


# ------------------------------------------------------------------------------------------------ 3. validate
def validate(plan: Path, out: Path, R: str | None = None) -> dict:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rec = dict(scrubbed={}, rejected=[], loopB_changes={}, focus={}, ideas=0, proposals=[])
    for name, cap in (('ideas.md', 40000), ('brief_notes.md', 6000), ('report_public.md', 8000), ('proposals.md', 30000)):
        f = plan / name
        if f.is_file():
            txt, n = scrub(f.read_text(errors='replace')[:cap])
            (out / name).write_text(txt); rec['scrubbed'][name] = n
    if (plan / 'report_private.md').is_file():
        shutil.copy(plan / 'report_private.md', out / 'report_private.md')          # humans only, never in the bucket
    if (out / 'ideas.md').is_file():
        rec['ideas'] = len(re.findall(r'^#{2,3} ', (out / 'ideas.md').read_text(), re.M))
    if (out / 'proposals.md').is_file():
        rec['proposals'] = re.findall(r'^#{2,3} (.+)$', (out / 'proposals.md').read_text(), re.M)[:20]
    try:
        fj = json.loads((plan / 'focus.json').read_text()) if (plan / 'focus.json').is_file() else {}
        for k, v in fj.items():
            if k in AGENT_NAMES and isinstance(v, str) and v.strip():
                t, n = scrub(v.strip()[:300])
                rec['focus'][k] = t.strip() if not n else ''
        rec['focus'] = {k: v for k, v in rec['focus'].items() if v}
        (out / 'focus.json').write_text(json.dumps(rec['focus'], indent=1))
    except (ValueError, AttributeError) as e:
        rec['rejected'].append(f'focus.json: {e}')
    cur = loopb_params(); new = json.loads(json.dumps(cur))
    try:
        lb = json.loads((plan / 'loopB.json').read_text()) if (plan / 'loopB.json').is_file() else {}
    except ValueError as e:
        lb = {}; rec['rejected'].append(f'loopB.json: {e}')
    for k, v in lb.items():
        if k == 'why':
            continue
        if len(rec['loopB_changes']) >= 2:
            rec['rejected'].append(f'loopB.{k}: more than 2 changes'); continue
        try:
            if k in RANGE:
                lo, hi = RANGE[k]; v = int(v) if k == 'steps' else float(v)
                ok = lo <= v <= hi
            elif k == 'rank':
                v = int(v); ok = v in (32, 64)
            elif k == 'teacher_terms':
                ok = v in ('articulation', 'all', 'none')     # 'none': pose from GT only
            elif k == 'mix':
                v = {d: float(x) for d, x in v.items()}
                ok = set(v) == set(DEFAULT_LOOPB['mix']) and all(0.5 * DEFAULT_LOOPB['mix'][d] <= x <= 1.5 * DEFAULT_LOOPB['mix'][d]
                                                                  for d, x in v.items())
                tot = sum(v.values()); v = {d: round(x / tot, 4) for d, x in v.items()}
            else:
                ok = False
        except (TypeError, ValueError, AttributeError):
            ok = False
        if not ok:
            rec['rejected'].append(f'loopB.{k}={v!r}: outside the whitelist'); continue
        if v != cur.get(k):
            new[k] = v; rec['loopB_changes'][k] = v
    rec['loopB_why'] = str(lb.get('why', ''))[:600]
    (out / 'loopB_params.json').write_text(json.dumps(new, indent=1))
    rec['action'], rec['action_why'] = check_action(plan, R, rec)
    (out / 'action.json').write_text(json.dumps(dict(action=rec['action'], why=rec['action_why']), indent=1))
    (out / 'record.json').write_text(json.dumps(rec, indent=1))
    return rec


MAX_LOOPB_RETRIES = 1     # per round: each extra GB is one more look at val


def check_action(plan: Path, R: str | None, rec: dict) -> tuple[str, str]:
    """Loop C's decision for the round it just reviewed: 'next_round' (default) or 'retry_loopB', i.e.
    redo this round's Loop B with the changed Loop B parameters, keeping its Loop A. Allowed only if: GB rejected the
    round; the round's Loop B has not been redone yet (rounds/<R>/round_v*.json counts earlier attempts);
    the round's teacher H_k* beat the round's base on val (merge.json: S >= 1 %, CI90 lower > 0, no regression), i.e.
    Loop A delivered and the distillation is what failed; and at least one Loop B parameter changed."""
    try:
        a = json.loads((plan / 'action.json').read_text()) if (plan / 'action.json').is_file() else {}
    except ValueError:
        a = {}
    want, why = str(a.get('action', 'next_round')), scrub(str(a.get('why', ''))[:800])[0].strip()   # goes to the bucket
    if want != 'retry_loopB':
        return 'next_round', why
    if R is None:
        return 'next_round', 'retry refused: no round given'
    rd = S / 'rounds' / R
    try:
        st = json.loads((rd / 'round.json').read_text())
        m = json.loads((rd / 'merge.json').read_text())
    except (OSError, ValueError) as e:
        return 'next_round', f'retry refused: {e}'
    v = st.get('verdict') or {}
    tf = (m.get('final') or {}).get(m.get('teacher')) or {}
    reasons = []
    if not v or v.get('accepted', True):
        reasons.append('GB did not reject the round')
    if len(list(rd.glob('round_v*.json'))) >= MAX_LOOPB_RETRIES:
        reasons.append(f'Loop B of {R} was already redone')
    if not (tf.get('S', 0) >= 0.01 and tf.get('lo', 0) > 0 and not tf.get('regressed')):
        reasons.append(f"the teacher did not beat the base on val by >= 1 % without regressions ({tf})")
    if not rec['loopB_changes']:
        reasons.append('no Loop B parameter changed')
    if reasons:
        return 'next_round', 'retry refused: ' + '; '.join(reasons)
    return 'retry_loopB', why


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--round', required=True)
    ap.add_argument('--minutes', type=int, default=60)
    ap.add_argument('--skip-session', action='store_true', help='rebuild dossier and re-validate an existing plan only')
    a = ap.parse_args()
    os.umask(0o022)
    R = a.round; rd = S / 'rounds' / R; lc = rd / 'loopC'; ws = lc / 'ws'
    cfg = dict(datasets=json.loads((rd / 'datasets.json').read_text()))
    for sub in ('plan', 'refs', 'home', 'spool'):
        (ws / sub).mkdir(parents=True, exist_ok=True)
    os.chmod(lc, 0o755); os.chmod(ws, 0o755)
    (ws / 'CLAUDE.md').write_text("# Workspace\n\nYour task description is in TASK.md in this directory. Read it first and follow it.\n")   # brief in TASK.md: see loopA/driver.py setup()
    (ws / 'TASK.md').write_text((S / 'loopC' / 'brief_loopC.md').read_text().replace('{root}', str(S)).replace('{minutes}', str(a.minutes))
                                  .replace('{round}', R))
    if (rd / 'teacher_tree').is_dir() and not (ws / 'tree').exists():
        shutil.copytree(rd / 'teacher_tree', ws / 'tree', ignore=shutil.ignore_patterns('__pycache__'))
    build_dossier(R, ws, cfg)
    for sub in ('plan', 'refs', 'home', 'spool', 'tree'):
        if (ws / sub).exists():
            driver.chown_tree(ws / sub)
    # `lt lit` / `lt code` log to <ws>/lit_log.jsonl and the workspace root is not the sandbox user's
    (ws / 'lit_log.jsonl').touch(exist_ok=True); driver.chown_tree(ws / 'lit_log.jsonl')
    log = None if a.skip_session else run_session(R, ws, cfg, a.minutes)
    rec = validate(ws / 'plan', lc / 'plan_validated', R)
    # the literature Loop C fetched becomes the next round's shared, read-only library
    lib = S / 'library' / R
    if (ws / 'refs').is_dir():
        shutil.copytree(ws / 'refs', lib, dirs_exist_ok=True)
        subprocess.run(['chmod', '-R', 'go-w,o+rX', str(lib)], check=False)
    pv = lc / 'plan_validated'
    pub = {n: pv / n for n in ('ideas.md', 'brief_notes.md', 'report_public.md', 'focus.json', 'proposals.md',
                               'loopB_params.json', 'record.json', 'action.json') if (pv / n).is_file()}
    summary = (pv / 'report_public.md').read_text()[:600] if (pv / 'report_public.md').is_file() else 'Loop C plan'
    bucket.add('loopC_plan', R, f'{R}_loopC', summary,
               dict(action=rec['action'], action_why=rec['action_why'], loopB_changes=rec['loopB_changes'], loopB_why=rec['loopB_why'], focus=rec['focus'], ideas=rec['ideas'],
                    proposals=rec['proposals'], rejected=rec['rejected']), files=pub)
    if log is not None:
        files = {'transcript.jsonl.gz': log}
        if (ws / 'lit_log.jsonl').is_file():
            files['lit_log.jsonl'] = ws / 'lit_log.jsonl'
        bucket.add('loopC_session', R, f'{R}_loopC_session', f'Loop C session transcript ({R})', files=files)
    (STATE / 'PLAN_LATEST.json').write_text(json.dumps(dict(round=R, dir=str(pv), library=str(lib),
                                                                   time=time.strftime('%F %T')), indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main()
