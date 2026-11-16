"""Loop A driver: one experiment = N coding agents (Claude Code or Codex CLI sessions, model routes in
loopA/endpoints.json) + the gate service.

  driver.py --exp runs/V1 --base T --agents a1:claude:claude-opus-5-5,a2:codex:gpt-6-sol,... --hours 3 --budget 8
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import re
import os
import signal
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid

_S = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_S / 'loopA')); sys.path.insert(0, str(_S))
import rsi_paths  # noqa: E402
import sandbox  # noqa: E402

R = _S / 'loopA'
UID = rsi_paths.SANDBOX_UID
KEYS = dict(l.strip().split('=', 1) for l in open(os.environ.get('RSI_KEYS_FILE', str(R / 'keys.env'))) if '=' in l)
AGENT_CARDS = os.environ.get('RSI_AGENT_CARDS', '0,1,2,3')    # Loop A card split (see run_agent's env)
GATE_CARDS = os.environ.get('RSI_GATE_CARDS', '4,5,6,7')
# Model routes: route name -> (Anthropic-compatible base URL, key). Kept out of the code: RSI_ENDPOINTS_FILE (JSON,
# {"route": ["https://base.url", "KEY_NAME_IN_KEYS_FILE"], ...}; see loopA/endpoints.example.json). A route whose key
# is missing from the keys file is skipped. The agent spec "a1:<route>:<model>" picks the route.
EP = {name: (url, KEYS[key]) for name, (url, key) in
      json.load(open(os.environ.get('RSI_ENDPOINTS_FILE', str(R / 'endpoints.json')))).items() if KEYS.get(key)}
QUOTA_PAT = ('rate_limit', 'QuotaExceeded', '"error_status":429', 'Request rejected (429)', 'usage quota')
TRANSIENT_PAT = ('"api_error_status":403', 'authentication_failed',   # transient 403s: another route
                 '"error_status":500', '"error_status":502', '"error_status":503', '"error_status":504',
                 'HTTP 500', 'HTTP 502', 'HTTP 503', 'HTTP 504', 'upstream_error', 'overloaded_error',
                 'Connection lost mid-response', 'ECONNRESET', 'socket hang up', 'connection reset by peer',
                 'unexpected status 500', 'unexpected status 502', 'unexpected status 503',
                 'unexpected status 504', 'Service temporarily unavailable')
CONFIG_PAT = ('"error_status":401', '"error_status":424', 'HTTP 424', 'authentication_failed', 'Invalid API Key', 'unrecognized_model',
              'may not exist or you may not have access', 'model_not_found', 'not supported when using Codex')


def parse_agent(spec: str):
    """'a1:prov:model[|prov:model...][#focus]' -> (name, [(prov, model), ...], focus). Options are tried in order:
    failover when one runs out of quota or turns out to be misconfigured."""
    spec, _, focus = spec.partition('#')
    name, rest = spec.split(':', 1)
    opts = [tuple(o.split(':', 1)) for o in rest.split('|') if o]
    return name, opts, focus or ''


def probe(prov: str, model: str) -> str:
    """Preflight: one tiny request. 'ok', 'quota' or 'config:<reason>'."""
    import urllib.request, urllib.error
    if prov not in EP:
        return 'config:no endpoint'
    base, key = EP[prov]
    if prov.startswith('codex'):
        sid = str(uuid.uuid4())
        body = json.dumps({'model': model, 'instructions': 'You are a coding assistant.',
                           'input': [{'role': 'user', 'content': [{'type': 'input_text', 'text': 'Reply exactly OK'}]}],
                           'stream': True, 'store': False, 'tools': [], 'tool_choice': 'auto',
                           'parallel_tool_calls': True}).encode()
        req = urllib.request.Request(base + '/responses', data=body, headers={
            'Authorization': 'Bearer ' + key, 'content-type': 'application/json', 'accept': 'text/event-stream',
            'originator': 'codex_cli_rs', 'user-agent': 'codex_cli_rs/0.164.0 (linux; x86_64)',
            'version': '0.164.0', 'OpenAI-Beta': 'responses=experimental', 'session_id': sid,
            'session-id': sid, 'x-client-request-id': sid, 'x-codex-window-id': sid + ':0'})
    else:
        body = json.dumps({'model': model, 'max_tokens': 16,
                           'messages': [{'role': 'user', 'content': 'Reply with OK.'}]}).encode()
        req = urllib.request.Request(base + '/v1/messages', data=body, headers={
            'x-api-key': key, 'Authorization': 'Bearer ' + key,
            'anthropic-version': '2023-06-01', 'content-type': 'application/json'})
    try:
        urllib.request.urlopen(req, timeout=90).read(); return 'ok'
    except urllib.error.HTTPError as e:
        txt = e.read().decode(errors='ignore')
        if e.code == 429 or any(p in txt for p in ('Quota', 'quota', 'rate_limit')):
            return 'quota'
        if e.code in (408, 425, 500, 502, 503, 504) or any(p in txt for p in TRANSIENT_PAT):
            return 'retry'
        return f'config:HTTP {e.code} {txt[:120]}'
    except Exception as e:                     # network hiccup: do not drop the option for that
        return 'ok' if 'timed out' in str(e) else f'config:{str(e)[:80]}'
VISION = (dict(LOOPA_VISION_URL=EP['vision'][0], LOOPA_VISION_KEY=EP['vision'][1], LOOPA_VISION_MODEL='doubao-seed-2.1-pro')
          if 'vision' in EP else {})   # the agents' image-reading tool (route 'vision'); none without it
BASE_TREE = _S / 'harness/trees/empty'   # round 1: the empty chain = 00
CODEX_ROOT = rsi_paths.CODEX_ROOT   # binary only, read-only
CODEX_BIN = CODEX_ROOT / 'bin/codex'


def chown_tree(p: Path):
    subprocess.run(['chown', '-R', '-h', f'{UID}:{UID}', str(p)], check=True)


def setup(exp: Path, cfg_in: dict, agents: list, brief_path: Path, base_tree: Path):
    """config.json for the gate (datasets, budget, rule), state.json with every agent's incumbent = the base
    node's predictions on dev and val, and one workspace per agent (./tree = the base harness tree)."""
    for d in ('spool', 'gate', 'archive', 'agents', 'logs'):
        (exp / d).mkdir(parents=True, exist_ok=True)
    os.chmod(exp, 0o755); os.chmod(exp / 'spool', 0o755); os.chmod(exp / 'agents', 0o755)
    if not (exp / 'config.json').exists():
        (exp / 'config.json').write_text(json.dumps(cfg_in, indent=1))
    cfg = json.loads((exp / 'config.json').read_text())
    state = json.loads((exp / 'state.json').read_text()) if (exp / 'state.json').exists() else dict(agents={})
    for a, _, _ in agents:
        state['agents'].setdefault(a, dict(used=0, incumbent_name='base',
                                           incumbent_dev={ds: d['base_dev_pred'] for ds, d in cfg['datasets'].items()},
                                           incumbent_val={ds: d['base_val_pred'] for ds, d in cfg['datasets'].items()}))
    (exp / 'state.json').write_text(json.dumps(state, indent=1))
    if not (exp / 'bucket.jsonl').exists():
        (exp / 'bucket.jsonl').write_text('')
    os.chmod(exp / 'bucket.jsonl', 0o644)
    brief0 = (Path(brief_path).read_text().replace('{budget}', str(cfg['budget_per_agent']))
             .replace('{root}', str(_S)).replace('{ace_eval}', str(rsi_paths.ACE_EVAL)))
    plan = Path(cfg['plan']) if cfg.get('plan') else None          # Loop C's plan for this round (already validated)
    if plan and (plan / 'brief_notes.md').is_file():
        brief0 += ('\n## Lessons from earlier rounds (written by Loop C, the research lead)\n\n'
                   + (plan / 'brief_notes.md').read_text()[:6000] + '\n')
    if plan and (plan / 'ideas.md').is_file():
        brief0 += ('\n## Idea cards for this round\n\nLoop C left idea cards with evidence and references: `lt ideas`. '
                   'They are suggestions, not orders; check them against `lt diagnose` and the bucket. The papers and '
                   f'code it read are under {_S}/library/ (read-only).\n')
    focus_over = json.loads((plan / 'focus.json').read_text()) if plan and (plan / 'focus.json').is_file() else {}
    for a, opts, focus in agents:
        model = opts[0][1]
        ftxt = focus_over.get(a) or FOCUS3D.get(focus, focus)
        brief = brief0.replace('{focus}', ftxt or 'none: pick the largest remaining error source from `lt diagnose`')
        ws = exp / 'agents' / a
        if not ws.exists():
            ws.mkdir(parents=True)
            shutil.copytree(base_tree, ws / 'tree', ignore=shutil.ignore_patterns('__pycache__'))
            (ws / 'notes.md').write_text(f'# Notes ({a}, model {model})\n')
            # the brief goes to TASK.md, read in-session: an API content filter rejected (403) the Loop C brief as a
            # CLAUDE.md system context every time, while the same text read by the agent from TASK.md passed
            (ws / 'TASK.md').write_text(brief)
            (ws / 'CLAUDE.md').write_text("# Workspace\n\nYour task description is in TASK.md in this directory. Read it first and follow it.\n")
            (ws / 'home').mkdir()
        (exp / 'spool' / a).mkdir(exist_ok=True)
        chown_tree(ws); chown_tree(exp / 'spool' / a)
    return cfg


# a focus hint per agent (agent spec suffix '#key'); the agent may still work elsewhere if the evidence says so
FOCUS3D = {
    'detect': 'detection: missed and spurious hands (edge hands, existence calibration, hands at segment borders)',
    'root': 'hand placement: root translation / depth (CT-p), especially at the start of a segment',
    'orient': 'global orientation (GO-p) and wrist-relative pose (MPJPE-p)',
    'fingers': 'finger articulation (PA-p, per-finger error)',
    'temporal': 'temporal consistency inside the 81-frame segment (jitter, sudden jumps, short dropouts)',
    'image': 'image evidence: refine 2D anchors / 3D with the RGB frames (e.g. the WiLoR model available read-only)',
    # the two columns no other focus covers
    'kp2d': '2D keypoints (EPE2D-p): the network\'s own 2D anchors (state joints_2d), which EPE2D-p scores: their bias '
            '(00 shows a constant offset of a few pixels that differs by recording), their consistency with the projected 3D '
            'hand, and refinement from image evidence',
    'oos': 'hands out of sight (MPJPE+OOS): the evaluator scores the wrist-relative pose of EVERY ground-truth hand-frame, '
           'in view or not, from the slot output whatever its existence score; 00 is about twice as wrong out of view (HOT3D '
           '22.7 vs 11.4 mm, ARCTIC 35.9 vs 15.2 mm), e.g. carry the pose through the frames where the hand leaves the image',
}


def ledger_add(exp: Path, agent: str, model: str, session_log: Path, lock):
    usage = dict(input=0, output=0, cache_read=0, turns=0)
    for line in session_log.read_text(errors='ignore').splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get('type') == 'result':
            u = d.get('usage') or {}
            usage['input'] += u.get('input_tokens', 0); usage['output'] += u.get('output_tokens', 0)
            usage['cache_read'] += u.get('cache_read_input_tokens', 0); usage['turns'] += d.get('num_turns') or 0
        elif d.get('type') == 'turn.completed':       # official Codex CLI --json
            u = d.get('usage') or {}
            usage['input'] += u.get('input_tokens', 0); usage['output'] += u.get('output_tokens', 0)
            usage['cache_read'] += u.get('cached_input_tokens', 0); usage['turns'] += 1
    with lock:
        p = exp / 'ledger.json'
        led = json.loads(p.read_text()) if p.exists() else {}
        e = led.setdefault(agent, dict(model=model, sessions=0, input=0, output=0, cache_read=0, turns=0))
        e['sessions'] += 1; e['model'] = model
        for k in ('input', 'output', 'cache_read', 'turns'):
            e[k] += usage[k]
        m = e.setdefault('by_model', {}).setdefault(model, dict(sessions=0, input=0, output=0, cache_read=0, turns=0))
        m['sessions'] += 1
        for k in ('input', 'output', 'cache_read', 'turns'):
            m[k] += usage[k]
        p.write_text(json.dumps(led, indent=1))


def read_deadline(exp: Path, default: float) -> float:
    """The Loop A deadline: <exp>/DEADLINE (unix epoch) when present (extends Loop A without a restart)."""
    try:
        return float((exp / 'DEADLINE').read_text().strip())
    except (OSError, ValueError):
        return default


def dlog(exp: Path, msg: str):
    with open(exp / 'logs' / 'driver.log', 'a') as d:
        d.write(f"{time.strftime('%F %T')} {msg}\n")


def alert(msg: str):
    rsi_paths.STATUS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(rsi_paths.STATUS_LOG, 'a') as st:
        st.write(f"- {time.strftime('%H:%M')} [driver] {msg}\n")


def run_agent(exp: Path, a: str, opts: list, focus: str, gpus: str, cache: Path, base: str, deadline: float, cfg: dict, lock):
    ws = exp / 'agents' / a
    # preflight every option once; a misconfigured one (dead key, model not activated / retired) is dropped now
    # instead of burning sessions (once 7 agents x 10 sessions were burnt on a dead key)
    usable = []
    for o in opts:
        r = probe(*o)
        dlog(exp, f'{a} preflight {o[0]}:{o[1]} -> {r}')
        if not r.startswith('config'):
            usable.append(o)
    if not usable:
        alert(f'FAILED {exp.name} {a}: no usable model among {opts} (preflight), agent not started')
        return
    cur, switches = 0, []
    dsv = list(cfg['datasets'].values())
    ro = [p for d in dsv for p in (d['dev_view'], d['dev_cache'], d['base_dev_pred'])]
    ro += [str(exp / 'state.json'), rsi_paths.WILOR_ROOT, os.environ.get('RSI_BUCKET', str(_S / 'bucket'))]
    if (_S / 'library').is_dir():
        ro.append(str(_S / 'library'))   # papers / code fetched by Loop C (read-only)
    if cfg.get('plan') and (Path(cfg['plan']) / 'ideas.md').is_file():
        ro.append(str(Path(cfg['plan']) / 'ideas.md'))
    ro = list(dict.fromkeys(ro))
    rw = [str(ws), str(exp / 'spool' / a)]
    d0 = dsv[0]                                   # HOT3D first; ARCTIC joins as a second dev dataset later
    env = dict(HOME=str(ws / 'home'), CLAUDE_CONFIG_DIR=str(ws / 'home' / '.claude'), API_TIMEOUT_MS='600000',
               DISABLE_TELEMETRY='1', CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1', DISABLE_AUTOUPDATER='1',
               LOOPA_OUTPUT_CAP='40000', BASH_DEFAULT_TIMEOUT_MS='3600000', BASH_MAX_TIMEOUT_MS='14400000',
               LOOPA_WS=str(ws), LOOPA_SPOOL=str(exp / 'spool' / a), LOOPA_DEV_VIEW=d0['dev_view'],
               LOOPA_DEV_CACHE=d0['dev_cache'], LOOPA_PARENT_DEV=d0['base_dev_pred'], LOOPA_BUCKET=str(exp / 'bucket.jsonl'),
               LOOPA_STATE=str(exp / 'state.json'), LOOPA_JOBS=str(cfg.get('agent_jobs', 16)), PYTHONPATH=str(_S),
               LOOPA_ONE_PENDING='1' if cfg.get('one_pending') else '0', LOOPA_MAX_PENDING=str(cfg.get('max_pending', 2)),
               WILOR_MINI_ROOT=rsi_paths.WILOR_ROOT,
               LOOPA_DATASETS=json.dumps({ds: dict(dev_view=d['dev_view'], dev_cache=d['dev_cache'], parent_dev=d['base_dev_pred'])
                                          for ds, d in cfg['datasets'].items()}),
               RSI_BUCKET=os.environ.get('RSI_BUCKET', str(_S / 'bucket')),
               LOOPA_IDEAS=str(Path(cfg['plan']) / 'ideas.md') if cfg.get('plan') else '/nonexistent',
               # cards split during Loop A: agents' own experiments and lt replays on AGENT_CARDS, the gate's
               # replays on GATE_CARDS. Under shared cards, image-model modules that fall back silently on CUDA OOM
               # (e.g. wilor, edge_rescue) can change a gate replay without crashing it.
               CUDA_VISIBLE_DEVICES=AGENT_CARDS)
    k = 0
    while True:
        deadline = read_deadline(exp, deadline)      # <exp>/DEADLINE (epoch) extends or shortens a running Loop A
        if time.time() >= deadline or k >= cfg.get('max_sessions', 6):
            break
        # no new session in the last minutes: one D_train replay (15-30 min) + a gate verdict cannot fit, and the agents
        # rightly close at once (sessions opened in the last minutes end within seconds)
        if k > 0 and deadline - time.time() < (cfg.get('submit_cutoff_min', 45) + cfg.get('min_session_minutes', 20)) * 60:
            break
        st = json.loads((exp / 'state.json').read_text())['agents'][a]
        if st['used'] >= cfg['budget_per_agent']:
            break
        k += 1
        prov, model = usable[cur]
        base_url, key = EP[prov]
        for env_key in ('ANTHROPIC_BASE_URL', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_MODEL',
                        'ANTHROPIC_SMALL_FAST_MODEL', 'CODEX_HOME', 'OPENAI_API_KEY'):
            env.pop(env_key, None)
        if prov.startswith('codex'):
            codex_home = ws / 'home' / '.codex'
            codex_home.mkdir(parents=True, exist_ok=True)
            (codex_home / 'config.toml').write_text(
                f'model_provider = "route"\nmodel = "{model}"\nmodel_reasoning_effort = "xhigh"\n\n'
                '[features]\ncode_mode_host = true\n\n'
                '[model_providers.route]\nname = "route"\n'
                f'base_url = "{base_url}"\nwire_api = "responses"\nrequires_openai_auth = true\n')
            (codex_home / 'auth.json').write_text(json.dumps({'OPENAI_API_KEY': key}) + '\n')
            os.chmod(codex_home / 'config.toml', 0o600); os.chmod(codex_home / 'auth.json', 0o600)
            chown_tree(codex_home)
            env.update(CODEX_HOME=str(codex_home))
        else:
            env.update(ANTHROPIC_BASE_URL=base_url, ANTHROPIC_AUTH_TOKEN=key,
                       ANTHROPIC_MODEL=model, ANTHROPIC_SMALL_FAST_MODEL=model)
        left_min = int((deadline - time.time()) / 60)
        prompt = (f'Read TASK.md in the current directory and follow it. This is session {k}. '
                  f'Wall-clock time left for your whole job: about {left_min} minutes; gate submissions left: '
                  f"{cfg['budget_per_agent'] - st['used']}. The gate stops taking submissions "
                  f"{cfg.get('submit_cutoff_min', 45)} minutes before the end; submit improvements well before that. "
                  + ('Your notes from earlier sessions are in notes.md; continue from them.' if k > 1 else
                     'Start with `lt bucket`, then `lt try`.'))
        log = exp / 'logs' / f'{a}_session{k}.jsonl'
        if log.exists():                             # restarted driver: never overwrite an earlier transcript
            log = exp / 'logs' / f'{a}_session{k}_{int(time.time())}.jsonl'
        remaining = max(60, int(deadline - time.time()))
        t_start = time.time()
        if prov.startswith('codex'):
            command = [str(CODEX_BIN), 'exec', '--json', '--dangerously-bypass-approvals-and-sandbox',
                       '--skip-git-repo-check', '--ephemeral', '-C', str(ws), prompt]
            command_ro = ro + [str(CODEX_ROOT)]
        else:
            # no turn limit (a 200-turn limit cut 10 of 42 sessions about 3.6 h into their work) and xhigh reasoning
            # for Claude models (the same level as the GPT agents; other models keep their default)
            command = [rsi_paths.CLAUDE_BIN, '-p', prompt, '--output-format', 'stream-json', '--verbose',
                       '--dangerously-skip-permissions'] + (['--effort', 'xhigh'] if model.startswith('claude') else [])
            command_ro = ro
        with open(log, 'w') as fh:
            try:
                sandbox.run(command, ro=command_ro, rw=rw, env=env, cwd=str(ws), stdout=fh,
                            stderr=subprocess.STDOUT, timeout=remaining)
            except subprocess.TimeoutExpired:
                fh.write(json.dumps(dict(type='driver', event='timeout')) + '\n')
        ledger_add(exp, a, model, log, lock)
        text = log.read_text(errors='ignore')
        tail = text[-4000:]                          # the session ended on a quota error (at once, or mid-way through work)
        if any(p in tail for p in TRANSIENT_PAT):
            k -= 1                                   # an upstream outage does not use up a session
            switches = [t for t in switches if time.time() - t < 900] + [time.time()]
            alt = next((i for i in list(range(cur + 1, len(usable))) + list(range(0, cur))
                        if probe(*usable[i]) == 'ok'), None)
            if alt is not None and len(switches) <= 2 * len(usable):
                dlog(exp, f'{a} transient upstream error on {usable[cur][0]}:{usable[cur][1]}; '
                          f'switching to {usable[alt][0]}:{usable[alt][1]}')
                cur = alt
                continue
            wait = min(120, max(10, int(deadline - time.time())))
            dlog(exp, f'{a} transient upstream error on {usable[cur][0]}:{usable[cur][1]}; retrying in {wait}s')
            time.sleep(wait)
            continue
        if any(p in tail for p in QUOTA_PAT):
            k -= 1                                   # a quota stop does not use up a session
            # switch to another option that still has quota instead of idling
            alt = next((i for i in list(range(cur + 1, len(usable))) + list(range(0, cur)) if probe(*usable[i]) == 'ok'), None)
            switches = [t for t in switches if time.time() - t < 900] + [time.time()]
            if alt is not None and len(switches) <= 2 * len(usable):    # no ping-pong: at most 2 switches per option in 15 min
                dlog(exp, f'{a} quota on {usable[cur][0]}:{usable[cur][1]}; switching to {usable[alt][0]}:{usable[alt][1]}')
                cur = alt
                continue
            # no alternative: wait for the reset if given
            wait, m = 600, re.search(r'reset at (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ?([+-]\d{4})', tail)
            if m:
                reset = datetime.strptime(f'{m.group(1)} {m.group(2)}', '%Y-%m-%d %H:%M:%S %z').timestamp()
                wait = max(60, min(reset - time.time() + 30, deadline - time.time()))
            with open(exp / 'logs' / 'driver.log', 'a') as d:
                d.write(f"{time.strftime('%F %T')} {a} hit API quota/rate limit; sleeping {wait / 60:.0f} min"
                        + (f" (quota resets {m.group(1)})" if m else '') + "\n")
            time.sleep(wait)
            continue
        # a configuration error (dead key, retired model) ends every session at once and would burn all sessions silently
        # (e.g. a revoked key or a retired model): stop this agent and raise an alert the watcher sees
        if any(p in tail for p in CONFIG_PAT):
            k -= 1
            dlog(exp, f'{a} API configuration error on {prov}:{model}; option dropped')
            usable.pop(cur)
            if not usable:
                alert(f'FAILED {exp.name} {a} ({prov}:{model}): API configuration error and no other option, agent stopped')
                break
            cur = cur % len(usable)
            continue
        with open(exp / 'logs' / 'driver.log', 'a') as d:
            d.write(f"{time.strftime('%F %T')} {a} session {k} ended\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', type=Path, required=True)
    ap.add_argument('--agents', required=True, help="a1:route:model[|route:model][#focus],... (rounds/roster.py)")
    ap.add_argument('--datasets', type=Path, required=True,
                    help='json {ds: {dev_view, dev_cache, val_view, val_cache, base_dev_pred, base_val_pred}}')
    ap.add_argument('--base-tree', type=Path, default=BASE_TREE)
    ap.add_argument('--brief', type=Path, default=R / 'brief.md')
    ap.add_argument('--hours', type=float, default=4)
    ap.add_argument('--deadline-epoch', type=float, default=None)
    ap.add_argument('--budget', type=int, default=10)
    ap.add_argument('--min-gain', type=float, default=0.003, help='GA: minimum composite S of one submission (0.3%)')
    ap.add_argument('--max-regress', type=float, default=0.02)
    ap.add_argument('--max-sessions', type=int, default=6)
    ap.add_argument('--gate-parallel', type=int, default=3)
    ap.add_argument('--submit-cutoff-min', type=int, default=30)
    ap.add_argument('--jobs', type=int, default=16, help='harness.run worker processes of one gate replay')
    ap.add_argument('--one-pending', action='store_true')
    ap.add_argument('--no-gate', action='store_true')
    ap.add_argument('--plan', type=Path, default=None, help="Loop C's validated plan dir (ideas.md, brief_notes.md, focus.json)")
    a = ap.parse_args()
    exp = a.exp.resolve(); exp.mkdir(parents=True, exist_ok=True)
    agents = [parse_agent(x) for x in a.agents.split(',')]
    cfg_in = dict(datasets=json.loads(a.datasets.read_text()), budget_per_agent=a.budget, min_gain=a.min_gain,
                  max_regress=a.max_regress, precheck=True, dedup=True, jobs=a.jobs, agent_jobs=a.jobs,
                  gate_parallel=a.gate_parallel, submit_cutoff_min=a.submit_cutoff_min, max_sessions=a.max_sessions,
                  one_pending=a.one_pending, base_tree=str(a.base_tree.resolve()), round=exp.name,
                  plan=str(a.plan.resolve()) if a.plan else None)
    cfg = setup(exp, cfg_in, agents, a.brief.resolve(), a.base_tree.resolve())
    gate = None if a.no_gate else subprocess.Popen([sys.executable, str(R / 'gate_service.py'), '--exp', str(exp)],
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=GATE_CARDS),
                            stdout=open(exp / 'logs' / 'gate_stdout.log', 'a'), stderr=subprocess.STDOUT, start_new_session=True)
    deadline = a.deadline_epoch or (time.time() + a.hours * 3600)
    if not (exp / 'DEADLINE').is_file():
        (exp / 'DEADLINE').write_text(f'{deadline:.0f}\n')
    lock = threading.Lock()
    th = [threading.Thread(target=run_agent, args=(exp, n, opts, focus, '', None, '', deadline, cfg, lock))
          for n, opts, focus in agents]
    [t.start() for t in th]; [t.join() for t in th]
    tags = [f'LOOPA_WS={exp}/agents/{n}\0'.encode() for n, _, _ in agents]
    for pd in Path('/proc').iterdir():
        try:
            if pd.name.isdigit() and pd.stat().st_uid == 23001 and any(t in (pd / 'environ').read_bytes() for t in tags):
                os.kill(int(pd.name), signal.SIGKILL)
        except OSError:
            pass
    if gate is None:
        return
    while any((r / 'READY').exists() and not (r / 'verdict.json').exists() for r in (exp / 'spool').glob('*/req_*')):
        time.sleep(20)
    (exp / 'STOP').write_text(''); gate.wait(timeout=120)
    print(json.dumps(json.loads((exp / 'state.json').read_text()), indent=1))


if __name__ == '__main__':
    main()
