"""The single, cross-round bucket (everything goes into the bucket, Loop A and Loop C alike, so every
agent sees how the harness and the process evolved).

  bucket/events.jsonl        one JSON line per event, append-only
  bucket/items/<id>/         the event's files: notes, diffs, transcripts (gz), Loop C plans

Event kinds:
  submission    a Loop A gate verdict (accept / reject / dev-precheck-fail / bad-files): note, modules, DEV per-column
                deltas, the coarse val S the agent itself was told; files: note.md, diff.patch
  agent_log     one Loop A agent at the end of the round: notes.md + session transcripts (its reasoning record)
  merge         greedy-merge steps (dev) and which tree became the teacher
  loopB         training run summary;   gate_B   round verdict (accepted / rejected only)
  loopC_plan    Loop C's analysis for the next round (public part), ideas, focus, Loop B changes, proposals
  plan_applied  what actually took effect at the start of a round
No val or test number is ever written here beyond the coarse val S each agent already receives from the gate.
  python rounds/bucket.py [--round R1] [--kind submission] [--grep W] [--all] [--show ID]
"""
from __future__ import annotations

import argparse
import difflib
import fcntl
import gzip
import json
import os
import shutil
import time
from pathlib import Path

S = Path(__file__).resolve().parents[1]
BUCKET = Path(os.environ.get('RSI_BUCKET', S / 'bucket'))


def _ensure():
    (BUCKET / 'items').mkdir(parents=True, exist_ok=True)
    for p in (BUCKET, BUCKET / 'items'):
        os.chmod(p, 0o755)
    ev = BUCKET / 'events.jsonl'
    if not ev.exists():
        ev.write_text('')
    os.chmod(ev, 0o644)


def add(kind: str, rnd: str, id_: str, summary: str, fields: dict | None = None, files: dict | None = None) -> None:
    """files: {name: str | bytes | Path}; a Path is copied (gzip-compressed when the name ends in .gz)."""
    _ensure()
    names = []
    if files:
        item = BUCKET / 'items' / id_
        item.mkdir(parents=True, exist_ok=True); os.chmod(item, 0o755)
        for name, c in files.items():
            t = item / name
            if isinstance(c, Path):
                if not c.is_file():
                    continue
                if name.endswith('.gz') and not c.name.endswith('.gz'):
                    with open(c, 'rb') as fi, gzip.open(t, 'wb') as fo:
                        shutil.copyfileobj(fi, fo)
                else:
                    shutil.copyfile(c, t)
            elif isinstance(c, bytes):
                t.write_bytes(c)
            else:
                t.write_text(c)
            os.chmod(t, 0o644); names.append(name)
    ev = dict(time=time.strftime('%F %T'), kind=kind, round=rnd, id=id_, summary=summary, **(fields or {}), files=names)
    with open(BUCKET / 'events.jsonl', 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(ev, default=float) + '\n')
        fcntl.flock(f, fcntl.LOCK_UN)


def tree_files(tree: Path | None) -> dict:
    if tree is None or not Path(tree).is_dir():
        return {}
    out = {}
    for p in sorted(Path(tree).rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py', '.json', '.md'):
            out[str(p.relative_to(tree))] = p.read_text(errors='replace')
    return out


def tree_diff(old: Path | None, new: Path) -> str:
    a, b = tree_files(old), tree_files(new)
    lines = []
    for name in sorted(set(a) | set(b)):
        if a.get(name) != b.get(name):
            lines += difflib.unified_diff(a.get(name, '').splitlines(True), b.get(name, '').splitlines(True),
                                          f'parent/{name}', f'new/{name}')
    return ''.join(lines) or '(no change)\n'


def read(path: Path | None = None) -> list[dict]:
    p = path or BUCKET / 'events.jsonl'
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def line(e: dict) -> str:
    k = e.get('kind')
    if k == 'submission':
        rel = ' '.join(f"{m}{100 * v:+.1f}%" for m, v in (e.get('dev_rel') or {}).items())
        return (f"[{e['time'][5:16]}] {e['round']} {e['id']} {e.get('verdict')} valS~{e.get('S_coarse', '-')} "
                f"dev[{rel}] | {','.join(e.get('modules') or [])} | {e.get('summary', '')[:300]}")
    return f"[{e['time'][5:16]}] {e['round']} {k} {e['id']} | {e.get('summary', '')[:400]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--round'); ap.add_argument('--kind'); ap.add_argument('--grep')
    ap.add_argument('--all', action='store_true'); ap.add_argument('--show')
    ap.add_argument('--file', type=Path)
    a = ap.parse_args()
    rows = read(a.file)
    if a.show:
        e = next((r for r in rows if r['id'] == a.show), None)
        if e is None:
            raise SystemExit(f'no event {a.show}')
        print(json.dumps(e, indent=1, default=float))
        base = (a.file.parent if a.file else BUCKET) / 'items' / a.show
        for n in e.get('files', []):
            if not n.endswith('.gz'):
                print(f'\n===== {n} =====\n' + (base / n).read_text(errors='replace')[:20000])
            else:
                print(f'\n===== {n} (gzip; zcat {base / n}) =====')
        return
    if a.round:
        rows = [r for r in rows if r.get('round') == a.round]
    if a.kind:
        rows = [r for r in rows if r.get('kind') == a.kind]
    if a.grep:
        rows = [r for r in rows if a.grep.lower() in json.dumps(r).lower()]
    if not a.all:
        rows = rows[-60:]
    print('\n'.join(line(r) for r in rows) or '(empty)')


if __name__ == '__main__':
    main()
