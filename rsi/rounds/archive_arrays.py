"""Move a finished stage's bulky per-segment arrays (.npz / .npy) off the working disk into one tar in the
archive (rsi_paths.ARCHIVE; on our host a 1 PB JuiceFS) instead of deleting them (keep everything the process produced).

  python rounds/archive_arrays.py --name R2_loopA --root loopA_runs/R2 [--root ...] [--glob 'loopB/cache/*/teacher_R2.npz']

Paths are stored relative to the rsi/ root; restore with  tar -xf <tar> -C <rsi root>
The local copies are removed only after the tar's member list and byte count match the selection; on any error the
local files stay. A manifest (path, size) is written next to the tar and to logs/archive/<name>.txt.
"""
import argparse
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

S = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S))
import rsi_paths  # noqa: E402
DEST = rsi_paths.ARCHIVE
STATUS = S / 'logs' / 'rounds.md'


def say(msg):
    with open(STATUS, 'a') as f:
        f.write(f"- {time.strftime('%m-%d %H:%M')} [archive] {msg}\n")


def select(roots, globs, keep_prefixes=()):
    files = []
    for r in roots:
        r = S / r
        if r.exists():
            # symlinks stay local (tar stores them as links, so the member check would never match; they take no
            # space)
            files += [f for f in r.rglob('*') if f.suffix in ('.npz', '.npy') and f.is_file() and not f.is_symlink()
                      and not any(str(f.relative_to(S)).startswith(k) for k in keep_prefixes)]
    for g in globs:
        files += [f for f in S.glob(g) if f.is_file() and not f.is_symlink()]
    return sorted(set(files))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', required=True)
    ap.add_argument('--root', action='append', default=[])
    ap.add_argument('--glob', action='append', default=[])
    ap.add_argument('--keep', action='append', default=[], help='path prefixes (relative to rsi/) that stay local')
    a = ap.parse_args()
    files = select(a.root, a.glob, tuple(a.keep))
    if not files:
        print(f'{a.name}: nothing to archive'); return          # not worth a status line
    rel = [str(f.relative_to(S)) for f in files]
    sizes = {r: (S / r).stat().st_size for r in rel}
    total = sum(sizes.values())
    DEST.mkdir(parents=True, exist_ok=True); (S / 'logs' / 'archive').mkdir(parents=True, exist_ok=True)
    tar = DEST / f'{a.name}.tar'
    if tar.exists():
        tar = DEST / f"{a.name}_{time.strftime('%Y%m%d_%H%M%S')}.tar"
    lst = S / 'logs' / 'archive' / f'{a.name}.list'
    lst.write_text('\n'.join(rel) + '\n')
    t0 = time.time()
    r = subprocess.run(['tar', '-cf', str(tar), '-C', str(S), '-T', str(lst)], capture_output=True, text=True)
    if r.returncode:
        say(f'{a.name}: tar FAILED ({r.stderr[-300:]}); local files kept'); sys.exit(1)
    with tarfile.open(tar) as t:
        mem = {m.name: m.size for m in t.getmembers() if m.isfile()}
    if mem != sizes:
        say(f'{a.name}: verification FAILED ({len(mem)} members vs {len(sizes)} files); local files kept'); sys.exit(1)
    man = '\n'.join(f'{p}\t{n}' for p, n in sizes.items()) + '\n'
    (DEST / f'{tar.stem}.manifest.tsv').write_text(man); (S / 'logs' / 'archive' / f'{a.name}.manifest.tsv').write_text(man)
    for p in rel:
        try:
            os.unlink(S / p)
        except OSError:
            pass
    say(f'{a.name}: {len(rel)} files, {total / 1e9:.1f} GB moved to {tar} in {time.time() - t0:.0f} s '
        f'(restore: tar -xf {tar} -C {S}); free {os.statvfs(S).f_bavail * os.statvfs(S).f_frsize / 1e9:.0f} GB')


if __name__ == '__main__':
    main()
