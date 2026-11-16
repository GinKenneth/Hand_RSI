"""Run a command as the unprivileged `rsiagent` user inside a whitelisted filesystem view.

A private mount namespace hides /data0, /data1, /mnt, /root and /home behind empty tmpfs and binds
back only the whitelisted paths (read-only unless listed as writable). The process keeps network
access (Claude Code must reach the model API) but cannot see validation/test labels, other people's
data, or anything outside the whitelist.

  sandbox.py --ro PATH ... --rw PATH ... --env K=V ... --cwd DIR -- CMD ARGS...
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rsi_paths  # noqa: E402
from ace_eval.mano import MANO_DIR  # noqa: E402

S = rsi_paths.ROOT
UID = GID = rsi_paths.SANDBOX_UID
HIDE = sorted({'/data0', '/data1', '/mnt', '/root', '/home', '/logs', '/source', '/deployment', '/srv', '/media',
               *(str(p) for p in Path('/').glob('data*') if p.is_dir())})   # every /data* volume (other people's projects)
COMMON_RO = [str(rsi_paths.VENV), MANO_DIR, str(Path(rsi_paths.CLAUDE_BIN).parent), str(rsi_paths.ACE_EVAL),
             str(S / 'rsi_paths.py'), str(S / 'harness/core.py'), str(S / 'harness/run.py'), str(S / 'harness/__init__.py'),
             str(S / 'eval'), str(S / 'loopA/tools'), str(S / 'rounds/bucket.py'),
             rsi_paths.WILOR_ROOT]   # WiLoR weights for the image-evidence modules, read-only in every sandbox
# val and test GT are never in any whitelist


def _interpreter_root(venv: str = str(rsi_paths.VENV)) -> list[str]:
    """The venv's base interpreter (pyvenv.cfg 'home'). E.g. /usr/bin, or a uv build under
    /home/ubuntu, which HIDE covers, so it has to be bound back read-only."""
    try:
        home = next(l.split('=', 1)[1].strip() for l in open(f'{venv}/pyvenv.cfg') if l.startswith('home'))
    except (OSError, StopIteration):
        return []
    root = str(Path(home).resolve().parent)
    return [root] if any(root == h or root.startswith(h + '/') for h in HIDE) else []


COMMON_RO += _interpreter_root()


def build_script(ro: list[str], rw: list[str], cmd: list[str], env: dict, cwd: str) -> str:
    stage = f'/run/loopa_stage_{uuid.uuid4().hex[:10]}'
    lines = ['set -e', 'umask 022', 'mount --make-rprivate /', f'mkdir -p {stage}', f'mount -t tmpfs tmpfs {stage}']
    binds = [(p, False) for p in ro] + [(p, True) for p in rw]
    for i, (p, writable) in enumerate(binds):
        s = f'{stage}/{i}'
        kind = 'd' if os.path.isdir(p) else 'f'
        lines += [f'mkdir -p {s}' if kind == 'd' else f'touch {s}', f'mount --bind {shlex.quote(p)} {s}']
        if not writable:
            lines.append(f'mount -o remount,bind,ro {s}')
    for h in HIDE:
        lines.append(f'[ -d {h} ] && mount -t tmpfs -o mode=755 tmpfs {h}')
    lines.append('mount -t tmpfs -o mode=1777 tmpfs /tmp')
    for i, (p, writable) in enumerate(binds):
        s = f'{stage}/{i}'
        if os.path.isdir(p):
            lines += [f'mkdir -p {shlex.quote(p)}', f'mount --bind {s} {shlex.quote(p)}']
        else:
            lines += [f'mkdir -p {shlex.quote(os.path.dirname(p))}', f'touch {shlex.quote(p)}', f'mount --bind {s} {shlex.quote(p)}']
    envs = ' '.join(shlex.quote(f'{k}={v}') for k, v in env.items())
    lines.append(f'cd {shlex.quote(cwd)}')
    lines.append(f'exec setpriv --reuid={UID} --regid={GID} --clear-groups env -i {envs} {" ".join(shlex.quote(c) for c in cmd)}')
    return '\n'.join(lines)


def run(cmd: list[str], *, ro=(), rw=(), env=None, cwd='/tmp', timeout=None, stdout=None, stderr=None,
        net: bool = True) -> subprocess.CompletedProcess:
    """net=False adds a private network namespace (loopback only): harness replays by the gate and Loop B need no
    network, and without it a module cannot fetch outside data at run time. Agent sessions keep net=True."""
    rw = list(dict.fromkeys(p for p in rw if os.path.exists(p)))          # de-duplicate: a second bind of the same
    ro = list(dict.fromkeys(p for p in [*COMMON_RO, *ro] if os.path.exists(p) and p not in rw))   # path fails (EROFS)
    base_env = dict(PATH=f'{rsi_paths.VENV}/bin:{Path(rsi_paths.CLAUDE_BIN).parent}:{S}/loopA/tools:/usr/local/bin:/usr/bin:/bin',
                    LANG='C.UTF-8', HOME=cwd, PYTHONDONTWRITEBYTECODE='1', LOOPA_PY=rsi_paths.PY,
                    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',   # torch >= 2.6: WiLoR's YOLO checkpoint
                    WILOR_MINI_ROOT=rsi_paths.WILOR_ROOT, ACE_EVAL_MANO_DIR=MANO_DIR,   # the env is cleared in the sandbox
                    # at most 8 BLAS / OpenMP threads per process (library default: one per core); harness/run.py
                    # workers use 1
                    OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', MKL_NUM_THREADS='8', NUMEXPR_NUM_THREADS='8')
    base_env.update(env or {})
    script = build_script(list(ro), list(rw), cmd, base_env, cwd)
    # own process group, killed as a whole on timeout (subprocess.run's timeout kills only unshare; a replay's harness.run
    # and its workers would be re-parented to init and run on for hours after the gate had given up)
    p = subprocess.Popen(['unshare', '--mount', *([] if net else ['--net']), '--propagation', 'private', 'bash', '-c', script],
                         stdout=stdout, stderr=stderr, text=True, start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, err = p.communicate()
        raise subprocess.TimeoutExpired(p.args, timeout, output=out, stderr=err)
    except BaseException:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        raise
    return subprocess.CompletedProcess(p.args, p.returncode, out, err)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ro', action='append', default=[])
    ap.add_argument('--rw', action='append', default=[])
    ap.add_argument('--env', action='append', default=[])
    ap.add_argument('--cwd', default='/tmp')
    ap.add_argument('cmd', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ['--'] else a.cmd
    r = run(cmd, ro=a.ro, rw=a.rw, env=dict(e.split('=', 1) for e in a.env), cwd=a.cwd)
    sys.exit(r.returncode)
