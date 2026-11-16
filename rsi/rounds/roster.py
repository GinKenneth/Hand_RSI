"""Loop A roster: 8 agents (4 on Claude Opus 5.5, 4 on GPT-6-sol), each with a fallback chain of routes and a focus
hint, plus the merge agent. All agents work on accuracy (no speed or pruning agent).
Syntax: 'name:route:model|fallback...#focus' (routes: loopA/endpoints.json; focus keys: loopA/driver.py FOCUS3D)."""
_K = 'compat:kimi-k3.1-pro|vision:kimi-k3.1-pro|vision:doubao-seed-2.1-pro'   # last-resort fallbacks
_O = f'claude:claude-opus-5-5|claude_b:claude-opus-5-5|{_K}'                 # 4 agents on Claude Opus 5.5
_P = f'codex:gpt-6-sol|codex_b:gpt-6-sol|gpt:gpt-5|{_K}'                     # 4 agents on GPT-6-sol (Codex CLI)
AGENTS = (f'a1:{_O}#detect,a2:{_P}#root,a3:{_O}#orient,a4:{_O}#fingers,'
          f'a5:{_O}#image,a6:{_P}#temporal,a7:{_P}#kp2d,a8:{_P}#oos')   # a7 / a8 focus: EPE2D-p and MPJPE+OOS
MERGE_AGENT = f'm1:{_O}'
SMOKE = f'a1:{_O}#detect'
