# Dispersion-Resampling Boltzmann Method

## Comms

- Ask when unsure. Push back if user wrong — explain why.
- Verify before claiming done.
- Direct path first. Fall back only if it fails.
- Caveman mode is the default voice. Code, commits, PRs, security warnings: normal English.

## Tokens-first workflow

Treat context as the scarce resource. Defaults:

- Read `docs/ARCHITECTURE.md` first to locate code. It is the map. Re-grepping the tree is wasteful.
- Use `tags` (ctags index) for symbol lookup before Grep. Regenerate via `pwsh scripts/tags.ps1` after structural changes.
- Use Grep with `-n` + `-C 3` rather than reading whole files.
- For >3-query explorations, delegate to `caveman:cavecrew-investigator` (or `Explore`). Subagent output is caveman-compressed → ~60% fewer tokens than inlining the searches.
- For 1–2 file mechanical edits, delegate to `caveman:cavecrew-builder`.
- Do NOT re-Read a file you just Edited; the tool errors loudly on failure.
- Do NOT re-run `nvidia-smi` / `uv --version` / `wp.get_cuda_devices()` GPU checks each turn. Hardware + stack facts live below; trust them.
- `.agent-workspace/` is scratch — write smoke / debug scripts there and delete when done. Don't pollute other folders.

## Hardware

- GPU: **NVIDIA RTX 5080 Laptop** — Blackwell, compute capability **(12, 0)**, arch tag **sm_120**, 16 GB VRAM (~15.9 GiB usable).
- CUDA runtime: 12.8 (driver 591.44, CUDA-13.1-capable).
- Memory budget per kernel: stay under ~14 GiB working set to leave headroom for display + other processes.

## Toolchain

- Python: 3.12.13 (installed via `uv`).
- Package manager: **`uv`** (winget-installed at `%LOCALAPPDATA%\Microsoft\WinGet\Packages\astral-sh.uv_*\uv.exe`). Project uses `uv sync` + `uv run`. `pyproject.toml` is the source of truth; do not hand-edit `uv.lock`.
- TLS: corp cert intercept on this machine — `[tool.uv] system-certs = true` is required and already set. Do not remove.

## Stack

- Compute: **NVIDIA Warp** (`warp-lang`) — Python-authored CUDA kernels. Smoke-verified on this
  GPU: Warp **1.14.0**, bundled **CUDA Toolkit 12.9**, device `cuda:0` **sm_120**, mempool on.
- Same kernels run on `cpu` (debug) and `cuda:0` (scale) — develop on CPU, scale on GPU, one
  codebase. Decision + rationale: `docs/decisions/0001-stack-warp.md`.
- **`wp.HashGrid`** is the spatial-hash / radius-neighbor primitive backing the method's "grid
  data structure" (heir of DiRe-CFD `MultiGrid`). Rebuild it each step (points move).
- Dynamic point counts (moments/components/control points change per step): Warp arrays are
  fixed-size → preallocate + compact, don't grow/shrink. `dx/α` density threshold bounds counts.
- Warp JIT kernel cache lives under `%LOCALAPPDATA%\NVIDIA\warp\Cache` — first compile per
  kernel is slow (~0.5 s), then cached.

## Commands

```pwsh
# Install / update deps (CPU + dev)
uv sync --extra dev

# Lint / format
uv run ruff check .
uv run ruff format .

# Regenerate ctags index
pwsh scripts/tags.ps1

# Run an experiment script
uv run python experiments/exp_<slug>.py

# Tests
uv run pytest
```

## Repo layout

```
direbm/            # the library (Warp). Modules mirror the method — see docs/ARCHITECTURE.md
research/          # research vision (idea.md canonical) + progress.md (running log of progress/findings/future directions)
docs/              # ARCHITECTURE.md + decisions/ (ADRs) + results/exp_<slug>.md
scripts/           # helper scripts
experiments/       # repeatable experiments (exp_<slug>.py) + tools
tests/             # pytest
.agent-workspace/  # ephemeral scratch (gitignored)
tags               # ctags index (gitignored)
```

## Conventions

- Public symbols re-exported from each package's `__init__.py`.
- Experiments: `experiments/exp_<slug>.py` + writeup at `docs/results/exp_<slug>.md`. Numbering monotonic.
- Comments: write the WHY when it isn't obvious. No what-comments.
- Research log: append progress, findings, and future directions to `research/progress.md` (newest first, ISO dates). Per-experiment detail still goes in `docs/results/exp_<slug>.md`; `progress.md` is the cross-experiment narrative.