# DiReBM — Dispersion-Resampling Boltzmann Method

A **latticeless** Lattice-Boltzmann method for fluid flow: keep the Boltzmann/BGK collision and
the discrete velocity set, but throw away the fixed spatial lattice. Distribution-function
values live at points chosen dynamically in (possibly unbounded) space, so resolution can vary
locally in space and time — at O(k·n) cost.

This is a revitalization and improvement of the prior project **DiRe-CFD** and of the 2020 MSc
TDK thesis *"The Dispersion-Resampling Boltzmann Method"* (ELTE IK). The headline improvement
target is **GPU parallelization**, which the original work flagged as its main open problem.

- **Pick up here:** [`HANDOFF.md`](HANDOFF.md) — full state + the next direction (3D GPU port).
- Method writeup: [`research/idea.md`](research/idea.md) (canonical).
- Progress log: [`research/progress.md`](research/progress.md).
- Code map: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Stack

Python 3.12 + [NVIDIA Warp](https://github.com/NVIDIA/warp) (Python-authored CUDA kernels,
built-in `wp.HashGrid` for radius-neighbor queries). Managed with [`uv`](https://docs.astral.sh/uv/).

```pwsh
uv sync --extra dev                       # install
uv run python experiments/exp_<slug>.py   # run an experiment
uv run ruff check .                        # lint
```

## License

Dual-licensed: MIT OR NPOSL-3.0.
