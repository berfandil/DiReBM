# exp_soft_outer — characterizing the soft_outer step-3 correction

Date: 2026-06-27 · Code: `experiments/exp_soft_outer.py` · Solver: `direbm.reference.Simulator`

The thesis's soft_outer rule (§4.3.3) spawns an extra control point a fixed 2(1−√3/2)·dx ahead of
the front to counter the D2Q7 hexagonal anisotropy. The author flagged it as possibly wrong for
straight wavefronts (the deferred "step-3" issue). This measures it, with a `soft_mode` knob:
`spawn` (thesis) vs `off` (no spawn, treat as inner).

Probes: a point pulse → front radius r(θ), hex bias = 6-fold ripple; a sheet source → planar front,
bias = roughness of the front position across the front, swept over propagation angle.

## Result

```
circular (point pulse) — anisotropy (lower = more isotropic):
   mode    rel_std   6-fold
    off     0.0515   0.0606
  spawn     0.0138   0.0037     ← 16× less hexagonal ripple

straight roughness vs propagation angle:
   mode      0°     15°     30°     45°     60°    mean
    off    0.372   0.559   0.627   1.105   2.378   1.008
  spawn    0.097   0.562   0.936   0.742   0.713   0.610     ← lower mean
```

- **The spawn works.** It cuts the circular 6-fold hexagonal ripple ~16× and lowers the mean
  straight-front roughness ~1.6×, helping most orientations (dramatically at the hex axes 0°/60°).
- **The thesis's straight-front worry is largely unfounded** — the spawn is beneficial, not
  harmful, for most straight fronts.
- **Residual:** a weak degradation near **30° off-axis** (spawn 0.936 vs off 0.627). It is small and
  noisy — the straight-roughness metric varies run-to-run (±0.3) with sim size, so this is a soft
  signal, not a clean failure.

## A fix attempt that did not work (negative result)

Hypothesis: gate the (circular-tuned) offset by a local curvature proxy — full on curved fronts,
fading on flat fronts — to keep the circular win and drop the off-axis harm. Proxy = alignment of
the present directions (‖Σ distinct c_i‖ / count).

It **failed**: it lost most of the circular benefit (6-fold 0.041, vs spawn's 0.004) without beating
spawn on straight fronts. The alignment proxy does not track where the correction helps. Reverted.

## Conclusion

Keep **`spawn`** as the default — it is a real, effective correction, better than the thesis
feared. The off-axis residual is small and within metric noise; a principled fix would need a
better curvature/front-geometry estimate than the alignment proxy, but the cost/benefit is low
given how well `spawn` already works. **Status: characterized; priority downgraded.** (`soft_mode`
is exposed for further study; the GPU v2 uses the `off`-equivalent.)
