"""Smoke tests: package imports and Warp can compile + run a kernel (CPU, no GPU needed)."""

import numpy as np

import direbm


def test_version():
    assert direbm.__version__


def test_warp_cpu_kernel():
    import warp as wp

    wp.init()

    @wp.kernel
    def add_one(x: wp.array(dtype=float)):
        i = wp.tid()
        x[i] = x[i] + 1.0

    a = wp.array(np.zeros(8, dtype=np.float32), dtype=float, device="cpu")
    wp.launch(add_one, dim=8, inputs=[a], device="cpu")
    assert np.allclose(a.numpy(), 1.0)
