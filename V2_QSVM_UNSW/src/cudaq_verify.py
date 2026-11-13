"""Standalone check (CPU statevector, no torch): the original CUDA-Q angle-encoding kernel
must give the same Gram matrix as the fast GPU kernel.   Usage: python cudaq_verify.py kernel_check.npz"""
import sys
import numpy as np
import cudaq

cudaq.set_target("qpp-cpu")
d = np.load(sys.argv[1])
x, ref = d["x"], d["K"]
n = x.shape[1]


@cudaq.kernel
def angle_encoding(n: int, x: list[float]):
    q = cudaq.qvector(n)
    for i in range(n):
        ry(x[i] * 3.14159265359, q[i])


S = np.array([np.array(cudaq.get_state(angle_encoding, n, xi.tolist())) for xi in x])
K = np.abs(S.conj() @ S.T) ** 2
print(f"CUDA-Q (qpp-cpu) kernel check | samples: {len(x)} | max |CUDA-Q - fast kernel| = {np.abs(K - ref).max():.2e}")
