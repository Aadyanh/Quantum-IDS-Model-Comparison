"""Standalone check (CPU statevector, no torch): a CUDA-Q kernel of the circuit must give the same <Z_i> as the GPU
simulator.   Usage: python cudaq_verify.py circuit_check.npz"""
import sys
import numpy as np
import cudaq

cudaq.set_target("qpp-cpu")
d = np.load(sys.argv[1])
x, theta, ref = d["x"], d["theta"], d["Z"]
L, n = theta.shape


@cudaq.kernel
def qnn(n: int, L: int, feat: list[float], theta: list[float]):
    q = cudaq.qvector(n)
    for i in range(n):
        ry(feat[i] * 3.14159265359, q[i])
    for l in range(L):
        for i in range(n - 1):
            cx(q[i], q[i + 1])
        for i in range(n):
            ry(theta[l * n + i], q[i])
    for i in range(n - 1):
        cx(q[i], q[i + 1])


th = theta.reshape(-1).astype(float).tolist()
Z = np.array([[cudaq.observe(qnn, cudaq.spin.z(i), n, L, xi.astype(float).tolist(), th).expectation()
               for i in range(n)] for xi in x])
print(f"CUDA-Q (qpp-cpu) circuit check | samples: {len(x)} | max |CUDA-Q - GPU simulator| = {np.abs(Z - ref).max():.2e}")
