
import sys, time, numpy as np
import cudaq

data = np.load(sys.argv[1])
x, th, ref = data["x"], data["th"], data["ref"]
m, n = x.shape
L = th.shape[0]

try:
    cudaq.set_target("nvidia")
    tgt = "nvidia (GPU)"
except Exception:
    cudaq.set_target("qpp-cpu")
    tgt = "qpp-cpu (CPU fallback)"

@cudaq.kernel
def circ(n: int, L: int, x: list[float], th: list[float]):
    q = cudaq.qvector(n)
    for i in range(n):
        ry(x[i] * 3.14159265359, q[i])
    for l in range(L):
        for i in range(n):
            rx(th[l * 2 * n + i], q[i])
            rz(th[l * 2 * n + n + i], q[i])
        for i in range(n - 1):
            cx(q[i], q[i + 1])

hams = [cudaq.spin.z(i) for i in range(n)]
ths = th.reshape(-1).astype(float).tolist()
out = np.zeros((m, n))
t0 = time.time()
for s in range(m):
    xs = x[s].astype(float).tolist()
    for i in range(n):
        out[s, i] = cudaq.observe(circ, hams[i], n, L, xs, ths).expectation()
err = np.abs(out - ref).max()
print(f"CUDA-Q target: {tgt} | samples: {m} | max |CUDA-Q - GPU simulator| = {err:.2e} | {time.time() - t0:.1f}s")
