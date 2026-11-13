"""
CUDA-Q cross-check. The trained circuit is re-run with real CUDA-Q in a SEPARATE process
(so PyTorch's and CUDA-Q's LLVM never share one process) and compared with the GPU simulator.
This file must not import cudaq at the top level.
"""
import os
import sys
import subprocess
import numpy as np
import torch
from src.config import RESULTS_DIR

CHECK_SCRIPT = r"""
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
"""


def verify_with_cudaq(model, Xte, device, n_samples=8):
    msg = "not run"
    try:
        model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(np.ascontiguousarray(Xte[:n_samples])).to(device)
            a = torch.sigmoid(model.fc1(xb))                    # circuit inputs in [0, 1]
            ref = model.q_layer(a)
        npz = os.path.join(RESULTS_DIR, "cudaq_check.npz")
        np.savez(npz, x=a.cpu().numpy().astype(np.float64),
                 th=model.q_layer.thetas.detach().cpu().numpy().astype(np.float64),
                 ref=ref.cpu().numpy().astype(np.float64))
        script = os.path.join(RESULTS_DIR, "cudaq_verify_run.py")
        with open(script, "w") as f:
            f.write(CHECK_SCRIPT)
        r = subprocess.run([sys.executable, script, npz], capture_output=True, text=True, timeout=900)
        lines = [l for l in r.stdout.strip().splitlines() if l.startswith("CUDA-Q target")]
        if lines:
            msg = lines[-1]
        elif "No module named" in r.stderr and "cudaq" in r.stderr:
            msg = "skipped (cudaq is not installed; pip install cuda-quantum-cu12 to enable)"
        else:
            msg = "failed: " + r.stderr[-400:]
    except Exception as e:
        msg = f"skipped ({e})"
    print("\nCUDA-Q cross-check:", msg)
    with open(os.path.join(RESULTS_DIR, "cudaq_check.txt"), "w") as f:
        f.write(msg + "\n")
    return msg
