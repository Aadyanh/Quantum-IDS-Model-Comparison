import os
import sys
import math
import subprocess
import numpy as np
import torch
import torch.nn as nn
from src.config import RESULTS_DIR


class QSVM(nn.Module):
    """6-qubit angle-encoding quantum kernel SVM in primal form.
    psi(x) = tensor_i [cos(pi*s*x_i/2), sin(pi*s*x_i/2)]  (= RY(pi*s*x_i)|0> on every qubit)
    K(x,x') = |<psi|psi'>|^2 = (psi . psi')^2  ->  f(x) = psi^T W psi + b,  RKHS norm = ||W||_F."""

    def __init__(self, scale=1.0, n_qubits=6):
        super().__init__()
        self.scale, self.n = scale, n_qubits
        self.W = nn.Parameter(torch.zeros(2 ** n_qubits, 2 ** n_qubits))
        self.b = nn.Parameter(torch.zeros(()))

    def features(self, x):                                # (B, n) in [0,1] -> (B, 2^n) quantum state
        ang = math.pi * self.scale * x
        a = torch.stack([torch.cos(ang / 2), torch.sin(ang / 2)], -1)      # (B, n, 2)
        phi = a[:, 0]
        for i in range(1, self.n):
            phi = (phi[:, :, None] * a[:, i, None, :]).reshape(len(x), -1)
        return phi

    def sym(self):
        return 0.5 * (self.W + self.W.T)

    def l2(self):
        return (self.sym() ** 2).sum()

    def forward(self, x):
        phi = self.features(x)
        return ((phi @ self.sym()) * phi).sum(1) + self.b


def qsvm_kernel(X1, X2, scale=1.0):
    """Gram matrix of the fidelity quantum kernel between two (small) sets, computed on the GPU."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = QSVM(scale, X1.shape[1])
    with torch.no_grad():
        p1 = m.features(torch.as_tensor(X1, dtype=torch.float32, device=dev))
        p2 = m.features(torch.as_tensor(X2, dtype=torch.float32, device=dev))
        return ((p1 @ p2.T) ** 2).cpu().numpy()


def cudaq_check(qsvm, Xte):
    """Runs src/cudaq_verify.py (original CUDA-Q statevector kernel) in a separate process."""
    try:
        x = torch.tensor(Xte[:100], dtype=torch.float64)
        m = QSVM(qsvm.scale, qsvm.n).double()
        phi = m.features(x)
        K = ((phi @ phi.T) ** 2).numpy()
        npz = os.path.join(RESULTS_DIR, "kernel_check.npz")
        np.savez(npz, x=(qsvm.scale * x).numpy(), K=K)
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cudaq_verify.py")
        r = subprocess.run([sys.executable, script, npz], capture_output=True, text=True, timeout=900)
        lines = [l for l in r.stdout.splitlines() if l.startswith("CUDA-Q")]
        if lines:
            print("\n" + lines[-1])
            return lines[-1]
        print("\nCUDA-Q check failed:\n", r.stderr[-800:])
    except Exception as e:
        print("\nCUDA-Q check skipped:", e)
    return "not run"
