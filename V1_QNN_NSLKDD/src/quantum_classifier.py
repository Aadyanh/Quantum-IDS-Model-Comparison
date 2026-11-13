import os
import sys
import math
import subprocess
import numpy as np
import torch
import torch.nn as nn
from src.config import RESULTS_DIR
from src.quantum_encoder import angle_encode


class QNN(nn.Module):
    """angle encoding RY(pi*x_i) on every qubit -> N_LAYERS x [CNOT chain, RY(theta) layer] -> CNOT chain -> <Z_i>.
    RY and CNOT are real, so the circuit is one real orthogonal (2^n x 2^n) matrix R(theta), built once per step on the
    GPU (differentiable) and applied to the whole batch with one matmul. Gradients are exact (autograd).
    readout='all'   : Linear(<Z_0..Z_7>)                      (every qubit and parameter can matter)
    readout='mean4' : scale * mean(<Z0>..<Z3>) + bias          (the original design)"""

    def __init__(self, n_qubits=8, n_layers=4, readout="all"):
        super().__init__()
        self.n, self.L, self.readout = n_qubits, n_layers, readout
        self.theta = nn.Parameter(torch.empty(n_layers, n_qubits).uniform_(-math.pi, math.pi))
        N = 2 ** n_qubits
        xs = np.arange(N)
        sh = n_qubits - 1 - np.arange(n_qubits)              # qubit 0 = most significant bit
        bits = (xs[:, None] >> sh[None, :]) & 1
        y = ((np.cumsum(bits, 1) % 2) << sh[None, :]).sum(1)  # cx(0,1),cx(1,2),... == prefix XOR
        finv = np.empty(N, dtype=np.int64)
        finv[y] = xs
        self.register_buffer("finv", torch.from_numpy(finv))
        self.register_buffer("zs", torch.from_numpy((1 - 2 * bits).astype(np.float32)))   # (N, n)
        if readout == "all":
            self.head = nn.Linear(n_qubits, 1)
        else:
            self.head = nn.Linear(1, 1)
            with torch.no_grad():
                self.head.weight.fill_(3.0)
                self.head.bias.zero_()

    def features(self, x):                                   # (B, n) in [0,1] -> (B, 2^n) product state
        return angle_encode(x)

    def build_R(self):
        R = torch.eye(2 ** self.n, device=self.theta.device, dtype=self.theta.dtype)
        for l in range(self.L):
            R = R.index_select(1, self.finv)                 # CNOT chain = one permutation
            c, s = torch.cos(self.theta[l] / 2), torch.sin(self.theta[l] / 2)
            U = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)    # (n, 2, 2) RY matrices
            Ug = U[0]
            for i in range(1, self.n):
                Ug = torch.kron(Ug, U[i])
            R = R @ Ug.T
        return R.index_select(1, self.finv)                  # final CNOT chain

    def zexp(self, x):                                       # <Z_i> for every qubit, (B, n)
        out = self.features(x) @ self.build_R()
        return (out * out) @ self.zs

    def forward(self, x):
        z = self.zexp(x)
        z = z if self.readout == "all" else z[:, :4].mean(1, keepdim=True)
        return self.head(z).squeeze(-1)                      # logits (B,)


def cudaq_check(model, Xte):
    """Runs src/cudaq_verify.py (a CUDA-Q kernel of the same circuit) in a separate process and compares <Z_i>."""
    try:
        m = QNN(model.n, model.L, "all").double()
        m.theta.data.copy_(model.theta.detach().cpu().double())
        x = torch.tensor(Xte[:64], dtype=torch.float64)
        with torch.no_grad():
            Z = m.zexp(x).numpy()
        npz = os.path.join(RESULTS_DIR, "circuit_check.npz")
        np.savez(npz, x=x.numpy(), theta=m.theta.detach().numpy(), Z=Z)
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
