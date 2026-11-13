import math
import numpy as np
import torch
import torch.nn as nn


def get_device():
    n_gpu = torch.cuda.device_count()
    device = torch.device("cuda" if n_gpu else "cpu")
    print(f"Device: {device} | GPUs: {n_gpu} | " + ", ".join(torch.cuda.get_device_name(i) for i in range(n_gpu)))
    if n_gpu == 0:
        print("WARNING: no GPU found. Turn on an accelerator (GPU T4 x2 / P100) in the notebook settings.")
    return device


def wrap_parallel(model, use_multi):
    if use_multi and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        return nn.DataParallel(model)
    return model


def predict(net, X, device, bsz=1024):
    """Logits (numpy) for a numpy feature matrix."""
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bsz):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + bsz])).to(device)
            out.append(net(xb).float().cpu())
    return torch.cat(out).numpy().ravel()


class QuantumLayer(nn.Module):
    """
    Exact, batched, differentiable statevector simulation of the same circuit family
    as the CUDA-Q kernel:  RY(pi*x_i) encoding -> [RX, RZ per qubit -> CNOT chain] x L
    Returns <Z_i> for every qubit i.  Gradients come from autograd (exact), so no
    parameter-shift circuits are needed.
    """
    def __init__(self, n_qubits=16, n_layers=2):
        super().__init__()
        self.n, self.L = n_qubits, n_layers
        self.thetas = nn.Parameter(torch.rand(n_layers, 2, n_qubits) * 2 * np.pi)  # [layer, (rx, rz), qubit]
        dim = 2 ** n_qubits
        xs = np.arange(dim)
        sh = n_qubits - 1 - np.arange(n_qubits)                 # qubit 0 = most significant bit
        bits = (xs[:, None] >> sh[None, :]) & 1                 # (dim, n)
        new_bits = np.cumsum(bits, axis=1) % 2                  # cx(0,1),cx(1,2).. == prefix XOR
        y = (new_bits << sh[None, :]).sum(1)
        finv = np.empty(dim, dtype=np.int64)
        finv[y] = xs                                            # new_state[y] = old_state[x]
        self.register_buffer("finv", torch.from_numpy(finv))
        self.register_buffer("zsign", torch.from_numpy((1 - 2 * bits).astype(np.float32)))

    @staticmethod
    def _unitaries(th):                                         # th: (2, n) -> (n, 2, 2) complex  (RZ @ RX)
        rx, rz = th[0], th[1]
        c, s = torch.cos(rx / 2), torch.sin(rx / 2)
        cp, sp = torch.cos(rz / 2), torch.sin(rz / 2)
        re = torch.stack([torch.stack([cp * c, -sp * s], -1), torch.stack([sp * s, cp * c], -1)], -2)
        im = torch.stack([torch.stack([-sp * c, -cp * s], -1), torch.stack([-cp * s, sp * c], -1)], -2)
        return torch.complex(re, im)

    def _apply_group(self, psi, Ug, s, k):
        B, R = psi.shape[0], 2 ** (self.n - s - k)
        if R == 1:
            return (psi.reshape(-1, 2 ** k) @ Ug.T).reshape(B, -1)
        return torch.matmul(Ug, psi.reshape(B * 2 ** s, 2 ** k, R)).reshape(B, -1)

    def forward(self, x):                                       # x: (B, n) in [0,1]
        B, n = x.shape
        a = x * (math.pi / 2)
        q = torch.stack([torch.cos(a), torch.sin(a)], -1)       # RY(pi*x)|0>
        psi = q[:, 0, :]
        for i in range(1, n):
            psi = (psi.unsqueeze(2) * q[:, i, :].unsqueeze(1)).reshape(B, -1)
        psi = torch.complex(psi, torch.zeros_like(psi))
        for l in range(self.L):
            U = self._unitaries(self.thetas[l])
            s = 0
            while s < n:                                        # fuse up to 4 single-qubit gates per matmul
                k = min(4, n - s)
                Ug = U[s]
                for j in range(s + 1, s + k):
                    Ug = torch.kron(Ug, U[j])
                psi = self._apply_group(psi, Ug, s, k)
                s += k
            psi = psi.index_select(1, self.finv)                # whole CNOT chain = one permutation
        probs = psi.real ** 2 + psi.imag ** 2
        return probs @ self.zsign                               # (B, n)  <Z_i>


class HybridQNN(nn.Module):
    def __init__(self, input_features, n_qubits=16, n_layers=2, hidden=64):
        super().__init__()
        if hidden:
            self.fc1 = nn.Sequential(nn.Linear(input_features, hidden), nn.ReLU(), nn.Linear(hidden, n_qubits))
        else:
            self.fc1 = nn.Linear(input_features, n_qubits)
        self.q_layer = QuantumLayer(n_qubits, n_layers)
        self.fc2 = nn.Linear(n_qubits, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))      # -> [0,1] encoding angles
        x = self.q_layer(x)                 # -> <Z_i> in [-1,1]
        return self.fc2(x)                  # logits (sigmoid is folded into BCEWithLogits)


# ---- self-test: simulator vs. an independent dense-matrix reference ----
def _dense_reference(x, th, n, L):
    I = np.eye(2, dtype=complex)

    def kall(ms):
        o = np.array([[1 + 0j]])
        for m in ms:
            o = np.kron(o, m)
        return o

    op = lambda m, i: kall([m if j == i else I for j in range(n)])
    ry = lambda a: np.array([[np.cos(a / 2), -np.sin(a / 2)], [np.sin(a / 2), np.cos(a / 2)]], dtype=complex)
    rx = lambda a: np.array([[np.cos(a / 2), -1j * np.sin(a / 2)], [-1j * np.sin(a / 2), np.cos(a / 2)]])
    rz = lambda a: np.diag([np.exp(-1j * a / 2), np.exp(1j * a / 2)])

    def cnot(c, t):
        d = 2 ** n
        P = np.zeros((d, d))
        for b in range(d):
            bt = [(b >> (n - 1 - k)) & 1 for k in range(n)]
            if bt[c]:
                bt[t] ^= 1
            P[sum(bt[k] << (n - 1 - k) for k in range(n)), b] = 1
        return P

    psi = np.zeros(2 ** n, complex)
    psi[0] = 1
    for i in range(n):
        psi = op(ry(np.pi * x[i]), i) @ psi
    for l in range(L):
        for i in range(n):
            psi = op(rx(th[l, 0, i]), i) @ psi
            psi = op(rz(th[l, 1, i]), i) @ psi
        for i in range(n - 1):
            psi = cnot(i, i + 1) @ psi
    Z = np.diag([1, -1]).astype(complex)
    return np.array([np.real(np.vdot(psi, op(Z, i) @ psi)) for i in range(n)])


def selftest():
    rng = np.random.default_rng(0)
    for n, L in [(9, 2), (4, 1)]:
        ql = QuantumLayer(n, L)
        th = rng.random((L, 2, n)) * 2 * np.pi
        x = rng.random((3, n))
        with torch.no_grad():
            ql.thetas.copy_(torch.tensor(th, dtype=torch.float32))
            out = ql(torch.tensor(x, dtype=torch.float32)).numpy()
        ref = np.stack([_dense_reference(x[i], th, n, L) for i in range(3)])
        err = np.abs(out - ref).max()
        assert err < 1e-4, f"Simulator mismatch (n={n}, L={L}): max err {err}"
    print("Simulator self-test passed (matches dense reference).")
