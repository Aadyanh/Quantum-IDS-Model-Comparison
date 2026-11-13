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


class QNN(nn.Module):
    """
    Same circuit as the V3 CUDA-Q kernel:
      amplitude-encode the L2-normalised vector -> N_LAYERS x [RX,RY,RZ on every qubit, CNOT chain] -> <Z0>
    The circuit depends on the input only through the initial state, so it is one fixed
    (2^n x 2^n) unitary per parameter setting. It is built once per step (on the GPU, differentiable)
    and applied to the whole batch with one matmul. Gradients are exact (autograd).
    A learnable scale+bias maps <Z0> in [-1,1] to a logit (monotone, so ROC/PR are unaffected).
    """
    def __init__(self, n_qubits=6, n_layers=4):
        super().__init__()
        self.n, self.L = n_qubits, n_layers
        self.thetas = nn.Parameter(torch.rand(n_layers * n_qubits * 3) * 2 * np.pi)  # layer*18 + qubit*3 + {rx,ry,rz}
        dim = 2 ** n_qubits
        xs = np.arange(dim)
        sh = n_qubits - 1 - np.arange(n_qubits)              # qubit 0 = most significant bit
        bits = (xs[:, None] >> sh[None, :]) & 1
        new_bits = np.cumsum(bits, axis=1) % 2               # cx(0,1),cx(1,2),... == prefix XOR
        y = (new_bits << sh[None, :]).sum(1)
        finv = np.empty(dim, dtype=np.int64)
        finv[y] = xs
        self.register_buffer("finv", torch.from_numpy(finv))
        self.register_buffer("z0", torch.from_numpy((1 - 2 * bits[:, 0]).astype(np.float32)))
        self.head = nn.Linear(1, 1)
        with torch.no_grad():
            self.head.weight.fill_(3.0)
            self.head.bias.zero_()

    @staticmethod
    def _mat(a, b, c, d):
        return torch.stack([torch.stack([a, b], -1), torch.stack([c, d], -1)], -2)

    def _unitary_rows(self):
        th = self.thetas.view(self.L, self.n, 3)
        R = torch.eye(2 ** self.n, dtype=torch.complex64, device=th.device)   # row b = basis state b
        for l in range(self.L):
            a, b, c = th[l, :, 0], th[l, :, 1], th[l, :, 2]
            cx, sx = torch.cos(a / 2), torch.sin(a / 2)
            cy, sy = torch.cos(b / 2), torch.sin(b / 2)
            cz, sz = torch.cos(c / 2), torch.sin(c / 2)
            zr = torch.zeros_like(cx)
            RX = torch.complex(self._mat(cx, zr, zr, cx), self._mat(zr, -sx, -sx, zr))
            ryr = self._mat(cy, -sy, sy, cy)
            RY = torch.complex(ryr, torch.zeros_like(ryr))
            RZ = torch.complex(self._mat(cz, zr, zr, cz), self._mat(-sz, zr, zr, sz))
            U = RZ @ RY @ RX                                 # rx first, then ry, then rz
            Ug = U[0]
            for i in range(1, self.n):
                Ug = torch.kron(Ug, U[i])
            R = R @ Ug.T                                     # apply to every basis-state row
            R = R.index_select(1, self.finv)                 # CNOT chain = one permutation
        return R

    def expval(self, x):                                     # x: (B, 2^n) L2-normalised, real
        R = self._unitary_rows()
        out = x.to(R.dtype) @ R
        probs = out.real ** 2 + out.imag ** 2
        return probs @ self.z0                               # <Z0>, shape (B,)

    def forward(self, x):
        return self.head(self.expval(x).unsqueeze(1))        # logits (B,1)


def predict(model, X, bsz=65536):
    """Logits for a tensor X that already lives on the model's device."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bsz):
            out.append(model(X[i:i + bsz]))
    return torch.cat(out)


# ---- self-test: simulator vs. an independent dense-matrix reference ----
def _dense_reference(x, th, n, L):
    I = np.eye(2, dtype=complex)

    def kall(ms):
        o = np.array([[1 + 0j]])
        for m in ms:
            o = np.kron(o, m)
        return o

    op = lambda m, i: kall([m if j == i else I for j in range(n)])
    rx = lambda a: np.array([[np.cos(a / 2), -1j * np.sin(a / 2)], [-1j * np.sin(a / 2), np.cos(a / 2)]])
    ry = lambda a: np.array([[np.cos(a / 2), -np.sin(a / 2)], [np.sin(a / 2), np.cos(a / 2)]], dtype=complex)
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

    psi = x.astype(complex)
    for l in range(L):
        for i in range(n):
            psi = op(rx(th[l, i, 0]), i) @ psi
            psi = op(ry(th[l, i, 1]), i) @ psi
            psi = op(rz(th[l, i, 2]), i) @ psi
        for i in range(n - 1):
            psi = cnot(i, i + 1) @ psi
    Z = np.diag([1, -1]).astype(complex)
    return float(np.real(np.vdot(psi, op(Z, 0) @ psi)))


def selftest():
    rng = np.random.default_rng(0)
    for n, L in [(6, 4), (4, 2)]:
        m = QNN(n, L)
        th = rng.random((L, n, 3)) * 2 * np.pi
        x = rng.random((3, 2 ** n))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        with torch.no_grad():
            m.thetas.copy_(torch.tensor(th.reshape(-1), dtype=torch.float32))
            out = m.expval(torch.tensor(x, dtype=torch.float32)).numpy()
        ref = np.array([_dense_reference(x[i], th, n, L) for i in range(3)])
        err = np.abs(out - ref).max()
        assert err < 1e-4, f"Simulator mismatch (n={n}, L={L}): max err {err}"
    print("Simulator self-test passed (matches dense reference).")
