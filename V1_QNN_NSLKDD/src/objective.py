import torch


@torch.no_grad()
def scores(model, X, chunk=65536):
    model.eval()
    return torch.cat([model(X[i:i + chunk]) for i in range(0, len(X), chunk)]).cpu().numpy()


def feature_influence(model, X, n=5000):
    """Share (%) of the model's sensitivity to each input feature: mean |d logit / d x_i|."""
    model.eval()
    x = X[:n].clone().requires_grad_(True)
    model(x).sum().backward()
    model.zero_grad(set_to_none=True)
    g = x.grad.abs().mean(0).cpu().numpy()
    return 100 * g / max(g.sum(), 1e-12)
