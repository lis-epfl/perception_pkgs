"""Fused RANSAC scoring (round 14): one upload of the correspondences and a TorchScript kernel per direction instead of ~20 elementwise launches.
count(corr_t, H, cth) -> int32 inlier count per hypothesis, same test as solver.score_many_t (cos of the angle between the transformed point
and the observed ray above cos(thr))."""
import torch, numpy as np

@torch.jit.script
def _cnt(X: torch.Tensor, C: torch.Tensor, D: torch.Tensor, h: torch.Tensor, cth: float, inverse: bool) -> torch.Tensor:
    cc = h[:, 0:1]; ss = h[:, 1:2]
    if not inverse:
        xa = cc * X[None, :, 0] - ss * X[None, :, 1] + h[:, 2:3]; ya = ss * X[None, :, 0] + cc * X[None, :, 1] + h[:, 3:4]; za = X[None, :, 2] + h[:, 4:5]
    else:
        ux = X[None, :, 0] - h[:, 2:3]; uy = X[None, :, 1] - h[:, 3:4]; uz = X[None, :, 2] - h[:, 4:5]; xa = cc * ux + ss * uy; ya = -ss * ux + cc * uy; za = uz
    wx = xa - C[None, :, 0]; wy = ya - C[None, :, 1]; wz = za - C[None, :, 2]
    cos = (wx * D[None, :, 0] + wy * D[None, :, 1] + wz * D[None, :, 2]) * torch.rsqrt(wx * wx + wy * wy + wz * wz + 1e-12)
    return (cos > cth).sum(1).to(torch.int32)

def upload(corr, dev='cuda'):
    """both directions' (X, C, D) blocks in one host-to-device copy."""
    mB = corr.hasB; mA = corr.hasA
    Z = np.concatenate([np.concatenate([corr.XB[mB], corr.cA[mB], corr.dA[mB]], 1), np.concatenate([corr.XA[mA], corr.cB[mA], corr.dB[mA]], 1)]).astype(np.float32)
    return torch.from_numpy(Z).to(dev), int(mB.sum())

def count(corr, H, thr):
    Z, nb = upload(corr, H.device); cth = float(np.cos(np.radians(thr))); out = torch.zeros(len(H), dtype=torch.int32, device=H.device)
    if nb: out += _cnt(Z[:nb, 0:3], Z[:nb, 3:6], Z[:nb, 6:9], H, cth, False)
    if len(Z) > nb: out += _cnt(Z[nb:, 0:3], Z[nb:, 3:6], Z[nb:, 6:9], H, cth, True)
    return out


@torch.jit.script
def _mask(X: torch.Tensor, C: torch.Tensor, D: torch.Tensor, h: torch.Tensor, cth: float, inverse: bool) -> torch.Tensor:
    cc = h[:, 0:1]; ss = h[:, 1:2]
    if not inverse:
        xa = cc * X[None, :, 0] - ss * X[None, :, 1] + h[:, 2:3]; ya = ss * X[None, :, 0] + cc * X[None, :, 1] + h[:, 3:4]; za = X[None, :, 2] + h[:, 4:5]
    else:
        ux = X[None, :, 0] - h[:, 2:3]; uy = X[None, :, 1] - h[:, 3:4]; uz = X[None, :, 2] - h[:, 4:5]; xa = cc * ux + ss * uy; ya = -ss * ux + cc * uy; za = uz
    wx = xa - C[None, :, 0]; wy = ya - C[None, :, 1]; wz = za - C[None, :, 2]
    cos = (wx * D[None, :, 0] + wy * D[None, :, 1] + wz * D[None, :, 2]) * torch.rsqrt(wx * wx + wy * wy + wz * wz + 1e-12)
    return cos > cth

def masks(corr, H, thr):
    """inlier masks (len(H), n_blocks) of hypotheses H over both directions' correspondence blocks (same order as count)."""
    Z, nb = upload(corr, H.device); cth = float(np.cos(np.radians(thr))); out = []
    if nb: out.append(_mask(Z[:nb, 0:3], Z[:nb, 3:6], Z[:nb, 6:9], H, cth, False))
    if len(Z) > nb: out.append(_mask(Z[nb:, 0:3], Z[nb:, 3:6], Z[nb:, 6:9], H, cth, True))
    return torch.cat(out, 1)
