"""VPoser, a learned prior on human poses, rebuilt here so no extra package is needed.

VPoser is a variational autoencoder trained on human poses. Optimising in its 32-dim latent
space instead of over joint angles keeps the fit near poses people really adopt. The network
is a few linear layers, so it is rebuilt in the same layout as the official package and the
published V02_05 checkpoint loads into it.

The pipeline does not use it: fit_smplx.py defaults to a plain L2 pull towards the neutral
pose (--pose_prior l2). It can be enabled with --pose_prior vposer once the checkpoint is
in models/vposer/V02_05/snapshots/.

Run from src/:  uv run python people/3D_People/vposer.py --selftest
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent
for _p in (str(SRC), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from geom import matrix_to_axis_angle, rot6d_to_matrix

VPOSER_DEFAULT = HERE / "models" / "vposer" / "V02_05" / "snapshots" / "V02_05_epoch=13_val_loss=0.03.ckpt"
NUM_BODY_JOINTS = 21
LATENT_DIM = 32
NUM_NEURONS = 512


class _NormalDistDecoder(nn.Module):
    """Mirrors human_body_prior.NormalDistDecoder: two Linears -> (mu, softplus(logvar))."""
    def __init__(self, nin, nout):
        super().__init__()
        self.mu = nn.Linear(nin, nout)
        self.logvar = nn.Linear(nin, nout)

    def forward(self, x):
        return self.mu(x), F.softplus(self.logvar(x))


class VPoser(nn.Module):
    """V02_05 VAE. Layer indices match the published checkpoint so it loads by `load_state_dict`."""
    def __init__(self):
        super().__init__()
        nf = NUM_BODY_JOINTS * 3                      # 63
        self.encoder_net = nn.Sequential(
            nn.Flatten(),                             # 0  (BatchFlatten)
            nn.BatchNorm1d(nf),                       # 1
            nn.Linear(nf, NUM_NEURONS),               # 2
            nn.LeakyReLU(),                           # 3
            nn.BatchNorm1d(NUM_NEURONS),              # 4
            nn.Dropout(0.1),                          # 5
            nn.Linear(NUM_NEURONS, NUM_NEURONS),      # 6
            nn.Linear(NUM_NEURONS, NUM_NEURONS),      # 7
            _NormalDistDecoder(NUM_NEURONS, LATENT_DIM),  # 8
        )
        self.decoder_net = nn.Sequential(
            nn.Linear(LATENT_DIM, NUM_NEURONS),       # 0
            nn.LeakyReLU(),                           # 1
            nn.Dropout(0.1),                          # 2
            nn.Linear(NUM_NEURONS, NUM_NEURONS),      # 3
            nn.LeakyReLU(),                           # 4
            nn.Linear(NUM_NEURONS, NUM_BODY_JOINTS * 6),  # 5 -> 126 (6D per joint)
            # index 6 (ContinousRotReprDecoder) has no params; done in decode() via geom.
        )

    def encode(self, pose_body):
        """pose_body axis-angle (B,63) or (B,21,3) -> latent mean mu (B,32)."""
        mu, _ = self.encoder_net(pose_body)
        return mu

    def decode(self, z):
        """latent (B,32) -> body pose axis-angle (B,63)."""
        d6 = self.decoder_net(z).reshape(-1, NUM_BODY_JOINTS, 6)
        R = rot6d_to_matrix(d6)                       # (B,21,3,3)
        aa = matrix_to_axis_angle(R)                  # (B,21,3)
        return aa.reshape(z.shape[0], NUM_BODY_JOINTS * 3)


def load_vposer(ckpt_path: Path = VPOSER_DEFAULT, device="cpu") -> VPoser:
    """Build VPoser and load the V02_05 checkpoint (encoder/decoder only), eval mode."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise SystemExit(f"[vposer] checkpoint not found: {ckpt_path}\n"
                         "  download VPoser v2 (V02_05) from https://smpl-x.is.tue.mpg.de and place it under "
                         f"{ckpt_path.parent}")
    # the VPoser checkpoint, see https://github.com/nghorbani/human_body_prior
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    keep = {}
    for k, v in sd.items():
        if k.startswith("vp_model.encoder_net.") or k.startswith("vp_model.decoder_net."):
            keep[k[len("vp_model."):]] = v
    model = VPoser()
    missing, unexpected = model.load_state_dict(keep, strict=False)
    # the only expected 'missing' is none; 'unexpected' should be empty too.
    real_missing = [m for m in missing if "num_batches_tracked" not in m]
    if real_missing or unexpected:
        raise SystemExit(f"[vposer] state_dict mismatch. missing={real_missing} unexpected={unexpected}")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _selftest() -> int:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vp = load_vposer(device=dev)
    n_loaded = sum(p.numel() for p in vp.parameters())
    print(f"[vposer] loaded V02_05 ({n_loaded:,} params) on {dev}; encoder+decoder only, eval mode")

    # decode(0) should be a calm mean pose (small axis-angles).
    z0 = torch.zeros(1, LATENT_DIM, device=dev)
    p0 = vp.decode(z0)
    print(f"[vposer] decode(0): pose_body shape {tuple(p0.shape)}  "
          f"max|aa| {float(p0.abs().max()):.3f} rad  mean|aa| {float(p0.abs().mean()):.3f}")
    assert p0.shape == (1, 63)
    assert float(p0.abs().max()) < 1.5, "mean pose unexpectedly extreme - decoder wiring suspect"

    # random latents decode to valid, bounded poses; batch works.
    torch.manual_seed(0)
    z = torch.randn(8, LATENT_DIM, device=dev)
    p = vp.decode(z)
    print(f"[vposer] decode(randn x8): shape {tuple(p.shape)}  "
          f"per-sample max|aa| {[round(float(x),2) for x in p.abs().reshape(8,-1).max(1).values]}")
    assert p.shape == (8, 63) and torch.isfinite(p).all()

    # feed a decoded pose through SMPL-X to confirm it produces a non-degenerate body.
    try:
        from fit_smplx import build_smplx
        m = build_smplx("neutral").to(dev)
        out = m(body_pose=p[:1].to(next(m.parameters()).dtype))
        v = out.vertices.detach()
        ext = float((v[0].max(0).values - v[0].min(0).values).max())
        print(f"[vposer] SMPL-X with a VPoser pose: vertices {tuple(v.shape)}  bbox extent {ext:.2f} m")
        assert 1.0 < ext < 3.0
    except SystemExit as e:
        print(f"[vposer] (skipped SMPL-X forward: {e})")

    print("[vposer] OK")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(_selftest())
    print("[vposer] self-contained VPoser v2 loader. Run --selftest to validate.")
