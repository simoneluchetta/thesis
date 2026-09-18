"""The neural radiance field (NeRF) network, just enough to load the trained model.

A NeRF is a small neural network that describes a scene. You give it a point in space and a
viewing direction, and it answers with two things: how "solid" the space is at that point
(the density) and what colour it looks from that direction.

The model in model/nerf.pt was trained on the arena earlier in the project.
It is not trained again here.
This file only rebuilds the same network so the saved weights can be loaded into it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# see the NeRF paper, section 5.1: https://arxiv.org/abs/2003.08934
class PositionalEncoding(nn.Module):
    """Turns a coordinate into sines and cosines of increasing frequency.

    A plain network is bad at learning fine detail from raw x, y, z. Feeding it these waves
    instead lets it represent sharp edges and textures.
    """

    def __init__(self, n_freqs: int):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freqs) * np.pi, persistent=False)
        self.out_mult = 1 + 2 * n_freqs

    def forward(self, x):
        out = [x]
        for f in self.freqs:
            out += [torch.sin(x * f), torch.cos(x * f)]
        return torch.cat(out, dim=-1)


class NeRF(nn.Module):
    """Eight layers for the density, then a small branch that adds the viewing direction for colour."""

    def __init__(self, pe_xyz=10, pe_dir=4, width=256, depth=8, skip=4):
        super().__init__()
        self.enc_x = PositionalEncoding(pe_xyz)
        self.enc_d = PositionalEncoding(pe_dir)
        in_x = 3 * self.enc_x.out_mult
        in_d = 3 * self.enc_d.out_mult
        self.skip = skip
        # halfway through, the encoded position is fed in again so it is not forgotten
        self.pts_layers = nn.ModuleList(
            [nn.Linear(in_x if i == 0 else width + (in_x if i == skip else 0), width) for i in range(depth)])
        self.sigma_head = nn.Linear(width, 1)
        self.feat = nn.Linear(width, width)
        self.rgb_hidden = nn.Linear(width + in_d, width // 2)
        self.rgb_head = nn.Linear(width // 2, 3)

    def forward(self, pts, dirs):
        x = self.enc_x(pts)
        h = x
        for i, lin in enumerate(self.pts_layers):
            if i == self.skip:
                h = torch.cat([h, x], dim=-1)
            h = F.relu(lin(h))
        sigma = self.sigma_head(h).squeeze(-1)
        rgb = F.relu(self.rgb_hidden(torch.cat([self.feat(h), self.enc_d(dirs)], dim=-1)))
        return sigma, torch.sigmoid(self.rgb_head(rgb))


def load_model(path, device):
    """Load model/nerf.pt. Returns the network and the settings it was trained with."""
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    net = NeRF(a["pe_xyz"], a["pe_dir"], a["width"], a["depth"]).to(device)
    net.load_state_dict(ck["net"])
    net.eval()
    return net, a
