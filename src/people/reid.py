"""Turns a crop of a person into a vector, so two crops can be compared by appearance.

Several encoders share one interface and all return a unit vector, so the same cosine
distance works for any of them. Cosine distance runs 0 to 2: near 0 means the same
appearance, about 1 means unrelated.

Encoders, best first: OSNet, a re-identification network (used when torchreid is installed);
a ResNet-50 ImageNet backbone from torchvision (no extra dependency); and an HSV colour
histogram of the torso, which needs nothing and always works. build_embedder('auto') picks
the best one available and logs which.

To enable OSNet: `uv add torchreid`, then put the osnet_ain_x1_0.pth weights in
people/models/ (git-ignored). Without the weights torchreid uses its ImageNet backbone.

Self-test (checks whichever backend is active):
    uv run python people/reid.py --selftest
    uv run python people/reid.py --selftest --backend hsv
    uv run python people/reid.py --crops_dir <dir-of-<identity>/<img>.jpg>   # real data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent          # src/people/
MODELS_DIR = ROOT / "models"

_EPS = 1e-8


def _log(msg: str) -> None:
    print(f"[reid] {msg}", file=sys.stderr)


# All encoders return unit vectors, so one cosine distance works for all of them.
def _l2norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > _EPS else v


def _l2norm_rows(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n < _EPS] = 1.0
    return m / n


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors. Zero vectors give 0 instead of NaN."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < _EPS:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 minus cosine similarity, clipped to [0, 2]: 0 for identical, about 1 for unrelated."""
    d = 1.0 - cosine_similarity(a, b)
    return float(min(2.0, max(0.0, d)))


class BaseEmbedder:
    """What every encoder has to provide: one crop in, one unit vector out."""

    name: str = "base"
    dim: int = 0

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        return self.embed_batch([crop_bgr])[0]

    def embed_batch(self, crops: list) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name} dim={self.dim}>"


class HSVHistEmbedder(BaseEmbedder):
    """Hue and saturation histogram of the torso band of the crop. Needs no extra dependency.

    Brightness is dropped, so a player in shadow still looks like the same player.
    """

    def __init__(self, h_bins: int = 32, s_bins: int = 32,
                 row_band=(0.15, 0.65), col_band=(0.20, 0.80)):
        self.h_bins, self.s_bins = int(h_bins), int(s_bins)
        self.row_band, self.col_band = row_band, col_band
        self.name = f"hsv_{self.h_bins}x{self.s_bins}"
        self.dim = self.h_bins * self.s_bins

    def _torso(self, crop_bgr: np.ndarray) -> np.ndarray:
        h, w = crop_bgr.shape[:2]
        r0, r1 = int(self.row_band[0] * h), int(self.row_band[1] * h)
        c0, c1 = int(self.col_band[0] * w), int(self.col_band[1] * w)
        roi = crop_bgr[max(r0, 0):max(r1, r0 + 1), max(c0, 0):max(c1, c0 + 1)]
        return roi if roi.size else crop_bgr

    def _hist(self, crop_bgr) -> np.ndarray:
        if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
            return np.zeros(self.dim, dtype=np.float32)
        roi = self._torso(crop_bgr)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [self.h_bins, self.s_bins], [0, 180, 0, 256])
        return _l2norm(hist.flatten())

    def embed_batch(self, crops: list) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._hist(c) for c in crops]).astype(np.float32)


class _DeepEmbedder(BaseEmbedder):
    """Shared preprocessing and batching for the torch encoders.

    Crops are resized to 256x128 (H, W), the usual person re-identification input shape.
    """

    input_hw = (256, 128)  # (H, W) - the person-ReID convention
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, device=None):
        import torch
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None  # set by subclass

    def _preprocess(self, crops: list):
        H, W = self.input_hw
        arr = []
        for c in crops:
            if c is None or getattr(c, "size", 0) == 0:
                c = np.zeros((H, W, 3), dtype=np.uint8)
            img = cv2.resize(c, (W, H), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = (img - self._MEAN) / self._STD
            arr.append(img.transpose(2, 0, 1))
        batch = np.ascontiguousarray(np.stack(arr), dtype=np.float32)
        return self._torch.from_numpy(batch).to(self.device)

    def embed_batch(self, crops: list) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)
        torch = self._torch
        ten = self._preprocess(crops)
        with torch.no_grad():
            feat = self.model(ten)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        feat = feat.reshape(feat.shape[0], -1).float().cpu().numpy()
        return _l2norm_rows(feat)


class TorchvisionCNNEmbedder(_DeepEmbedder):
    """ResNet-50 ImageNet backbone with the classifier removed, used as an appearance encoder.

    Not trained for re-identification, but it needs nothing that is not already installed.
    """

    def __init__(self, device=None, arch: str = "resnet50"):
        super().__init__(device=device)
        import torch.nn as nn
        from torchvision import models
        if arch != "resnet50":
            raise ValueError(f"unsupported arch {arch!r}")
        weights = models.ResNet50_Weights.IMAGENET1K_V2  # downloads on first use
        net = models.resnet50(weights=weights)
        net.fc = nn.Identity()
        net.eval().to(self.device)
        self.model = net
        self.dim = 2048
        self.name = "cnn_resnet50_imagenet"


class TimmEmbedder(_DeepEmbedder):
    """Any timm model as an encoder, useful for comparing encoders.

    Weights come from the HF hub on first use and are cached. Two things differ from the ReID
    backbones: the input size follows the model's own config and the crop is letterboxed to a
    square instead of squashed; and the mean/std come from the model config too, because CLIP
    and SigLIP do not use ImageNet statistics.

    Names are timm model ids, e.g. `timm:vit_small_patch14_dinov2.lvd142m` or
    `timm:vit_base_patch16_clip_224.openai`.
    """

    def __init__(self, model_name: str, device=None, size: int = 0):
        super().__init__(device=device)
        import timm
        # see: https://huggingface.co/docs/timm/reference/models
        probe = timm.create_model(model_name, pretrained=False, num_classes=0)
        cfg = timm.data.resolve_data_config({}, model=probe)
        side = size or int(cfg["input_size"][-1])
        # round the side down to a whole number of patches
        patch = getattr(getattr(probe, "patch_embed", None), "patch_size", None)
        if patch:
            pp = patch[0] if isinstance(patch, (tuple, list)) else int(patch)
            side = max(pp, (side // pp) * pp)
        del probe
        # Passing img_size lets timm interpolate the position embeddings to this size.
        # Models that do not take the argument keep their trained resolution.
        try:
            net = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=side)
        except TypeError:
            net = timm.create_model(model_name, pretrained=True, num_classes=0)
            side = int(cfg["input_size"][-1])
        net.eval().to(self.device)
        self._MEAN = np.asarray(cfg["mean"], dtype=np.float32)
        self._STD = np.asarray(cfg["std"], dtype=np.float32)
        self.input_hw = (side, side)
        self.model = net
        self.dim = int(net.num_features)
        self.name = f"timm_{model_name}"

    def _preprocess(self, crops: list):
        """Letterbox each crop to a square, then normalise with the model's mean and std."""
        H, W = self.input_hw
        arr = []
        for c in crops:
            if c is None or getattr(c, "size", 0) == 0:
                c = np.zeros((H, W, 3), dtype=np.uint8)
            h, w = c.shape[:2]
            sc = min(H / max(h, 1), W / max(w, 1))
            nh, nw = max(1, int(round(h * sc))), max(1, int(round(w * sc)))
            r = cv2.resize(c, (nw, nh), interpolation=cv2.INTER_AREA)
            pad = np.zeros((H, W, 3), dtype=r.dtype)
            y0, x0 = (H - nh) // 2, (W - nw) // 2
            pad[y0:y0 + nh, x0:x0 + nw] = r
            img = cv2.cvtColor(pad, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = (img - self._MEAN) / self._STD
            arr.append(img.transpose(2, 0, 1))
        batch = np.ascontiguousarray(np.stack(arr), dtype=np.float32)
        return self._torch.from_numpy(batch).to(self.device)


def _load_reid_state_dict(net, path: str) -> int:
    """Load a torchreid zoo checkpoint into `net` and return the number of tensors loaded.

    Uses weights_only=False because the zoo checkpoints pickle a numpy scalar (torch>=2.6
    refuses them otherwise). Strips any 'module.' prefix and copies only shape-matching keys,
    so the unused classifier head is skipped."""
    import torch
    # weights_only=False because the checkpoint holds more than tensors
    # see: https://pytorch.org/docs/stable/generated/torch.load.html
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model_sd = net.state_dict()
    matched = {}
    for k, v in sd.items():
        kk = k[7:] if k.startswith("module.") else k
        if kk in model_sd and model_sd[kk].shape == v.shape:
            matched[kk] = v
    if not matched:
        raise RuntimeError(f"no matching tensors loaded from {path} (wrong checkpoint?)")
    model_sd.update(matched)
    net.load_state_dict(model_sd)
    return len(matched)


class OSNetEmbedder(_DeepEmbedder):
    """OSNet re-identification network, available when torchreid is installed.

    Uses the ReID weights from people/models/ if present, else torchreid's ImageNet backbone.
    Run the self-test before trusting its output.
    """

    def __init__(self, device=None, model_name: str = "osnet_ain_x1_0", model_path=None):
        super().__init__(device=device)
        # OSNet comes from torchreid, see https://kaiyangzhou.github.io/deep-person-reid/
        # torchreid moved this import between releases, so try both. If torchreid is missing
        # the error propagates and 'auto' falls back to the next backend.
        try:
            from torchreid.reid.models import build_model
        except ModuleNotFoundError:
            from torchreid.models import build_model
        weights = model_path
        if weights is None:
            cand = MODELS_DIR / f"{model_name}.pth"
            weights = cand if cand.exists() else None
        # skip the ImageNet backbone download when ReID weights are on disk (they replace it anyway)
        net = build_model(name=model_name, num_classes=1, loss="softmax", pretrained=weights is None)
        if weights is not None:
            n = _load_reid_state_dict(net, str(weights))
            _log(f"OSNet ReID weights loaded from {weights} ({n} tensors)")
        else:
            _log("OSNet using the ImageNet-pretrained backbone (no ReID weights in "
                 f"{MODELS_DIR}); for full cross-camera ReID drop osnet_ain_x1_0.pth there.")
        net.eval().to(self.device)
        self.model = net
        # read the output size from the model itself
        with self._torch.no_grad():
            d = self.model(self._preprocess([np.zeros((8, 4, 3), np.uint8)]))
            if isinstance(d, (tuple, list)):
                d = d[0]
            self.dim = int(d.reshape(1, -1).shape[1])
        self.name = model_name


_BACKENDS = {
    "osnet": OSNetEmbedder,
    "cnn": TorchvisionCNNEmbedder,
    "hsv": HSVHistEmbedder,
}
_AUTO_ORDER = ["osnet", "cnn", "hsv"]

# Short names for some timm encoders, with the input side in pixels. The side is fixed here
# because DINOv2 asks for 518 px, far more than a person crop of about 200 px can use.
_TIMM_ALIASES = {
    "dinov2": ("vit_small_patch14_dinov2.lvd142m", 224),
    "dinov2b": ("vit_base_patch14_dinov2.lvd142m", 224),
    "clip": ("vit_base_patch16_clip_224.openai", 224),
    "siglip": ("vit_base_patch16_siglip_224", 224),
}


def build_embedder(name: str = "auto", device=None) -> BaseEmbedder:
    """Build an encoder by name, or the best available with 'auto'.

    'auto' tries osnet -> cnn -> hsv and returns the first that builds. An explicit name
    that fails raises, with no silent downgrade."""
    name = (name or "auto").lower()
    # timm models can be named directly
    if name.startswith("timm:") or name in _TIMM_ALIASES:
        if name in _TIMM_ALIASES:
            model_id, side = _TIMM_ALIASES[name]
        else:                                   # timm:<model_id>[@<side>]
            spec = name.split(":", 1)[1]
            model_id, _, sz = spec.partition("@")
            side = int(sz) if sz else 0
        emb = TimmEmbedder(model_id, device=device, size=side)
        _log(f"using timm backend ({emb.name}, dim={emb.dim}, input {emb.input_hw})")
        return emb
    if name != "auto" and name not in _BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choose from {list(_BACKENDS)}, "
                         f"'timm:<model_id>', {list(_TIMM_ALIASES)} or 'auto'")

    order = _AUTO_ORDER if name == "auto" else [name]
    errors = {}
    for n in order:
        try:
            emb = _BACKENDS[n](device=device) if n != "hsv" else _BACKENDS[n]()
            if name == "auto" and n != _AUTO_ORDER[0]:
                _log(f"using '{n}' backend (higher-priority backends unavailable: "
                     f"{', '.join(f'{k} [{v}]' for k, v in errors.items())})")
            else:
                _log(f"using '{n}' backend ({emb.name}, dim={emb.dim})")
            return emb
        except Exception as e:  # noqa: BLE001 - report and try the next backend
            errors[n] = f"{type(e).__name__}: {e}"
            if name != "auto":
                raise
    raise RuntimeError(f"no embedder could be built: {errors}")  # hsv should never fail


def _person(shirt_bgr, pants_bgr) -> np.ndarray:
    """A synthetic person image for the self-test.

    The vertical shading stripe gives the deep encoders some texture to work with.
    """
    img = np.zeros((256, 128, 3), dtype=np.uint8)
    img[:40] = (90, 120, 150)          # head (skin-ish, BGR)
    img[40:150] = shirt_bgr            # torso
    img[150:256] = pants_bgr           # legs
    img[:, 58:66] = (img[:, 58:66] * 0.65).astype(np.uint8)  # shading stripe
    return img


def _vary(base: np.ndarray, rng) -> np.ndarray:
    """The same synthetic person with small brightness, noise and shift changes, as from another camera."""
    out = base.astype(np.float32)
    out *= rng.uniform(0.85, 1.15)                                   # brightness
    out += rng.normal(0, 6.0, size=out.shape)                        # sensor noise
    shift = int(rng.integers(-6, 7))                                 # sub-crop shift
    out = np.roll(out, shift, axis=1)
    return np.clip(out, 0, 255).astype(np.uint8)


def _load_crops_dir(crops_dir: Path):
    """Real crops, taking each sub-folder to be one person."""
    labels, crops = [], []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for sub in sorted(p for p in crops_dir.iterdir() if p.is_dir()):
        for img_path in sorted(sub.iterdir()):
            if img_path.suffix.lower() in exts:
                img = cv2.imread(str(img_path))
                if img is not None:
                    labels.append(sub.name)
                    crops.append(img)
    return labels, crops


def _separation_report(emb: BaseEmbedder, labels, crops) -> dict:
    """Distances between crops of the same person (intra) and of different people (inter)."""
    vecs = emb.embed_batch(crops)
    n = len(labels)
    intra, inter = [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = cosine_distance(vecs[i], vecs[j])
            (intra if labels[i] == labels[j] else inter).append(d)
    intra = np.asarray(intra, dtype=np.float64)
    inter = np.asarray(inter, dtype=np.float64)
    return {
        "backend": emb.name, "dim": emb.dim, "n_crops": n,
        "intra_mean": float(intra.mean()) if intra.size else float("nan"),
        "intra_max": float(intra.max()) if intra.size else float("nan"),
        "inter_mean": float(inter.mean()) if inter.size else float("nan"),
        "inter_min": float(inter.min()) if inter.size else float("nan"),
        "n_intra": int(intra.size), "n_inter": int(inter.size),
        # a positive margin means one threshold separates them completely
        "margin": (float(inter.min() - intra.max())
                   if intra.size and inter.size else float("nan")),
    }


def _selftest(backend: str, device=None, crops_dir: Path = None) -> int:
    if crops_dir is not None:
        labels, crops = _load_crops_dir(crops_dir)
        if len({*labels}) < 2:
            _log(f"need >=2 identity sub-folders with images in {crops_dir}")
            return 2
        _log(f"loaded {len(crops)} real crops across {len({*labels})} identities")
    else:
        rng = np.random.default_rng(0)
        specs = [("A", (200, 60, 40), (60, 40, 30)),     # blue shirt / dark pants
                 ("B", (40, 50, 200), (200, 200, 210))]  # red shirt  / light pants
        labels, crops = [], []
        for lab, shirt, pants in specs:
            base = _person(shirt, pants)
            for _ in range(5):
                labels.append(lab)
                crops.append(_vary(base, rng))
        _log(f"synthetic: {len(crops)} crops across {len(specs)} identities")

    emb = build_embedder(backend, device=device)
    rep = _separation_report(emb, labels, crops)
    print("\n=== reid self-test ===")
    for k, v in rep.items():
        print(f"  {k:11}: {v}")
    ok = (rep["inter_mean"] > rep["intra_mean"] * 1.05)  # inter must clearly exceed intra
    sep = rep["margin"] > 0
    print(f"  ranking    : inter_mean > intra_mean*1.05 -> {'PASS' if ok else 'FAIL'}")
    print(f"  separable  : inter_min > intra_max         -> {'YES' if sep else 'no (overlap)'}")
    print("=== " + ("OK" if ok else "FAILED") + " ===")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="auto", choices=["auto", "osnet", "cnn", "hsv"],
                    help="embedding backend (default: auto = osnet->cnn->hsv)")
    ap.add_argument("--device", default=None, help="torch device override (e.g. cpu, cuda)")
    ap.add_argument("--selftest", action="store_true",
                    help="run the synthetic identity-separation self-test")
    ap.add_argument("--crops_dir", default=None,
                    help="run the self-test on REAL crops: <dir>/<identity>/<img>.jpg")
    args = ap.parse_args()

    if args.selftest or args.crops_dir:
        cd = Path(args.crops_dir).resolve() if args.crops_dir else None
        raise SystemExit(_selftest(args.backend, device=args.device, crops_dir=cd))

    # nothing asked for, so just say which encoder is available here
    emb = build_embedder(args.backend, device=args.device)
    print(f"resolved backend: {emb!r}")


if __name__ == "__main__":
    main()
