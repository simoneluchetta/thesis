# -*- coding: utf-8 -*-
"""Rebuilds the person identities from the fused floor positions.

It works in three steps: tracking, splitting and joining.

Tracking walks through the frames with one Kalman filter per person and assigns the
detections to the tracks with the Hungarian method. Where the best assignment is only just
better than the second best, the track is cut there.

Splitting cuts each piece again wherever its appearance vectors change from one person to
another (an MMD two-sample test).

Joining puts the pieces together into whole people with a minimum-cost path cover, so two
pieces alive at the same instant never get the same identity. The appearance gate is
calibrated on this clip from pairs known to be the same person and pairs known to be
different. Teammates often look alike to the appearance model, so swaps inside one
team are left to jersey_repair.py, which uses the shirt numbers.

Input  : people_fused*.json with the per-person `appearance` vectors (work/people/fused.json).
Output : same schema with `id` == `track_id`, plus `pos_smooth` per person-instant, an
         `assembly` metadata block and an `uncertain_links` list.

How to run? --> uv run python -m people.assemble --in  ../work/people/fused.json --out ../work/people/assembled.json
"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

BIG = 1e6          # cost of a forbidden pair in the assignment matrices (scipy needs a finite number)
_LAST_APP = [0.0]  # appearance distance of the most recent link_cost call (used by the margin penalty)
_APP_DIM = [0]     # length of the appearance vectors in the current input (0 = no embeddings at all)


# Motion model: a Kalman filter, plus a smoother that runs backwards over a finished track.
# see: https://en.wikipedia.org/wiki/Kalman_filter
class KF:
    """Kalman filter for one person: position and velocity on the floor, with their uncertainty.

    The state is [x, y, vx, vy] in metres and m/s. The process noise is the piecewise white
    acceleration model: `q` is the acceleration power spectral density in m^2/s^3, so the
    uncertainty grows the longer a track goes without a detection.
    """

    def __init__(self, pos, q, p0_pos=0.25, p0_vel=9.0):
        self.x = np.array([pos[0], pos[1], 0.0, 0.0], float)
        self.P = np.diag([p0_pos, p0_pos, p0_vel, p0_vel]).astype(float)
        self.q = float(q)

    @staticmethod
    def _F(dt):
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        return F

    def _Q(self, dt):
        q, d2, d3, d4 = self.q, dt * dt, dt ** 3, dt ** 4
        Q = np.zeros((4, 4))
        Q[0, 0] = Q[1, 1] = q * d4 / 4.0
        Q[2, 2] = Q[3, 3] = q * d2
        Q[0, 2] = Q[2, 0] = Q[1, 3] = Q[3, 1] = q * d3 / 2.0
        return Q

    def predict(self, dt):
        F = self._F(dt)
        return F @ self.x, F @ self.P @ F.T + self._Q(dt)

    def step(self, dt):
        self.x, self.P = self.predict(dt)

    def update(self, z, r):
        H = np.zeros((2, 4)); H[0, 0] = H[1, 1] = 1.0
        R = np.eye(2) * (r * r)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        y = np.asarray(z, float) - H @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    @staticmethod
    def innovation(xp, Pp, z, r):
        """Distance between a detection `z` and the predicted state.

        Returns (Mahalanobis distance in standard deviations, Euclidean distance in metres).
        Matching uses the first one, so an uncertain track gets a wider gate.
        """
        H = np.zeros((2, 4)); H[0, 0] = H[1, 1] = 1.0
        S = H @ Pp @ H.T + np.eye(2) * (r * r)
        y = np.asarray(z, float) - H @ xp
        m2 = float(y @ np.linalg.solve(S, y))
        return math.sqrt(max(m2, 0.0)), float(np.hypot(*y))


def rts_smooth(ts_sec, zs, rs, q):
    """Smooth a finished tracklet with a Rauch-Tung-Striebel backward pass.

    Returns (means (N,4), covariances (N,4,4)); each mean is [x, y, vx, vy]. The joining
    step extrapolates from these rather than from a single noisy endpoint.
    see: https://en.wikipedia.org/wiki/Kalman_filter#Rauch-Tung-Striebel
    """
    n = len(zs)
    kf = KF(zs[0], q)
    kf.update(zs[0], rs[0])
    xf = [kf.x.copy()]; Pf = [kf.P.copy()]
    xp_l = [kf.x.copy()]; Pp_l = [kf.P.copy()]
    for i in range(1, n):
        dt = max(ts_sec[i] - ts_sec[i - 1], 1e-6)
        xp, Pp = kf.predict(dt)
        xp_l.append(xp.copy()); Pp_l.append(Pp.copy())
        kf.x, kf.P = xp, Pp
        kf.update(zs[i], rs[i])
        xf.append(kf.x.copy()); Pf.append(kf.P.copy())
    xs = [None] * n; Ps = [None] * n
    xs[-1], Ps[-1] = xf[-1], Pf[-1]
    for i in range(n - 2, -1, -1):
        dt = max(ts_sec[i + 1] - ts_sec[i], 1e-6)
        F = KF._F(dt)
        C = Pf[i] @ F.T @ np.linalg.inv(Pp_l[i + 1])
        xs[i] = xf[i] + C @ (xs[i + 1] - xp_l[i + 1])
        Ps[i] = Pf[i] + C @ (Ps[i + 1] - Pp_l[i + 1]) @ C.T
    return np.stack(xs), np.stack(Ps)


# Appearance statistics: do two sets of appearance vectors come from the same person?
def median_sigma(X, cap=2000, rng=None):
    """Median distance between appearance vectors, used as the width of the RBF kernel.

    Computed once on a global sample, so every comparison in the run uses the same scale.
    """
    rng = rng or np.random.default_rng(0)
    if len(X) > cap:
        X = X[rng.choice(len(X), cap, replace=False)]
    D = np.sqrt(np.maximum(_sqdist(X, X), 0.0))
    iu = np.triu_indices(len(X), 1)
    m = float(np.median(D[iu])) if len(iu[0]) else 1.0
    return max(m, 1e-3)


def _sqdist(A, B):
    return (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)


def rbf(A, B, sigma):
    return np.exp(-np.maximum(_sqdist(A, B), 0.0) / (2.0 * sigma * sigma))


_KAA_POOL = [0.5]      # pooled within-tracklet self-affinity E[k(a,a')], estimated once per run
_KAA_SHRINK = [6.0]    # how strongly a short tracklet is pulled toward the pool (in pseudo pairs)


def self_affinity(A, sigma, shrink=True):
    """Mean kernel value over the distinct pairs of one set, E[k(a,a')].

    A short set has few pairs, so its estimate is shrunk toward the pooled value `_KAA_POOL`
    with `_KAA_SHRINK` pseudo pairs. A set of one sample returns the pool. This keeps short
    tracklets from looking different from everything just because they are short.
    """
    n = len(A)
    npair = n * (n - 1) / 2.0
    if n < 2:
        return _KAA_POOL[0]
    K = rbf(A, A, sigma)
    own = float((K.sum() - np.trace(K)) / (n * (n - 1)))
    if not shrink:
        return own
    m0 = _KAA_SHRINK[0]
    return (npair * own + m0 * _KAA_POOL[0]) / (npair + m0)


# the MMD two-sample test, see Gretton et al. 2012: https://jmlr.org/papers/v13/gretton12a.html
def mmd2(A, B, sigma, kaa=None, kbb=None):
    """Squared MMD between two sets of appearance vectors; 0 means the same distribution.

    MMD^2 = E k(a,a') + E k(b,b') - 2 E k(a,b), with the self terms over distinct pairs
    (the unbiased form), so two short sets of the same person are not pushed apart.
    The value can go slightly negative; callers clamp it where a cost is needed.
    """
    kaa = self_affinity(A, sigma) if kaa is None else kaa
    kbb = self_affinity(B, sigma) if kbb is None else kbb
    kab = float(rbf(A, B, sigma).mean())
    return kaa + kbb - 2.0 * kab


def _scatter_tables(K):
    """Prefix sums of the kernel matrix, so the scatter of any segment costs a few lookups.

    Scatter of segment [i,j) is  sum_a k(a,a)  -  (1/(j-i)) * sum_{a,b} k(a,b),
    the total squared kernel distance of its points from their own mean.
    """
    n = K.shape[0]
    d = np.cumsum(np.concatenate([[0.0], np.diag(K)]))          # (n+1,)
    C = np.zeros((n + 1, n + 1))
    C[1:, 1:] = K.cumsum(0).cumsum(1)
    return d, C


def _scatter(d, C, i, j):
    if j - i <= 0:
        return 0.0
    blk = C[j, j] - C[i, j] - C[j, i] + C[i, i]
    return float(d[j] - d[i] - blk / (j - i))


def kcp_split(K, min_seg, max_depth=6):
    """Recursive change-point search on a kernel matrix, scored by scatter gain.

    Returns (cut indices, normalised gain of each cut). The gain is the fraction of the
    segment's scatter that the cut removes, a number in [0,1]. Not called by the pipeline,
    which uses `mmd_split`.
    """
    n = K.shape[0]
    d, C = _scatter_tables(K)
    cuts, gains = [], []

    def rec(i, j, depth):
        if depth >= max_depth or (j - i) < 2 * min_seg:
            return
        s_all = _scatter(d, C, i, j)
        if s_all <= 1e-12:
            return
        best, best_m = -1.0, None
        for m in range(i + min_seg, j - min_seg + 1):
            v = s_all - (_scatter(d, C, i, m) + _scatter(d, C, m, j))
            if v > best:
                best, best_m = v, m
        if best_m is None:
            return
        g = best / s_all
        cuts.append(best_m); gains.append(g)
        rec(i, best_m, depth + 1)
        rec(best_m, j, depth + 1)

    rec(0, n, 0)
    order = np.argsort(cuts)
    return [cuts[i] for i in order], [gains[i] for i in order]


def split_mmds(K, min_seg):
    """MMD^2 between the two halves of a tracklet, for every possible cut point at once.

    Returns (values, cut indices); both are empty when the tracklet is shorter than
    2 * min_seg. The threshold on these values is the calibrated appearance gate: a tracklet
    whose two halves are further apart than two different people usually are holds two people.
    """
    n = K.shape[0]
    if n < 2 * min_seg:
        return np.zeros(0), np.zeros(0, int)
    d = np.cumsum(np.concatenate([[0.0], np.diag(K)]))
    C = np.zeros((n + 1, n + 1))
    C[1:, 1:] = K.cumsum(0).cumsum(1)
    ms = np.arange(min_seg, n - min_seg + 1)
    m = ms.astype(float)
    nb = float(n) - m
    s_aa = C[ms, ms] - d[ms]
    s_bb = (C[n, n] - C[ms, n] - C[n, ms] + C[ms, ms]) - (d[n] - d[ms])
    s_ab = C[ms, n] - C[ms, ms]
    kaa = s_aa / np.maximum(m * (m - 1.0), 1e-9)
    kbb = s_bb / np.maximum(nb * (nb - 1.0), 1e-9)
    kab = s_ab / np.maximum(m * nb, 1e-9)
    return kaa + kbb - 2.0 * kab, ms


def max_split_mmd(K, min_seg):
    """The best cut point of a tracklet and its MMD^2; (0.0, None) when it is too short."""
    v, ms = split_mmds(K, min_seg)
    if not len(v):
        return 0.0, None
    i = int(np.argmax(v))
    return float(v[i]), int(ms[i])


def mmd_split(K, min_seg, gate, max_depth=6):
    """Cut a tracklet recursively wherever the split MMD^2 is above `gate`. Returns the cut indices."""
    cuts = []

    def rec(i, j, depth):
        if depth >= max_depth or (j - i) < 2 * min_seg:
            return
        v, m = max_split_mmd(K[i:j, i:j], min_seg)
        if m is None or v <= gate:
            return
        cuts.append(i + m)
        rec(i, i + m, depth + 1)
        rec(i + m, j, depth + 1)

    rec(0, K.shape[0], 0)
    return sorted(cuts)


def kcp_best_gain(K, min_seg):
    """Largest normalised scatter gain of any single cut. Not called by the pipeline."""
    n = K.shape[0]
    if n < 2 * min_seg:
        return 0.0
    d, C = _scatter_tables(K)
    s_all = _scatter(d, C, 0, n)
    if s_all <= 1e-12:
        return 0.0
    best = max(s_all - (_scatter(d, C, 0, m) + _scatter(d, C, m, n))
               for m in range(min_seg, n - min_seg + 1))
    return float(best / s_all)


# Tracking: build the tracklets frame by frame and cut them where the assignment was unsure.
def kin_nis(sam, pos_of, r_of, fps, q):
    """Standardised innovation of the Kalman filter at each step of a tracklet.

    A swap between two people shows up as a jump: the position leaves one body and lands on
    another. The jump is divided by the innovation covariance, so a detection with a large
    `spread_m` is allowed a larger residual. Only used with --kin_split.
    """
    if len(sam) < 2:
        return np.zeros(0)
    kf = KF(pos_of[sam[0]], q)
    kf.update(pos_of[sam[0]], r_of[sam[0]])
    out = np.zeros(len(sam) - 1)
    for i in range(1, len(sam)):
        dt = max((sam[i][0] - sam[i - 1][0]) / fps, 1e-6)
        xp, Pp = kf.predict(dt)
        m, _ = KF.innovation(xp, Pp, pos_of[sam[i]], r_of[sam[i]])
        out[i - 1] = m
        kf.x, kf.P = xp, Pp
        kf.update(pos_of[sam[i]], r_of[sam[i]])
    return out


def kin_best_cut(sam, pos_of, r_of, fps, q, min_seg):
    """The step with the largest standardised jump that leaves `min_seg` samples on both sides.

    Returns (jump value, cut index), or (0.0, None). One cut per tracklet per pass.
    """
    nis = kin_nis(sam, pos_of, r_of, fps, q)
    if nis.size == 0:
        return 0.0, None
    lo, hi = min_seg - 1, len(sam) - min_seg
    if hi <= lo:
        return 0.0, None
    seg = nis[lo:hi]
    if seg.size == 0:
        return 0.0, None
    k = int(np.argmax(seg))
    return float(seg[k]), lo + k + 1


def build_tracklets(ts, frames, args):
    """Tracking, forward pass: assign the detections of each frame to the live tracks.

    Each frame is solved with the Hungarian method on a cost of motion plus appearance.
    Nothing is cut here. A sample is only marked ambiguous when the winning assignment beats
    the runner-up by less than --amb_margin; `cut_tracks` turns those marks into cuts.
    Returns {track id: [(t, detection index, ambiguous)]}.
    """
    fps = args.fps
    live = []
    raw = {}
    next_tid = 0
    for t in ts:
        dets = frames.get(str(t), [])
        if not dets:
            continue
        Z = np.array([d["pos"][:2] for d in dets], float)
        RR = np.array([max(args.r_min, args.r_spread * float(d.get("spread_m") or 0.4))
                       for d in dets], float)
        A = np.zeros((len(dets), _APP_DIM[0]), np.float32) if _APP_DIM[0] else None
        Aok = np.zeros(len(dets), bool)
        if A is not None:
            for j, d in enumerate(dets):
                if d.get("_app") is not None:
                    A[j] = d["_app"]; Aok[j] = True

        live = [tr for tr in live if (t - tr["t_last"]) / fps <= args.max_coast_s]
        nT, nD = len(live), len(dets)
        cost = np.full((nT, nD + nT), BIG, float)
        for i in range(nT):
            cost[i, nD + i] = args.miss_cost
        for i, tr in enumerate(live):
            dt = max((t - tr["t_last"]) / fps, 1e-6)
            xp, Pp = tr["kf"].predict(dt)
            for j in range(nD):
                maha, euc = KF.innovation(xp, Pp, Z[j], RR[j])
                if maha > args.gate_maha or euc > args.gate_max_m:
                    continue
                c = args.w_mot * (maha / args.gate_maha)
                if A is not None and Aok[j] and tr["tpl"] is not None:
                    c += args.w_app * (float(1.0 - np.dot(tr["tpl"], A[j])) / args.app_scale)
                cost[i, j] = c
        # see: https://en.wikipedia.org/wiki/Hungarian_algorithm
        # and: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
        ri, ci = linear_sum_assignment(cost) if nT else ([], [])

        used_det = set()
        for i, j in zip(ri, ci):
            if j >= nD or cost[i, j] >= BIG:
                continue
            tr = live[i]
            row = np.delete(cost[i, :nD], j) if nD > 1 else np.array([BIG])
            col = np.delete(cost[:, j], i) if nT > 1 else np.array([BIG])
            margin = min(row.min() if row.size else BIG,
                         col.min() if col.size else BIG) - cost[i, j]
            dt = max((t - tr["t_last"]) / fps, 1e-6)
            tr["kf"].step(dt)
            tr["kf"].update(Z[j], RR[j])
            tr["t_last"] = t
            if A is not None and Aok[j]:
                a = A[j]
                tr["tpl"] = a.copy() if tr["tpl"] is None else _unit(
                    args.app_ema * tr["tpl"] + (1 - args.app_ema) * a)
            raw[tr["tid"]].append((t, j, bool(margin < args.amb_margin)))
            used_det.add(j)

        for j in range(nD):
            if j in used_det:
                continue
            kf = KF(Z[j], args.q)
            kf.update(Z[j], RR[j])
            raw[next_tid] = [(t, j, False)]
            live.append({"kf": kf, "t_last": t, "tid": next_tid,
                         "tpl": (A[j].copy() if (A is not None and Aok[j]) else None)})
            next_tid += 1
    return raw


def cut_tracks(raw, args):
    """Cut the raw tracks at the edges of each ambiguous stretch and at long gaps.

    Two athletes crossing are confusable for tens of frames, so the whole ambiguous stretch
    becomes its own tracklet and the joining step decides who it belongs to. Stretches shorter than
    --amb_min_run samples are ignored, and stretches separated by fewer than --amb_merge_run
    clean samples are merged. A track is also cut wherever it coasted longer than --cut_gap_s.
    Returns ({tracklet id: [(t, detection index)]}, counts).
    """
    segs, n_amb, n_gap = {}, 0, 0
    nid = 0
    for tid, sam in sorted(raw.items()):
        sam.sort()
        n = len(sam)
        amb = np.array([a for _, _, a in sam], bool)
        # merge ambiguous stretches that are close together, and drop very short ones
        i = 0
        while i < n:
            if amb[i]:
                j = i
                while j < n and amb[j]:
                    j += 1
                k = j
                while k < n and not amb[k]:
                    k += 1
                if j < n and (k - j) < args.amb_merge_run and k < n:
                    amb[j:k] = True          # bridge a short clean gap between two episodes
                    i = j
                    continue
                if (j - i) < args.amb_min_run:
                    amb[i:j] = False         # too short to count
                i = j
            else:
                i += 1
        # cut at the edges of each ambiguous stretch, and wherever the track coasted a long way
        cuts = set()
        for i in range(1, n):
            if amb[i] != amb[i - 1]:
                cuts.add(i)
                n_amb += 1
            if (sam[i][0] - sam[i - 1][0]) / args.fps > args.cut_gap_s:
                cuts.add(i)
                n_gap += 1
        prev = 0
        for c in sorted(cuts) + [n]:
            if c <= prev:
                continue
            segs[nid] = [(t, j) for t, j, _ in sam[prev:c]]
            nid += 1
            prev = c
    return segs, {"raw_tracks": len(raw), "ambiguity_cuts": n_amb, "gap_cuts": n_gap}


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


# Joining: the tracklets become whole people through a minimum-cost path cover.
class Chain:
    """One person as currently believed: its samples in time order, with appearance and smoothed motion."""

    __slots__ = ("segs", "samples", "t0", "t1", "emb", "kaa", "sm_x", "sm_P", "sm_t")

    def __init__(self, samples, emb):
        self.samples = samples                     # [(t, det_index)], time-sorted
        self.t0, self.t1 = samples[0][0], samples[-1][0]
        self.emb = emb                             # (n,512) subsample of appearance
        self.kaa = None
        self.sm_x = self.sm_P = self.sm_t = None


def _fit_chain(ch, pos_of, r_of, fps, q):
    ts_sec = np.array([t / fps for t, _ in ch.samples], float)
    zs = np.array([pos_of[s] for s in ch.samples], float)
    rs = np.array([r_of[s] for s in ch.samples], float)
    if len(zs) == 1:
        ch.sm_x = np.array([[zs[0, 0], zs[0, 1], 0.0, 0.0]])
        ch.sm_P = np.array([np.diag([0.25, 0.25, 9.0, 9.0])])
    else:
        ch.sm_x, ch.sm_P = rts_smooth(ts_sec, zs, rs, q)
    ch.sm_t = ts_sec


def _extrapolate(x, P, dt, q):
    F = KF._F(dt)
    kf = KF([0, 0], q)
    return F @ x, F @ P @ F.T + kf._Q(dt)


# tolerance (m) when matching a human verdict to a junction; the recorded positions are rounded to 1 cm
_CON_TOL_M = 0.05


def link_cost(a, b, sigma, args, why=None):
    """Cost of linking tracklet `b` after tracklet `a`, or None when the link is not allowed.

    Geometry (a Kalman extrapolation across the gap) mostly acts as a veto; the appearance
    distance (MMD^2) carries most of the cost. `why` counts the reason for each veto.
    """
    def _no(tag):
        if why is not None:
            why[tag] += 1
        return None

    # A human verdict on this junction overrides every gate below: "different" always refuses
    # the link, "same" always accepts it, geometric vetoes included.
    con = getattr(args, "_constraints", None)
    if con:
        # (t_end_a, t_start_b) alone is not a unique key, so the endpoints are matched too,
        # against the `at`/`to` positions that `uncertain_links` recorded for this junction.
        for at, to, v in con.get((int(a.t1), int(b.t0)), ()):
            if at is not None and to is not None:
                if (abs(float(a.sm_x[-1][0]) - at[0]) > _CON_TOL_M
                        or abs(float(a.sm_x[-1][1]) - at[1]) > _CON_TOL_M
                        or abs(float(b.sm_x[0][0]) - to[0]) > _CON_TOL_M
                        or abs(float(b.sm_x[0][1]) - to[1]) > _CON_TOL_M):
                    continue
            if v == "different":
                return _no("HUMAN_cut")
            if v == "same":
                if why is not None:
                    why["HUMAN_link"] += 1
                _LAST_APP[0] = 0.0
                return -1.0e6

    gap_s = (b.t0 - a.t1) / args.fps
    if gap_s <= 0 or gap_s > args.max_gap_s:
        return _no("gap")
    # forward extrapolation of a; backward extrapolation of b; take the more forgiving of the two
    xa, Pa = _extrapolate(a.sm_x[-1], a.sm_P[-1], gap_s, args.q)
    xb = b.sm_x[0].copy(); xb[2:] *= -1.0
    xb2, Pb = _extrapolate(xb, b.sm_P[0], gap_s, args.q)
    m_f, e_f = KF.innovation(xa, Pa, b.sm_x[0][:2], args.r_min)
    m_b, e_b = KF.innovation(xb2, Pb, a.sm_x[-1][:2], args.r_min)
    maha, euc = min(m_f, m_b), min(e_f, e_b)
    if maha > args.link_maha:
        return _no("maha")
    if euc > args.link_max_m:
        return _no("euclid")
    if a.emb is None or b.emb is None or len(a.emb) == 0 or len(b.emb) == 0:
        app = args.app_missing
    else:
        app = max(mmd2(a.emb, b.emb, sigma, a.kaa, b.kaa), 0.0)
        if app > args.mmd_gate:
            return _no("appearance")
    _LAST_APP[0] = app
    c = (args.w_app * app / args.mmd_scale
         + args.w_mot * maha / args.link_maha
         + args.w_gap * gap_s / args.max_gap_s)
    if c >= args.link_cost_max:
        return _no("cost")
    if why is not None:
        why["ADMITTED"] += 1
    return c


def calibrate(chains, sigma, args, nn_of):
    """Collect MMD^2 values for tracklet pairs whose answer is known from geometry alone.

    Positive pairs: a ends and b starts within --cal_gap_s, their endpoints are within
    --cal_pos_m, and nobody else was within --cal_clear_m of either endpoint, so b can only
    be a continuation of a. Negative pairs: a and b overlap in time for at least
    --cal_overlap_frames, so they are different people. Returns (positives, negatives).
    """
    order = sorted(range(len(chains)), key=lambda k: chains[k].t0)
    t0s = [chains[k].t0 for k in order]
    pos, neg = [], []
    for i, a in enumerate(chains):
        if a.emb is None or len(a.emb) < 2:
            continue
        lo = bisect.bisect_right(t0s, a.t1)
        hi = bisect.bisect_right(t0s, a.t1 + args.cal_gap_s * args.fps)
        for k in range(lo, hi):
            j = order[k]
            b = chains[j]
            if i == j or b.emb is None or len(b.emb) < 2:
                continue
            d = float(np.hypot(*(b.sm_x[0][:2] - a.sm_x[-1][:2])))
            if d > args.cal_pos_m:
                continue
            if min(nn_of.get(a.samples[-1], 0.0), nn_of.get(b.samples[0], 0.0)) < args.cal_clear_m:
                continue
            pos.append(mmd2(a.emb, b.emb, sigma, a.kaa, b.kaa))
    for i, a in enumerate(chains):
        if a.emb is None or len(a.emb) < 2:
            continue
        for j in range(i + 1, len(chains)):
            b = chains[j]
            if b.emb is None or len(b.emb) < 2:
                continue
            if b.t0 > a.t1 or a.t0 > b.t1:
                continue                                     # no temporal overlap
            ov = min(a.t1, b.t1) - max(a.t0, b.t0)
            if ov < args.cal_overlap_frames:
                continue
            neg.append(mmd2(a.emb, b.emb, sigma, a.kaa, b.kaa))
            if len(neg) >= args.cal_max_neg:
                break
        if len(neg) >= args.cal_max_neg:
            break
    return np.array(pos, float), np.array(neg, float)


def assemble(chains, sigma, args):
    """Minimum-cost path cover over the chains, solved as a Hungarian assignment on an N x 2N matrix.

    Column j < N means "chain j is my successor"; column N+i means "I stop here". Every chain
    gets at most one successor and one predecessor, and every allowed link goes forward in
    time, so no cycle is possible. Returns (successor map, edge count, veto counts, regret).
    """
    N = len(chains)
    M = np.full((N, 2 * N), BIG, float)
    for i in range(N):
        M[i, N + i] = args.link_cost_max                # cost of not linking = the accept threshold
    # only b's whose start falls inside a's admissible gap window can ever link, so index the
    # chains by start time and scan that window instead of all N^2 pairs
    order = sorted(range(N), key=lambda k: chains[k].t0)
    t0s = [chains[k].t0 for k in order]
    span = args.max_gap_s * args.fps
    n_edges = 0
    why = collections.Counter()
    raw = {}                       # (i,j) -> (cost, appearance distance)
    for i, a in enumerate(chains):
        lo = bisect.bisect_right(t0s, a.t1)
        hi = bisect.bisect_right(t0s, a.t1 + span)
        for k in range(lo, hi):
            j = order[k]
            if i == j:
                continue
            c = link_cost(a, chains[j], sigma, args, why)
            if c is not None:
                raw[(i, j)] = (c, _LAST_APP[0])

    # Ambiguity penalty: a low appearance distance means little when every candidate has one.
    # A link is charged --w_amb when its margin over the next-best is below --app_margin.
    # Off by default.
    if args.app_margin > 0 and raw:
        best_row, best_col = collections.defaultdict(list), collections.defaultdict(list)
        for (i, j), (c, ap) in raw.items():
            best_row[i].append(ap)
            best_col[j].append(ap)
        for k in best_row:
            best_row[k].sort()
        for k in best_col:
            best_col[k].sort()
        for (i, j), (c, ap) in raw.items():
            r = best_row[i]
            cl = best_col[j]
            mr = (r[1] - r[0]) if len(r) > 1 else args.app_margin
            mc = (cl[1] - cl[0]) if len(cl) > 1 else args.app_margin
            # only the winner of a row/column gets to claim the margin; a runner-up has none
            m = min(mr if ap <= r[0] + 1e-12 else 0.0,
                    mc if ap <= cl[0] + 1e-12 else 0.0)
            pen = args.w_amb * max(0.0, args.app_margin - m) / args.app_margin
            raw[(i, j)] = (c + pen, ap)

    for (i, j), (c, ap) in raw.items():
        if c < args.link_cost_max:
            M[i, j] = c
            n_edges += 1
        else:
            why["ambiguous"] += 1
    # count the chains that had no admissible successor at all
    has_edge = (M[:, :N] < args.link_cost_max).any(axis=1)
    why["_chains_with_no_candidate"] = int((~has_edge).sum())
    # a minimum path cover solved as an assignment problem
    # see: https://en.wikipedia.org/wiki/Path_cover
    # and: https://stackoverflow.com/questions/69290319/how-to-solve-an-assignment-problem-like-hungarian-linear-sum-assignment-with-a
    ri, ci = linear_sum_assignment(M)
    succ = {}
    for i, j in zip(ri, ci):
        if j < N and M[i, j] < args.link_cost_max:
            succ[i] = j

    # Regret of each chosen link: how much more the best assembly without it would cost.
    # Links with a small regret were not forced by the evidence, and go in `uncertain_links`.
    base = float(M[ri, ci].sum())
    regret = {}
    if args.regret:
        # every link is examined by default; --regret_max caps it, most expensive first
        items = sorted(succ.items(), key=lambda kv: -M[kv[0], kv[1]])[:args.regret_max]
        for i, j in items:
            keep = M[i, j]
            M[i, j] = BIG
            r2, c2 = linear_sum_assignment(M)
            regret[(i, j)] = float(M[r2, c2].sum() - base)
            M[i, j] = keep
    margins = regret
    return succ, n_edges, why, margins


def phantom_merge(chains, sigma, args, pos_of, ncams_of):
    """Merge two chains that are one athlete counted twice by the fusion for a few frames.

    A pair is a candidate when they share at most --phantom_frac of the shorter chain's
    instants, their appearance MMD^2 is under the gate, and at every shared instant they are
    within --phantom_max_m. The merge is refused if the merged sequence has a step faster than
    --phantom_max_speed (m/s). Returns (chains as sample lists, counts).
    """
    n = len(chains)
    tsets = [set(t for t, _ in c.samples) for c in chains]
    cand = []
    for i in range(n):
        if chains[i].emb is None or len(chains[i].emb) < 2:
            continue
        for j in range(i + 1, n):
            if chains[j].emb is None or len(chains[j].emb) < 2:
                continue
            sh = tsets[i] & tsets[j]
            if not sh:
                continue
            frac = len(sh) / max(min(len(tsets[i]), len(tsets[j])), 1)
            if frac > args.phantom_frac:
                continue
            m = mmd2(chains[i].emb, chains[j].emb, sigma, chains[i].kaa, chains[j].kaa)
            if m > args.mmd_gate:
                continue
            pi = {t: p for t, p in zip([t for t, _ in chains[i].samples],
                                       [pos_of[s] for s in chains[i].samples])}
            pj = {t: p for t, p in zip([t for t, _ in chains[j].samples],
                                       [pos_of[s] for s in chains[j].samples])}
            d = [float(np.hypot(*(pi[t] - pj[t]))) for t in sh]
            if max(d) > args.phantom_max_m:
                continue
            cand.append((m, i, j, len(sh), float(np.median(d))))
    cand.sort()

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    groups = {i: set(tsets[i]) for i in range(n)}
    members = {i: [i] for i in range(n)}
    merged, rejected_speed = 0, 0
    for m, i, j, ns, med in cand:
        a, b = find(i), find(j)
        if a == b:
            continue
        sh = groups[a] & groups[b]
        if len(sh) / max(min(len(groups[a]), len(groups[b])), 1) > args.phantom_frac:
            continue
        # The merge must not teleport. Being close at the few shared instants is not enough,
        # so build the merged sequence and refuse it if any step is faster than a person can run.
        sam = sorted({s: None for k in members[a] + members[b]
                      for s in chains[k].samples}.keys())
        keep = {}
        for sx in sam:
            t = sx[0]
            if t not in keep or ncams_of.get(sx, 0) > ncams_of.get(keep[t], 0):
                keep[t] = sx
        seq = sorted(keep.values())
        bad = False
        for u in range(1, len(seq)):
            dt = (seq[u][0] - seq[u - 1][0]) / args.fps
            if dt <= 0:
                continue
            v = float(np.hypot(*(pos_of[seq[u]] - pos_of[seq[u - 1]]))) / dt
            if v > args.phantom_max_speed:
                bad = True
                break
        if bad:
            rejected_speed += 1
            continue
        parent[b] = a
        groups[a] = groups[a] | groups[b]
        members[a] = members[a] + members[b]
        merged += 1
    if not merged:
        return chains, {"phantom_merges": 0, "phantom_duplicates_split_off": 0,
                        "phantom_rejected_on_speed": rejected_speed}

    out, extra = [], []
    buckets = {}
    for i in range(n):
        buckets.setdefault(find(i), []).append(i)
    for root, members in buckets.items():
        sam = [s for i in members for s in chains[i].samples]
        keep, dup = {}, []
        for s in sam:
            t = s[0]
            if t not in keep:
                keep[t] = s
            elif ncams_of.get(s, 0) > ncams_of.get(keep[t], 0):
                dup.append(keep[t]); keep[t] = s
            else:
                dup.append(s)
        out.append(sorted(keep.values()))
        extra.extend([[d] for d in dup])
    return out + extra, {"phantom_merges": merged,
                         "phantom_duplicates_split_off": sum(len(e) for e in extra),
                         "phantom_rejected_on_speed": rejected_speed}


# driver
def run(data, args):
    ts = sorted(int(t) for t in data.get("timestamps", []))
    frames = data.get("frames", {})
    t_start = time.time()

    # appearance vectors, L2-normalised once
    emb_all, key_of = [], {}
    for t in ts:
        for j, d in enumerate(frames.get(str(t), [])):
            a = d.get("appearance")
            if a is not None:
                v = _unit(np.asarray(a, dtype=np.float32))
                d["_app"] = v
                key_of[(t, j)] = len(emb_all)
                emb_all.append(v)
            else:
                d["_app"] = None
    emb_all = np.stack(emb_all) if emb_all else np.zeros((0, 1), np.float32)
    _APP_DIM[0] = int(emb_all.shape[1]) if len(emb_all) else 0
    sigma = median_sigma(emb_all) if len(emb_all) else 1.0
    print(f"[assemble] {len(ts)} instants, {sum(len(frames.get(str(t), [])) for t in ts)} "
          f"person-instants, {len(emb_all)} with appearance; RBF sigma = {sigma:.4f}")
    if not len(emb_all):
        print("[assemble] WARNING: no appearance vectors in this input, so the assembly runs on "
              "geometry alone and swaps identities much more often.")
        print("[assemble]          Use a fused file made without --no_appearance.")

    # tracking
    raw = build_tracklets(ts, frames, args)
    segs, stA = cut_tracks(raw, args)
    lens = sorted(len(v) for v in segs.values())
    print(f"[assemble] A  {stA['raw_tracks']} raw tracks -> {len(segs)} tracklets "
          f"({stA['ambiguity_cuts']} ambiguity cuts, {stA['gap_cuts']} gap cuts); "
          f"length median {lens[len(lens)//2]}, singletons {sum(1 for x in lens if x == 1)}")

    pos_of = {(t, j): np.asarray(frames[str(t)][j]["pos"][:2], float)
              for t in ts for j in range(len(frames.get(str(t), [])))}
    r_of = {(t, j): max(args.r_min, args.r_spread * float(frames[str(t)][j].get("spread_m") or 0.4))
            for t in ts for j in range(len(frames.get(str(t), [])))}

    # pooled within-tracklet self-affinity, the fallback for short tracklets
    def _emb(sam, cap):
        idx = [key_of.get(x) for x in sam]
        idx = [i for i in idx if i is not None]
        return emb_all[_thin(idx, cap)] if idx else None

    # `_emb` is None when a tracklet has no embeddings at all (for example when the fusion ran
    # with --no_appearance). Then the assembly runs on geometry alone instead of crashing.
    owns = [a for a in (self_affinity(e, sigma, shrink=False)
                        for e in (_emb(sam, args.emb_max_n)
                                  for sam in segs.values() if len(sam) >= 4)
                        if e is not None and len(e) >= 2)
            if a is not None]
    if owns:
        _KAA_POOL[0] = float(np.median(owns))
    print(f"[assemble] B  pooled within-tracklet self-affinity E[k(a,a')] = "
          f"{_KAA_POOL[0]:.4f} over {len(owns)} tracklets of 4+ samples")

    # calibrate the appearance gate on the raw tracklets, before anything uses it
    nn_of = _nearest_neighbour(ts, frames)
    cal = None
    if args.calibrate:
        pre = []
        for sam in sorted(segs.values(), key=lambda v: v[0][0]):
            e = _emb(sam, args.emb_max_n)
            ch = Chain(sam, e)
            if e is not None and len(e):
                ch.kaa = self_affinity(e, sigma)
            _fit_chain(ch, pos_of, r_of, args.fps, args.q)
            pre.append(ch)
        pos, neg = calibrate(pre, sigma, args, nn_of)
        if len(pos) >= args.cal_min_pos and len(neg) >= 20:
            g = float(np.quantile(pos, args.cal_pos_q))
            cal = {"n_positive": int(len(pos)), "n_negative": int(len(neg)),
                   "pos_median": round(float(np.median(pos)), 4),
                   "pos_q": round(g, 4),
                   "neg_median": round(float(np.median(neg)), 4),
                   "separation": round(float(np.median(neg) - np.median(pos)), 4),
                   "false_merge_rate_at_gate": round(float(np.mean(neg <= g)), 4)}
            args.mmd_gate = g
            args.mmd_scale = max(g, 1e-3)
            print(f"[assemble] cal {len(pos)} same-person pairs (MMD median "
                  f"{np.median(pos):.4f}) vs {len(neg)} different-person pairs "
                  f"(median {np.median(neg):.4f}); gate <- q{args.cal_pos_q:.2f} = {g:.4f}, "
                  f"admitting {100*np.mean(neg <= g):.1f} % of the different-people pairs")
        else:
            print(f"[assemble] cal too few calibration pairs ({len(pos)} pos / {len(neg)} neg); "
                  f"keeping mmd_gate {args.mmd_gate}")

    # splitting: cut a tracklet wherever its two halves look like different people
    split_gate = args.split_gate if args.split_gate is not None else \
        args.mmd_gate * args.split_gate_factor
    split_segs, n_splits, nid = {}, 0, 0
    for sid, sam in sorted(segs.items()):
        cuts = []
        if len(sam) >= 2 * args.kcp_min_seg:
            idx = [key_of.get(x) for x in sam]
            if all(i is not None for i in idx):
                sel = _thin(list(range(len(idx))), args.kcp_max_n)
                X = emb_all[[idx[i] for i in sel]]
                cuts = sorted({sel[m] for m in
                               mmd_split(rbf(X, X, sigma), args.kcp_min_seg, split_gate,
                                         args.kcp_max_depth) if 0 < m < len(sel)})
        prev = 0
        for cpt in cuts + [len(sam)]:
            if cpt <= prev:
                continue
            split_segs[nid] = sam[prev:cpt]
            nid += 1
            prev = cpt
        n_splits += len(cuts)
    print(f"[assemble] B  split gate {split_gate:.4f} -> {n_splits} appearance change-points, "
          f"{len(split_segs)} tracklets")

    # splitting on motion as well (opt-in): cut where the movement changes character
    stB2 = {}
    if args.kin_split:
        # Calibrate on clean tracklets: no one came within --cal_clear_m of them, so they
        # cannot contain a swap.
        clean = []
        for sid, sam in sorted(split_segs.items()):
            if len(sam) < 2 * args.kin_min_seg:
                continue
            if min((nn_of.get(x, 1e9) for x in sam), default=0.0) < args.cal_clear_m:
                continue
            v, _ = kin_best_cut(sam, pos_of, r_of, args.fps, args.q, args.kin_min_seg)
            if v > 0:
                clean.append(v)
        if args.kin_gate is not None:
            gate = args.kin_gate
            src = "given"
        elif len(clean) >= 8:
            gate = float(np.quantile(clean, args.kin_cal_q)) * args.split_gate_factor
            src = (f"q{args.kin_cal_q} of {len(clean)} clean tracklets "
                   f"x{args.split_gate_factor}")
        else:
            gate = None
            src = f"only {len(clean)} clean tracklets - NOT ENOUGH, kinematic split disabled"
        print(f"[assemble] B2 kinematic gate {gate if gate is None else round(gate, 3)} ({src})")

        if gate is not None:
            new_segs, nid2, n_kin = {}, 0, 0
            for sid, sam in sorted(split_segs.items()):
                cut = None
                if len(sam) >= 2 * args.kin_min_seg:
                    v, k = kin_best_cut(sam, pos_of, r_of, args.fps, args.q, args.kin_min_seg)
                    if k is not None and v >= gate:
                        cut = k
                        n_kin += 1
                if cut is None:
                    new_segs[nid2] = sam; nid2 += 1
                else:
                    new_segs[nid2] = sam[:cut]; nid2 += 1
                    new_segs[nid2] = sam[cut:]; nid2 += 1
            print(f"[assemble] B2 {n_kin} kinematic change-points -> {len(new_segs)} tracklets")
            split_segs = new_segs
            stB2 = {"gate": gate, "n_clean_cal": len(clean), "n_cuts": n_kin}

    # joining: iterated exact assembly
    chains = []
    for sid, sam in sorted(split_segs.items()):
        idx = [key_of.get(s) for s in sam]
        idx = [i for i in idx if i is not None]
        emb = emb_all[_thin(idx, args.emb_max_n)] if idx else None
        ch = Chain(sam, emb)
        if emb is not None and len(emb):
            ch.kaa = self_affinity(emb, sigma)
        _fit_chain(ch, pos_of, r_of, args.fps, args.q)
        chains.append(ch)

    rounds, uncertain = [], []
    for rd in range(args.rounds):
        succ, n_edges, why, margins = assemble(chains, sigma, args)
        cens = "  ".join(f"{k}:{v}" for k, v in why.most_common() if not k.startswith("_"))
        print(f"[assemble] C  round {rd+1} vetoes: {cens}   "
              f"| chains with no admissible successor at all: "
              f"{why['_chains_with_no_candidate']}/{len(chains)}")
        if not succ:
            rounds.append({"round": rd + 1, "chains_in": len(chains), "links": 0,
                           "edges": n_edges})
            print(f"[assemble] C  round {rd+1}: {len(chains)} chains, {n_edges} admissible "
                  f"edges, 0 links -> converged")
            break
        for (i, j), m in margins.items():
            a, b = chains[i], chains[j]
            uncertain.append({
                "round": rd + 1,
                "regret": round(m, 4),
                "t_end_a": int(a.t1), "t_start_b": int(b.t0),
                "gap_s": round((b.t0 - a.t1) / args.fps, 2),
                "at": [round(float(v), 2) for v in a.sm_x[-1][:2]],
                "to": [round(float(v), 2) for v in b.sm_x[0][:2]],
                "n_a": len(a.samples), "n_b": len(b.samples),
            })
        pred = {v: k for k, v in succ.items()}
        heads = [i for i in range(len(chains)) if i not in pred]
        merged = []
        for h in heads:
            sam, i = [], h
            while True:
                sam.extend(chains[i].samples)
                if i not in succ:
                    break
                i = succ[i]
            sam.sort()
            idx = [key_of.get(s) for s in sam]
            idx = [k for k in idx if k is not None]
            emb = emb_all[_thin(idx, args.emb_max_n)] if idx else None
            ch = Chain(sam, emb)
            if emb is not None and len(emb):
                ch.kaa = self_affinity(emb, sigma)
            _fit_chain(ch, pos_of, r_of, args.fps, args.q)
            merged.append(ch)
        rounds.append({"round": rd + 1, "chains_in": len(chains), "links": len(succ),
                       "edges": n_edges, "chains_out": len(merged)})
        print(f"[assemble] C  round {rd+1}: {len(chains)} chains, {n_edges} admissible edges, "
              f"{len(succ)} links -> {len(merged)} chains")
        chains = merged

    # resolve phantom co-detections, then rebuild the affected chains
    ph = {"phantom_merges": 0, "phantom_duplicates_split_off": 0}
    if args.phantom_merge:
        ncams_of = {(t, j): int(frames[str(t)][j].get("n_cams", 0))
                    for t in ts for j in range(len(frames.get(str(t), [])))}
        newsam, ph = phantom_merge(chains, sigma, args, pos_of, ncams_of)
        if ph["phantom_merges"]:
            chains = []
            for sam in newsam:
                sam = sorted(sam)
                idx = [key_of.get(x) for x in sam]
                idx = [k for k in idx if k is not None]
                e = emb_all[_thin(idx, args.emb_max_n)] if idx else None
                ch = Chain(sam, e)
                if e is not None and len(e):
                    ch.kaa = self_affinity(e, sigma)
                _fit_chain(ch, pos_of, r_of, args.fps, args.q)
                chains.append(ch)
        print(f"[assemble] D  phantom co-detections: {ph['phantom_merges']} merges, "
              f"{ph['phantom_duplicates_split_off']} duplicate samples split off, "
              f"{ph['phantom_rejected_on_speed']} candidates refused because the merge would "
              f"teleport -> {len(chains)} chains")

    # verify the invariant rather than trusting it
    for ci, ch in enumerate(chains):
        tt = [t for t, _ in ch.samples]
        assert len(tt) == len(set(tt)), f"chain {ci} occupies one instant twice - exclusion broken"

    # write the result
    chains.sort(key=lambda c: (c.t0, c.t1))
    assign, smooth = {}, {}
    for cid, ch in enumerate(chains):
        for k, s in enumerate(ch.samples):
            assign[s] = cid
            smooth[s] = ch.sm_x[k][:2]

    out_frames, track_len = {}, {}
    for t in ts:
        rows = []
        for j, d in enumerate(frames.get(str(t), [])):
            tid = assign.get((t, j))
            rec = {k: v for k, v in d.items() if k not in ("appearance", "_app")}
            if tid is None:                      # never happens; a person must never be dropped
                tid = -1
            rec["id"] = tid
            rec["track_id"] = tid
            sm = smooth.get((t, j))
            if sm is not None:
                rec["pos_smooth"] = [float(sm[0]), float(sm[1]), 0.0]
                if args.use_smoothed:
                    rec["pos_raw"] = list(d["pos"])
                    rec["pos"] = [float(sm[0]), float(sm[1]), 0.0]
            rows.append(rec)
            track_len[tid] = track_len.get(tid, 0) + 1
        out_frames[str(t)] = rows

    # least confident links first
    uncertain.sort(key=lambda r: r["regret"])
    n_close = sum(1 for r in uncertain if r["regret"] < args.uncertain_margin)

    out = {"timestamps": ts, "frames": out_frames}
    for k, v in data.items():
        if k not in ("timestamps", "frames"):
            out[k] = v
    stats = {
        "n_timestamps": len(ts),
        "n_tracklets_stageA": len(segs),
        "n_tracklets_after_split": len(split_segs),
        "n_appearance_changepoints": n_splits,
        "split_gate": round(float(split_gate), 5),
        "n_tracks": len(chains),
        "rbf_sigma": round(float(sigma), 5),
        "kaa_pool": round(float(_KAA_POOL[0]), 5),
        "rounds": rounds,
        "track_lengths": dict(sorted(track_len.items())),
        "seconds": round(time.time() - t_start, 1),
        "n_links_total": len(uncertain),
        "n_links_uncertain": n_close,
        "calibration": cal,
        **ph,
        "mmd_gate": round(args.mmd_gate, 5),
        **stA,
    }
    out["assembly"] = {k: v for k, v in stats.items() if k != "track_lengths"}
    # store the command line too, so a later comparison can tell a code change from a flag change
    out["assembly"]["argv"] = sys.argv[1:]
    if stB2:
        stats["kinematic_split"] = stB2
    out["uncertain_links"] = uncertain[:args.uncertain_keep]
    print(f"[assemble] E  {n_close} of {len(uncertain)} examined links have regret below "
          f"{args.uncertain_margin} - the evidence did not force them;")
    print(f"[assemble]      the {min(len(uncertain), args.uncertain_keep)} least confident are in "
          f"`uncertain_links` - that is the list a human should adjudicate, not the whole clip.")
    return out, stats


def _thin(idx, cap):
    """Evenly thin a list of indices to at most `cap` entries (keeps the ends)."""
    if len(idx) <= cap:
        return list(idx)
    sel = np.linspace(0, len(idx) - 1, cap).round().astype(int)
    return [idx[i] for i in sorted(set(sel.tolist()))]


def _nearest_neighbour(ts, frames):
    """Distance to the closest other person, per person-instant."""
    out = {}
    for t in ts:
        P = np.array([d["pos"][:2] for d in frames.get(str(t), [])], float)
        n = len(P)
        if n == 0:
            continue
        if n == 1:
            out[(t, 0)] = 1e9
            continue
        D = np.sqrt(np.maximum(_sqdist(P, P), 0.0))
        np.fill_diagonal(D, np.inf)
        for j in range(n):
            out[(t, j)] = float(D[j].min())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True,
                    help="people_fused*.json (must still carry the `appearance` vectors)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=25.0)
    # tracking: Gaussian state + ambiguity cutting
    ap.add_argument("--q", type=float, default=16.0,
                    help="acceleration PSD (m^2/s^3) of the constant-velocity model; sets how fast "
                         "the gate opens while a track coasts")
    ap.add_argument("--r_min", type=float, default=0.06,
                    help="floor on the measurement sigma (m); a moving fused position jitters by "
                         "about 0.07 m at 25 fps")
    ap.add_argument("--r_spread", type=float, default=0.15,
                    help="measurement sigma also scales with the multi-view spread of that "
                         "detection, so a 2-camera guess is trusted less than a 7-camera one")
    ap.add_argument("--gate_maha", type=float, default=4.0, help="Mahalanobis association gate")
    ap.add_argument("--gate_max_m", type=float, default=2.5, help="absolute association gate (m)")
    ap.add_argument("--max_coast_s", type=float, default=2.0,
                    help="how long a track may coast unobserved before the forward pass drops it")
    ap.add_argument("--miss_cost", type=float, default=0.9,
                    help="cost of leaving a live track unobserved this frame")
    ap.add_argument("--amb_margin", type=float, default=0.15,
                    help="a sample is ambiguous when the winning assignment beats the runner-up "
                         "by less than this. Higher = more cutting = more work for the joining step, but "
                         "fewer swaps carried forward. 0 disables cutting entirely (for A/B).")
    ap.add_argument("--amb_min_run", type=int, default=3,
                    help="ignore ambiguous runs shorter than this many samples (flicker, not a "
                         "crossing)")
    ap.add_argument("--amb_merge_run", type=int, default=6,
                    help="merge two ambiguous episodes separated by fewer clean samples than this")
    ap.add_argument("--cut_gap_s", type=float, default=0.8,
                    help="also cut wherever the track coasted longer than this (s)")
    ap.add_argument("--app_ema", type=float, default=0.9)
    ap.add_argument("--app_scale", type=float, default=0.35,
                    help="cosine distance that counts as one unit of appearance cost")
    # splitting: kernel change-point detection
    ap.add_argument("--kcp_min_seg", type=int, default=8,
                    help="shortest segment a change-point may create (samples)")
    ap.add_argument("--kin_split", action="store_true",
                    help="also split a tracklet at the most swap-like kinematic step (the filter "
                         "innovation standardised by its own covariance). Off by default: it did "
                         "not help on this clip")
    ap.add_argument("--kin_gate", type=float, default=None,
                    help="standardised-innovation gate. Default: calibrated from tracklets with no one "
                         "within --cal_clear_m")
    ap.add_argument("--kin_min_seg", type=int, default=8)
    ap.add_argument("--kin_cal_q", type=float, default=0.99,
                    help="quantile of the clean-tracklet statistic used as the gate")
    ap.add_argument("--kcp_max_n", type=int, default=240,
                    help="thin a long tracklet to at most this many samples for the kernel matrix")
    ap.add_argument("--kcp_max_depth", type=int, default=6)
    ap.add_argument("--split_gate", type=float, default=None,
                    help="MMD^2 between a tracklet's two halves above which it is split. Default "
                         "= split_gate_factor x the calibrated same-person gate.")
    ap.add_argument("--split_gate_factor", type=float, default=2.0,
                    help="a split is a destructive edit, so it is held to a stricter standard "
                         "than a merge: by default the halves must be twice as far apart as the "
                         "same-person gate before the tracklet is cut")
    # joining: global assembly
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--max_gap_s", type=float, default=8.0,
                    help="longest gap a link may bridge. Generous, since the appearance gate "
                         "and the one-successor constraint carry the risk, and a long re-attach "
                         "is what turns two half-identities into one.")
    ap.add_argument("--link_maha", type=float, default=6.0)
    ap.add_argument("--link_max_m", type=float, default=9.0)
    ap.add_argument("--mmd_gate", type=float, default=0.30,
                    help="hard veto: two tracklets whose appearance distributions differ by more "
                         "squared-MMD than this are never the same person")
    ap.add_argument("--mmd_scale", type=float, default=0.15)
    ap.add_argument("--app_missing", type=float, default=0.12,
                    help="MMD stand-in when one side has no embeddings at all")
    ap.add_argument("--emb_max_n", type=int, default=48,
                    help="embeddings sampled per chain for the MMD (kernel mean embedding is a "
                         "sample mean, so a few dozen is plenty and it keeps the pass fast)")
    ap.add_argument("--w_app", type=float, default=1.0)
    ap.add_argument("--w_mot", type=float, default=0.4)
    ap.add_argument("--w_gap", type=float, default=0.2)
    ap.add_argument("--link_cost_max", type=float, default=None,
                    help="accept threshold on the link cost. Default = w_app+w_mot+w_gap, i.e. "
                         "exactly the cost of a link sitting on every gate at once - so the HARD "
                         "GATES decide admissibility and the cost only ranks the survivors. "
                         "Lower it to make the assembly more conservative.")
    ap.add_argument("--calibrate", dest="calibrate", action="store_true", default=True,
                    help="learn the appearance gate from known same/different pairs (default)")
    ap.add_argument("--no_calibrate", dest="calibrate", action="store_false")
    ap.add_argument("--cal_gap_s", type=float, default=0.4)
    ap.add_argument("--cal_pos_m", type=float, default=0.6)
    ap.add_argument("--cal_clear_m", type=float, default=3.0,
                    help="a junction with no one else this close is the same person")
    ap.add_argument("--cal_overlap_frames", type=int, default=3)
    ap.add_argument("--cal_max_neg", type=int, default=20000)
    ap.add_argument("--cal_min_pos", type=int, default=25)
    ap.add_argument("--constraints", default="",
                    help="JSON of junctions a human answered (optional, the pipeline does not use it). `same` "
                         "force-links and overrides the geometric vetoes, `different` force-cuts. "
                         "Keyed by (t_end_a, t_start_b), the pair `uncertain_links` records")
    ap.add_argument("--cal_pos_q", type=float, default=0.98,
                    help="quantile of the same-person MMDs used as the appearance gate")
    ap.add_argument("--app_margin", type=float, default=0.0,
                    help="appearance margin (MMD^2) a link is expected to have over its nearest "
                         "rival. Links with less are charged --w_amb, pro rata, so they are taken "
                         "only when the geometry can carry them alone. 0 disables. This targets "
                         "WITHIN-TEAM swaps, which no appearance threshold can reach because "
                         "teammates sit below the same-person median.")
    ap.add_argument("--w_amb", type=float, default=0.8,
                    help="cost charged for a link whose appearance evidence does not discriminate")
    ap.add_argument("--phantom_merge", dest="phantom_merge", action="store_true", default=True,
                    help="merge identities kept apart only by a handful of duplicate detections "
                         "(default on; see phantom_merge)")
    ap.add_argument("--no_phantom_merge", dest="phantom_merge", action="store_false")
    ap.add_argument("--phantom_frac", type=float, default=0.06,
                    help="max share of the shorter track's instants two chains may share and "
                         "still be considered one person counted twice")
    ap.add_argument("--phantom_max_speed", type=float, default=9.0,
                    help="a phantom merge is refused if the merged sequence contains a step faster "
                         "than this (m/s) - the check that distinguishes one person counted twice "
                         "from two people who are rarely co-detected")
    ap.add_argument("--phantom_max_m", type=float, default=1.6,
                    help="at every shared instant the two must be at least this close; two different "
                         "people who co-exist are metres apart")
    ap.add_argument("--regret", dest="regret", action="store_true", default=True,
                    help="compute each link's REGRET (cost of the best assembly without it) so the "
                         "least-forced decisions can be handed to a human. Costs one re-solve per "
                         "link examined; a few seconds at this size.")
    ap.add_argument("--no_regret", dest="regret", action="store_false")
    ap.add_argument("--regret_max", type=int, default=5000,
                    help="cap on links examined (most expensive first). The default covers every "
                         "link at this clip's size; lower it only if the re-solves become slow.")
    ap.add_argument("--uncertain_margin", type=float, default=0.05,
                    help="a link whose regret is below this could have been dropped for almost no "
                         "cost - the evidence did not force it")
    ap.add_argument("--uncertain_keep", type=int, default=200,
                    help="how many of the least-confident links to write into `uncertain_links`")
    ap.add_argument("--use_smoothed", action="store_true",
                    help="write the RTS-smoothed position into `pos` (raw kept as `pos_raw`)")
    args = ap.parse_args()
    if args.link_cost_max is None:
        args.link_cost_max = args.w_app + args.w_mot + args.w_gap

    args._constraints = {}
    if args.constraints:
        doc = json.loads(Path(args.constraints).read_text(encoding="utf-8"))
        rows = doc["constraints"] if isinstance(doc, dict) else doc
        n_s = n_d = n_loose = 0
        for r in rows:
            at, to = r.get("at"), r.get("to")
            if at is None or to is None:
                n_loose += 1
            args._constraints.setdefault(
                (int(r["t_end_a"]), int(r["t_start_b"])), []).append((at, to, r["verdict"]))
            n_s += r["verdict"] == "same"
            n_d += r["verdict"] == "different"
        print(f"[assemble] {n_s + n_d} human verdicts loaded from "
              f"{Path(args.constraints).name}: {n_s} must-link, {n_d} cannot-link")
        if n_loose:
            print(f"[assemble] WARNING: {n_loose} carry no `at`/`to`, so they match on the frame "
                  f"pair alone and could hit the wrong junction")

    inp = Path(args.inp).resolve()
    data = json.loads(inp.read_text(encoding="utf-8"))
    out, stats = run(data, args)
    Path(args.out).resolve().write_text(json.dumps(out), encoding="utf-8")
    print(f"[assemble] {inp.name} -> {Path(args.out).name}")
    for k, v in stats.items():
        if k == "track_lengths":
            continue
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
