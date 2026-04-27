"""
uncertainty.py
==============================================================================
Phase-5 uncertainty quantification and calibration module for the
"Lightweight Multi-Modal Tiny LLM Framework for Privacy-Preserving Academic
Assistance in University Environments" research codebase.

This file is a thin, dependency-free *consumer* of the rest of the system:
it accepts raw probability distributions and confidence/accuracy arrays from
``classifier.py`` / ``models.py`` / ``summarizer.py`` and never touches their
internals. It is the calibration layer that ``evaluate.py`` (Phase 6) will
plug into.

Provides
--------
1. **Semantic Predictive Uncertainty (SPU)** -- mean pairwise Jensen-Shannon
   Divergence across N stochastic forward passes::

       SPU(x) = mean_{i<j} JSD(P_i || P_j)

   bounded in [0, ln(2)] in nats.

2. **Bloom-level uncertainty** -- normalised entropy of the LDL distribution::

       U_bloom = H(p) / log(K),     K = |Bloom levels| (default 6)

   bounded in [0, 1].

3. **Expected Calibration Error (ECE)** with configurable bin count::

       ECE = sum_k (|B_k|/N) * |conf_k - acc_k|

4. **Reliability diagram data** -- bin centres, bin accuracy, bin confidence,
   bin counts. Plotting is intentionally *not* done here.

5. **Deterministic stochastic forward pass runner** -- seed = base_seed +
   run_index, with a flexible model_fn signature contract.

Constraints
-----------
* CPU only, < 1 GB RAM (numpy only; ~10 KB working set for typical usage).
* Deterministic seeding: random / numpy / torch == 42.
* No external APIs, no training, no modification of earlier-phase files.
"""

from __future__ import annotations

import logging
import math
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ----------------------------------------------------------------------------
# Reproducibility (mandated global rule)
# ----------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)
try:
    import torch  # noqa: F401
    torch.manual_seed(42)
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logger = logging.getLogger("uncertainty")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(levelname)s: %(message)s",
    )


def _ok(msg: str) -> None:
    enc = (getattr(sys.stdout, "encoding", None) or "utf-8").lower()
    mark = "\u2714" if "utf" in enc else "[OK]"
    try:
        print(f"{mark} {msg}")
    except UnicodeEncodeError:  # pragma: no cover
        print(f"[OK] {msg}")


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
EPS_DEFAULT = 1e-12
DEFAULT_BLOOM_K = 6
JSD_MAX = math.log(2.0)  # natural-log upper bound of JSD


# ----------------------------------------------------------------------------
# Numerically stable distribution helpers
# ----------------------------------------------------------------------------
def _as_prob(p: Any, axis: int = -1, eps: float = EPS_DEFAULT) -> np.ndarray:
    """Coerce input to a non-negative probability array along ``axis``.

    * Casts to float64 for numerical stability.
    * Rejects NaN/inf, negative entries, and all-zero rows.
    * Renormalises so the requested axis sums to 1.
    """
    arr = np.asarray(p, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("probability array is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("probability array contains NaN/inf")
    if (arr < -1e-9).any():
        raise ValueError("probability array contains negative entries")
    arr = np.clip(arr, 0.0, None)
    s = arr.sum(axis=axis, keepdims=True)
    if (s < eps).any():
        raise ValueError("probability vector sums to (essentially) 0")
    return arr / s


def _entropy(p: np.ndarray, axis: int = -1, eps: float = EPS_DEFAULT) -> np.ndarray:
    """H(p) = -Σ p log p in nats. p must already be a valid distribution."""
    safe = np.clip(p, eps, 1.0)
    return -(p * np.log(safe)).sum(axis=axis)


def _kl(p: np.ndarray, q: np.ndarray, axis: int = -1, eps: float = EPS_DEFAULT) -> np.ndarray:
    """KL(p || q) in nats with masking of zero p entries."""
    log_p = np.log(np.clip(p, eps, 1.0))
    log_q = np.log(np.clip(q, eps, 1.0))
    return (p * (log_p - log_q)).sum(axis=axis)


def _jsd(p: np.ndarray, q: np.ndarray, axis: int = -1, eps: float = EPS_DEFAULT) -> float:
    """Jensen-Shannon Divergence in nats, JSD(p, q) ∈ [0, ln 2]."""
    m = 0.5 * (p + q)
    val = 0.5 * _kl(p, m, axis=axis, eps=eps) + 0.5 * _kl(q, m, axis=axis, eps=eps)
    # Numerical floor / ceiling
    return float(min(JSD_MAX, max(0.0, float(val))))


# ----------------------------------------------------------------------------
# Output containers
# ----------------------------------------------------------------------------
@dataclass
class CalibrationReport:
    ece: float
    bin_centers: np.ndarray
    bin_accuracy: np.ndarray   # NaN where bin is empty
    bin_confidence: np.ndarray # NaN where bin is empty
    bin_counts: np.ndarray     # int counts per bin


@dataclass
class UncertaintySummary:
    """Aggregate uncertainty signals for one prediction (or a batch)."""

    spu: float                  # mean pairwise JSD across stochastic outputs
    bloom_uncertainty: float    # normalised entropy in [0, 1]
    bloom_entropy: float        # raw entropy (nats)
    confidence: float           # 1 - bloom_uncertainty
    n_samples: int = 0          # number of stochastic outputs used for SPU
    metadata: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# UncertaintyEngine
# ----------------------------------------------------------------------------
class UncertaintyEngine:
    """Stateless uncertainty + calibration utilities (CPU, pure NumPy)."""

    def __init__(
        self,
        K: int = DEFAULT_BLOOM_K,
        n_bins: int = 10,
        eps: float = EPS_DEFAULT,
    ) -> None:
        if K < 2:
            raise ValueError("K must be >= 2 (need at least 2 classes)")
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        if eps <= 0:
            raise ValueError("eps must be > 0")
        self.K = int(K)
        self.n_bins = int(n_bins)
        self.eps = float(eps)

    # ------------------------------------------------------------------ #
    # 1) Semantic Predictive Uncertainty (SPU) via mean pairwise JSD
    # ------------------------------------------------------------------ #
    def compute_jsd_uncertainty(
        self,
        outputs: Sequence[Union[Sequence[float], np.ndarray]],
        pairwise: bool = True,
    ) -> float:
        """Return SPU = mean pairwise JSD across N predictive distributions.

        If ``pairwise=True`` (default per spec) we compute the mean of
        ``JSD(P_i || P_j)`` over all C(N, 2) unordered pairs. With
        ``pairwise=False`` we use the *generalised* JSD::

            JSD = H((1/N) Σ P_i) - (1/N) Σ H(P_i)

        which is mathematically equivalent up to a constant for fixed N
        and avoids the O(N^2) loop. Both are non-negative.

        Returns 0.0 when fewer than 2 outputs are provided.
        """
        if len(outputs) < 2:
            return 0.0
        # Stack into (N, K) and validate as distributions on the last axis
        try:
            P = np.stack([_as_prob(p, axis=-1, eps=self.eps) for p in outputs], axis=0)
        except ValueError as e:
            raise ValueError(f"invalid stochastic output: {e}") from e
        # Ensure all same K
        if P.ndim != 2:
            raise ValueError(
                f"stochastic outputs must be 1-D distributions; got shape {P.shape}"
            )

        N = P.shape[0]
        if pairwise:
            total = 0.0
            count = 0
            for i in range(N):
                for j in range(i + 1, N):
                    total += _jsd(P[i], P[j], eps=self.eps)
                    count += 1
            return float(total / max(1, count))
        else:
            M = P.mean(axis=0)
            H_M = float(_entropy(M, eps=self.eps))
            mean_H = float(_entropy(P, axis=1, eps=self.eps).mean())
            return float(min(JSD_MAX, max(0.0, H_M - mean_H)))

    # ------------------------------------------------------------------ #
    # 2) Bloom-level uncertainty (normalised entropy)
    # ------------------------------------------------------------------ #
    def compute_bloom_uncertainty(
        self,
        probs: Union[Sequence[float], np.ndarray],
    ) -> float:
        """Return ``H(p) / log(K)`` in [0, 1] for a single Bloom distribution."""
        p = _as_prob(np.asarray(probs, dtype=np.float64).reshape(-1), eps=self.eps)
        if p.size != self.K:
            raise ValueError(
                f"expected {self.K}-dim distribution, got {p.size}"
            )
        H = float(_entropy(p, eps=self.eps))
        max_H = math.log(self.K)
        return float(min(1.0, max(0.0, H / max_H)))

    def compute_bloom_uncertainty_batch(
        self,
        probs: Union[Sequence[Sequence[float]], np.ndarray],
    ) -> np.ndarray:
        """Vectorised version of :meth:`compute_bloom_uncertainty`."""
        P = _as_prob(np.asarray(probs, dtype=np.float64), axis=-1, eps=self.eps)
        if P.ndim != 2 or P.shape[1] != self.K:
            raise ValueError(
                f"expected (N, {self.K}) batch, got shape {P.shape}"
            )
        H = _entropy(P, axis=1, eps=self.eps)
        return (H / math.log(self.K)).clip(0.0, 1.0).astype(np.float64)

    # ------------------------------------------------------------------ #
    # 3) Expected Calibration Error (ECE)
    # ------------------------------------------------------------------ #
    def compute_ece(
        self,
        confidences: Union[Sequence[float], np.ndarray],
        accuracies: Union[Sequence[int], np.ndarray],
    ) -> float:
        """Return ECE over equal-width confidence bins."""
        c, a = self._validate_calib(confidences, accuracies)
        edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        N = len(c)
        ece = 0.0
        for k in range(self.n_bins):
            lo, hi = edges[k], edges[k + 1]
            in_bin = (
                (c >= lo) & (c <= hi)
                if k == self.n_bins - 1
                else (c >= lo) & (c < hi)
            )
            n_k = int(in_bin.sum())
            if n_k == 0:
                continue
            avg_conf = float(c[in_bin].mean())
            avg_acc = float(a[in_bin].mean())
            ece += (n_k / N) * abs(avg_conf - avg_acc)
        return float(ece)

    # ------------------------------------------------------------------ #
    # 4) Reliability diagram data (no plotting)
    # ------------------------------------------------------------------ #
    def reliability_data(
        self,
        confidences: Union[Sequence[float], np.ndarray],
        accuracies: Union[Sequence[int], np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Return arrays ready for a reliability-diagram plot.

        Empty bins yield NaN for bin_accuracy / bin_confidence and 0 for
        bin_counts, so downstream code can mask them.
        """
        c, a = self._validate_calib(confidences, accuracies)
        edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        bin_acc = np.full(self.n_bins, np.nan, dtype=np.float64)
        bin_conf = np.full(self.n_bins, np.nan, dtype=np.float64)
        bin_counts = np.zeros(self.n_bins, dtype=np.int64)

        for k in range(self.n_bins):
            lo, hi = edges[k], edges[k + 1]
            in_bin = (
                (c >= lo) & (c <= hi)
                if k == self.n_bins - 1
                else (c >= lo) & (c < hi)
            )
            n_k = int(in_bin.sum())
            bin_counts[k] = n_k
            if n_k > 0:
                bin_acc[k] = float(a[in_bin].mean())
                bin_conf[k] = float(c[in_bin].mean())

        return {
            "bin_centers": centers,
            "bin_accuracy": bin_acc,
            "bin_confidence": bin_conf,
            "bin_counts": bin_counts,
        }

    def calibration_report(
        self,
        confidences: Union[Sequence[float], np.ndarray],
        accuracies: Union[Sequence[int], np.ndarray],
    ) -> CalibrationReport:
        """Convenience wrapper combining ECE + reliability arrays."""
        rd = self.reliability_data(confidences, accuracies)
        return CalibrationReport(
            ece=self.compute_ece(confidences, accuracies),
            bin_centers=rd["bin_centers"],
            bin_accuracy=rd["bin_accuracy"],
            bin_confidence=rd["bin_confidence"],
            bin_counts=rd["bin_counts"],
        )

    # ------------------------------------------------------------------ #
    # 5) Deterministic stochastic forward pass runner
    # ------------------------------------------------------------------ #
    def run_stochastic_forward(
        self,
        model_fn: Callable[..., Any],
        x: Any,
        n: int = 5,
        temperature: float = 1.0,
        base_seed: int = 42,
    ) -> List[Any]:
        """Run ``model_fn(x)`` ``n`` times under deterministic seeds.

        Seed contract: the i-th call uses ``seed = base_seed + i`` (per spec).
        Global ``random`` / ``numpy`` / ``torch`` seeds are also set so that
        callees that do not accept seed kwargs still behave deterministically.

        ``model_fn`` is called with the most specific signature available:
            1. ``model_fn(x, seed=seed, temperature=temperature)``
            2. ``model_fn(x, seed=seed)``
            3. ``model_fn(x, temperature=temperature)``
            4. ``model_fn(x)``
        """
        if not callable(model_fn):
            raise TypeError("model_fn must be callable")
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        outputs: List[Any] = []
        for i in range(n):
            seed = int(base_seed) + i
            random.seed(seed)
            np.random.seed(seed)
            if torch is not None:
                try:
                    torch.manual_seed(seed)
                except Exception:  # pragma: no cover
                    pass

            for kwargs in (
                {"seed": seed, "temperature": temperature},
                {"seed": seed},
                {"temperature": temperature},
                {},
            ):
                try:
                    out = model_fn(x, **kwargs)
                    break
                except TypeError:
                    continue
            else:  # pragma: no cover - unreachable, last attempt has {}
                out = model_fn(x)
            outputs.append(out)
        return outputs

    # ------------------------------------------------------------------ #
    # 6) Aggregate summary (one-call uncertainty bundle)
    # ------------------------------------------------------------------ #
    def aggregate_summary(
        self,
        bloom_distribution: Optional[Union[Sequence[float], np.ndarray]] = None,
        stochastic_outputs: Optional[Sequence[Union[Sequence[float], np.ndarray]]] = None,
    ) -> UncertaintySummary:
        """Combine Bloom-uncertainty and SPU into one container."""
        bloom_unc = 0.0
        bloom_H = 0.0
        conf = 1.0
        if bloom_distribution is not None:
            bloom_unc = self.compute_bloom_uncertainty(bloom_distribution)
            bloom_H = bloom_unc * math.log(self.K)
            conf = 1.0 - bloom_unc

        spu = 0.0
        n_s = 0
        if stochastic_outputs is not None and len(stochastic_outputs) >= 2:
            spu = self.compute_jsd_uncertainty(list(stochastic_outputs))
            n_s = len(stochastic_outputs)

        return UncertaintySummary(
            spu=float(spu),
            bloom_uncertainty=float(bloom_unc),
            bloom_entropy=float(bloom_H),
            confidence=float(conf),
            n_samples=int(n_s),
        )

    # ------------------------------------------------------------------ #
    # Internal: input validation for calibration arrays
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_calib(
        confidences: Union[Sequence[float], np.ndarray],
        accuracies: Union[Sequence[int], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        c = np.asarray(confidences, dtype=np.float64).reshape(-1)
        a = np.asarray(accuracies, dtype=np.float64).reshape(-1)
        if c.size != a.size:
            raise ValueError(
                f"confidences ({c.size}) and accuracies ({a.size}) length mismatch"
            )
        if c.size == 0:
            raise ValueError("empty calibration arrays")
        if not np.all(np.isfinite(c)) or not np.all(np.isfinite(a)):
            raise ValueError("non-finite entries in calibration arrays")
        if (c < -1e-9).any() or (c > 1.0 + 1e-9).any():
            raise ValueError("confidences must lie in [0, 1]")
        c = np.clip(c, 0.0, 1.0)
        if (np.abs(a - np.round(a)) > 1e-9).any():
            raise ValueError("accuracies must be 0/1 indicators")
        a = np.round(a).astype(np.int64)
        if ((a < 0) | (a > 1)).any():
            raise ValueError("accuracies must be 0 or 1")
        return c, a


# ============================================================================
# SELF-TEST
# ----------------------------------------------------------------------------
# Validates:
#   * JSD: zero for identical distributions; positive for differing ones;
#     bounded by ln(2); permutation-invariant; pairwise vs generalised both
#     non-negative.
#   * Bloom uncertainty: 1 for uniform, 0 for one-hot; raises on K mismatch.
#   * ECE: 0 by construction for synthetic perfectly-calibrated data; large
#     (~0.45) for systematically over-confident data; raises on bad inputs.
#   * Reliability data shapes and bin-count totals.
#   * Stochastic forward: returns n outputs, each a valid distribution;
#     reruns with same base_seed are bit-identical; supports model_fns that
#     ignore seed/temperature (signature fallback chain).
#   * aggregate_summary combines both signals with correct n_samples.
#   * Constructor and method input validation.
# ============================================================================
def _self_test() -> None:
    eng = UncertaintyEngine(K=6, n_bins=10)

    # ------------------------------------------------------------------ #
    # 1. JSD-based SPU
    # ------------------------------------------------------------------ #
    p = np.array([0.7, 0.1, 0.05, 0.05, 0.05, 0.05])
    q = np.array([0.05, 0.05, 0.05, 0.05, 0.10, 0.70])

    # Identical distributions -> SPU = 0
    assert eng.compute_jsd_uncertainty([p, p, p]) < 1e-12

    # Differing distributions -> SPU > 0 and <= ln(2)
    val = eng.compute_jsd_uncertainty([p, q, p, q, p])
    assert 0.05 < val <= JSD_MAX + 1e-9, f"unexpected SPU={val}"

    # Permutation invariance (pairwise mean is set-symmetric)
    val_swap = eng.compute_jsd_uncertainty([q, p, q, p, q])
    # Note: [p,q,p,q,p] has 3 p's and 2 q's; [q,p,q,p,q] has 2 p's and 3 q's.
    # Pair multiset is the same up to ordering, so means should be equal.
    assert abs(val - val_swap) < 1e-9

    # Generalised JSD path is also non-negative and finite
    g = eng.compute_jsd_uncertainty([p, q, p, q, p], pairwise=False)
    assert 0.0 <= g <= JSD_MAX + 1e-9

    # Single-element list -> 0
    assert eng.compute_jsd_uncertainty([p]) == 0.0

    # ------------------------------------------------------------------ #
    # 2. Bloom-level uncertainty
    # ------------------------------------------------------------------ #
    uniform = np.full(6, 1.0 / 6.0)
    assert abs(eng.compute_bloom_uncertainty(uniform) - 1.0) < 1e-9

    onehot = np.zeros(6); onehot[2] = 1.0
    assert eng.compute_bloom_uncertainty(onehot) < 1e-6

    # K mismatch must raise
    try:
        eng.compute_bloom_uncertainty(np.array([0.5, 0.5]))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on K mismatch")

    # Batched version: shape and bounds
    batch = np.stack([uniform, onehot, p])
    u_batch = eng.compute_bloom_uncertainty_batch(batch)
    assert u_batch.shape == (3,)
    assert abs(u_batch[0] - 1.0) < 1e-9 and u_batch[1] < 1e-6
    assert 0.0 < u_batch[2] < 1.0

    # ------------------------------------------------------------------ #
    # 3. Expected Calibration Error (ECE)
    # ------------------------------------------------------------------ #
    # Construct a *deterministic* perfectly-calibrated distribution:
    # for each bin k (centre c_k), n_ones = round(100 * c_k) ones and the
    # rest zeros. Then conf_avg == c_k and acc_avg == c_k exactly per bin.
    confs: List[float] = []
    accs: List[int] = []
    for k in range(10):
        c_k = (k + 0.5) / 10.0
        n_ones = int(round(100 * c_k))
        confs.extend([c_k] * 100)
        accs.extend([1] * n_ones + [0] * (100 - n_ones))
    ece_calib = eng.compute_ece(confs, accs)
    # All within-bin gaps are 0 by construction; ECE must be ~0.
    assert ece_calib < 1e-9, f"calibrated ECE not zero: {ece_calib}"

    # Systematic overconfidence: predict 0.95 always, but accuracy ~ 0.5
    rng = np.random.default_rng(0)
    overconf_c = np.full(2000, 0.95)
    overconf_a = (rng.uniform(size=2000) < 0.5).astype(int)
    ece_over = eng.compute_ece(overconf_c, overconf_a)
    assert ece_over > 0.40, f"overconfident ECE too low: {ece_over}"

    # ------------------------------------------------------------------ #
    # 4. Reliability data shapes and counts
    # ------------------------------------------------------------------ #
    rd = eng.reliability_data(confs, accs)
    assert set(rd.keys()) == {
        "bin_centers", "bin_accuracy", "bin_confidence", "bin_counts"
    }
    assert rd["bin_centers"].shape == (10,)
    assert rd["bin_accuracy"].shape == (10,)
    assert rd["bin_confidence"].shape == (10,)
    assert rd["bin_counts"].shape == (10,)
    assert int(rd["bin_counts"].sum()) == len(confs) == 1000
    # No NaNs in this synthetic set (every bin populated)
    assert np.all(np.isfinite(rd["bin_accuracy"]))
    assert np.all(np.isfinite(rd["bin_confidence"]))

    # ------------------------------------------------------------------ #
    # 5. Deterministic stochastic forward
    # ------------------------------------------------------------------ #
    def dummy_model(x: str, seed: int = 0, temperature: float = 1.0) -> np.ndarray:
        r = np.random.default_rng(seed)
        z = r.normal(size=6) / max(temperature, 1e-3)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    outs1 = eng.run_stochastic_forward(
        dummy_model, "the input", n=5, temperature=1.0, base_seed=42
    )
    assert len(outs1) == 5
    for o in outs1:
        assert isinstance(o, np.ndarray) and o.shape == (6,)
        assert np.isclose(o.sum(), 1.0)
        assert (o >= 0).all()

    # Determinism: re-run with same base_seed -> bit-identical
    outs2 = eng.run_stochastic_forward(
        dummy_model, "the input", n=5, temperature=1.0, base_seed=42
    )
    for a, b in zip(outs1, outs2):
        assert np.allclose(a, b, atol=0, rtol=0), "stochastic forward not deterministic"

    # Different base_seed -> different outputs
    outs3 = eng.run_stochastic_forward(
        dummy_model, "the input", n=5, temperature=1.0, base_seed=100
    )
    assert not np.allclose(outs1[0], outs3[0])

    # Signature fallback: model_fn that doesn't accept seed/temperature
    def signature_free_model(x: str) -> np.ndarray:
        # Reads global numpy seed (set by run_stochastic_forward).
        z = np.random.normal(size=6)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    outs_sf = eng.run_stochastic_forward(signature_free_model, "x", n=3, base_seed=7)
    assert len(outs_sf) == 3
    outs_sf2 = eng.run_stochastic_forward(signature_free_model, "x", n=3, base_seed=7)
    for a, b in zip(outs_sf, outs_sf2):
        assert np.allclose(a, b)  # global-seed determinism

    # SPU on stochastic outputs
    spu = eng.compute_jsd_uncertainty(outs1)
    assert 0.0 <= spu <= JSD_MAX + 1e-9

    # ------------------------------------------------------------------ #
    # 6. aggregate_summary
    # ------------------------------------------------------------------ #
    summ = eng.aggregate_summary(bloom_distribution=p, stochastic_outputs=outs1)
    assert 0.0 <= summ.bloom_uncertainty <= 1.0
    assert 0.0 <= summ.confidence <= 1.0
    assert abs(summ.confidence + summ.bloom_uncertainty - 1.0) < 1e-9
    assert summ.n_samples == 5
    assert math.isfinite(summ.spu) and summ.spu >= 0.0
    # Empty inputs -> zero/default summary
    summ_empty = eng.aggregate_summary()
    assert summ_empty.bloom_uncertainty == 0.0
    assert summ_empty.confidence == 1.0
    assert summ_empty.n_samples == 0

    # ------------------------------------------------------------------ #
    # 7. CalibrationReport wrapper
    # ------------------------------------------------------------------ #
    rep = eng.calibration_report(confs, accs)
    assert isinstance(rep, CalibrationReport)
    assert rep.ece == ece_calib
    assert rep.bin_centers.shape == (10,)
    assert int(rep.bin_counts.sum()) == 1000

    # ------------------------------------------------------------------ #
    # 8. Bad-input handling
    # ------------------------------------------------------------------ #
    for kwargs in ({"K": 1}, {"n_bins": 1}, {"eps": 0.0}, {"eps": -1.0}):
        try:
            UncertaintyEngine(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for kwargs={kwargs}")

    bad_pairs = [
        ([0.5, 1.5], [1, 0]),   # confidence > 1
        ([0.5, -0.1], [1, 0]),  # confidence < 0
        ([0.5, 0.5], [1, 0, 1]),  # length mismatch
        ([0.5, 0.5], [1, 0.5]),   # non-binary accuracy
        ([], []),                 # empty
    ]
    for c_arg, a_arg in bad_pairs:
        try:
            eng.compute_ece(c_arg, a_arg)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for ECE inputs {(c_arg, a_arg)}")

    try:
        eng.run_stochastic_forward(dummy_model, "x", n=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for n=0")
    try:
        eng.run_stochastic_forward(dummy_model, "x", n=3, temperature=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for temperature=0")
    try:
        eng.run_stochastic_forward("not_callable", "x")  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for non-callable model_fn")

    logger.info(
        "SPU(p,q,p,q,p)=%.4f  bloom_unc(p)=%.4f  ECE_calib=%.2e  ECE_over=%.3f",
        val, eng.compute_bloom_uncertainty(p), ece_calib, ece_over,
    )

    _ok("UncertaintyEngine sanity check passed")


if __name__ == "__main__":
    _self_test()
