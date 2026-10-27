"""
Measures: WMAPE, R2, zero detection, prediction intervals and pair accuracy.
"""

import numpy as np

from .definitions import ACTIVE_THRESHOLD, ALPHA, EPS


def wmape(t, p):
    s = np.abs(t).sum()
    return float(np.abs(p - t).sum() / s) if s > EPS else np.nan


def r2(t, p):
    ss = ((t - t.mean()) ** 2).sum()
    return float(1 - ((t - p) ** 2).sum() / ss) if ss > EPS else np.nan


def balanced_accuracy(t, p):
    if t.min() == t.max():
        return np.nan
    return float(0.5 * ((p[t == 1] == 1).mean() + (p[t == 0] == 0).mean()))


def conformal_q(errors, alpha=ALPHA):
    """Split-conformal quantile of the calibration errors, with the finite-sample correction."""
    n = len(errors)
    if n == 0:
        return 0.0
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(errors, level, method="higher"))


def interval(pred, q):
    """Prediction interval on the log scale: [(1+p)e^-q - 1, (1+p)e^q - 1], not below 0."""
    pred, q = np.asarray(pred, float), np.asarray(q, float)
    return np.maximum(np.expm1(np.log1p(pred) - q), 0.0), np.expm1(np.log1p(pred) + q)


def mondrian_q(true_cal, pred_cal, group, decision, position, n_groups, calibration_sizes, alpha=ALPHA):
    """Conformal q per technology group AND per zero / non-zero decision, from the first n calibration runs
    (position = rank of the run in the fixed calibration order). Errors on the log scale.
    Returns {n: array (groups x 2)}; a decision class without calibration rows gets the q of the whole group."""
    err = np.abs(np.log1p(true_cal) - np.log1p(pred_cal))
    q = {}
    for n in calibration_sizes:
        qn = np.zeros((n_groups, 2))
        for g in range(n_groups):
            in_g = (group == g) & (position < n)
            fallback = conformal_q(err[in_g], alpha)
            for k in (0, 1):
                rows = in_g & (decision == k)
                qn[g, k] = conformal_q(err[rows], alpha) if rows.any() else fallback
        q[n] = qn
    return q


def interval_measures(t, lo, hi, suffix):
    """Coverage and relative width of the intervals: on all rows, on the non-zero rows (used in the paper;
    width / 2 = the +/- half-width), and the mean width on zero rows in the output's units."""
    nz = t > ACTIVE_THRESHOLD
    inside = (t >= lo - EPS) & (t <= hi + EPS)
    width = hi - lo
    return {f"coverage_{suffix}": float(inside.mean()),
            f"width_{suffix}": float(width.sum() / max(np.abs(t).sum(), EPS)),
            f"coverage_nonzero_{suffix}": float(inside[nz].mean()) if nz.any() else np.nan,
            f"width_nonzero_{suffix}": float(width[nz].sum() / np.abs(t[nz]).sum()) if nz.any() else np.nan,
            f"width_zero_{suffix}": float(width[~nz].mean()) if (~nz).any() else np.nan}


def test_measures(true, pred, decision, group, groups, bounds, extra):
    """One row per technology group and one for ALL groups: WMAPE, R2, MAE, zero detection and the
    interval measures for every calibration size in bounds ({n: (lo, hi)})."""
    rows = []
    for g in list(range(len(groups))) + ["ALL"]:
        m = np.ones(len(true), bool) if g == "ALL" else group == g
        t, p = true[m], pred[m]
        nz = t > ACTIVE_THRESHOLD
        row = {**extra, "group": "ALL" if g == "ALL" else str(groups[g]), "rows": int(m.sum()),
               "share_nonzero": float(nz.mean()), "wmape": wmape(t, p), "r2": r2(t, p),
               "mae": float(np.abs(p - t).mean()), "zero_bal_acc": balanced_accuracy(nz.astype(np.uint8), decision[m])}
        for n, (lo, hi) in bounds.items():
            row.update(interval_measures(t, lo[m], hi[m], f"cal{n}"))
        rows.append(row)
    return rows


def clear_pair_accuracy(t, p, tol):
    """Share of pairs with a true difference > tol that the prediction orders correctly (a tie in the
    prediction counts as half), and the share of such pairs among all pairs."""
    dt = t[:, None] - t[None, :]
    keep = dt > tol                                   # each clearly different pair once
    n_pairs = len(t) * (len(t) - 1) / 2
    n = int(keep.sum())
    if n == 0:
        return np.nan, 0.0
    dp = (p[:, None] - p[None, :])[keep]
    return float(((dp > 0).sum() + 0.5 * (dp == 0).sum()) / n), n / n_pairs


def weighted_median(values, weights):
    """Value at which half of the total weight is passed (the mean of the two neighbours if exactly half,
    so that with equal weights it is the ordinary median)."""
    order = np.argsort(values, kind="stable")
    v, w = np.asarray(values, float)[order], np.asarray(weights, float)[order]
    cum, half = np.cumsum(w), w.sum() / 2
    i = int(np.searchsorted(cum, half - 1e-12 * w.sum()))
    if np.isclose(cum[i], half) and i + 1 < len(v):
        return float((v[i] + v[i + 1]) / 2)
    return float(v[i])
