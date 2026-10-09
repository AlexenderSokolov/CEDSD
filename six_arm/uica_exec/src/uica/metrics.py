"""Metrics with explicit score direction and undefined single-class results."""
from __future__ import annotations

import numpy as np


def _arrays(labels, scores):
    # Validate before casting: values such as 0.4 must not silently become real.
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores) or not len(labels):
        raise ValueError("Expected one finite prediction for every nonempty label record")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Labels must be binary and all predictions finite")
    return labels.astype(np.int64), scores


def _roc(labels, scores):
    labels, scores = _arrays(labels, scores)
    if len(np.unique(labels)) < 2:
        return None
    order = np.argsort(-scores, kind="stable")
    y, s = labels[order], scores[order]
    end = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    true_positive = np.r_[0, np.cumsum(y)[end]]
    false_positive = np.r_[0, (end + 1) - np.cumsum(y)[end]]
    thresholds = np.r_[np.nextafter(s[0], np.inf), s[end]]
    fpr = false_positive / np.sum(labels == 0)
    fnr = 1 - true_positive / np.sum(labels == 1)
    difference = fpr - fnr
    index = int(np.searchsorted(difference, 0, side="left"))
    if difference[index] == 0:
        eer = float(fpr[index])
    else:
        fraction = -difference[index - 1] / (difference[index] - difference[index - 1])
        eer = float(fpr[index - 1] + fraction * (fpr[index] - fpr[index - 1]))
    integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    auc = float(integrate(1 - fnr, fpr))
    return fpr, fnr, thresholds, eer, auc


def choose_threshold(labels, scores):
    roc = _roc(labels, scores)
    if roc is None:
        raise ValueError("A working threshold requires both classes in validation")
    fpr, fnr, thresholds, _, _ = roc
    valid = np.flatnonzero(np.isfinite(thresholds))
    index = min(valid, key=lambda i: (abs(fpr[i] - fnr[i]), fpr[i] + fnr[i], -thresholds[i]))
    return float(thresholds[index])


def binary_metrics(labels, scores, threshold):
    labels, scores = _arrays(labels, scores)
    if not np.isfinite(threshold):
        raise ValueError("Working threshold must be finite and fixed")
    predicted = scores >= threshold
    real, fake = labels == 0, labels == 1
    fpr = float(predicted[real].mean()) if real.any() else None
    fnr = float((~predicted[fake]).mean()) if fake.any() else None
    roc = _roc(labels, scores)
    return {"n": len(labels), "n_real": int(real.sum()), "n_spoof": int(fake.sum()),
            "eer": roc[3] if roc is not None else None, "auc": roc[4] if roc is not None else None,
            "threshold": float(threshold), "fpr": fpr, "fnr": fnr,
            "tnr": 1 - fpr if fpr is not None else None, "tpr": 1 - fnr if fnr is not None else None,
            "accuracy": float((predicted == labels).mean()), "score_mean": float(scores.mean()),
            "score_std": float(scores.std()), "score_direction": "higher_is_spoof"}


def weighted_ce_sum(logits, labels, weights):
    import torch.nn.functional as functional
    return functional.cross_entropy(logits, labels, weight=weights, reduction="sum")


def paired_bootstrap_eer(labels, candidate_scores, baseline_scores, groups, n_bootstrap=1000, seed=1701):
    labels, candidate = _arrays(labels, candidate_scores)
    _, baseline = _arrays(labels, baseline_scores)
    groups = np.asarray(groups)
    if len(groups) != len(labels) or any(g is None or g == "" for g in groups):
        raise ValueError("Bootstrap requires a known group for each paired sample")
    a, b = _roc(labels, candidate), _roc(labels, baseline)
    if a is None or b is None:
        return {"delta": None, "ci95": None, "valid_replicates": 0, "reason": "single_class"}
    identities = sorted(set(groups.tolist()))
    indices = {g: np.flatnonzero(groups == g) for g in identities}
    rng = np.random.default_rng(seed)
    differences = []
    attempts = 0
    while len(differences) < n_bootstrap and attempts < n_bootstrap * 10:
        attempts += 1
        selected = np.concatenate([indices[identities[i]] for i in rng.integers(len(identities), size=len(identities))])
        a_rep, b_rep = _roc(labels[selected], candidate[selected]), _roc(labels[selected], baseline[selected])
        if a_rep is not None and b_rep is not None:
            differences.append(a_rep[3] - b_rep[3])
    return {"delta": a[3] - b[3], "ci95": np.quantile(differences, [0.025, 0.975]).tolist() if differences else None,
            "valid_replicates": len(differences), "requested_replicates": n_bootstrap,
            "groups": len(identities), "seed": seed, "resampling_unit": "speaker",
            "delta_direction": "candidate_minus_baseline"}
