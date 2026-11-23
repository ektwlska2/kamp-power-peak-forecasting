import numpy as np
import pandas as pd

PEAK = 190  # analysis threshold in recorded units (not a verified contract limit)


def score(y: pd.DataFrame, p: pd.DataFrame, peak=PEAK) -> dict:
    """y, p: (hours x 4 quarters) aligned frames."""
    a, f = y.to_numpy(float), p.to_numpy(float)
    e = f - a
    pk = a >= peak
    return {
        "MAE": np.abs(e).mean(),
        "RMSE": np.sqrt((e ** 2).mean()),
        "bias": e.mean(),
        "peak_MAE": np.abs(e[pk]).mean() if pk.any() else np.nan,
        "peak_shortfall": np.clip(-e[pk], 0, None).mean() if pk.any() else np.nan,
        "n_hours": len(a),
    }


def peak_hour_f1(y: pd.DataFrame, p: pd.DataFrame, peak=PEAK, cutoff=None) -> dict:
    """Hour-level peak detection: actual hour is a peak if any quarter >= peak;
    flagged if forecast hourly max >= cutoff (default: peak)."""
    cutoff = peak if cutoff is None else cutoff
    act = (y.max(axis=1) >= peak).to_numpy()
    flag = (p.max(axis=1) >= cutoff).to_numpy()
    tp, fp, fn = (act & flag).sum(), (~act & flag).sum(), (act & ~flag).sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"peak_hours": int(act.sum()), "TP": int(tp), "FP": int(fp), "FN": int(fn),
            "precision": prec, "recall": rec, "F1": f1}
