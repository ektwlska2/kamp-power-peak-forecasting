"""Data loading, timestamp reconstruction and copied-day detection.

Minimal-cleaning policy: raw values are preserved (zeros kept). Only the
hour index is rebuilt from row order, because 48 rows on 2021-07-13/15 carry
invalid `시간` labels while every date still has exactly 24 rows.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "okm_augumented_2021.csv"

Q_COLS = ["15분", "30분", "45분", "60분"]

RENAME = {
    "날짜": "date_raw", "시간": "hour_raw", "15분": "q1", "30분": "q2", "45분": "q3",
    "60분": "q4", "평균": "mean_load", "생산량": "production", "기온": "temp",
    "풍속": "wind", "습도": "humidity", "강수량": "rain", "전기요금(계절)": "tariff",
    "day": "dow", "d": "dom", "m": "month", "공장인원": "staff", "인건비": "labor_rate",
}
Q = ["q1", "q2", "q3", "q4"]


def load_raw(path=RAW_CSV) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def load(path=RAW_CSV) -> pd.DataFrame:
    """Return the dataset with a reconstructed hourly timestamp.

    Adds: `hour` (row order within date), `ts`, `hour_label_invalid` flag.
    """
    raw = load_raw(path)
    df = raw.rename(columns=RENAME)
    sizes = df.groupby("date_raw").size()
    if not (sizes == 24).all():
        raise ValueError("expected 24 rows per date for order-based hour reconstruction")
    df["hour"] = df.groupby("date_raw").cumcount()
    df["hour_label_invalid"] = df["hour_raw"] != df["hour"]
    df["date"] = pd.to_datetime(df["date_raw"].astype(str), format="%Y%m%d")
    df["ts"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")
    df = df.sort_values("ts").reset_index(drop=True)
    df["peak_q"] = df[Q].max(axis=1)
    return df


def day_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """One row per date: the 96 quarter readings as a tuple key."""
    return df.groupby("date")[Q].apply(lambda x: tuple(x.to_numpy().ravel()))


def copy_map(df: pd.DataFrame) -> pd.DataFrame:
    """Classify each date by whether its full 96-reading electricity profile
    is shared with another date.

    status: 'unique'  – profile appears once
            'first'   – earliest date of a repeated profile (source unknown)
            'copy'    – later date repeating an earlier profile
    """
    key = day_profiles(df)
    gid = pd.Series(pd.factorize(key)[0], index=key.index, name="group")
    size = gid.map(gid.value_counts())
    first = ~key.duplicated()
    status = np.where(size == 1, "unique", np.where(first, "first", "copy"))
    prod_key = df.groupby("date")["production"].apply(tuple)
    prod_same = gid.map(prod_key.groupby(gid).nunique() == 1)
    out = pd.DataFrame({"group": gid, "group_size": size, "status": status,
                        "production_also_copied": prod_same & (size > 1)})
    # exact copy of the date exactly 7 days earlier (weekly-lag contamination)
    for lag in (1, 7):
        prev = key.reindex(key.index - pd.Timedelta(days=lag))
        out[f"equals_minus{lag}"] = [isinstance(p, tuple) and p == k
                                     for p, k in zip(prev, key)]
    return out


def with_copy_status(df: pd.DataFrame) -> pd.DataFrame:
    cm = copy_map(df)
    df = df.copy()
    df["copy_status"] = df["date"].map(cm["status"])
    df["is_copy_day"] = df["copy_status"] != "unique"
    df["equals_minus7"] = df["date"].map(cm["equals_minus7"])
    return df
