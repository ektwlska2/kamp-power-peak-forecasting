"""Training weights for repeated-profile (copied) days.

Rule-based weight chosen by the pre-declared protocol (exp13): every distinct
electricity profile counts as one day. A repeated-profile day gets weight
1 / (number of its group's dates inside the training window); unique days 1.
"""
import pandas as pd

from .data import copy_map


def copy_weights(df: pd.DataFrame, train_mask: pd.Series) -> pd.Series:
    cm = copy_map(df)
    grp = df["date"].map(cm["group"])
    tr_dates = df.loc[train_mask, "date"].drop_duplicates()
    n_in_train = tr_dates.map(cm["group"]).value_counts()
    return (1.0 / grp.map(n_in_train).fillna(1)).clip(upper=1.0).rename("weight")
