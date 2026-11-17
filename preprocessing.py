"""Reusable cleaning step. Everything is LEARNED on train only, then applied to train and test."""
import numpy as np
import pandas as pd

MISSING = "?"


class Cleaner:
    """Clip numerics to train percentiles, impute (median / mode), treat '?' as missing."""

    def __init__(self, num_cols, cat_cols, low=1, high=99):
        self.num_cols, self.cat_cols = list(num_cols), list(cat_cols)
        self.low, self.high = low, high

    @staticmethod
    def _cat(s):
        s = s.astype(object)
        return s.where(s.ne(MISSING), np.nan)          # '?' -> NaN

    def fit(self, train):
        num = train[self.num_cols].apply(pd.to_numeric, errors="coerce")
        self.low_ = num.quantile(self.low / 100)
        self.high_ = num.quantile(self.high / 100)
        self.median_ = num.median()
        self.mode_ = {c: self._cat(train[c]).mode(dropna=True).iloc[0] for c in self.cat_cols}
        return self

    def transform(self, df):
        out = df.copy()
        num = out[self.num_cols].apply(pd.to_numeric, errors="coerce")
        out[self.num_cols] = num.clip(self.low_, self.high_, axis=1).fillna(self.median_)
        for c in self.cat_cols:
            out[c] = self._cat(out[c]).fillna(self.mode_[c])
        return out


def add_features(df):
    out = df.copy()
    out["capital_net"] = out["capital-gain"] - out["capital-loss"]
    out["hours_per_age"] = out["hours-per-week"] / np.maximum(out["age"] - 16, 1)
    return out
