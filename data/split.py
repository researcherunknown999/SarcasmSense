from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split_if_missing(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> pd.DataFrame:
    """
    If df['split'] is missing/empty, create train/val/test using stratification
    on the combined label string y_sarc-y_shift-y_target.
    """
    df = df.copy()
    if "split" in df.columns and (df["split"].astype(str).str.strip() != "").any():
        # If at least some are set, assume user provided splits
        # (You can enforce all set later if desired.)
        return df

    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1.0")

    strat = (
        df["y_sarc"].astype(int).astype(str)
        + "-"
        + df["y_shift"].astype(int).astype(str)
        + "-"
        + df["y_target"].astype(int).astype(str)
    )

    df_train, df_tmp = train_test_split(
        df, test_size=(1.0 - train_frac), random_state=seed, stratify=strat
    )

    tmp_frac = 1.0 - train_frac
    test_rel = test_frac / tmp_frac  # fraction of tmp assigned to test

    strat_tmp = (
        df_tmp["y_sarc"].astype(int).astype(str)
        + "-"
        + df_tmp["y_shift"].astype(int).astype(str)
        + "-"
        + df_tmp["y_target"].astype(int).astype(str)
    )

    df_val, df_test = train_test_split(
        df_tmp, test_size=test_rel, random_state=seed, stratify=strat_tmp
    )

    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"
    return pd.concat([df_train, df_val, df_test], axis=0).reset_index(drop=True)
