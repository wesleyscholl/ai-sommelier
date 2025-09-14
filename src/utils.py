import pandas as pd
import numpy as np
from typing import Optional


def load_wine_dataset(path: str) -> pd.DataFrame:
    """
    Loads a wine dataset CSV and does light cleaning.
    Expects columns: title(or wine), variety, country (or region), description, price.
    """
    df = pd.read_csv(path)
    # standardize some column names if present
    for col in ("title", "wine", "name"):
        if col in df.columns:
            df = df.rename(columns={col: "title"})
            break
    for col in ("country", "region", "region_1"):
        if col in df.columns:
            df = df.rename(columns={col: "country"})
            break

    # Ensure required columns exist; fill missing with placeholders
    if "description" not in df.columns:
        df["description"] = df.get("title", "").astype(str)
    if "price" not in df.columns:
        df["price"] = np.nan
    if "variety" not in df.columns:
        df["variety"] = "Unknown"

    # simple cleaning: drop rows with no description
    df = df[df["description"].notna()].copy()
    # ensure price numeric
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # create a text field used for embedding/search
    df["text_for_embedding"] = (
        df["title"].fillna("")
        + " | "
        + df["variety"].fillna("")
        + " | "
        + df["country"].fillna("")
        + " | "
        + df["description"].fillna("")
    )

    df = df.reset_index(drop=True)
    return df


def top_k_indices(similarities: np.ndarray, k: int):
    """
    Given an array of similarities, return top-k indices (descending)
    """
    if k >= len(similarities):
        return np.argsort(-similarities)
    return np.argpartition(-similarities, k)[:k]
