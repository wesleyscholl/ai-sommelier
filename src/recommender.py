import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from typing import List, Optional, Tuple
import os
import json
from tqdm import tqdm

MODEL_NAME = "all-MiniLM-L6-v2"


class Recommender:
    """
    Simple content-based recommender using sentence-transformers embeddings + NearestNeighbors (cosine).
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embeddings = None  # (N, D)
        self.df = None
        self.nn = None

    def fit(self, df: pd.DataFrame, text_column: str = "text_for_embedding"):
        """
        Compute embeddings for the text column and fit a nearest neighbors index.
        """
        self.df = df.reset_index(drop=True).copy()
        texts = self.df[text_column].astype(str).tolist()
        # compute embeddings in batches to be memory-friendly
        emb = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        self.embeddings = emb
        # fit NearestNeighbors with cosine metric (scikit expects metric='cosine' which computes 1 - cosine_similarity)
        self.nn = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="auto")
        self.nn.fit(self.embeddings)

    def recommend(
        self,
        query: str,
        top_k: int = 5,
        price_min: Optional[float] = None,
        price_max: Optional[float] = None,
        variety: Optional[List[str]] = None,
        text_column: str = "text_for_embedding",
    ) -> pd.DataFrame:
        """
        Return top_k recommended wines for a textual query and simple filters.
        """
        if self.embeddings is None or self.df is None:
            raise RuntimeError("Recommender not fitted. Call fit(df) first.")

        q_emb = self.model.encode([query], convert_to_numpy=True)
        # NearestNeighbors returns distances where smaller = closer (cosine distance)
        distances, indices = self.nn.kneighbors(q_emb, n_neighbors=min(100, len(self.df)))
        indices = indices[0]
        distances = distances[0]
        # convert distances -> similarity
        similarities = 1.0 - distances

        # create result df
        res = self.df.iloc[indices].copy()
        res["_similarity"] = similarities
        # Apply filters
        if price_min is not None:
            res = res[res["price"].notna() & (res["price"] >= price_min)]
        if price_max is not None:
            res = res[res["price"].notna() & (res["price"] <= price_max)]
        if variety:
            variety_lower = {v.lower() for v in variety}
            res = res[res["variety"].fillna("").str.lower().isin(variety_lower)]

        # Keep top_k by similarity after filtering
        res = res.sort_values("_similarity", ascending=False).head(top_k)
        # include id/original index
        res = res.reset_index(drop=True)
        return res

    def save_embeddings(self, path: str):
        """
        Save embeddings and metadata to npz for faster reload.
        """
        if self.embeddings is None or self.df is None:
            raise RuntimeError("No embeddings to save. Call fit() first.")
        np.savez_compressed(
            path,
            embeddings=self.embeddings,
            texts=self.df["text_for_embedding"].astype(str).values,
            metadata=self.df.to_json(orient="records"),
        )

    def load_embeddings(self, path: str):
        """
        Load previously saved embeddings (npz) and rebuild NearestNeighbors.
        """
        data = np.load(path, allow_pickle=True)
        embeddings = data["embeddings"]
        metadata_json = data["metadata"].item() if isinstance(data["metadata"], np.ndarray) else data["metadata"]
        # metadata_json may be a JSON string or bytes
        try:
            df = pd.read_json(metadata_json, orient="records")
        except Exception:
            # fallback: load from texts only
            df = pd.DataFrame({"text_for_embedding": data["texts"].tolist()})
        self.embeddings = embeddings
        self.df = df
        self.nn = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="auto")
        self.nn.fit(self.embeddings)
