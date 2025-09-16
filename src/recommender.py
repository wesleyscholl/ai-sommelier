import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from typing import List, Optional, Tuple
import os
import json
import warnings

MODEL_NAME = "all-MiniLM-L6-v2"


class Recommender:
    """
    Production-optimized content-based recommender using sentence-transformers + NearestNeighbors.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._initialize_model()
        self.embeddings = None
        self.df = None
        self.nn = None

    def _initialize_model(self):
        """Initialize SentenceTransformer with robust error handling."""
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            warnings.warn(f"Standard model initialization failed: {str(e)}. Trying alternatives...")
            self._initialize_model_fallback()

    def _initialize_model_fallback(self):
        """Fallback model initialization for deployment environments."""
        import ssl
        
        try:
            # Disable SSL verification for deployment environments
            ssl._create_default_https_context = ssl._create_unverified_context
            os.environ['CURL_CA_BUNDLE'] = ''
            os.environ['REQUESTS_CA_BUNDLE'] = ''
            
            self.model = SentenceTransformer(self.model_name)
        except Exception as ssl_error:
            # Final attempt with explicit cache directory
            cache_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_cache"))
            os.makedirs(cache_dir, exist_ok=True)
            
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=cache_dir,
            )

    def fit(self, df: pd.DataFrame, text_column: str = "text_for_embedding", batch_size: int = 16, progress_callback=None):
        """
        Compute embeddings with production-optimized processing.
        
        Args:
            df: DataFrame with text to embed
            text_column: Column name containing text to embed
            batch_size: Number of samples to process at once (optimized for Streamlit Cloud)
            progress_callback: Optional callback function(progress_fraction, message)
        """
        self.df = df.reset_index(drop=True).copy()
        texts = self.df[text_column].astype(str).tolist()
        total_samples = len(texts)
        
        if total_samples == 0:
            raise ValueError("Dataset is empty")
        
        # Get embedding dimension efficiently
        try:
            embedding_dim = int(getattr(self.model, "get_sentence_embedding_dimension")())
        except Exception:
            # Fallback to single sample encoding
            sample_emb = self.model.encode(texts[0], convert_to_numpy=True).reshape(1, -1)
            embedding_dim = sample_emb.shape[1]

        # Pre-allocate array for memory efficiency
        all_embeddings = np.zeros((total_samples, embedding_dim), dtype=np.float32)
        
        # Process in optimized batches
        processed = 0
        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            batch_texts = texts[start_idx:end_idx]
            
            if progress_callback:
                progress = 0.1 + 0.7 * (processed / total_samples)
                progress_callback(progress, f"Processing {processed + len(batch_texts)}/{total_samples} wines")
            
            try:
                batch_embeddings = self.model.encode(
                    batch_texts, 
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    device='cpu'  # Force CPU for deployment stability
                )
                all_embeddings[start_idx:end_idx] = batch_embeddings
                processed += len(batch_texts)
                
            except Exception as e:
                # Graceful degradation: process individually
                for i, text in enumerate(batch_texts):
                    try:
                        single_emb = self.model.encode([text], convert_to_numpy=True, device='cpu')
                        all_embeddings[start_idx + i] = single_emb[0]
                        processed += 1
                    except Exception:
                        # Skip problematic samples with zero embedding
                        warnings.warn(f"Skipping problematic sample at index {start_idx + i}")
        
        self.embeddings = all_embeddings
        
        # Build efficient search index
        if progress_callback:
            progress_callback(0.9, "Building search index...")
        
        n_neighbors = min(100, len(self.df))  # Much larger for full dataset utilization
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="auto")
        self.nn.fit(self.embeddings)
        
        if progress_callback:
            progress_callback(1.0, "✅ Ready!")

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
        Return top_k recommended wines with enhanced filtering and ranking.
        """
        if self.embeddings is None or self.df is None:
            raise RuntimeError("Recommender not fitted. Call fit(df) first.")

        # Encode query
        q_emb = self.model.encode([query], convert_to_numpy=True, device='cpu')
        
        # Get many more candidates for full dataset utilization
        n_candidates = min(1000, len(self.df))  # Much larger candidate pool
        distances, indices = self.nn.kneighbors(q_emb, n_neighbors=n_candidates)
        indices = indices[0]
        distances = distances[0]
        
        # Convert to similarities (cosine distance -> similarity)
        similarities = 1.0 - distances

        # Create result DataFrame
        res = self.df.iloc[indices].copy()
        res["_similarity"] = similarities
        
        # Apply filters with robust handling
        if price_min is not None:
            res = res[res["price"].notna() & (res["price"] >= price_min)]
        if price_max is not None:
            res = res[res["price"].notna() & (res["price"] <= price_max)]
        if variety and len(variety) > 0:
            # More flexible variety matching - check if any variety pattern matches
            variety_patterns = [v.strip().lower() for v in variety if v.strip()]
            if variety_patterns:  # Only filter if we have actual variety patterns
                mask = res["variety"].fillna("").str.lower().apply(
                    lambda x: any(pattern in x or x in pattern for pattern in variety_patterns)
                )
                res = res[mask]

        # Sort by similarity and return top results
        res = res.sort_values("_similarity", ascending=False).head(top_k)
        return res.reset_index(drop=True)

    def save_embeddings(self, path: str):
        """
        Save embeddings and metadata to npz for faster reload.
        """
        if self.embeddings is None or self.df is None:
            raise RuntimeError("No embeddings to save. Call fit() first.")
        # Save model name too so loaders can detect mismatches
        np.savez_compressed(
            path,
            embeddings=self.embeddings,
            texts=self.df["text_for_embedding"].astype(str).values,
            metadata=self.df.to_json(orient="records"),
            model_name=self.model_name,
        )

    def load_embeddings(self, path: str):
        """
        Load previously saved embeddings (npz) and rebuild NearestNeighbors.
        """
        data = np.load(path, allow_pickle=True)
        embeddings = data["embeddings"]

        # metadata may be saved as a string/bytes/ndarray; handle common cases
        raw_meta = data.get("metadata")
        df = None
        if raw_meta is not None:
            try:
                if isinstance(raw_meta, np.ndarray):
                    meta_val = raw_meta.item()
                else:
                    meta_val = raw_meta
                # if bytes, decode
                if isinstance(meta_val, (bytes, bytearray)):
                    meta_val = meta_val.decode("utf-8")
                df = pd.read_json(meta_val, orient="records")
            except Exception:
                # fallback: try to reconstruct minimal dataframe from texts
                try:
                    texts = data.get("texts")
                    if texts is not None:
                        df = pd.DataFrame({"text_for_embedding": texts.tolist()})
                except Exception:
                    df = pd.DataFrame({"text_for_embedding": []})
        else:
            df = pd.DataFrame({"text_for_embedding": data.get("texts", []).tolist()})

        # Attach loaded data
        self.embeddings = embeddings
        self.df = df.reset_index(drop=True)

        # Rebuild nearest neighbors (respect dataset size)
        n_neighbors = min(10, len(self.df)) if len(self.df) > 0 else 1
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="auto")
        self.nn.fit(self.embeddings)
