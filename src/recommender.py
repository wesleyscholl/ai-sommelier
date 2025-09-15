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
        
        # Handle potential SSL certificate issues (common in Streamlit Cloud)
        try:
            # Try normal initialization first
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            # If normal initialization fails, try with SSL workarounds
            import ssl
            import warnings
            
            warnings.warn(f"Model initialization failed with error: {str(e)}. Trying SSL workaround.")
            
            # Disable SSL verification - only use in trusted environments like Streamlit Cloud
            try:
                # Create unverified SSL context
                ssl._create_default_https_context = ssl._create_unverified_context
                
                # Clear potential SSL environment variables that might interfere
                os.environ['CURL_CA_BUNDLE'] = ''
                os.environ['REQUESTS_CA_BUNDLE'] = ''
                
                # Try again with SSL verification disabled
                self.model = SentenceTransformer(model_name)
            except Exception as ssl_error:
                # If that also fails, try with explicit cache directory
                warnings.warn(f"SSL workaround failed with: {str(ssl_error)}. Trying with explicit cache.")
                
                # Create cache directory in the current project
                cache_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_cache"))
                os.makedirs(cache_dir, exist_ok=True)
                
                # Final attempt with cache specification and no download if not present
                self.model = SentenceTransformer(
                    model_name,
                    cache_folder=cache_dir,
                )
                
        self.embeddings = None  # (N, D)
        self.df = None
        self.nn = None

    def fit(self, df: pd.DataFrame, text_column: str = "text_for_embedding", batch_size: int = 32, progress_callback=None):
        """
        Compute embeddings for the text column and fit a nearest neighbors index.
        
        Args:
            df: DataFrame with text to embed
            text_column: Column name containing text to embed
            batch_size: Number of samples to process at once (lower=less memory)
            progress_callback: Optional callback function(progress_fraction, message) to report progress
        """
        # Clean data and prepare for processing
        self.df = df.reset_index(drop=True).copy()
        texts = self.df[text_column].astype(str).tolist()
        total_samples = len(texts)
        
        # Pre-allocate embedding array (saves memory vs. list appending)
        # Get embedding dimension from a sample
        sample_emb = self.model.encode(texts[0], convert_to_numpy=True).reshape(1, -1)
        embedding_dim = sample_emb.shape[1]
        all_embeddings = np.zeros((total_samples, embedding_dim), dtype=np.float32)
        
        # Process in batches to manage memory
        start_idx = 0
        while start_idx < total_samples:
            if progress_callback:
                progress = start_idx / total_samples
                progress_callback(progress, f"Processing batch {start_idx//batch_size + 1}/{(total_samples-1)//batch_size + 1}")
            
            # Get current batch
            end_idx = min(start_idx + batch_size, total_samples)
            batch_texts = texts[start_idx:end_idx]
            
            # Encode batch
            try:
                batch_embeddings = self.model.encode(
                    batch_texts, 
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                
                # Store in the pre-allocated array
                all_embeddings[start_idx:end_idx] = batch_embeddings
                
            except Exception as e:
                if progress_callback:
                    progress_callback(start_idx / total_samples, f"Error in batch: {str(e)}. Retrying...")
                # On error, try one by one to identify problematic samples
                for i, text in enumerate(batch_texts):
                    try:
                        single_emb = self.model.encode([text], convert_to_numpy=True)
                        all_embeddings[start_idx + i] = single_emb[0]
                    except Exception:
                        # If a single sample fails, use zero embedding
                        if progress_callback:
                            progress_callback(0, f"Warning: Skipping problematic sample at index {start_idx + i}")
            
            # Move to next batch
            start_idx = end_idx
        
        # Store final embeddings
        self.embeddings = all_embeddings
        
        # Fit NearestNeighbors with cosine metric
        if progress_callback:
            progress_callback(1.0, "Building nearest neighbors index...")
        self.nn = NearestNeighbors(n_neighbors=min(10, len(self.df)), metric="cosine", algorithm="auto")
        self.nn.fit(self.embeddings)
        
        if progress_callback:
            progress_callback(1.0, "Embedding computation complete!")

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
