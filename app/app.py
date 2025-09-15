import streamlit as st
import os
import sys

# Add the project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_wine_dataset
from src.recommender import Recommender
from src.sommelier import Sommelier
import tempfile
import time
import numpy as np

# Define a fallback optimized recommender in case the updated src/recommender.py isn't loaded
class OptimizedRecommender(Recommender):
    """
    Fallback implementation with optimized batch processing.
    Only used if the main Recommender class hasn't been updated.
    """
    
    def fit(self, df, text_column="text_for_embedding", batch_size=32, progress_callback=None):
        """
        Optimized version with batch processing and progress tracking.
        """
        # Basic implementation - no batching but with progress updates
        self.df = df.reset_index(drop=True).copy()
        texts = self.df[text_column].astype(str).tolist()
        
        if progress_callback:
            progress_callback(0.3, "Computing embeddings...")
            
        # Use the standard encode method but with progress bar
        self.embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        if progress_callback:
            progress_callback(0.9, "Building nearest neighbors index...")
            
        # Fit NearestNeighbors
        self.nn = NearestNeighbors(n_neighbors=min(10, len(self.df)), metric="cosine", algorithm="auto")
        self.nn.fit(self.embeddings)
        
        if progress_callback:
            progress_callback(1.0, "Embedding computation complete!")

st.set_page_config(page_title="AI Wine Sommelier", layout="centered")

st.title("🤵🏻‍♂️🍷 AI Wine Sommelier")

data_path = st.sidebar.text_input("Path to wine CSV", value="data/wine_reviews.csv")
embed_file = st.sidebar.text_input("Optional embeddings file (.npz)", value="data/embeddings.npz")

# Add sampling option for faster testing
st.sidebar.header("Performance Options")
use_sample = st.sidebar.checkbox("Use data sample (faster)", value=False)
if use_sample:
    sample_size = st.sidebar.slider("Sample size", min_value=100, max_value=5000, value=1000, step=100)

# Add batch size option for memory management
batch_size = st.sidebar.slider("Batch size", min_value=8, max_value=128, value=32, step=8, 
                              help="Lower values use less memory but process slower")

if not os.path.exists(data_path):
    st.sidebar.warning("Data file not found. Place a CSV at data/wine_reviews.csv or change the path.")
else:
    with st.spinner("Loading dataset..."):
        df = load_wine_dataset(data_path)
        # Apply sampling if enabled
        if use_sample and len(df) > sample_size:
            df = df.sample(sample_size, random_state=42)
    st.sidebar.success(f"Loaded {len(df)} rows")

# Build or load recommender
@st.cache_resource(show_spinner=False)
def build_recommender(data_path: str, embed_file: str = None, batch_size: int = 32):
    """
    Build or load a recommender model with optimized performance.
    
    Args:
        data_path: Path to the wine dataset CSV
        embed_file: Optional path to pre-computed embeddings
        batch_size: Batch size for embedding computation (lower = less memory)
    """
    # Create UI elements for progress tracking
    status_container = st.sidebar.container()
    progress_bar = status_container.progress(0)
    status_text = status_container.empty()
    
    try:
        # Define progress callback function
        def update_progress(fraction, message):
            # Update progress bar and message
            try:
                progress_bar.progress(fraction)
                status_text.text(message)
            except Exception:
                # Failsafe if progress bar was removed
                pass
        
        # Load dataset
        update_progress(0.1, "Loading dataset...")
        df_local = load_wine_dataset(data_path)
        update_progress(0.2, f"Dataset loaded: {len(df_local)} wines")
        
        # Try to initialize recommender with optimized version first
        try:
            # Try to create an instance of our recommender with the batch_size parameter
            test_recommender = Recommender()
            test_fit_method = getattr(test_recommender, 'fit')
            import inspect
            fit_params = inspect.signature(test_fit_method).parameters
            
            if 'batch_size' in fit_params and 'progress_callback' in fit_params:
                # Good! The updated version is available
                rec = test_recommender
                update_progress(0.2, "Using optimized recommender...")
            else:
                # Fall back to our local implementation
                rec = OptimizedRecommender()
                update_progress(0.2, "Using fallback optimized recommender...")
        except Exception:
            # Something went wrong, use fallback
            rec = OptimizedRecommender()
            update_progress(0.2, "Using fallback optimized recommender...")
        
        # Try to load pre-computed embeddings if available
        if embed_file and os.path.exists(embed_file):
            try:
                update_progress(0.3, "Loading pre-computed embeddings...")
                rec.load_embeddings(embed_file)
                update_progress(1.0, "Embeddings loaded successfully!")
            except Exception as e:
                update_progress(0.3, f"Failed to load embeddings: {str(e)}")
                update_progress(0.4, "Computing new embeddings...")
                
                # Use the appropriate fit method based on what we detected
                if isinstance(rec, OptimizedRecommender):
                    rec.fit(df_local, batch_size=batch_size, progress_callback=update_progress)
                else:
                    try:
                        # Try optimized method first
                        rec.fit(df_local, batch_size=batch_size, progress_callback=update_progress)
                    except (TypeError, ValueError):
                        # Fall back to basic method if optimization fails
                        update_progress(0.4, "Using standard embedding computation...")
                        rec.fit(df_local)
        else:
            update_progress(0.3, "No pre-computed embeddings found")
            update_progress(0.4, "Computing embeddings (this may take a while)...")
            
            # Use the appropriate fit method based on what we detected
            if isinstance(rec, OptimizedRecommender):
                rec.fit(df_local, batch_size=batch_size, progress_callback=update_progress)
            else:
                try:
                    # Try optimized method first
                    rec.fit(df_local, batch_size=batch_size, progress_callback=update_progress)
                except (TypeError, ValueError):
                    # Fall back to basic method if optimization fails
                    update_progress(0.4, "Using standard embedding computation...")
                    rec.fit(df_local)
        
        time.sleep(1)  # Let users see the completion message
        progress_bar.empty()  # Remove progress bar when done
        status_text.empty()  # Remove status text when done
        
        return rec
        
    except Exception as e:
        # Handle any unexpected errors
        progress_bar.empty()
        status_text.error(f"Error building recommender: {str(e)}")
        raise e

try:
    recommender = build_recommender(data_path, embed_file, batch_size=batch_size)
    sommelier = Sommelier(recommender)
    st.sidebar.success("✅ Recommender system ready!")
except Exception as e:
    st.sidebar.error(f"Failed to initialize recommender: {str(e)}")
    st.error("❌ The recommender system could not be initialized. Please check the error message in the sidebar.")
    st.stop()  # Stop execution if recommender initialization fails

st.header("Red, white, or something sparkling?")
user_text = st.text_input("Describe your wine, meal or cheese to pair with...", "a medium-bodied red to go with steak, under $30")

col1, col2 = st.columns(2)
with col1:
    budget_min = st.number_input("Min price (optional)", value=0.0, step=1.0)
with col2:
    budget_max = st.number_input("Max price (optional)", value=30.0, step=1.0)

variety_input = st.text_input("Prefer a grape/variety? (comma-separated)", value="chardonnay, pinot noir")

top_k = st.slider("How many suggestions?", 1, 8, 3)

if st.button("Recommend"):
    if not user_text.strip():
        st.error("Please enter a request.")
    else:
        with st.spinner("Finding wines..."):
            variety_list = [v.strip() for v in variety_input.split(",") if v.strip()] or None
            # Fix for minimum price - convert to float and validate
            price_min = float(budget_min) if budget_min > 0 else None
            price_max = float(budget_max) if budget_max > 0 else None
            res = sommelier.recommend_and_explain(
                user_text=user_text,
                top_k=top_k,
                price_min=price_min,
                price_max=price_max,
                variety=variety_list,
            )
        st.markdown("### Recommendations")
        for i, c in enumerate(res["candidates"], start=1):
            st.markdown(f"**{i}. {c['title']}** — {c['variety']} — ${c.get('price','?')}")
            st.markdown(f"*{c['description'][:300]}*")
            st.markdown("---")

        st.markdown("### Sommelier explanation")
        st.write(res["explanation"])

st.sidebar.header("LLM (optional)")
if os.environ.get("GEMINI_API_KEY"):
    st.sidebar.success("Gemini API key detected — Gemini explanations enabled")
else:
    st.sidebar.info("No GEMINI_API_KEY; explanations use templates.")
