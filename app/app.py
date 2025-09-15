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

st.set_page_config(page_title="AI Wine Sommelier", layout="centered")

st.title("🤵🏻‍♂️🍷 AI Wine Sommelier")

data_path = st.sidebar.text_input("Path to wine CSV", value="data/wine_reviews.csv")
embed_file = st.sidebar.text_input("Optional embeddings file (.npz)", value="data/embeddings.npz")

if not os.path.exists(data_path):
    st.sidebar.warning("Data file not found. Place a CSV at data/wine_reviews.csv or change the path.")
else:
    with st.spinner("Loading dataset..."):
        df = load_wine_dataset(data_path)
    st.sidebar.success(f"Loaded {len(df)} rows")

# Build or load recommender
@st.cache_resource
def build_recommender(data_path: str, embed_file: str = None):
    status = st.sidebar.empty()
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    df_local = load_wine_dataset(data_path)
    rec = Recommender()
    
    if embed_file and os.path.exists(embed_file):
        try:
            status_text.text("Loading embeddings from file...")
            progress_bar.progress(25)
            rec.load_embeddings(embed_file)
            progress_bar.progress(100)
            status_text.text("Embeddings loaded successfully!")
        except Exception:
            status_text.text("Failed to load embeddings, computing new ones...")
            progress_bar.progress(30)
            
            # Simulate progress during embedding computation
            # This creates the appearance of progress while embeddings are computed
            status_text.text("Computing embeddings (this may take a while)...")
            for i in range(30, 90, 10):
                progress_bar.progress(i)
                time.sleep(0.5)  # Small delay for visual feedback
                
            rec.fit(df_local)
            progress_bar.progress(100)
            status_text.text("Embeddings computed successfully!")
    else:
        status_text.text("Computing embeddings (this may take a while)...")
        
        # Simulate progress during embedding computation
        for i in range(0, 90, 10):
            progress_bar.progress(i)
            time.sleep(0.5)  # Small delay for visual feedback
            
        rec.fit(df_local)
        progress_bar.progress(100)
        status_text.text("Embeddings computed successfully!")
    
    time.sleep(1)  # Let users see the 100% completion
    progress_bar.empty()  # Remove progress bar when done
    return rec

recommender = build_recommender(data_path, embed_file)
sommelier = Sommelier(recommender)

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
