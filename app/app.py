import streamlit as st
import os
import sys
import time
import numpy as np
from pathlib import Path
import threading

# Add the project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_wine_dataset
from src.recommender import Recommender
from src.sommelier import Sommelier

# Helper function for delayed toast messages
def delayed_toast(message, icon, delay_seconds=0):
    """Show a toast message after a specified delay."""
    def show_toast():
        time.sleep(delay_seconds)
        st.toast(message, icon=icon)
    
    thread = threading.Thread(target=show_toast)
    thread.daemon = True
    thread.start()

# Country flag emoji mapping
def get_country_flag(country):
    """Get flag emoji for wine country."""
    flag_map = {
        'Argentina': '🇦🇷', 'Australia': '🇦🇺', 'Austria': '🇦🇹', 'Bulgaria': '🇧🇬',
        'Brazil': '🇧🇷', 'Canada': '🇨🇦', 'Chile': '🇨🇱', 'Croatia': '🇭🇷',
        'Cyprus': '🇨🇾', 'Czech Republic': '🇨🇿', 'England': '🏴󠁧󠁢󠁥󠁮󠁧󠁿', 'France': '🇫🇷',
        'Georgia': '🇬🇪', 'Germany': '🇩🇪', 'Greece': '🇬🇷', 'Hungary': '🇭🇺',
        'India': '🇮🇳', 'Israel': '🇮🇱', 'Italy': '🇮🇹', 'Lebanon': '🇱🇧',
        'Luxembourg': '🇱🇺', 'Macedonia': '🇲🇰', 'Moldova': '🇲🇩', 'Morocco': '🇲🇦',
        'New Zealand': '🇳🇿', 'Peru': '🇵🇪', 'Portugal': '🇵🇹', 'Romania': '🇷🇴',
        'Serbia': '🇷🇸', 'Slovenia': '🇸🇮', 'South Africa': '🇿🇦', 'Spain': '🇪🇸',
        'Switzerland': '🇨🇭', 'Turkey': '🇹🇷', 'Ukraine': '🇺🇦', 'Uruguay': '🇺🇾',
        'US': '🇺🇸', 'USA': '🇺🇸', 'United States': '🇺🇸'
    }
    return flag_map.get(country, '🍷')  # Default wine emoji if country not found

# Configure page for production
st.set_page_config(
    page_title="AI Wine Sommelier",
    page_icon="🍷",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Streamlined Recommender for production
class ProductionRecommender(Recommender):
    """Production-optimized recommender with better memory management."""
    
    def fit(self, df, text_column="text_for_embedding", batch_size=16, progress_callback=None):
        """Memory-optimized fitting with smaller batches for Streamlit Cloud."""
        self.df = df.reset_index(drop=True).copy()
        texts = self.df[text_column].astype(str).tolist()
        total_samples = len(texts)
        
        if progress_callback:
            progress_callback(0.1, f"Processing {total_samples} wines...")
        
        # Get embedding dimension efficiently
        try:
            embedding_dim = int(getattr(self.model, "get_sentence_embedding_dimension")())
        except Exception:
            sample_emb = self.model.encode(texts[0], convert_to_numpy=True).reshape(1, -1)
            embedding_dim = sample_emb.shape[1]

        all_embeddings = np.zeros((total_samples, embedding_dim), dtype=np.float32)
        
        # Process in smaller batches for stability
        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            batch_texts = texts[start_idx:end_idx]
            
            if progress_callback:
                progress = 0.1 + 0.7 * (start_idx / total_samples)
                progress_callback(progress, f"Batch {start_idx//batch_size + 1}/{(total_samples-1)//batch_size + 1}")
            
            try:
                batch_embeddings = self.model.encode(
                    batch_texts, 
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    device='cpu'  # Force CPU for stability
                )
                all_embeddings[start_idx:end_idx] = batch_embeddings
            except Exception as e:
                if progress_callback:
                    progress_callback(progress, f"Retrying batch due to error...")
                # Fallback: process one by one
                for i, text in enumerate(batch_texts):
                    try:
                        single_emb = self.model.encode([text], convert_to_numpy=True, device='cpu')
                        all_embeddings[start_idx + i] = single_emb[0]
                    except Exception:
                        pass  # Skip problematic samples
        
        self.embeddings = all_embeddings
        
        if progress_callback:
            progress_callback(0.9, "Building search index...")
        
        try:
            from sklearn.neighbors import NearestNeighbors
            self.nn = NearestNeighbors(n_neighbors=min(10, len(self.df)), metric="cosine", algorithm="auto")
            self.nn.fit(self.embeddings)
        except ImportError:
            # If sklearn not available, use a simple fallback
            if progress_callback:
                progress_callback(0.9, "Using fallback similarity search...")
            self.nn = None
        
        if progress_callback:
            progress_callback(1.0, "Ready!")

st.title("🤵🏻‍♂️🍷 AI Wine Sommelier")
st.markdown("*Find your perfect wine with AI-powered recommendations*")

# Add custom CSS for dark wine theme and toast styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #2D1B3D 0%, #1E1E1E 100%);
    }
    .wine-card {
        background: rgba(212, 175, 55, 0.1);
        border-left: 4px solid #D4AF37;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
    }
    .metric-container {
        background: rgba(45, 27, 61, 0.7);
        padding: 0.5rem;
        border-radius: 6px;
        margin: 0.2rem;
    }
    /* Custom toast styling to match wine theme */
    .stToast {
        background: linear-gradient(135deg, #2D1B3D 0%, #45274A 100%) !important;
        border: 1px solid #D4AF37 !important;
        color: #D4AF37 !important;
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.3) !important;
    }
    .stToast > div {
        color: #FFFFFF !important;
    }
    .stToast [data-testid="toastContainer"] {
        background: linear-gradient(135deg, #2D1B3D 0%, #45274A 100%) !important;
        border: 1px solid #D4AF37 !important;
    }
</style>
""", unsafe_allow_html=True)

# Smart configuration with production defaults
with st.sidebar:
    st.header("🍷 AI Wine Sommelier")

    st.markdown("---")

    # Quick examples (keep visible)
    st.markdown("**✨ Quick Examples:**")
    examples = [
        "Bold red for BBQ under $25",
        "Crisp white for seafood", 
        "Elegant wine for special dinner",
        "Sweet wine for dessert"
    ]
    
    for i, example in enumerate(examples):
        if st.button(f"🍷 {example}", key=f"example_{i}", use_container_width=True):
            # Force page rerun with new query
            st.session_state['example_query'] = example
            st.rerun()

    st.markdown("---")
    
    # Collapsed configuration section
    with st.expander("⚙️ Configuration", expanded=False):
        data_path = st.text_input("Wine dataset path", value="data/wine_reviews.csv")
        embed_file = st.text_input("Embeddings cache", value="data/embeddings.npz", 
                                  help="Pre-computed embeddings for faster loading")
        
        # Performance settings
        use_sample = st.checkbox("Use sample for testing", value=False, 
                                help="Enable only for testing - uses full 130K+ dataset by default")
        if use_sample:
            sample_size = st.slider("Sample size", 500, 10000, 5000, 500)
        batch_size = st.slider("Processing batch size", 8, 32, 16, 4,
                              help="Lower = less memory, slower processing")

    # Sidebar information
    with st.sidebar:

        # About section in collapsible expander
        with st.expander("ℹ️ About AI Wine Sommelier", expanded=False):
            
            st.markdown("""
            **🍷 Features:**
            - 🧠 AI-powered semantic search
            - 📊 Full 130K+ wine database
            - 💰 Smart price filtering ($0-$300+)
            - 🍇 Flexible variety matching
            - 🤖 AI sommelier explanations
            - 🎯 1000+ candidates per search
            
            **💡 Tips:**
            - Be specific in descriptions
            - Mention food pairings
            - Include price preferences  
            - Try different varieties
            """)

# Initialize data loading with error handling
@st.cache_data
def load_dataset(path, use_sample=False, sample_size=2000):
    """Load and optionally sample the wine dataset."""
    try:
        if not os.path.exists(path):
            st.error(f"Dataset not found at {path}")
            return None
        
        df = load_wine_dataset(path)
        if use_sample and len(df) > sample_size:
            df = df.sample(sample_size, random_state=42).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Failed to load dataset: {str(e)}")
        return None

# Load dataset
df = load_dataset(data_path, use_sample, sample_size if use_sample else None)
if df is None:
    st.stop()

# Build or load recommender with enhanced caching
@st.cache_resource(show_spinner=False)
def build_recommender(data_path: str, embed_file: str = None, batch_size: int = 16, sample_size: int = None):
    """Production-optimized recommender builder with robust caching."""
    
    # Progress tracking
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    def update_progress(fraction, message):
        try:
            progress_bar.progress(fraction)
            status_text.text(message)
        except Exception:
            pass
    
    try:
        # Load dataset with sampling
        update_progress(0.1, "Loading dataset...")
        df_local = load_wine_dataset(data_path)
        if sample_size and len(df_local) > sample_size:
            df_local = df_local.sample(sample_size, random_state=42).reset_index(drop=True)
        
        update_progress(0.2, f"Dataset ready: {len(df_local):,} wines")
        
        # Initialize recommender with fallback
        try:
            rec = ProductionRecommender()
        except Exception as e:
            # Fallback to original Recommender if ProductionRecommender fails
            rec = Recommender()
            update_progress(0.2, "Using standard recommender...")
        
        # Try loading cached embeddings
        if embed_file and os.path.exists(embed_file):
            try:
                update_progress(0.3, "Loading cached embeddings...")
                rec.load_embeddings(embed_file)
                update_progress(1.0, "✅ Embeddings loaded from cache!")
                
                # Clear progress after brief display
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                return rec
                
            except Exception as e:
                update_progress(0.3, f"Cache failed: {str(e)[:50]}...")
        
        # Compute embeddings
        update_progress(0.4, "Computing embeddings (this may take a few minutes)...")
        rec.fit(df_local, batch_size=batch_size, progress_callback=update_progress)
        
        # Save embeddings for future use
        try:
            cache_path = embed_file or "data/embeddings.npz"
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            rec.save_embeddings(cache_path)
            update_progress(1.0, f"✅ Embeddings cached to {cache_path}")
        except Exception:
            update_progress(1.0, "✅ Recommender ready!")
        
        # Clear progress UI
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return rec
        
    except Exception as e:
        progress_bar.empty()
        status_text.error(f"❌ Error: {str(e)}")
        raise e

# Initialize recommender system
try:
    recommender = build_recommender(
        data_path, 
        embed_file, 
        batch_size=batch_size,
        sample_size=sample_size if use_sample else None
    )
    sommelier = Sommelier(recommender)
    
    # Staggered auto-hiding toast notifications (5 seconds each, 1 second apart)
    st.toast("🎯 AI Sommelier ready!", icon="🎯")  # Shows immediately
    delayed_toast(f"✅ Dataset loaded: {len(df):,} wines", "✅", 1)  # Shows after 1 second
    
    # API status toast (shows after 2 seconds if enabled)
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        delayed_toast("🤖 AI explanations enabled", "🤖", 2)  # Shows after 2 seconds
    else:
        st.info("💡 Set GOOGLE_API_KEY for AI explanations")
        
except Exception as e:
    st.sidebar.error(f"❌ Initialization failed: {str(e)}")
    st.toast("Sommelier failed to initialize", icon="❌")
    st.error("The AI sommelier could not be initialized. Please check your configuration.")
    st.stop()

# User input with better UX
user_text = st.text_input(
    "Describe your perfect wine or food pairing:",
    placeholder="e.g., a medium-bodied red for steak dinner under $30",
    value=st.session_state.get('example_query', "a medium-bodied red to go with steak, under $30"),
    key="wine_query"
)

# Filters in columns for better layout
col1, col2, col3 = st.columns(3)
with col1:
    budget_min = st.number_input("Min price ($)", value=0.0, step=5.0, min_value=0.0)
with col2:
    budget_max = st.number_input("Max price ($)", value=50.0, step=5.0, min_value=0.0)
with col3:
    top_k = st.selectbox("Number of recommendations", [3, 5, 8], index=0)

variety_input = st.text_input(
    "Preferred varieties (optional):",
    placeholder="e.g., Cabernet Sauvignon, Pinot Noir",
    help="Leave blank for all varieties, or specify comma-separated grape types",
    key="variety_input"
)

# Recommendation button and results
if st.button("🔍 Find My Wine", type="primary"):
    if not user_text.strip():
        st.error("Please describe what wine you're looking for.")
    else:
        with st.spinner("🍷 Finding your perfect wines..."):
            # Clean variety input - only use if actually provided
            variety_list = None
            if variety_input and variety_input.strip():
                variety_list = [v.strip() for v in variety_input.split(",") if v.strip()]
            
            price_min = float(budget_min) if budget_min > 0 else None
            price_max = float(budget_max) if budget_max > 0 else None
            
            try:
                res = sommelier.recommend_and_explain(
                    user_text=user_text,
                    top_k=top_k,
                    price_min=price_min,
                    price_max=price_max,
                    variety=variety_list,
                )
                
                # Display recommendations with enhanced formatting
                st.markdown("## 🎯 Your Wine Recommendations")

                if not res["candidates"]:
                    st.warning("No wines found matching your criteria. Try adjusting your filters or description.")
                    st.info("💡 **Tips:** Try broader terms, remove variety filters, or increase price range")
                else:
                    for i, wine in enumerate(res["candidates"], 1):
                        # Get country flag
                        country = wine.get('country', 'Unknown')
                        flag = get_country_flag(country)
                        
                        # Use custom styling for wine cards
                        wine_html = f"""
                        <div class="wine-card">
                            <h4>🍷 {i}. {wine['title']}</h4>
                        </div>
                        """
                        st.markdown(wine_html, unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.write(f"**{wine['variety']}** from {country} {flag}")
                        with col2:
                            price_str = f"${wine.get('price', '?')}" if wine.get('price') else "Price not available"
                            st.markdown(f"<div class='metric-container'>💰 {price_str}</div>", unsafe_allow_html=True)
                        with col3:
                            similarity = wine.get('similarity', 0)
                            st.markdown(f"<div class='metric-container'>🎯 {similarity:.1%} match</div>", unsafe_allow_html=True)
                        
                        # Description
                        if wine.get('description'):
                            desc = wine['description']
                            # [:200] + "..." if len(wine['description']) > 200 else wine['description']
                            st.write(f"*{desc}*")
                        
                        st.markdown("---")
                
                # Sommelier explanation
                if res["explanation"]:
                    st.markdown("## 🤵🏻‍♂️ Sommelier's Notes")
                    st.write(res["explanation"])
                    
            except Exception as e:
                st.error(f"An error occurred while finding recommendations: {str(e)}")
