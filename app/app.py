import streamlit as st
import os
import sys
import time
import re
import numpy as np
from pathlib import Path

# Add the project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_wine_dataset
from src.recommender import Recommender
from src.sommelier import Sommelier

def load_css(file_path):
    """Load CSS from external file and inject into Streamlit app."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            css_content = f.read()
        st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file not found: {file_path}")
    except Exception as e:
        st.error(f"Error loading CSS: {str(e)}")

def format_sommelier_text(text):
    """Format sommelier explanation text with enhanced HTML styling."""
    if not text:
        return ""
    
    # Convert basic markdown formatting to HTML
    formatted_text = text
    
    # Convert **bold** to styled spans
    formatted_text = re.sub(r'\*\*(.*?)\*\*', r'<span class="wine-name">\1</span>', formatted_text)
    
    # Convert numbered lists to styled lists
    formatted_text = re.sub(r'^(\d+)\.\s+', r'<div class="wine-entry"><span class="wine-number">\1.</span> ', formatted_text, flags=re.MULTILINE)
    
    # Close wine-entry divs at the end of each entry (before next number or end)
    formatted_text = re.sub(r'(?=\n\d+\.|\n*$)', '</div>', formatted_text)
    
    # Clean up any double closing divs
    formatted_text = formatted_text.replace('</div></div>', '</div>')
    
    # Add subtle spacing and styling
    formatted_text = formatted_text.replace('\n\n', '<br><br>')
    formatted_text = formatted_text.replace('\n', '<br>')
    
    return formatted_text

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

# Load custom CSS from external file
css_path = os.path.join(os.path.dirname(__file__), 'styles.css')
load_css(css_path)

# Smart configuration with production defaults
with st.sidebar:
    st.header("🍷 AI Wine Sommelier")

    st.markdown("---")

    # Quick examples (keep visible)
    st.markdown("**✨ Quick Examples:**")
    examples = [
        {"text": "🍗 Bold red for BBQ under $25", "min_price": 0.0, "max_price": 25.0, "variety": ""},
        {"text": "🐠 Crisp white for seafood", "min_price": 0.0, "max_price": 50.0, "variety": "Sauvignon Blanc, Pinot Grigio"}, 
        {"text": "🍾 Elegant wine for special dinner", "min_price": 30.0, "max_price": 100.0, "variety": ""},
        {"text": "🍰 Sweet wine for dessert", "min_price": 0.0, "max_price": 40.0, "variety": "Riesling, Port"}
    ]
    
    for i, example in enumerate(examples):
        if st.button(f"{example['text']}", key=f"example_{i}", use_container_width=True):
            # Set all form values and trigger auto-search
            st.session_state['example_query'] = example['text']
            st.session_state['auto_search'] = True
            st.session_state['example_min_price'] = example['min_price']
            st.session_state['example_max_price'] = example['max_price']
            st.session_state['example_variety'] = example['variety']
            st.rerun()

    st.markdown("---")
    
    # Collapsed configuration section
    with st.expander("⚙️  Configuration", expanded=False):
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
        with st.expander("ℹ️  About", expanded=False):
            
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

            **🔒 Privacy:**
            - We do not store any personal data
            - All interactions are anonymous
            - You can reset the chat at any time
            - No account or login required

            **👨🏻‍💻 Created by:**
            
            - *Wesley Scholl*
            - [GitHub](https://github.com/wesleyscholl/ai-sommelier)
            - [LinkedIn](https://www.linkedin.com/in/wesleyscholl/)
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
    
    # Show toasts only on first load
    if 'toasts_shown' not in st.session_state:
        st.session_state['toasts_shown'] = True
        
        # Simple success toasts without threading
        st.toast("AI Sommelier ready!", icon="🤵🏻‍♂️")
        st.toast(f"Dataset loaded: {len(df):,} wines", icon="✅")
        
        # API status toast if enabled
        if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
            st.toast("AI explanations enabled", icon="🤖")
    
    # Always show API info message if no key
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        st.info("💡 Set GOOGLE_API_KEY for AI explanations")
        
except Exception as e:
    st.sidebar.error(f"❌ Initialization failed: {str(e)}")
    st.toast("Sommelier failed to initialize", icon="❌")
    st.error("The AI sommelier could not be initialized. Please check your configuration.")
    st.stop()

with st.container(key="input-form"):
    # User input with better UX wrapped in elegant card
    st.markdown("""
        <div class="form-header">
            <h3>🍷 Find Your Perfect Wine</h3>
            <p>Tell us what you're looking for and we'll find the perfect match</p>
        </div>
    """, unsafe_allow_html=True)
    
    user_text = st.text_input(
        "Describe your perfect wine, food or cheese pairing:",
        placeholder="e.g., a medium-bodied red for steak dinner under $30",
        value=st.session_state.get('example_query', "a medium-bodied red to go with steak, under $30"),
        key="wine_query"
    )

    # Filters in columns for better layout
    col1, col2, col3 = st.columns(3)
    with col1:
        budget_min = st.number_input("Min price ($)", 
                                    value=st.session_state.get('example_min_price', 0.0), 
                                    step=5.0, min_value=0.0)
    with col2:
        budget_max = st.number_input("Max price ($)", 
                                    value=st.session_state.get('example_max_price', 50.0), 
                                    step=5.0, min_value=0.0)
    with col3:
        top_k = st.selectbox("Number of recommendations", [3, 5, 8], index=0)

    variety_input = st.text_input(
        "Preferred varieties (optional):",
        placeholder="e.g., Cabernet Sauvignon, Pinot Noir",
        value=st.session_state.get('example_variety', ""),
        help="Leave blank for all varieties, or specify comma-separated grape types",
        key="variety_input"
    )

    st.write("")
    # Recommendation button and results
    search_triggered = st.button("🔍 Find My Wine", type="primary") or st.session_state.get('auto_search', False)

# Clear auto_search flag after using it
if st.session_state.get('auto_search', False):
    st.session_state['auto_search'] = False

if search_triggered:
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
                st.write("")
                st.markdown("## Your Wine Recommendations")

                if not res["candidates"]:
                    st.warning("No wines found matching your criteria. Try adjusting your filters or description.")
                    st.info("💡 **Tips:** Try broader terms, remove variety filters, or increase price range")
                else:
                    # Display wines in single full-width column
                    for i, wine in enumerate(res["candidates"]):
                        country = wine.get('country', 'Unknown')
                        flag = get_country_flag(country)
                        
                        # Wine card with full width
                        wine_html = f"""
                        <div class="wine-card">
                            <div class="wine-card-content">
                                <div style="flex-shrink: 0;">
                                    <div style="width: 80px; height: 120px; background: linear-gradient(145deg, #4a5568, #2d3748); 
                                                border-radius: 8px; display: flex; align-items: center; justify-content: center; 
                                                border: 2px solid #D4AF37; box-shadow: 0 2px 8px rgba(0,0,0,0.3);">
                                        <span style="font-size: 2rem;">🍷</span>
                                    </div>
                                </div>
                                <div style="flex: 1; min-width: 0;">
                                    <h4 style="margin: 0 0 0.5rem 0; color: #D4AF37;">{i+1}. {wine['title']}</h4>
                                    <p style="margin: 0 0 0.5rem 0; font-weight: bold;">{wine['variety']} from {country} {flag}</p>
                                    <div style="display: flex; gap: 1rem; margin-bottom: 0.5rem;">
                                        <span style="background: rgba(45, 27, 61, 0.7); padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.9rem;">
                                            💰 ${wine.get('price', '?') if wine.get('price') else 'N/A'}
                                        </span>
                                        <span style="background: rgba(45, 27, 61, 0.7); padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.9rem;">
                                            🎯 {wine.get('similarity', 0):.1%} match
                                        </span>
                                    </div>
                                </div>
                            </div>
                            <div class="wine-card-description">
                                {wine.get('description', '')}
                            </div>
                        </div>
                        """
                        st.markdown(wine_html, unsafe_allow_html=True)
                        
                        # Add spacing between cards
                        st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)
                
                # Sommelier explanation with elegant styling
                if res["explanation"]:
                    formatted_explanation = format_sommelier_text(res["explanation"])
                    sommelier_html = f"""
                    <div class="sommelier-card">
                        <div class="sommelier-header">
                            <div class="sommelier-avatar">🤵🏻‍♂️</div>
                            <div>
                                <h2 class="sommelier-title">Sommelier's Notes</h2>
                                <p class="sommelier-subtitle">Expert wine pairing insights</p>
                            </div>
                        </div>
                        <div class="sommelier-content">
                            <div class="sommelier-quote">
                                {formatted_explanation}
                            </div>
                        </div>
                    </div>
                    """
                    st.markdown(sommelier_html, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"An error occurred while finding recommendations: {str(e)}")

        # Clear example session state after search
        for key in ['example_min_price', 'example_max_price', 'example_variety']:
            if key in st.session_state:
                del st.session_state[key]
