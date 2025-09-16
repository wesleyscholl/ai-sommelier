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
        display: flex;
        flex-direction: column;
        min-height: 200px;
        width: 100%;
        box-sizing: border-box;
    }
    
    /* Remove column-specific styling since we're using single column */
    .stColumn {
        display: flex !important;
        flex-direction: column !important;
    }
    
    .stColumn > div {
        display: flex !important;
        flex-direction: column !important;
    }
    
    .wine-card-content {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        flex: 1;
    }
    
    .wine-card-description {
        margin-top: auto;
        padding-top: 0.5rem;
        font-style: italic;
        color: #E2E8F0;
        line-height: 1.4;
        font-size: 0.9rem;
    }
    .metric-container {
        background: rgba(45, 27, 61, 0.7);
        padding: 0.5rem;
        border-radius: 6px;
        margin: 0.2rem;
    }
    
    /* Elegant Sommelier Notes Card */
    .sommelier-card {
        background: linear-gradient(145deg, #2D1B3D 0%, #45274A 50%, #2D1B3D 100%);
        border: 3px solid #D4AF37;
        border-radius: 20px;
        padding: 2.5rem;
        margin: 3rem 0;
        box-shadow: 
            0 10px 40px rgba(212, 175, 55, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }
    
    .sommelier-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 6px;
        background: linear-gradient(90deg, 
            #D4AF37 0%, 
            #F4E99B 25%, 
            #D4AF37 50%, 
            #F4E99B 75%, 
            #D4AF37 100%);
        animation: shimmer 4s ease-in-out infinite;
    }
    
    .sommelier-card::after {
        content: '';
        position: absolute;
        top: 10px;
        left: 10px;
        right: 10px;
        bottom: 10px;
        border: 1px solid rgba(212, 175, 55, 0.2);
        border-radius: 15px;
        pointer-events: none;
    }
    
    @keyframes shimmer {
        0%, 100% { 
            opacity: 0.6;
            transform: translateX(-100%);
        }
        50% { 
            opacity: 1;
            transform: translateX(0%);
        }
    }
    
    .sommelier-header {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 2px solid rgba(212, 175, 55, 0.3);
        position: relative;
    }
    
    .sommelier-header::after {
        content: '✦';
        position: absolute;
        bottom: -8px;
        left: 50%;
        transform: translateX(-50%);
        color: #D4AF37;
        font-size: 1rem;
        background: linear-gradient(145deg, #2D1B3D, #45274A);
        padding: 0 0.5rem;
    }
    
    .sommelier-avatar {
        width: 80px;
        height: 80px;
        background: linear-gradient(145deg, #D4AF37, #F4E99B, #D4AF37);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.5rem;
        box-shadow: 
            0 8px 25px rgba(212, 175, 55, 0.4),
            inset 0 2px 4px rgba(255, 255, 255, 0.2);
        border: 3px solid rgba(255, 255, 255, 0.1);
        position: relative;
    }
    
    .sommelier-avatar::before {
        content: '';
        position: absolute;
        top: -3px;
        left: -3px;
        right: -3px;
        bottom: -3px;
        border-radius: 50%;
        background: linear-gradient(45deg, #D4AF37, transparent, #D4AF37);
        z-index: -1;
        animation: rotate 6s linear infinite;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .sommelier-title {
        color: #F4E99B !important;
        font-size: 2.2rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 
            0 2px 4px rgba(0,0,0,0.05),
            0 0 10px rgba(212, 175, 55, 0.1);
        background: linear-gradient(270deg, #D4AF37, #F4E99B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .sommelier-subtitle {
        color: #E2E8F0;
        font-size: 1.1rem;
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
        font-style: italic;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    .sommelier-content {
        color: #F7FAFC;
        line-height: 1.8;
        font-size: 1.1rem;
        text-align: justify;
        position: relative;
    }
    
    .sommelier-quote {
        border-left: 6px solid #D4AF37;
        padding: 1.5rem;
        margin: 1.5rem 0;
        font-style: italic;
        color: #F0F4F8;
        background: linear-gradient(135deg, 
            rgba(212, 175, 55, 0.08) 0%, 
            rgba(212, 175, 55, 0.03) 100%);
        border-radius: 0 12px 12px 0;
        position: relative;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .sommelier-quote::before {
        content: '"';
        position: absolute;
        top: -10px;
        left: 10px;
        font-size: 4rem;
        color: #D4AF37;
        opacity: 0.3;
        font-family: serif;
        line-height: 1;
    }
    
    .sommelier-quote::after {
        content: '"';
        position: absolute;
        bottom: -20px;
        right: 20px;
        font-size: 4rem;
        color: #D4AF37;
        opacity: 0.3;
        font-family: serif;
        line-height: 1;
    }
    
    /* Enhanced wine text formatting */
    .wine-name {
        color: #D4AF37 !important;
        font-weight: bold !important;
        font-size: 1.1em !important;
        text-shadow: 0 1px 3px rgba(212, 175, 55, 0.4) !important;
        background: linear-gradient(45deg, #D4AF37, #F4E99B) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
    }
    
    .wine-entry {
        margin: 1.2rem 0 !important;
        padding: 0.8rem !important;
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.05) 0%, rgba(45, 27, 61, 0.1) 100%) !important;
        border-left: 3px solid rgba(212, 175, 55, 0.4) !important;
        border-radius: 0 8px 8px 0 !important;
        line-height: 1.6 !important;
    }
    
    .wine-number {
        color: #F4E99B !important;
        font-weight: bold !important;
        font-size: 1.1em !important;
        margin-right: 0.5rem !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
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
    
    /* Enhanced Gold/Yellow Styling for UI Elements */
    
    /* Main title styling */
    .stApp h1 {
        color: #D4AF37 !important;
        text-shadow: 0 2px 8px rgba(212, 175, 55, 0.4) !important;
        background: linear-gradient(45deg, #D4AF37, #F4E99B, #D4AF37) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        font-weight: bold !important;
    }
    
    /* Section headers styling */
    .stApp h2, .stApp h3 {
        color: #F4E99B !important;
        text-shadow: 0 1px 4px rgba(212, 175, 55, 0.3) !important;
        padding-bottom: 0.5rem !important;
    }
    
    /* Sidebar header styling */
    .stSidebar h1, .stSidebar h2, .stSidebar h3 {
        color: #D4AF37 !important;
        text-shadow: 0 1px 3px rgba(212, 175, 55, 0.4) !important;
    }
    
    /* Primary button styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(145deg, #D4AF37 0%, #F4E99B 50%, #D4AF37 100%) !important;
        border: 2px solid #D4AF37 !important;
        color: #1E1E1E !important;
        font-weight: bold !important;
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(145deg, #F4E99B 0%, #D4AF37 50%, #F4E99B 100%) !important;
        box-shadow: 0 6px 20px rgba(212, 175, 55, 0.6) !important;
        transform: translateY(-2px) !important;
        color: #000000 !important;
        transform: scale(1.1) !important;
    }
    
    /* Secondary/example buttons styling */
    .stButton > button {
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.15) 0%, rgba(244, 233, 155, 0.1) 100%) !important;
        border: 1px solid rgba(212, 175, 55, 0.4) !important;
        color: #F4E99B !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.25) 0%, rgba(244, 233, 155, 0.2) 100%) !important;
        border: 1px solid #D4AF37 !important;
        color: #D4AF37 !important;
        box-shadow: 0 3px 10px rgba(212, 175, 55, 0.3) !important;
    }
    
    /* Input field styling with gold outline */
    .stTextInput > div > div > input {
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.05) 0%, rgba(45, 27, 61, 0.3) 100%) !important;
        border: 2px solid rgba(212, 175, 55, 0.3) !important;
        color: #F4E99B !important;
        border-radius: 8px !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stTextInput > div > div > input:focus {
        border: 2px solid #D4AF37 !important;
        box-shadow: 0 0 10px rgba(212, 175, 55, 0.4) !important;
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.1) 0%, rgba(45, 27, 61, 0.2) 100%) !important;
    }

    /* Number input styling */
    .stNumberInput input {
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.05) 0%, rgba(45, 27, 61, 0.3) 100%) !important;
        border: 2px solid rgba(212, 175, 55, 0.3) !important;
        color: #F4E99B !important;
        border-radius: 8px !important;
    }
    
    .stNumberInput input:focus {
        border: 2px solid #D4AF37 !important;
        box-shadow: 0 0 8px rgba(212, 175, 55, 0.4) !important;
    }
    
    /* Selectbox styling */
    [data-baseweb="select"] {
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.05) 0%, rgba(45, 27, 61, 0.3) 100%) !important;
        border: 2px solid rgba(212, 175, 55, 0.3) !important;
        color: #F4E99B !important;
        border-radius: 8px !important;
    }
    
    /* Label styling */
    .stTextInput > label, .stNumberInput > label, .stSelectbox > label {
        color: #D4AF37 !important;
        font-weight: 600 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.1) 0%, rgba(45, 27, 61, 0.2) 100%) !important;
        border: 1px solid rgba(212, 175, 55, 0.3) !important;
        color: #D4AF37 !important;
        border-radius: 8px !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.2) 0%, rgba(45, 27, 61, 0.3) 100%) !important;
        border: 1px solid #D4AF37 !important;
    }
    
    /* Sidebar separator styling */
    .stSidebar hr {
        border-color: rgba(212, 175, 55, 0.4) !important;
        box-shadow: 0 1px 3px rgba(212, 175, 55, 0.2) !important;
    }
    
    /* Markdown text enhancements in sidebar */
    .stSidebar .stMarkdown p strong {
        color: #D4AF37 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #D4AF37 0%, #F4E99B 50%, #D4AF37 100%) !important;
    }
    
    /* Checkbox styling */
    .stCheckbox > label > div > div {
        border: 2px solid rgba(212, 175, 55, 0.3) !important;
    }
    
    .stCheckbox > label > div > div > div {
        background: #D4AF37 !important;
    }
    
    /* Info/warning message styling */
    .stInfo {
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.1) 0%, rgba(45, 27, 61, 0.2) 100%) !important;
        border-left: 4px solid #D4AF37 !important;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: #D4AF37 !important;
    }
    
    .stSlider > div > div > div {
        background: rgba(212, 175, 55, 0.3) !important;
    }
    
    /* Input Form Card Styling */
    .input-form-card {
        background: linear-gradient(145deg, rgba(212, 175, 55, 0.08) 0%, rgba(45, 27, 61, 0.15) 100%);
        border: 2px solid rgba(212, 175, 55, 0.3) !important;
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 
            0 8px 32px rgba(212, 175, 55, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        backdrop-filter: blur(10px);
    }

    .input-form-card:hover {
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.2) 0%, rgba(45, 27, 61, 0.3) 100%) !important;
        border: 1px solid #D4AF37 !important;
    }
    
    .form-header {
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(212, 175, 55, 0.2);
    }
    
    .form-header h3 {
        color: #D4AF37 !important;
        font-size: 1.5rem !important;
        margin: 0 0 0.5rem 0 !important;
        text-shadow: 0 2px 4px rgba(212, 175, 55, 0.3) !important;
        background: linear-gradient(45deg, #D4AF37, #F4E99B) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
    }
    
    .form-header p {
        color: #E2E8F0 !important;
        font-style: italic !important;
        margin: 0 !important;
        opacity: 0.9 !important;
    }

    .stExpander {
        background: rgba(45, 27, 61, 0.6) !important;
        border: 1px solid rgba(212, 175, 55, 0.3) !important;
        border-radius: 8px !important;
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
        with st.expander("ℹ️  About AI Wine Sommelier", expanded=False):
            
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

# User input with better UX wrapped in elegant card
st.markdown("""
<div class="input-form-card">
    <div class="form-header">
        <h3>🍷 Find Your Perfect Wine</h3>
        <p>Tell us what you're looking for and we'll find the perfect match</p>
    </div>
""", unsafe_allow_html=True)

with st.container():
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

    # Recommendation button and results
    search_triggered = st.button("🔍 Find My Wine", type="primary") or st.session_state.get('auto_search', False)

# Close the input form card
st.markdown("</div>", unsafe_allow_html=True)

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
                st.write("")
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
