# 🚀 Production Deployment Guide

## Quick Deploy to Streamlit Cloud

1. **Fork this repository** to your GitHub account

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select this repository
   - Set main file: `app/app.py`
   - Deploy!

3. **Environment Variables (Optional):**
   - Add `GOOGLE_API_KEY` or `GEMINI_API_KEY` for AI explanations
   - Get key from [Google AI Studio](https://aistudio.google.com/)

## Features ✨

- **Fast Loading**: Optimized embeddings cache system
- **Smart Sampling**: Demo mode with 2000 wines for faster performance  
- **Robust Error Handling**: Graceful degradation and fallbacks
- **Production Ready**: Streamlined for Streamlit Cloud deployment
- **Enhanced UX**: Clean interface with quick examples and filters

## Performance Optimizations

- ⚡ **Embedding Caching**: Automatic save/load of computed embeddings
- 🧠 **Memory Efficient**: Optimized batch processing (16 samples default)
- 🎯 **Smart Sampling**: Use sample mode for demos and testing
- 💾 **CPU Optimized**: Forced CPU processing for deployment stability
- 🔍 **Enhanced Search**: Increased neighbor candidates for better results

## Configuration Options

### In Sidebar:
- **Sample Mode**: Enable for faster demo with 2K wines
- **Batch Size**: Lower = less memory usage
- **Embeddings Cache**: Automatic caching for subsequent runs

### Environment Variables:
- `GOOGLE_API_KEY`: Enable AI-powered explanations
- `GEMINI_API_KEY`: Alternative key name

## Production Features

✅ **SSL Handling**: Robust model downloading for deployment environments  
✅ **Error Recovery**: Graceful degradation when services fail  
✅ **Memory Management**: Optimized for Streamlit Cloud resource limits  
✅ **User Experience**: Clean interface with helpful examples  
✅ **Caching Strategy**: Smart embedding persistence  

## Quick Examples

Try these in the app:
- "Bold red for BBQ under $25"
- "Crisp white for seafood"  
- "Elegant wine for special dinner"
- "Sweet wine for dessert"

## Technical Stack

- **Frontend**: Streamlit with custom config
- **ML Backend**: SentenceTransformers (all-MiniLM-L6-v2)
- **Search**: Scikit-learn NearestNeighbors with cosine similarity
- **AI**: Google Gemini for explanations (optional)
- **Data**: 130K+ wine reviews with smart sampling

---

Ready to impress employers and industry contacts! 🍷✨