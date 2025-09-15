# 🍷 Quick Start Guide

## Test Locally

1. **Install dependencies:**
```bash
pip install streamlit pandas numpy scikit-learn sentence-transformers google-generativeai
```

2. **Run the app:**
```bash
cd /Users/wscholl/ai-sommelier
streamlit run app/app.py
```

3. **Test features:**
   - Enable "Use sample for demo" in sidebar 
   - Try quick example buttons
   - Test custom queries

## Fixes Applied ✅

### Dark Wine Theme
- Deep purple/gold color scheme
- Custom CSS styling for wine cards
- Professional dark mode appearance

### Functionality Fixes
- ✅ Fixed quick example buttons (now use `st.rerun()`)
- ✅ Fixed variety filtering (more flexible matching)
- ✅ Added debug info in sidebar
- ✅ Improved error handling with fallbacks
- ✅ Enhanced recommendation display

### Common Issues & Solutions

**"No wines found":**
- Leave variety field blank for broader results
- Try higher price ranges ($0-$100)
- Use more general descriptions

**Slow loading:**
- Enable "Use sample for demo" mode
- Reduce batch size to 8
- Pre-compute embeddings will cache

**Quick examples not working:**
- Should now work with `st.rerun()`
- Check session state updates in sidebar

## Deploy to Streamlit Cloud

1. Push changes to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Main file: `app/app.py`
4. Optional: Add `GOOGLE_API_KEY` environment variable

The app now has a beautiful dark wine theme and should work correctly! 🍷✨