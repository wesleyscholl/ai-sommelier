# AI Wine Sommelier 🤵🏻‍♂️🍷

An AI-powered wine recommendation system that helps customers find the perfect bottle based on taste preferences, grape varietals, food and cheese pairings, or mood.

## Technical Overview

The AI Wine Sommelier application leverages several state-of-the-art artificial intelligence technologies to provide intelligent wine recommendations based on natural language descriptions and preferences. This application demonstrates the practical application of modern NLP (Natural Language Processing) techniques to create a sophisticated recommendation system accessible through an intuitive interface.


### Core AI Technologies
1. Semantic Text Embeddings with SentenceTransformer
- Model Used: all-MiniLM-L6-v2 from the Sentence-Transformers library
- Technology: This transformer-based model converts wine descriptions and user queries into high-dimensional vector embeddings (768 dimensions) that capture semantic meaning beyond simple keywords
- Advantage: Enables understanding of context, synonyms, and related concepts in wine descriptions
- Implementation: Direct integration with the HuggingFace Sentence-Transformers library for state-of-the-art text encoding
2. Content-Based Recommendation Engine
- Algorithm: Nearest Neighbors search with cosine similarity metric
- Implementation: scikit-learn's NearestNeighbors with cosine distance for efficient vector similarity computation
- Process: User queries are embedded in the same vector space as wine descriptions, allowing the system to find semantically similar wines regardless of exact keyword matches
- Filtering: Additional dimensional filtering for price range, grape variety, and other attributes
3. Natural Language Generation with Google Gemini
- Model: Google Gemini 1.5 Flash
- Application: Generates natural, sommelier-style explanations for wine recommendations
- Context-Awareness: Incorporates user requests, wine characteristics, and tasting notes to craft personalized explanations
- Fallback System: Template-based explanations when Gemini API is unavailable

### Technical Architecture
The application employs a hybrid AI architecture combining multiple models:

1. Embedding Layer: Transforms raw text descriptions into numerical vectors

Uses Sentence-BERT architecture for contextual understanding
Dimensionality: 768 (based on the MiniLM model)
Efficiently encodes both wine descriptions and user queries

2. Retrieval Layer: Implements efficient similarity search

Indexed vector database for fast retrieval
Support for complex filtering criteria (price, variety)
Maintains original metadata alongside vectors

3. Explanation Generation Layer:

Connects to Google's Gemini API
Prompt engineering to ensure concise, relevant explanations
Structured output formatting

4. Caching System:

Streamlit's caching mechanism for model persistence
Embeddings storage/retrieval for performance optimization

### AI Development Considerations
The application implements several AI best practices:

1. Robustness: The system includes template-based explanation fallbacks when external AI services are unavailable

2. Efficiency: Vector caching and batched processing reduce computational overhead

3. Explainability: The system doesn't just recommend wines but explains why they match the user's request

4. Adaptability: The modular design allows for easy model swapping or upgrading as better AI technologies become available

### Future AI Enhancement Potential
The architecture supports several avenues for AI advancement:

- Fine-tuning the embedding model on wine-specific language
- Adding multi-modal capabilities to incorporate wine label images
- Implementing personalized recommendations based on user preference history
- Incorporating domain-specific wine knowledge graphs

This application demonstrates how multiple AI technologies can be integrated to create a practical, user-friendly application that brings expert-level wine knowledge to everyone through natural language interaction.


## Performance Optimization Guide

The embedding process can be resource-intensive, especially with large datasets. Here are tips for optimizing performance:

### For Faster Development/Testing
- **Use Data Sampling**: Enable the "Use data sample" option in the sidebar to work with a smaller subset of wines.
- **Adjust Sample Size**: Use the slider to find a balance between coverage and speed (1000-2000 wines is usually sufficient for testing).
- **Pre-compute Embeddings**: Generate embeddings offline and save them to a file:
  ```python
  from src.recommender import Recommender
  from src.utils import load_wine_dataset
  
  # Load data
  df = load_wine_dataset("data/wine_reviews.csv")
  
  # Create and fit recommender
  rec = Recommender()
  rec.fit(df)
  
  # Save embeddings for faster loading
  rec.save_embeddings("data/embeddings.npz")
  ```

### For Memory Optimization
- **Adjust Batch Size**: Lower the batch size slider in the sidebar if you encounter memory issues.
- **Recommended Settings**:
  - 8-16: For very limited memory environments (e.g., shared hosting)
  - 32: Good balance for most deployments
  - 64-128: For environments with ample memory

### Troubleshooting
If the app crashes during embedding computation:
1. Try using a smaller data sample
2. Reduce the batch size
3. Check for SSL certificate issues if deployed on Streamlit Cloud
4. Pre-compute embeddings locally and upload them to your deployment

## LLM Setup (optional)
To enable natural-language explanations from Gemini:

1. Get a Gemini AI Studio API key: https://aistudio.google.com/
2. Export it in your shell:
   ```bash
   export GEMINI_API_KEY="your_key_here"
    ```