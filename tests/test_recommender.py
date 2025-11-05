"""Comprehensive unit tests for Recommender class"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from src.recommender import Recommender
from src.utils import load_wine_dataset


def make_dummy_df():
    """Create a small test dataset"""
    data = [
        {"title": "Sunny Pinot", "variety": "Pinot Noir", "country": "USA", "description": "Light red with cherry and raspberry notes.", "price": 18},
        {"title": "Big Cab", "variety": "Cabernet Sauvignon", "country": "France", "description": "Full-bodied, blackcurrant, tannic; great with steak.", "price": 28},
        {"title": "Ocean White", "variety": "Sauvignon Blanc", "country": "New Zealand", "description": "Crisp, citrus, grassy; good with fish.", "price": 15},
        {"title": "Velvet Malbec", "variety": "Malbec", "country": "Argentina", "description": "Dark fruit, smooth, spicy finish.", "price": 20},
    ]
    df = pd.DataFrame(data)
    # Add text_for_embedding column
    df["text_for_embedding"] = df["description"]
    return df


class TestRecommenderInitialization:
    """Test Recommender initialization"""
    
    def test_default_initialization(self):
        """Test default recommender initialization"""
        rec = Recommender()
        assert rec.model_name == "all-MiniLM-L6-v2"
        assert rec.model is not None
        assert rec.embeddings is None
        assert rec.df is None
        assert rec.nn is None
    
    def test_custom_model_initialization(self):
        """Test initialization with custom model"""
        rec = Recommender(model_name="all-MiniLM-L6-v2")
        assert rec.model_name == "all-MiniLM-L6-v2"


class TestRecommenderFit:
    """Test Recommender fit functionality"""
    
    def test_fit_basic(self):
        """Test basic fit operation"""
        df = make_dummy_df()
        rec = Recommender()
        rec.fit(df, text_column="description")
        
        assert rec.embeddings is not None
        assert rec.df is not None
        assert rec.nn is not None
        assert len(rec.embeddings) == len(df)
        assert rec.embeddings.shape[1] > 0  # Has embedding dimensions
    
    def test_fit_with_custom_text_column(self):
        """Test fit with custom text column"""
        df = make_dummy_df()
        rec = Recommender()
        rec.fit(df, text_column="text_for_embedding")
        
        assert rec.embeddings is not None
        assert len(rec.embeddings) == 4
    
    def test_fit_with_empty_dataframe(self):
        """Test that fit raises error with empty dataframe"""
        df = pd.DataFrame({"text_for_embedding": []})
        rec = Recommender()
        
        with pytest.raises(ValueError, match="Dataset is empty"):
            rec.fit(df, text_column="text_for_embedding")
    
    def test_fit_with_batch_processing(self):
        """Test fit with custom batch size"""
        df = make_dummy_df()
        rec = Recommender()
        rec.fit(df, text_column="description", batch_size=2)
        
        assert len(rec.embeddings) == len(df)
    
    def test_fit_with_progress_callback(self):
        """Test fit with progress callback"""
        df = make_dummy_df()
        rec = Recommender()
        
        progress_calls = []
        def callback(progress, message):
            progress_calls.append((progress, message))
        
        rec.fit(df, text_column="description", progress_callback=callback)
        
        assert len(progress_calls) > 0
        # Check that progress increases
        assert progress_calls[-1][0] == 1.0


class TestRecommenderRecommend:
    """Test Recommender recommend functionality"""
    
    def test_recommend_basic(self):
        """Test basic recommendation"""
        df = make_dummy_df()
        rec = Recommender()
        rec.fit(df, text_column="description")
        
        # Query for steak should favor Big Cab
        res = rec.recommend("wine for steak", top_k=2)
        assert len(res) <= 2
        assert "_similarity" in res.columns
        titles = res["title"].tolist()
        assert any("Cab" in str(x) or "Cabernet" in str(x) for x in titles)
    
    def test_recommend_without_fit(self):
        """Test that recommend raises error without fit"""
        rec = Recommender()
        
        with pytest.raises(RuntimeError, match="not fitted"):
            rec.recommend("test query")
    
    def test_recommend_with_price_filter(self):
        """Test recommendation with price filtering"""
        df = make_dummy_df()
        rec = Recommender()
        rec.fit(df, text_column="description")
        
        # Filter for wines under $20
        res = rec.recommend("light white for fish", top_k=5, price_max=16)
        
        assert len(res) > 0
        assert all(res["price"] <= 16)
        assert any("Ocean White" in r for r in res["title"])
    
    def test_recommend_with_price_range(self):
        """Test recommendation with price range"""
        df = make_dummy_df()
        rec = Recommender()
        rec.fit(df, text_column="description")
        
        res = rec.recommend("red wine", top_k=5, price_min=17, price_max=25)
        
        assert all((res["price"] >= 17) & (res["price"] <= 25))
    
    def test_recommend_with_variety_filter(self):
        """Test recommendation with variety filtering"""
        df = make_dummy_df()
        rec = Recommender()
        rec.fit(df, text_column="description")
        
        res = rec.recommend("red wine", top_k=5, variety=["Pinot Noir"])
        
        assert len(res) > 0
        assert all("pinot" in str(v).lower() for v in res["variety"])
    
    def test_recommend_with_multiple_varieties(self):
        """Test recommendation with multiple variety filters"""
        df = make_dummy_df()
        rec = Recommender()
        rec.fit(df, text_column="description")
        
        res = rec.recommend("red wine", top_k=5, variety=["Pinot Noir", "Malbec"])
        
        assert len(res) > 0
        assert all(any(v.lower() in str(row_v).lower() for v in ["pinot", "malbec"]) 
                  for row_v in res["variety"])
    
    def test_recommend_similarity_scores(self):
        """Test that similarity scores are included and sorted"""
        df = make_dummy_df()
        rec = Recommender()
        rec.fit(df, text_column="description")
        
        res = rec.recommend("fruity red wine", top_k=3)
        
        assert "_similarity" in res.columns
        assert all(res["_similarity"] >= 0)
        assert all(res["_similarity"] <= 1)
        # Check that results are sorted by similarity (descending)
        similarities = res["_similarity"].tolist()
        assert similarities == sorted(similarities, reverse=True)
    
    def test_recommend_top_k_limit(self):
        """Test that top_k parameter limits results"""
        df = make_dummy_df()
        rec = Recommender()
        rec.fit(df, text_column="description")
        
        res = rec.recommend("wine", top_k=2)
        assert len(res) <= 2


class TestRecommenderSaveLoad:
    """Test Recommender save/load functionality"""
    
    def test_save_embeddings(self):
        """Test saving embeddings to file"""
        df = make_dummy_df()
        rec = Recommender()
        rec.fit(df, text_column="description")
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            try:
                rec.save_embeddings(f.name)
                assert os.path.exists(f.name)
                assert os.path.getsize(f.name) > 0
            finally:
                os.unlink(f.name)
    
    def test_save_without_fit(self):
        """Test that save raises error without fit"""
        rec = Recommender()
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            try:
                with pytest.raises(RuntimeError, match="No embeddings to save"):
                    rec.save_embeddings(f.name)
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)
    
    def test_load_embeddings(self):
        """Test loading previously saved embeddings"""
        df = make_dummy_df()
        rec1 = Recommender()
        rec1.fit(df, text_column="description")
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            try:
                rec1.save_embeddings(f.name)
                
                # Load in new recommender
                rec2 = Recommender()
                rec2.load_embeddings(f.name)
                
                assert rec2.embeddings is not None
                assert rec2.df is not None
                assert rec2.nn is not None
                assert len(rec2.embeddings) == len(df)
                
                # Test that loaded recommender works
                res = rec2.recommend("red wine", top_k=2)
                assert len(res) > 0
            finally:
                os.unlink(f.name)
    
    def test_save_load_roundtrip(self):
        """Test that save/load preserves functionality"""
        df = make_dummy_df()
        rec1 = Recommender()
        rec1.fit(df, text_column="description")
        
        res1 = rec1.recommend("fruity wine", top_k=2)
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            try:
                rec1.save_embeddings(f.name)
                
                rec2 = Recommender()
                rec2.load_embeddings(f.name)
                res2 = rec2.recommend("fruity wine", top_k=2)
                
                # Results should be similar (same order)
                assert list(res1["title"]) == list(res2["title"])
            finally:
                os.unlink(f.name)


class TestRecommenderEdgeCases:
    """Test edge cases and error handling"""
    
    def test_single_wine_dataset(self):
        """Test with dataset containing single wine"""
        df = pd.DataFrame({
            "title": ["Single Wine"],
            "variety": ["Merlot"],
            "country": ["USA"],
            "description": ["Good wine"],
            "price": [20],
            "text_for_embedding": ["Good wine"]
        })
        
        rec = Recommender()
        rec.fit(df, text_column="text_for_embedding")
        res = rec.recommend("wine", top_k=1)
        
        assert len(res) == 1
        assert res.iloc[0]["title"] == "Single Wine"
    
    def test_null_price_values(self):
        """Test handling of null price values"""
        df = make_dummy_df()
        df.loc[0, "price"] = np.nan
        
        rec = Recommender()
        rec.fit(df, text_column="description")
        res = rec.recommend("wine", top_k=4, price_min=10)
        
        # Should filter out null prices when price filter applied
        assert all(res["price"].notna())
    
    def test_empty_query(self):
        """Test recommendation with empty query"""
        df = make_dummy_df()
        rec = Recommender()
        rec.fit(df, text_column="description")
        
        res = rec.recommend("", top_k=2)
        assert len(res) > 0  # Should still return results
    
    def test_variety_filter_no_matches(self):
        """Test variety filter with no matching wines"""
        df = make_dummy_df()
        rec = Recommender()
        rec.fit(df, text_column="description")
        
        res = rec.recommend("wine", top_k=5, variety=["Riesling"])
        
        assert len(res) == 0  # No Riesling in test data
