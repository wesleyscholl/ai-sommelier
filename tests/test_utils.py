"""Comprehensive unit tests for utils.py"""
import pytest
import pandas as pd
import numpy as np
from src.utils import load_wine_dataset, top_k_indices
import tempfile
import os


class TestLoadWineDataset:
    """Test suite for load_wine_dataset function"""
    
    def test_load_basic_dataset(self):
        """Test loading a properly formatted dataset"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('title,variety,country,description,price\n')
            f.write('Wine A,Pinot Noir,USA,Light red wine,25.99\n')
            f.write('Wine B,Chardonnay,France,Crisp white wine,18.50\n')
            f.name
            
        try:
            df = load_wine_dataset(f.name)
            
            assert len(df) == 2
            assert 'title' in df.columns
            assert 'variety' in df.columns
            assert 'country' in df.columns
            assert 'description' in df.columns
            assert 'price' in df.columns
            assert 'text_for_embedding' in df.columns
            
            # Check text_for_embedding format
            assert 'Wine A' in df.iloc[0]['text_for_embedding']
            assert 'Pinot Noir' in df.iloc[0]['text_for_embedding']
            assert 'USA' in df.iloc[0]['text_for_embedding']
            assert 'Light red wine' in df.iloc[0]['text_for_embedding']
        finally:
            os.unlink(f.name)
    
    def test_column_name_standardization(self):
        """Test that alternative column names are standardized"""
        # Test 'wine' -> 'title'
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('wine,variety,region,description,price\n')
            f.write('Test Wine,Merlot,California,Full bodied,30\n')
            f.name
            
        try:
            df = load_wine_dataset(f.name)
            assert 'title' in df.columns
            assert df.iloc[0]['title'] == 'Test Wine'
            assert 'country' in df.columns
            assert df.iloc[0]['country'] == 'California'
        finally:
            os.unlink(f.name)
    
    def test_missing_price_column(self):
        """Test handling when price column is missing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('title,variety,country,description\n')
            f.write('Wine C,Syrah,Australia,Bold red\n')
            f.name
            
        try:
            df = load_wine_dataset(f.name)
            assert 'price' in df.columns
            assert pd.isna(df.iloc[0]['price'])
        finally:
            os.unlink(f.name)
    
    def test_missing_variety_column(self):
        """Test handling when variety column is missing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('title,country,description,price\n')
            f.write('Wine D,Italy,Sparkling wine,20\n')
            f.name
            
        try:
            df = load_wine_dataset(f.name)
            assert 'variety' in df.columns
            assert df.iloc[0]['variety'] == 'Unknown'
        finally:
            os.unlink(f.name)
    
    def test_price_conversion_to_numeric(self):
        """Test that price values are converted to numeric"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('title,variety,country,description,price\n')
            f.write('Wine E,Riesling,Germany,Sweet white,25.50\n')
            f.write('Wine F,Pinot Grigio,Italy,Dry white,invalid\n')
            f.name
            
        try:
            df = load_wine_dataset(f.name)
            assert pd.api.types.is_numeric_dtype(df['price'])
            assert df.iloc[0]['price'] == 25.50
            assert pd.isna(df.iloc[1]['price'])  # invalid price becomes NaN
        finally:
            os.unlink(f.name)
    
    def test_empty_description_filtering(self):
        """Test that rows with empty descriptions are dropped"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('title,variety,country,description,price\n')
            f.write('Wine G,Merlot,USA,Good wine,20\n')
            f.write('Wine H,Cabernet,France,,25\n')
            f.name
            
        try:
            df = load_wine_dataset(f.name)
            assert len(df) == 1
            assert df.iloc[0]['title'] == 'Wine G'
        finally:
            os.unlink(f.name)
    
    def test_text_for_embedding_with_nulls(self):
        """Test text_for_embedding creation with null values"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('title,variety,country,description,price\n')
            f.write('Wine I,,USA,Smooth red,30\n')
            f.name
            
        try:
            df = load_wine_dataset(f.name)
            text = df.iloc[0]['text_for_embedding']
            assert 'Wine I' in text
            assert 'USA' in text
            assert 'Smooth red' in text
            # Should handle null variety gracefully
            assert ' |  | ' in text or text.count('|') >= 2
        finally:
            os.unlink(f.name)


class TestTopKIndices:
    """Test suite for top_k_indices function"""
    
    def test_basic_top_k(self):
        """Test basic top-k selection"""
        similarities = np.array([0.1, 0.9, 0.3, 0.7, 0.5])
        k = 3
        indices = top_k_indices(similarities, k)
        
        assert len(indices) == k
        # Should include index 1 (0.9), 3 (0.7), and 4 (0.5)
        assert 1 in indices
        assert 3 in indices
        assert 4 in indices
    
    def test_k_equals_length(self):
        """Test when k equals array length"""
        similarities = np.array([0.5, 0.2, 0.8, 0.1])
        k = 4
        indices = top_k_indices(similarities, k)
        
        assert len(indices) == 4
        # Should return all indices sorted by similarity
        expected_order = [2, 0, 1, 3]  # descending by similarity
        assert list(indices) == expected_order
    
    def test_k_greater_than_length(self):
        """Test when k is greater than array length"""
        similarities = np.array([0.3, 0.7, 0.5])
        k = 5
        indices = top_k_indices(similarities, k)
        
        assert len(indices) == 3  # Should return all available
        expected_order = [1, 2, 0]
        assert list(indices) == expected_order
    
    def test_single_element(self):
        """Test with single element array"""
        similarities = np.array([0.5])
        k = 1
        indices = top_k_indices(similarities, k)
        
        assert len(indices) == 1
        assert indices[0] == 0
    
    def test_all_equal_values(self):
        """Test with all equal similarity values"""
        similarities = np.array([0.5, 0.5, 0.5, 0.5])
        k = 2
        indices = top_k_indices(similarities, k)
        
        assert len(indices) == k
        # Any 2 indices should be fine since all values are equal
        assert all(i >= 0 and i < 4 for i in indices)
    
    def test_negative_similarities(self):
        """Test with negative similarity values"""
        similarities = np.array([-0.5, 0.2, -0.1, 0.0])
        k = 2
        indices = top_k_indices(similarities, k)
        
        assert len(indices) == k
        assert 1 in indices  # 0.2
        assert 3 in indices  # 0.0
    
    def test_large_array(self):
        """Test with larger array to verify performance"""
        np.random.seed(42)
        similarities = np.random.rand(1000)
        k = 10
        indices = top_k_indices(similarities, k)
        
        assert len(indices) == k
        # Verify these are actually top-k
        top_values = similarities[indices]
        assert all(top_values >= np.partition(similarities, -k)[-k])
