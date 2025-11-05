"""Comprehensive unit tests for Sommelier class"""
import pytest
import pandas as pd
import os
from unittest.mock import Mock, patch, MagicMock
from src.sommelier import Sommelier
from src.recommender import Recommender


def make_dummy_df():
    """Create a small test dataset"""
    data = [
        {"title": "Sunny Pinot", "variety": "Pinot Noir", "country": "USA", 
         "description": "Light red with cherry notes.", "price": 18},
        {"title": "Big Cab", "variety": "Cabernet Sauvignon", "country": "France", 
         "description": "Full-bodied, great with steak.", "price": 28},
        {"title": "Ocean White", "variety": "Sauvignon Blanc", "country": "New Zealand", 
         "description": "Crisp, citrus, good with fish.", "price": 15},
    ]
    df = pd.DataFrame(data)
    df["text_for_embedding"] = df["description"]
    return df


class TestSommelierInitialization:
    """Test Sommelier initialization"""
    
    def test_initialization_without_api_key(self):
        """Test initialization without Google API key"""
        # Clear any existing API keys
        old_key = os.environ.get("GOOGLE_API_KEY")
        old_gemini_key = os.environ.get("GEMINI_API_KEY")
        if old_key:
            del os.environ["GOOGLE_API_KEY"]
        if old_gemini_key:
            del os.environ["GEMINI_API_KEY"]
        
        try:
            df = make_dummy_df()
            rec = Recommender()
            rec.fit(df, text_column="description")
            
            somm = Sommelier(rec)
            
            assert somm.recommender is not None
            assert somm.gemini is None  # Should be None without API key
            assert somm.google_api_key is None
        finally:
            # Restore keys
            if old_key:
                os.environ["GOOGLE_API_KEY"] = old_key
            if old_gemini_key:
                os.environ["GEMINI_API_KEY"] = old_gemini_key
    
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key_123"})
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_initialization_with_api_key(self, mock_model, mock_configure):
        """Test initialization with Google API key"""
        df = make_dummy_df()
        rec = Recommender()
        rec.fit(df, text_column="description")
        
        somm = Sommelier(rec)
        
        assert somm.google_api_key == "test_key_123"
        mock_configure.assert_called_once_with(api_key="test_key_123")
    
    def test_custom_gemini_model(self):
        """Test initialization with custom Gemini model"""
        df = make_dummy_df()
        rec = Recommender()
        rec.fit(df, text_column="description")
        
        somm = Sommelier(rec, gemini_model="gemini-1.5-pro")
        
        assert somm.gemini_model == "gemini-1.5-pro"


class TestSommelierFormatCandidates:
    """Test _format_candidates method"""
    
    def test_format_basic_candidates(self):
        """Test formatting basic wine candidates"""
        df = make_dummy_df()
        rec = Recommender()
        rec.fit(df, text_column="description")
        somm = Sommelier(rec)
        
        # Create mock results with similarity scores
        results_df = df.copy()
        results_df["_similarity"] = [0.9, 0.8, 0.7]
        
        candidates = somm._format_candidates(results_df)
        
        assert len(candidates) == 3
        assert candidates[0]["title"] == "Sunny Pinot"
        assert candidates[0]["variety"] == "Pinot Noir"
        assert candidates[0]["country"] == "USA"
        assert candidates[0]["price"] == 18
        assert candidates[0]["similarity"] == 0.9
    
    def test_format_candidates_with_missing_data(self):
        """Test formatting candidates with missing data"""
        df = pd.DataFrame({
            "title": ["Wine A"],
            "_similarity": [0.8]
        })
        
        rec = Recommender()
        rec.fit(make_dummy_df(), text_column="description")
        somm = Sommelier(rec)
        
        candidates = somm._format_candidates(df)
        
        assert candidates[0]["title"] == "Wine A"
        assert candidates[0]["variety"] == "Unknown"
        assert candidates[0]["country"] == "Unknown"
        assert candidates[0]["price"] is None
        assert candidates[0]["similarity"] == 0.8


class TestSommelierSafePrice:
    """Test _safe_price method"""
    
    def test_safe_price_valid_float(self):
        """Test safe price with valid float"""
        df = make_dummy_df()
        rec = Recommender()
        rec.fit(df, text_column="description")
        somm = Sommelier(rec)
        
        assert somm._safe_price(25.99) == 25.99
        assert somm._safe_price(10) == 10.0
    
    def test_safe_price_none(self):
        """Test safe price with None"""
        df = make_dummy_df()
        rec = Recommender()
        rec.fit(df, text_column="description")
        somm = Sommelier(rec)
        
        assert somm._safe_price(None) is None
    
    def test_safe_price_invalid(self):
        """Test safe price with invalid values"""
        df = make_dummy_df()
        rec = Recommender()
        rec.fit(df, text_column="description")
        somm = Sommelier(rec)
        
        assert somm._safe_price("invalid") is None
        assert somm._safe_price([]) is None


class TestSommelierBuildPrompt:
    """Test _build_prompt method"""
    
    def test_build_prompt_basic(self):
        """Test building basic prompt"""
        df = make_dummy_df()
        rec = Recommender()
        rec.fit(df, text_column="description")
        somm = Sommelier(rec)
        
        candidates = [
            {"title": "Test Wine", "variety": "Merlot", "country": "USA",
             "price": 20.0, "description": "Smooth red wine", "similarity": 0.9}
        ]
        
        prompt = somm._build_prompt("red wine for dinner", candidates)
        
        assert "red wine for dinner" in prompt
        assert "Test Wine" in prompt
        assert "Merlot" in prompt
        assert "USA" in prompt
        assert "$20" in prompt
        assert "sommelier" in prompt.lower()
    
    def test_build_prompt_multiple_candidates(self):
        """Test prompt with multiple candidates"""
        df = make_dummy_df()
        rec = Recommender()
        rec.fit(df, text_column="description")
        somm = Sommelier(rec)
        
        candidates = [
            {"title": "Wine A", "variety": "Pinot", "country": "USA",
             "price": 15.0, "description": "Light", "similarity": 0.9},
            {"title": "Wine B", "variety": "Cab", "country": "France",
             "price": 25.0, "description": "Bold", "similarity": 0.8},
            {"title": "Wine C", "variety": "Merlot", "country": "Italy",
             "price": 20.0, "description": "Smooth", "similarity": 0.7},
        ]
        
        prompt = somm._build_prompt("dinner wine", candidates)
        
        # Should include top 3
        assert "Wine A" in prompt
        assert "Wine B" in prompt
        assert "Wine C" in prompt
    
    def test_build_prompt_no_price(self):
        """Test prompt building when price is None"""
        df = make_dummy_df()
        rec = Recommender()
        rec.fit(df, text_column="description")
        somm = Sommelier(rec)
        
        candidates = [
            {"title": "Test Wine", "variety": "Merlot", "country": "USA",
             "price": None, "description": "Smooth", "similarity": 0.9}
        ]
        
        prompt = somm._build_prompt("wine", candidates)
        
        assert "Test Wine" in prompt
        assert "price varies" in prompt


class TestSommelierTemplateExplanation:
    """Test _template_explanation method"""
    
    def test_template_explanation_basic(self):
        """Test basic template explanation"""
        df = make_dummy_df()
        rec = Recommender()
        rec.fit(df, text_column="description")
        somm = Sommelier(rec)
        
        candidates = [
            {"title": "Sunny Pinot", "variety": "Pinot Noir", "country": "USA",
             "price": 18.0, "description": "Light red", "similarity": 0.9}
        ]
        
        explanation = somm._template_explanation("light red wine", candidates)
        
        assert "Sunny Pinot" in explanation
        assert "Pinot Noir" in explanation
        assert "USA" in explanation
        assert "$18" in explanation
        assert "light red wine" in explanation
    
    def test_template_explanation_no_candidates(self):
        """Test template explanation with no candidates"""
        df = make_dummy_df()
        rec = Recommender()
        rec.fit(df, text_column="description")
        somm = Sommelier(rec)
        
        explanation = somm._template_explanation("nonexistent wine", [])
        
        assert "No wines found" in explanation
        assert "nonexistent wine" in explanation
    
    def test_template_explanation_multiple_wines(self):
        """Test template with multiple wines"""
        df = make_dummy_df()
        rec = Recommender()
        rec.fit(df, text_column="description")
        somm = Sommelier(rec)
        
        candidates = [
            {"title": "Wine A", "variety": "Pinot", "country": "USA",
             "price": 15.0, "description": "Light", "similarity": 0.9},
            {"title": "Wine B", "variety": "Cab", "country": "France",
             "price": 25.0, "description": "Bold", "similarity": 0.8},
        ]
        
        explanation = somm._template_explanation("dinner wine", candidates)
        
        assert "Wine A" in explanation
        assert "Wine B" in explanation


class TestSommelierGenerateExplanation:
    """Test generate_explanation method"""
    
    def test_generate_explanation_without_gemini(self):
        """Test explanation generation without Gemini (fallback)"""
        df = make_dummy_df()
        rec = Recommender()
        rec.fit(df, text_column="description")
        
        # Ensure no API key
        old_key = os.environ.get("GOOGLE_API_KEY")
        if old_key:
            del os.environ["GOOGLE_API_KEY"]
        
        try:
            somm = Sommelier(rec)
            
            candidates = [
                {"title": "Test Wine", "variety": "Merlot", "country": "USA",
                 "price": 20.0, "description": "Smooth", "similarity": 0.9}
            ]
            
            explanation = somm.generate_explanation("red wine", candidates)
            
            assert isinstance(explanation, str)
            assert len(explanation) > 0
            assert "Test Wine" in explanation
        finally:
            if old_key:
                os.environ["GOOGLE_API_KEY"] = old_key
    
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"})
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_generate_explanation_with_gemini_success(self, mock_model_class, mock_configure):
        """Test explanation generation with Gemini success"""
        # Setup mocks
        mock_response = MagicMock()
        mock_response.text = "This is a great wine pairing for your needs."
        
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        df = make_dummy_df()
        rec = Recommender()
        rec.fit(df, text_column="description")
        somm = Sommelier(rec)
        
        candidates = [
            {"title": "Test Wine", "variety": "Merlot", "country": "USA",
             "price": 20.0, "description": "Smooth", "similarity": 0.9}
        ]
        
        explanation = somm.generate_explanation("red wine", candidates)
        
        assert explanation == "This is a great wine pairing for your needs."
    
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"})
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_generate_explanation_gemini_failure_fallback(self, mock_model_class, mock_configure):
        """Test fallback when Gemini fails"""
        # Setup mock to raise exception
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("API Error")
        mock_model_class.return_value = mock_model
        
        df = make_dummy_df()
        rec = Recommender()
        rec.fit(df, text_column="description")
        somm = Sommelier(rec)
        somm.gemini = mock_model
        
        candidates = [
            {"title": "Test Wine", "variety": "Merlot", "country": "USA",
             "price": 20.0, "description": "Smooth", "similarity": 0.9}
        ]
        
        explanation = somm.generate_explanation("red wine", candidates)
        
        # Should fall back to template
        assert isinstance(explanation, str)
        assert "Test Wine" in explanation
    
    def test_generate_explanation_empty_candidates(self):
        """Test explanation generation with no candidates"""
        df = make_dummy_df()
        rec = Recommender()
        rec.fit(df, text_column="description")
        somm = Sommelier(rec)
        
        explanation = somm.generate_explanation("nonexistent wine", [])
        
        assert "No wines found" in explanation
