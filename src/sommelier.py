import os
import logging
from typing import List, Dict, Optional
from .recommender import Recommender

# Configure logging for production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Sommelier:
    """
    Production-ready sommelier with enhanced error handling and fallback systems.
    """

    def __init__(self, recommender: Recommender, gemini_model: str = "gemini-1.5-flash"):
        self.recommender = recommender
        self.gemini_model = gemini_model
        self.google_api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.gemini = None
        
        if self.google_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.google_api_key)
                self.gemini = genai.GenerativeModel(self.gemini_model)
                logger.info("Gemini AI initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}")
                self.gemini = None

    def _format_candidates(self, df):
        """Format recommendation results with enhanced data handling."""
        candidates = []
        for _, row in df.iterrows():
            # Robust data extraction with defaults
            candidates.append({
                "title": str(row.get("title", "Unknown Wine")),
                "variety": str(row.get("variety", "Unknown")),
                "country": str(row.get("country", "Unknown")),
                "price": self._safe_price(row.get("price")),
                "description": str(row.get("description", "")),
                "similarity": float(row.get("_similarity", 0.0)),
            })
        return candidates

    def _safe_price(self, price):
        """Safely handle price data with various formats."""
        if price is None or (hasattr(price, '__iter__') and len(str(price)) == 0):
            return None
        try:
            return float(price)
        except (ValueError, TypeError):
            return None

    def generate_explanation(
        self, user_text: str, candidates: List[Dict], max_tokens: int = 300
    ) -> str:
        """
        Generate sommelier explanations with robust fallback handling.
        """
        if self.gemini and candidates:
            try:
                prompt = self._build_prompt(user_text, candidates)
                response = self.gemini.generate_content(prompt)
                if response and response.text:
                    return response.text.strip()
                else:
                    logger.warning("Gemini returned empty response")
            except Exception as e:
                logger.error(f"Gemini generation failed: {e}")
        
        # Always provide template fallback
        return self._template_explanation(user_text, candidates)

    def _build_prompt(self, user_text: str, candidates: List[Dict]) -> str:
        """Build optimized prompt for Gemini."""
        prompt = (
            "You are a professional sommelier. Provide concise, expert wine recommendations.\n\n"
            f"Customer request: {user_text}\n\n"
            "Recommended wines:\n"
        )
        
        for i, wine in enumerate(candidates[:3], 1):  # Limit to top 3 for prompt efficiency
            price_str = f"${wine['price']:.0f}" if wine['price'] else "price varies"
            prompt += (
                f"{i}. {wine['title']} - {wine['variety']} from {wine['country']} ({price_str})\n"
                f"   Notes: {wine['description'][:150]}...\n\n"
            )
        
        prompt += (
            "For each wine, explain in 1-2 sentences why it matches the request. "
            "Include specific tasting notes and pairing suggestions. Keep responses under 40 words each."
        )
        return prompt

    def _template_explanation(self, user_text: str, candidates: List[Dict]) -> str:
        """Enhanced template explanation with better formatting."""
        if not candidates:
            return f"No wines found matching '{user_text}'. Try adjusting your criteria or being more specific."
        
        parts = [f"Based on your request for '{user_text}', here are my recommendations:\n"]
        
        for i, wine in enumerate(candidates, 1):
            title = wine.get("title", "Unknown Wine")
            variety = wine.get("variety", "")
            country = wine.get("country", "")
            price = wine.get("price")
            
            price_str = f"${price:.0f}" if price else "price not listed"
            location = f" from {country}" if country != "Unknown" else ""
            
            # Extract key descriptors from description
            description = wine.get("description", "")
            key_notes = self._extract_key_notes(description)
            
            parts.append(
                f"{i}. **{title}** ({variety}{location}) - {price_str}\n"
                f"   {key_notes}\n"
            )
        
        return "\n".join(parts)

    def _extract_key_notes(self, description: str) -> str:
        """Extract key tasting notes from wine description."""
        if not description:
            return "A delightful wine with unique character."
        
        # Simple extraction of first sentence or key phrases
        sentences = description.split('.')
        first_sentence = sentences[0].strip() if sentences else description
        
        # Limit length and ensure it ends properly
        if len(first_sentence) > 100:
            first_sentence = first_sentence[:97] + "..."
        elif not first_sentence.endswith('.'):
            first_sentence += "."
            
        return first_sentence

    def recommend_and_explain(
        self,
        user_text: str,
        top_k: int = 3,
        price_min: Optional[float] = None,
        price_max: Optional[float] = None,
        variety: Optional[List[str]] = None,
    ) -> Dict:
        """
        Main recommendation method with comprehensive error handling.
        """
        try:
            # Get recommendations
            df = self.recommender.recommend(
                query=user_text,
                top_k=top_k,
                price_min=price_min,
                price_max=price_max,
                variety=variety,
            )
            
            # Format results
            candidates = self._format_candidates(df)
            
            # Generate explanation
            explanation = self.generate_explanation(user_text, candidates)
            
            return {
                "candidates": candidates,
                "explanation": explanation,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Recommendation failed: {e}")
            return {
                "candidates": [],
                "explanation": f"I apologize, but I encountered an error while searching for wines: {str(e)}",
                "status": "error"
            }
