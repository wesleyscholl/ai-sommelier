import os
import google.generativeai as genai
from typing import List, Dict, Optional
from .recommender import Recommender


class Sommelier:
    """
    Uses a Recommender to find wines and Gemini to generate natural explanations.
    Falls back to template if GEMINI_API_KEY is not set.
    """

    def __init__(self, recommender: Recommender, gemini_model: str = "gemini-1.5-flash"):
        self.recommender = recommender
        self.gemini_model = gemini_model
        self.GEMINI_api_key = os.environ.get("GEMINI_API_KEY")
        self.gemini = None
        if self.GEMINI_api_key:
            genai.configure(api_key=self.GEMINI_api_key)
            self.gemini = genai.GenerativeModel(self.gemini_model)

    def _format_candidates(self, df):
        candidates = []
        for _, row in df.iterrows():
            candidates.append(
                {
                    "title": row.get("title", ""),
                    "variety": row.get("variety", ""),
                    "country": row.get("country", ""),
                    "price": row.get("price", None),
                    "description": row.get("description", ""),
                    "similarity": float(row.get("_similarity", 0.0)),
                }
            )
        return candidates

    def generate_explanation(
        self, user_text: str, candidates: List[Dict], max_tokens: int = 300
    ) -> str:
        """
        If GEMINI_API_KEY is present, ask Gemini for explanations.
        Otherwise, fallback to templates.
        """
        if self.gemini:
            prompt = (
                "You are an expert sommelier. Write short, friendly wine recommendations.\n\n"
                f"User request: {user_text}\n\n"
                "Candidates:\n"
            )
            for i, c in enumerate(candidates, start=1):
                prompt += f"{i}. {c['title']} — {c['variety']} ({c.get('country','')}) — ${c.get('price','?')}\n   Notes: {c.get('description','')}\n\n"
            prompt += (
                "For each candidate, give a 1–2 sentence reason why it fits the request, "
                "including a brief tasting note. Keep each under 40 words."
            )

            try:
                response = self.gemini.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                return f"(Gemini generation failed: {e})\n\n" + self._template_explanation(user_text, candidates)
        else:
            return self._template_explanation(user_text, candidates)

    def _template_explanation(self, user_text: str, candidates: List[Dict]) -> str:
        parts = [f"Request: {user_text}\n"]
        for i, c in enumerate(candidates, start=1):
            title = c.get("title") or "Unknown wine"
            variety = c.get("variety") or ""
            price = c.get("price")
            price_str = f"${price:.0f}" if price is not None and not (price != price) else "unknown price"
            note = c.get("description", "")[:200]
            reason = f"{variety} with {note.split('.')[0]}" if variety else note.split(".")[0]
            parts.append(f"{i}. {title} ({variety}) — {price_str}\n   Why: {reason}\n")
        return "\n".join(parts)

    def recommend_and_explain(
        self,
        user_text: str,
        top_k: int = 3,
        price_min: Optional[float] = None,
        price_max: Optional[float] = None,
        variety: Optional[List[str]] = None,
    ) -> Dict:
        df = self.recommender.recommend(
            query=user_text,
            top_k=top_k,
            price_min=price_min,
            price_max=price_max,
            variety=variety,
        )
        candidates = self._format_candidates(df)
        explanation = self.generate_explanation(user_text, candidates)
        return {"candidates": candidates, "explanation": explanation}
