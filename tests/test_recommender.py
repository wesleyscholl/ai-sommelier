import pandas as pd
from src.recommender import Recommender
from src.utils import load_wine_dataset
import tempfile
import numpy as np

def make_dummy_df():
    data = [
        {"title": "Sunny Pinot", "variety": "Pinot Noir", "country": "USA", "description": "Light red with cherry and raspberry notes.", "price": 18},
        {"title": "Big Cab", "variety": "Cabernet Sauvignon", "country": "France", "description": "Full-bodied, blackcurrant, tannic; great with steak.", "price": 28},
        {"title": "Ocean White", "variety": "Sauvignon Blanc", "country": "New Zealand", "description": "Crisp, citrus, grassy; good with fish.", "price": 15},
        {"title": "Velvet Malbec", "variety": "Malbec", "country": "Argentina", "description": "Dark fruit, smooth, spicy finish.", "price": 20},
    ]
    return pd.DataFrame(data)

def test_recommend_basic():
    df = make_dummy_df()
    rec = Recommender()
    rec.fit(df, text_column="description")
    # query for steak should favor "Big Cab"
    res = rec.recommend("wine for steak", top_k=2)
    titles = [t for t in res["title"].tolist()]
    assert any("Cab" in str(x) or "Cabernet" in str(x) for x in titles), "Expected Cabernet in top results"

def test_price_filtering():
    df = make_dummy_df()
    rec = Recommender()
    rec.fit(df, text_column="description")
    res = rec.recommend("light white for fish", top_k=5, price_max=16)
    # Expect Ocean White to be present (price 15)
    assert any(r for r in res["title"] if "Ocean White" in r), "Ocean White should be in results with price_max=16"
