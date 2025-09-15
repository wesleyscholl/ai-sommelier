import numpy as np
import pandas as pd
from src.recommender import Recommender

class FakeModel:
    def __init__(self, dim=8):
        self._dim = dim
    def get_sentence_embedding_dimension(self):
        return self._dim
    def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
        # produce deterministic embeddings based on hash of text
        if isinstance(texts, str):
            texts = [texts]
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i,t in enumerate(texts):
            h = abs(hash(t))
            for j in range(self._dim):
                out[i,j] = ((h >> (j*3)) & 0xFF) / 255.0
        return out


def make_dummy_df():
    data = [
        {"title": "Sunny Pinot", "variety": "Pinot Noir", "country": "USA", "description": "Light red with cherry and raspberry notes.", "price": 18},
        {"title": "Big Cab", "variety": "Cabernet Sauvignon", "country": "France", "description": "Full-bodied, blackcurrant, tannic; great with steak.", "price": 28},
        {"title": "Ocean White", "variety": "Sauvignon Blanc", "country": "New Zealand", "description": "Crisp, citrus, grassy; good with fish.", "price": 15},
        {"title": "Velvet Malbec", "variety": "Malbec", "country": "Argentina", "description": "Dark fruit, smooth, spicy finish.", "price": 20},
    ]
    return pd.DataFrame(data)

if __name__ == '__main__':
    df = make_dummy_df()
    rec = Recommender()
    # inject fake model
    rec.model = FakeModel(dim=8)
    print('Fitting recommender to dummy data (mock model)...')
    rec.fit(df, text_column='description', batch_size=2)
    print('Fitting complete. Running recommend for "wine for steak"')
    res = rec.recommend('wine for steak', top_k=2)
    print(res[['title','_similarity']])
