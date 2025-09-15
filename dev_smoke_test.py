from src.recommender import Recommender
import pandas as pd

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
    print('Fitting recommender to dummy data...')
    rec.fit(df, text_column='description', batch_size=2)
    print('Fitting complete. Running recommend for "wine for steak"')
    res = rec.recommend('wine for steak', top_k=2)
    print(res[['title','_similarity']])
