# 01_data_explore

This markdown file gives quick commands to explore the dataset in a notebook.

Suggested steps (run in Jupyter or convert to a notebook):

```python
from src.utils import load_wine_dataset
df = load_wine_dataset("data/wine_reviews.csv")
df.head()
df.description.str.len().describe()
df.variety.value_counts().head(30)
```
    
Check price distribution:

```python
import matplotlib.pyplot as plt
plt.hist(df.price.dropna(), bins=50)
plt.show()
```

Inspect some example tasting notes:

```python
for i, row in df.sample(5).iterrows():
    print(row.title, row.variety, row.price)
    print(row.description[:300], "\n---\n")
```