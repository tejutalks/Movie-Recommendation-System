import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load dataset
movies = pd.read_csv("movies.csv")

# Create embeddings
embeddings = model.encode(movies["description"].tolist())

def recommend(movie_name, top_k=3):
    if movie_name not in movies["title"].values:
        return "Movie not found!"

    idx = movies[movies["title"] == movie_name].index[0]
    similarity = cosine_similarity([embeddings[idx]], embeddings)[0]

    similar_indices = similarity.argsort()[-top_k-1:-1][::-1]

    return movies.iloc[similar_indices]["title"].values

# Test
print("Recommendations for Inception:")
print(recommend("Inception"))