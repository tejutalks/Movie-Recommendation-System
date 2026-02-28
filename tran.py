from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Load pretrained model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample movie dataset
data = {
    "title": [
        "Inception",
        "Interstellar",
        "The Dark Knight",
        "Titanic",
        "The Avengers"
    ],
    "description": [
        "A thief who steals corporate secrets using dream-sharing technology.",
        "A team travels through a wormhole in space to save humanity.",
        "Batman faces the Joker in Gotham City.",
        "A love story on the Titanic ship.",
        "Superheroes team up to save the world."
    ]
}

df = pd.DataFrame(data)

# Convert descriptions into embeddings
embeddings = model.encode(df["description"].tolist())

# Function to recommend movies
def recommend(movie_name, top_n=3):
    if movie_name not in df["title"].values:
        return "Movie not found!"
    
    index = df[df["title"] == movie_name].index[0]
    
    similarity_scores = cosine_similarity(
        [embeddings[index]], embeddings
    )[0]
    
    similar_indices = np.argsort(similarity_scores)[::-1][1:top_n+1]
    
    return df["title"].iloc[similar_indices].tolist()


# Example usage
print("Recommendations for Inception:")
print(recommend("Inception"))