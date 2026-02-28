import streamlit as st
import pandas as pd
import ast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎬 Movie Recommendation System")

# Load datasets
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge datasets
movies = movies.merge(credits, on="title")

# Keep important columns
movies = movies[['movie_id','title','overview','genres','cast','crew']]
movies.dropna(inplace=True)

# Convert stringified JSON to list
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: convert(x)[:3])

def fetch_director(text):
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            return i['name']
    return ""

movies['crew'] = movies['crew'].apply(fetch_director)

# Combine all features
movies['tags'] = movies['overview'] + " " + movies['genres'].apply(lambda x: " ".join(x)) + " " + movies['cast'].apply(lambda x: " ".join(x)) + " " + movies['crew']

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Generate embeddings
@st.cache_data
def get_embeddings():
    return model.encode(movies['tags'].tolist())

embeddings = get_embeddings()

# Recommendation function
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    similarity = cosine_similarity([embeddings[index]], embeddings)[0]
    distances = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]].title for i in distances]

# User selection
selected_movie = st.selectbox("Select a movie", movies['title'].values)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    for movie in recommendations:
        st.write("👉", movie)