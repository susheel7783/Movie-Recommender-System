import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("top10K-TMDB-movies.csv")
data = data[['id', 'title', 'genre', 'overview']]
data['tags'] = "The genre of movie is " + data['genre'] + " and a short overview of it is " + data['overview']
data = data.drop(columns=['genre', 'overview'])

cv = CountVectorizer(max_features=10000, stop_words='english')
vector = cv.fit_transform(data['tags'].values.astype('U')).toarray()

similarity = cosine_similarity(vector)

def recommend(movie):
    index = data[data['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movies = []
    for i in distances[1:6]:
        recommended_movies.append(data.iloc[i[0]].title)
    return recommended_movies

st.title('Movie Recommendation System')
st.write("Select a movie from the suggestions:")

movie_name = st.selectbox('Select a Movie', data['title'].tolist())

if movie_name:
    st.write(f"Recommendations based on {movie_name}:")
    suggestions = recommend(movie_name)
    for movie in suggestions:
        st.write(movie)