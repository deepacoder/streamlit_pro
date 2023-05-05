import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import requests

def load_lottieurl(url="https://assets6.lottiefiles.com/packages/lf20_khzniaya.json"):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = "https://assets6.lottiefiles.com/packages/lf20_khzniaya.json"
lottie_json = load_lottieurl(lottie_url)
st_lottie(lottie_json)

st.title("Recommender Systems")
 
st.write("""
### Project description
Creating movie recommendations based on Machine Learning using Item Based Collaborative filtering and User Based Collaborative filtering. """)

import os
import pickle
import pandas as pd

def get_movie_recommendation(movies, ratings, movie_name, n):
    #datacleaning
    for column in movies:
        if column == 'title':
            movies['title'] = movies.title.str.lower()
            movies[['name', 'year']] = movies['title'].str.split(' \(', 1, expand=True)
            movies['year'] = movies['year'].str.replace("\)", "", regex=True) 
        if column == 'genre':
            movies['genres'] = movies['genres'].str.replace("[^a-zA-Z0-9_]", ",", regex=True)
    #merge dataframes
    movies_ratings_df = pd.merge(movies,ratings, on = 'movieId')
    #keep necessary columns
    movies_ratings_df = movies_ratings_df[['movieId','userId','name','year','genres', 'rating']]
    #getaverageratings
    avgratings_df = pd.DataFrame(data=movies_ratings_df.groupby('name')['rating'].mean())
    avgratings_df['count'] = movies_ratings_df.groupby('name')['rating'].count()
    #create pivot table
    user_movie_matrix = movies_ratings_df.pivot_table(index='userId', columns ='name',     values='rating')
    #get similar movies
    movie_to_compare = user_movie_matrix.loc[:, movie_name]
    similar_movies = user_movie_matrix.corrwith(movie_to_compare)
    corr_movies = pd.DataFrame(data=similar_movies, columns=['correlation'])
    corr_movies.dropna(inplace=True)
    corr_movies = corr_movies.join(avgratings_df['count'])
    top10_corr = corr_movies[corr_movies['correlation']>=0.8].sort_values('count',       ascending=False).head(n)
    return top10_corr

# some code to get user input and data
movie_name = st.text_input("Enter a movie title")
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
n =  st.number_input("Enter the number of recommendations", min_value=1, max_value=10, value=5, step=1)


# use the get_movie_recommendation function to get recommendations
recommendations = get_movie_recommendation(movies, ratings, movie_name, n)
    

# display the recommendations
st.write("Here are some movies you might like:")
st.table(recommendations)