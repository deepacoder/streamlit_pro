#!/usr/bin/env python
# coding: utf-8

# # Prepare data

# In[1]:


import pandas as pd


# In[2]:


movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')


# # Data Cleaning

# In[3]:


#Convert to lower case
movies['title'] = movies.title.str.lower()


# In[4]:


#split title name into name and year
movies[['name', 'year']] = movies['title'].str.split(' \(', 1, expand=True)


# In[5]:


movies.head(3)


# In[6]:


#Replace Special characters
movies['year'] = movies['year'].str.replace("\)", "", regex=True) 
movies['year'] 


# In[7]:


movies['genres'] = movies['genres'].str.replace("[^a-zA-Z0-9_]", ",", regex=True)
movies['genres']


# In[8]:


#Ratings Df whats in it
ratings.head(3)


# In[9]:


#Merge the two Dataframes
movies_ratings_df = pd.merge(movies,ratings, on = 'movieId')
movies_ratings_df


# In[10]:


#Keep relevant columns
movies_ratings_df = movies_ratings_df[['movieId','userId','name','year','genres', 'rating']]


# In[11]:


movies_ratings_df


# In[12]:


import matplotlib as plt
import seaborn as sns


# In[13]:


#Groupby movies with mean of ratings and sort values
movies_ratings_df.groupby(['name'])['rating'].mean().sort_values(ascending=False)


# In[14]:


#Groupby and count the number of ratings per movie
movies_ratings_df.groupby('name')['rating'].count().sort_values(ascending=False)


# In[15]:


#New dataframe with name and avg ratings
avgratings_df = pd.DataFrame(data=movies_ratings_df.groupby('name')['rating'].mean())


# In[16]:


#No.of ratings per movie add column
avgratings_df['count'] = movies_ratings_df.groupby('name')['rating'].count()


# In[17]:


avgratings_df


# In[18]:


avgratings_df.hist(bins=70)


# In[19]:


import seaborn as sns
sns.scatterplot(data=avgratings_df, x='rating', y='count')


# # Create user movie matrix

# In[20]:


user_movie_matrix= movies_ratings_df.pivot_table(index='userId', columns ='name', values='rating')
user_movie_matrix


# In[21]:


#top 10 movies based on ratings and counts
avgratings_df.sort_values('count', ascending=False).head(10)


# In[22]:


#Forrest Gump is the highest rated movie with most number of high ratings


# In[23]:


jurassic_park_rating = user_movie_matrix['jurassic park']
jurassic_park_rating


# In[24]:


similar_to_jurassic_park = user_movie_matrix.corrwith(jurassic_park_rating)
similar_to_jurassic_park.head(5)


# In[25]:


corr_jurrpark = pd.DataFrame(data=similar_to_jurassic_park, columns=['correlation'])
corr_jurrpark.dropna(inplace=True)
corr_jurrpark.head()


# In[26]:


corr_jurrpark = corr_jurrpark.join(avgratings_df['count'])


# In[27]:


corr_jurrpark.head()


# In[28]:


movies_like_jurrpark = corr_jurrpark[corr_jurrpark['count']> 50].sort_values('correlation', ascending=False)
movies_like_jurrpark.head(6)


# 
# # sort by high correlation
# 

# In[29]:


movies_like_jurrpark_corr = corr_jurrpark[corr_jurrpark['correlation']>=0.8].sort_values('count', ascending=False)
movies_like_jurrpark_corr.head(10)


# In[41]:


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


# In[42]:


get_movie_recommendation(movies, ratings, 'zootopia', 10)


# In[ ]:





# In[43]:


# store the trained pipeline
import pickle
with open('recommendersys.pkl', 'wb') as file:
    pickle.dump(get_movie_recommendation, file)


# In[ ]:





# In[ ]:





# In[35]:





# In[ ]:




