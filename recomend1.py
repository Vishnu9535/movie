import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

credits=pd.read_csv('credits.csv')
movies=pd.read_csv('movies.csv')
credits.columns=['id','tittle','cast','crew']
movies= movies.merge(credits,on='id')
# print(movies.head(5))
#using the imdb formula
#w=(v/v+m.R)+(m/v+m.c) where 
#v is the number of votes for the movie;
# m is the minimum votes required to be listed in the chart;
# R is the average rating of the movie; And
# C is the mean vote across the whole report
C= movies['vote_average'].mean()
m= movies['vote_count'].quantile(0.9)
movies_qualify=movies.copy().loc[movies['vote_count']  >= m]#returns data of all  movies  from movies that has vote count above 90 percent ie m
print(movies_qualify.shape)

def weighted_rating(qualify_movies,m=m,C=C):
    v=qualify_movies['vote_count']
    R=qualify_movies['vote_average']

    return (v/(v+m) *R)+(m/(m+v) * C)  #using imdb formula\
movies_qualify['score']=movies_qualify.apply(weighted_rating,axis=1)
print(movies_qualify[['title', 'vote_count', 'vote_average', 'score']].head(10))
#new column is created and all data is sent rown wise and movies_qualify score would get score of each movie by sending it to the weightes training
movies_qualify = movies_qualify.sort_values('score', ascending=False)

view=movies_qualify.sort_values('score')# arranging the movies in desending order
plt.figure(figsize=(12,4))

plt.barh(view['title'].head(6),view['score'].head(6), align='center',
        color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("score populatity")
plt.title("recomended movies")

#then weighted score is caluculated  using tf-id multiplications
#formula for tf is  Term frequency is the number of times a term appears in the overview divided by the total number of terms
#formula for  idf is The IDF value for a term is calculated as the logarithm of the total number of documents divided by the number of documents containing that term
# weighted score is  TF*IDF 

#Remove all english stop words such as 'the', 'a'
tfid = TfidfVectorizer(stop_words='english')
movies['overview']=movies['overview'].fillna(' ')
tfid_matrix = tfid.fit_transform(movies['overview'])
# print(tfid_matrix.shape)
#then we get cosine similarity scores by this we will be getting similarities between two movies ,thus this can be done through linear kernal

cos_sim = linear_kernel(tfid_matrix, tfid_matrix)
#cos sim =a.b/|a||b|
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
#create a dict() of key value pairs of title and its index
# print(indices)

def get_recommendations(title, cosine_sim=cos_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # top 10 most similar movies
    return movies['title'].iloc[movie_indices]
x=get_recommendations('The Dark Knight Rises')
print(x)