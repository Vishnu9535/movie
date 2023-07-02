import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

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
print(tfid_matrix.shape)




