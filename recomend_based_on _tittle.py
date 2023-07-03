import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

credits=pd.read_csv('credits.csv')
movies=pd.read_csv('movies.csv')
credits.columns=['id','tittle','cast','crew']
movies= movies.merge(credits,on='id')

#weighted score is caluculated  using tf-id multiplications
#formula for tf is  Term frequency is the number of times a term appears in the overview divided by the total number of terms
#formula for  idf is The IDF value for a term is calculated as the logarithm of the total number of documents divided by the number of documents containing that term
# weighted score is  TF*IDF 

tfid = TfidfVectorizer(stop_words='english')#Remove all english stop words such as 'the', 'a'
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
    movie_index=[]
    for i in sim_scores:
        movie_index.append(i[0]) 

    # top 10 most similar movies
    return movies['title'].iloc[movie_index]
x=get_recommendations('The Dark Knight Rises')
print(x)

