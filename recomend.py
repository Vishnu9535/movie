from ast import literal_eval
from cgi import print_directory

import numpy as np
import pandas as pd

credits=pd.read_csv("credits.csv")
movies=pd.read_csv("movies.csv")
movies.head()
credits.head()
#print(credits.columns)
credits.columns=['id','title','cast','crew']
#print(credits.columns)
#print(movies.columns)
movies=movies.merge(credits,on="id")
movies.head()
# print(movies.columns)
#print(movies)
features=["cast","crew","keywords","genres"]
#print(movies[features])
for feature in features:
    movies[feature]=movies[feature].apply(literal_eval)
    #print(movies[feature])
movies[features].head(10)
#print(movies[features])
print(movies.columns)
def get_director(crew):
    for i in crew:
        if i["job"]=="Director":
            return(i["name"])
    np.nan
def get_list(name):
    if isinstance(name,list):
        names=[i["name"] for i in name]  #list comprehension
        # print(names)
        
        if len(names) > 3:
            names=names[:3]
        #print(names)
        return names
    return []
movies["director"]=movies["crew"].apply(get_director)
#print(movies["director"])
features=["cast","keywords","genres"]
for feature in features:
    movies[feature]=movies[feature].apply(get_list)
    #print(movies[feature])
movies[['title_x', 'cast', 'director', 'keywords', 'genres']].head()
#print(movies[features])
def clean_data(row):
    if isinstance(row,list):
        return [str.lower(i.replace(" ","")) for i in row]
    else:
        if isinstance(row,str):
            return[str.lower(i.replace(" ","")) for i in row]
        else:
            return ""
features=['cast','keywords','director','genres']
for feature in features:
    movies[feature]=movies[feature].apply(clean_data)

# print(movies['keywords'])
def create_soup(movies):
    x= ' '.join(movies['keywords']) + ' ' + ' '.join(movies['cast']) + ' ' +' '.join(movies['director']) + ' ' + ' '.join(movies['genres'])
    # print(x)
    return(x)

movies["soup"] = movies.apply(create_soup, axis=1)
print(movies["soup"].head())
print(movies["soup"])