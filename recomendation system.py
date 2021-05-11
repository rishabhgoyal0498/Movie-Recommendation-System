#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


columns_names=["user_id","item_id","rating","timestamp"]
df= pd.read_csv("u.data",sep="\t",names=columns_names)


# In[4]:


df.head()


# In[5]:


movies_title=pd.read_csv("u.item",sep="\|",header=None)


# movies_title.shape()

# movies_title.shape()

# In[6]:


movies_title.shape


# In[7]:


movies_title=movies_title[[0,1]]


# In[8]:


movies_title.head()


# In[9]:


movies_title.columns=['item_id','title']


# In[10]:


movies_title.head()


# In[11]:


df=pd.merge(df,movies_title,on="item_id")


# In[12]:


df


# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[14]:


df.groupby('title').mean()


# In[15]:


ratings=pd.DataFrame(df.groupby('title').mean()['rating'])


# In[16]:


ratings.head()


# In[17]:


ratings['number of ratings']= pd.DataFrame(df.groupby('title').count()['rating'])


# In[21]:


ratings.sort_values(by='rating',ascending=False)


# In[23]:


plt.figure(figsize=(10,6))
plt.hist(ratings['number of ratings'],bins=70)
plt.show()


# In[24]:


plt.hist(ratings['rating'],bins=70)
plt.show()


# In[29]:


sns.jointplot(x='rating',y='number of ratings',data=ratings,alpha=0.4)


# In[30]:


#MOVIE RECOMMENDATION


# In[31]:


moviemat=df.pivot_table(index='user_id',columns='title',values='rating')


# In[32]:


moviemat


# In[33]:


starwars_user_ratings=moviemat['Star Wars (1977)']
starwars_user_ratings.head()


# In[36]:


similar_to_starwars=moviemat.corrwith(starwars_user_ratings)
similar_to_starwars


# In[37]:


corr_to_starwars=pd.DataFrame(similar_to_starwars,columns=['Correlation'])


# In[38]:


corr_to_starwars.dropna(inplace=True)


# In[40]:


corr_to_starwars.head()


# In[42]:


corr_to_starwars.sort_values('Correlation',ascending=False).head(10)


# In[43]:


corr_to_starwars=corr_to_starwars.join(ratings['number of ratings'])
corr_to_starwars.head()


# In[45]:


corr_to_starwars[corr_to_starwars['number of ratings']>100].sort_values('Correlation',ascending=False).head(10)


# In[46]:


## PREDICT FUNCTION


# In[56]:


def predict_movies(movie_name):
    movie_user_ratings=moviemat[movie_name]
    similar_to_movie=moviemat.corrwith(movie_user_ratings)
    
    corr_movie=pd.DataFrame(similar_to_movie,columns=['Correlation'])
    corr_movie.dropna(inplace=True)
    
    corr_movie=corr_movie.join(ratings['number of ratings'])
    predictions=corr_movie[corr_movie['number of ratings']>100].sort_values('Correlation',ascending=False)
    return predictions


# In[57]:


predictions=predict_movies("Titanic (1997)")


# In[59]:


predictions.head(10)

