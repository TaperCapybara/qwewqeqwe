#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from sklearn import preprocessing


# In[3]:


data = pd.read_csv('weather.csv')
data


# In[4]:


data = data.drop(labels = 'RespondentID', axis = 1)
data = data.drop(labels = 'A specific website or app (please provide the answer)', axis = 1)

data['Do you typically check a daily weather report?'] = data['Do you typically check a daily weather report?'].replace(['Yes','No', '-'], [2, 1, 0])
data['How do you typically check the weather?'] = data['How do you typically check the weather?'].replace(['The default weather app on your phone',
       'A specific website or app (please provide the answer)',
       'The Weather Channel', 'Internet search', 'Local TV News',
       'Newspaper', 'Radio weather', 'Newsletter', '-'], [1,2,3,4,5,6,7, 8,0])


# In[5]:


data['If you had a smartwatch (like the soon to be released Apple Watch), how likely or unlikely would you be to check the weather on that device?'] = data['If you had a smartwatch (like the soon to be released Apple Watch), how likely or unlikely would you be to check the weather on that device?'].replace(['Very likely', 'Somewhat likely', 'Very unlikely',
       'Somewhat unlikely', '-'], [1,2,3, 4, 0])
data['Age'] = data['Age'].replace(['30 - 44', '18 - 29', '45 - 59', '60+', '-'], [1,2,3, 4, 0])
data['What is your gender?'] = data['What is your gender?'].replace(['Male', 'Female', '-'], [2, 1, 0])
data['How much total combined money did all members of your HOUSEHOLD earn last year?'] = data['How much total combined money did all members of your HOUSEHOLD earn last year?'].replace(['$50,000 to $74,999', 'Prefer not to answer',
       '$100,000 to $124,999', '$150,000 to $174,999',
       '$25,000 to $49,999', '$0 to $9,999', '$10,000 to $24,999',
       '$75,000 to $99,999', '$200,000 and up', '$175,000 to $199,999',
       '$125,000 to $149,999', '-'], [1,2,3,4,5,6,7,8,9,10, 11, 0])
data['US Region'] = data['US Region'].replace(['South Atlantic', 'Middle Atlantic', 'West South Central',
       'Pacific', 'West North Central', 'East North Central', 'Mountain',
       'New England', 'East South Central', '-'], [1,2,3,4,5,6,7,8, 9, 0])


# In[6]:


data


# In[7]:


for column in data.columns:
    missing = np.mean(data[column].isna()*100)
    print(f" {column} : {round(missing,1)}%")


# In[8]:


scaler=preprocessing.MinMaxScaler()
data1 = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
data1


# In[9]:


from sklearn.metrics import silhouette_score


# In[10]:


models = []
score1 = []
score2 = []
for i in range(2,10):
    model = KMeans(n_clusters=i, random_state=123, init='k-means++').fit(data)
    models.append(model)
    score1.append(model.inertia_)
    score2.append(silhouette_score(data, model.labels_))


# In[11]:


plt.grid()
plt.plot(np.arange(2,10), score1, marker = 'o')
plt.show


# In[12]:


plt.grid()
plt.plot(np.arange(2,10), score2, marker = 'o')
plt.show


# In[13]:


model1 = KMeans(n_clusters = 2, random_state = 123, init = 'k-means++')
model1.fit(data)
model1.cluster_centers_


# In[14]:


labels=model1.labels_
data['Claster'] = labels
data['Claster'].value_counts()


# In[15]:


# В задании ещё проверить на какую-то ошибку


# In[16]:


from sklearn.cluster import AgglomerativeClustering


# In[17]:


model2 = AgglomerativeClustering(6, compute_distances=True)
clastering = model2.fit(data)
data['Claster']=clastering.labels_
data['Claster'].value_counts()


# In[18]:


fig = go.Figure(data=[go.Scatter3d(x=data['Age'], y=data['US Region'], z=data['How do you typically check the weather?'], mode='markers', 
                                  marker_color=data['Claster'], marker_size=4)])
fig.show()


# In[19]:


from sklearn.cluster import DBSCAN


# In[29]:


model3 = DBSCAN(eps=4, min_samples=3).fit(data)
data['Claster'] = model3.labels_
data['Claster'].value_counts()


# In[30]:


fig = go.Figure(data=[go.Scatter3d(x=data['Age'], y=data['US Region'], z=data['How do you typically check the weather?'], mode='markers', 
                                  marker_color=data['Claster'], marker_size=4)])
fig.show()


# In[ ]:




