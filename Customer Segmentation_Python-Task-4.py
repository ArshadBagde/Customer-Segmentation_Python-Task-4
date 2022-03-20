#!/usr/bin/env python
# coding: utf-8

# #  Project:- Customer Segmentation_Python- Project
# Organization:- Infopillar Solution

# # Data Science Internship
# Author:- Arshad R. Bagde

# In[ ]:





# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


# Supress Warnings
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


# Data display coustomization
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', -1)


# In[5]:


# To perform Hierarchical clustering
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


# In[6]:


# import all libraries and dependencies for machine learning
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
from math import isnan


# # Data Preparation

# ## Data Loading

# In[7]:


mall= pd.read_csv("Mall_Customers.csv")
mall.head()


# In[8]:


mall.shape


# In[9]:


mall.info()


# In[10]:


mall.describe()


# # Duplicate Check

# In[11]:


mall_d= mall.copy()
mall_d.drop_duplicates(subset=None,inplace=True)


# In[12]:


mall_d.shape


# In[13]:


mall.shape


# # Data Cleaning

# Null Percentage: Columns

# In[14]:


(mall.isnull().sum() * 100 / len(mall)).value_counts(ascending=False)


# Null Count: Columns

# In[15]:


mall.isnull().sum()


# Null Percentage: Rows

# In[16]:


(mall.isnull().sum(axis=1) * 100 / len(mall)).value_counts(ascending=False)


# Null Count: Rows

# In[17]:


mall.isnull().sum(axis=1).value_counts(ascending=False)


# # Exploratory Data Analytics

# Univariate Analysis

# **Gender**

# In[18]:


plt.figure(figsize = (5,5))
gender = mall['Gender'].sort_values(ascending = False)
ax = sns.countplot(x='Gender', data= mall)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
plt.xticks(rotation=90)
plt.show()


# Data is not balanced, 27% more Females have participated  than males 

# **Age**

# In[19]:



plt.figure(figsize = (20,5))
gender = mall['Age'].sort_values(ascending = False)
ax = sns.countplot(x='Age', data= mall)
for p in ax.patches:
   ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.show()


# Audience are from Age 18 to 70

# **Annual Income (k$)**

# In[20]:


plt.figure(figsize = (25,5))
gender = mall['Annual Income (k$)'].sort_values(ascending = False)
ax = sns.countplot(x='Annual Income (k$)', data= mall)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.show()


# Audience are from Annual Income(k$) range between 15 to 137

# **Spending Score (1-100)**

# In[21]:


plt.figure(figsize = (27,5))
gender = mall['Spending Score (1-100)'].sort_values(ascending = False)
ax = sns.countplot(x='Spending Score (1-100)', data= mall)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.show()


# Audience are having Spending Score (1-100) between 1 to 99 

# In[22]:


# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (5,5))
sns.heatmap(mall.corr(), annot = True, cmap="rainbow")
plt.savefig('Correlation')
plt.show()


# - Age and Spending Score (1-100) are moderately correlated with correlation of -0.33

# In[23]:


sns.pairplot(mall,corner=True,diag_kind="kde")
plt.show()


# ## Outlier Analysis

# In[24]:


# Data before Outlier Treatment 
mall.describe()


# In[25]:


f, axes = plt.subplots(1,3, figsize=(15,5))
s=sns.violinplot(y=mall.Age,ax=axes[0])
axes[0].set_title('Age')
s=sns.violinplot(y=mall['Annual Income (k$)'],ax=axes[1])
axes[1].set_title('Annual Income (k$)')
s=sns.violinplot(y=mall['Spending Score (1-100)'],ax=axes[2])
axes[2].set_title('Spending Score (1-100)')
plt.show()


# There is an outlier in Annual Income (k$) field but Income & Spending Score(1-100) has no outliers 

# ## We use Percentile Capping (Winsorization) for outliers handling

# In[26]:


Q3 = mall['Annual Income (k$)'].quantile(0.99)
Q1 = mall['Annual Income (k$)'].quantile(0.01)
mall['Annual Income (k$)'][mall['Annual Income (k$)']<=Q1]=Q1
mall['Annual Income (k$)'][mall['Annual Income (k$)']>=Q3]=Q3


# In[27]:


# Data After Outlier Treatment 
mall.describe()


# In[28]:


f, axes = plt.subplots(1,3, figsize=(15,5))
s=sns.violinplot(y=mall.Age,ax=axes[0])
axes[0].set_title('Age')
s=sns.violinplot(y=mall['Annual Income (k$)'],ax=axes[1])
axes[1].set_title('Annual Income (k$)')
s=sns.violinplot(y=mall['Spending Score (1-100)'],ax=axes[2])
axes[2].set_title('Spending Score (1-100)')
plt.show()


# In[29]:


# Dropping CustomerID,Gender field to form cluster

mall_c = mall.drop(['CustomerID','Gender'],axis=1,inplace=True)


# In[30]:


mall.head()


# # Hopkins Statistics Test

# The Hopkins statistic (introduced by Brian Hopkins and John Gordon Skellam) is a way of measuring the cluster tendency of a data set.It acts as a statistical hypothesis test where the null hypothesis is that the data is generated by a Poisson point process and are thus uniformly randomly distributed. A value close to 1 tends to indicate the data is highly clustered, random data will tend to result in values around 0.5, and uniformly distributed data will tend to result in values close to 0.
# 
# • If the value is between {0.01, ...,0.3}, the data is regularly spaced.
# 
# • If the value is around 0.5, it is random.
# 
# • If the value is between {0.7, ..., 0.99}, it has a high tendency to cluster.

# In[31]:


def hopkins(X):
    d = X.shape[1]
    n = len(X)
    m = int(0.1 * n) 
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    HS = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(HS):
        print(ujd, wjd)
        HS = 0
 
    return HS


# In[32]:


# Hopkins score
Hopkins_score=round(hopkins(mall),2)


# In[33]:


print("{} is a good Hopkins score for Clustering.".format(Hopkins_score))


# # Rescaling the Features

# Most software packages use SVD to compute the principal components and assume that the data is scaled and centred, so it is important to do standardisation/normalisation. There are two common ways of rescaling:
# 
# - Min-Max scaling
# - Standardisation (mean-0, sigma-1)
# 
# Here, we will use Standardisation Scaling.

# In[34]:


# Standarisation technique for scaling
scaler = StandardScaler()
mall_scaled = scaler.fit_transform(mall)


# In[35]:


mall_scaled


# In[36]:


mall_df1 = pd.DataFrame(mall_scaled, columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
mall_df1.head()


# # Model Building

# ## K- means Clustering

# K-means clustering is one of the simplest and popular unsupervised machine learning algorithms.
# 
# The algorithm works as follows:
# 
# First we initialize k points, called means, randomly. We categorize each item to its closest mean and we update the mean’s coordinates, which are the averages of the items categorized in that mean so far. We repeat the process for a given number of iterations and at the end, we have our clusters.

# ## Finding the Optimal Number of Clusters

# Elbow Curve to get the right number of Clusters

# A fundamental step for any unsupervised algorithm is to determine the optimal number of clusters into which the data may be clustered. The Elbow Method is one of the most popular methods to determine this optimal value of k.

# In[37]:


# Elbow curve method to find the ideal number of clusters.
clusters=list(range(2,8))
ssd = []
for num_clusters in clusters:
    model_clus = KMeans(n_clusters = num_clusters, max_iter=150,random_state= 50)
    model_clus.fit(mall_df1)
    ssd.append(model_clus.inertia_)

plt.plot(clusters,ssd);


# Looking at the above elbow curve it looks good to proceed with 4 clusters.

# ## Silhouette Analysis

# silhouette score=(p−q)/max(p,q)
# 
# p is the mean distance to the points in the nearest cluster that the data point is not a part of
# 
# q is the mean intra-cluster distance to all the points in its own cluster.
# 
# The value of the silhouette score range lies between -1 to 1.
# 
# A score closer to 1 indicates that the data point is very similar to other data points in the cluster,
# 
# A score closer to -1 indicates that the data point is not similar to the data points in its cluster.

# In[38]:


# Silhouette score analysis to find the ideal number of clusters for K-means clustering

range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50,random_state= 100)
    kmeans.fit(mall_df1)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(mall_df1, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))


# We will opt for 4 as cluster

# In[39]:


#K-means with k=4 clusters

cluster = KMeans(n_clusters=4, max_iter=150, random_state= 50)
cluster.fit(mall_df1)


# In[40]:


# Cluster labels

cluster.labels_


# In[41]:


# Assign the label

mall_d['Cluster_Id'] = cluster.labels_
mall_d.head()


# In[42]:


## Number of customers in each cluster
mall_d['Cluster_Id'].value_counts(ascending=True)


# In[43]:


mall_d.columns


# It seems there are good number of countries in each clusters.

# In[44]:


plt.figure(figsize = (20,15))
plt.subplot(3,1,1)
sns.scatterplot(x = 'Age', y = 'Annual Income (k$)',hue='Cluster_Id',data = mall_d,legend='full',palette="Set1")
plt.subplot(3,1,2)
sns.scatterplot(x = 'Annual Income (k$)', y = 'Spending Score (1-100)',hue='Cluster_Id', data = mall_d,legend='full',palette="Set1")
plt.subplot(3,1,3)
sns.scatterplot(x = 'Spending Score (1-100)', y = 'Age',hue='Cluster_Id',data= mall_d,legend='full',palette="Set1")
plt.show()


# In[45]:


#Violin plot on Original attributes to visualize the spread of the data

fig, axes = plt.subplots(1,3, figsize=(20,5))

sns.violinplot(x = 'Cluster_Id', y = 'Age', data = mall_d,ax=axes[0])
sns.violinplot(x = 'Cluster_Id', y = 'Annual Income (k$)', data = mall_d,ax=axes[1])
sns.violinplot(x = 'Cluster_Id', y = 'Spending Score (1-100)', data=mall_d,ax=axes[2])
plt.show()


# In[46]:


mall_d.head()


# In[47]:


mall_d[['Age', 'Annual Income (k$)','Spending Score (1-100)','Cluster_Id']].groupby('Cluster_Id').mean()


# Cluster 0  are those people whose 
# - Avg Age : 54
# - Avg Annual Income (k$) : 47.7k
# - Avg Spending Score (1-100) : 40 
# 
# We can label them Medium Spender 

# In[48]:


group_0= mall_d[mall_d['Cluster_Id']==0]
group_0.head()


# In[49]:


fig, axes = plt.subplots(1,3, figsize=(20,5))

sns.violinplot(x = 'Gender', y = 'Age', data = group_0,ax=axes[0])
sns.violinplot(x = 'Gender', y = 'Annual Income (k$)', data = group_0,ax=axes[1])
sns.violinplot(x = 'Gender', y = 'Spending Score (1-100)', data=group_0,ax=axes[2])
plt.show()


# - Mean Age of this cluster for Male is more than Females
# - Males earn more than females
# - Mean Spending Score (1-100) is same for both gender 

# Cluster 1  are those people whose 
# - Avg Age : 25
# - Avg Annual Income (k$) : 40 k
# - Avg Spending Score (1-100) : 60 
# 
# We can label them Large Spender

# In[50]:


group_1= mall_d[mall_d['Cluster_Id']==1]
group_1.head()


# In[51]:


fig, axes = plt.subplots(1,3, figsize=(20,5))

sns.violinplot(x = 'Gender', y = 'Age', data = group_1,ax=axes[0])
sns.violinplot(x = 'Gender', y = 'Annual Income (k$)', data = group_1,ax=axes[1])
sns.violinplot(x = 'Gender', y = 'Spending Score (1-100)', data=group_1,ax=axes[2])
plt.show()


# - Mean Age of this cluster are same for both genders 
# - Males earn more than females
# - Mean Spending Score (1-100) is more for males 

# Cluster 2 are those people whose 
# - Avg Age : 32
# - Avg Annual Income (k$) : 86 k
# - Avg Spending Score (1-100) : 81
# 
# We can label them Extra Spender

# In[52]:


group_2= mall_d[mall_d['Cluster_Id']==2]
group_2.head()


# In[53]:


fig, axes = plt.subplots(1,3, figsize=(20,5))

sns.violinplot(x = 'Gender', y = 'Age', data = group_2,ax=axes[0])
sns.violinplot(x = 'Gender', y = 'Annual Income (k$)', data = group_2,ax=axes[1])
sns.violinplot(x = 'Gender', y = 'Spending Score (1-100)', data=group_2,ax=axes[2])
plt.show()


# - Age range for males are higher than females 
# - Males earn more than females
# - Mean Spending Score (1-100) is more for males 

# Cluster 3 are those people whose 
# - Avg Age : 40
# - Avg Annual Income (k$) : 86.5 k
# - Avg Spending Score (1-100) : 19
# 
# We can label them Low Spender

# In[54]:


group_3= mall_d[mall_d['Cluster_Id']==3]
group_3.head()


# In[55]:


fig, axes = plt.subplots(1,3, figsize=(20,5))

sns.violinplot(x = 'Gender', y = 'Age', data = group_3,ax=axes[0])
sns.violinplot(x = 'Gender', y = 'Annual Income (k$)', data = group_3,ax=axes[1])
sns.violinplot(x = 'Gender', y = 'Spending Score (1-100)', data=group_3,ax=axes[2])
plt.show()


# - Age range for males are higher than females 
# - Annual Income range for males are lower than females 
# - Mean Spending Score (1-100) is more for females 

# In[56]:


mall_d[['Age', 'Annual Income (k$)','Spending Score (1-100)','Cluster_Id']].groupby('Cluster_Id').mean()


# Final Points 
# 
# - Target Cluster 1 with more offers 
# - Reward Cluster 2 people for being  loyal customer.
# - Improve the services to  attract Cluster 3 
# - Target Cluster 0 with better employees support 
