#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


#Read dataset
df = pd.read_csv("adult.csv")


# In[5]:


df


# In[6]:


df.isin(['?']).sum()


# In[7]:


#Replace ? with NaN 
df['workclass'] = df['workclass'].replace('?', np.nan)
df['occupation'] = df['occupation'].replace('?', np.nan)
df['native-country'] = df['native-country'].replace('?', np.nan)


# In[8]:


#Drop all rows that contain a missing value
df.dropna(how='any', inplace=True)


# In[9]:


#Check duplicate values in dataframe now
print(f"There are {df.duplicated().sum()} duplicate values")
df = df.drop_duplicates()


# In[10]:


df.columns


# In[11]:


#Drop non-relevant columns

df.drop(['fnlwgt','educational-num','marital-status','relationship','race'], axis=1,inplace=True)


# In[12]:





# In[13]:


#Extract X and y from the dataframe , income column is the target column, rest columns are features
X = df.loc[:,['age', 'workclass', 'education', 'occupation', 'gender', 'capital-gain',
       'capital-loss', 'hours-per-week', 'native-country']]
y = df.loc[:,'income']


# In[15]:


# Since y is a binary categorical column we will use label encoder to convert it into numerical columns with values 0 and 1
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)
y = pd.DataFrame(y)
y.head()


# In[16]:


#First identify caterogical features and numeric features
numeric_features = X.select_dtypes('number')
categorical_features = X.select_dtypes('object')
categorical_features


# In[19]:


#Convert categorical features into numeric
converted_categorical_features = pd.get_dummies(categorical_features)
converted_categorical_features


# In[20]:


#combine the converted categorical features and the numeric features together into a new dataframe called "newX"
all_features = [converted_categorical_features, numeric_features]
newX = pd.concat(all_features,axis=1, join='inner')
newX.shape


# In[24]:


#Do a train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(newX, y, test_size=0.33, random_state=42)


# In[25]:


# Load Decision Tree Classifier, max_depth = 5 and fit it with X-train and y-train
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)


# In[26]:


# Make predictions
y_pred = clf.predict(X_test)


# In[27]:


#Evaluate the performance of fitting
from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test))


# In[28]:


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(14,14))
plot_tree(clf, fontsize=10, filled=True)
plt.title("Decision tree trained on the selected features")
plt.show()


# In[ ]:




