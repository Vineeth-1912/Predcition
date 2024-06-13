#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv("IRIS.csv")


# In[3]:


df


# In[5]:


df.isnull().sum()


# In[22]:


x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
x,y


# In[8]:


import plotly.express as px


# In[23]:


fig=px.scatter(df,x="petal_length",y="petal_width",color='species')
fig


# In[9]:


df.head()


# In[36]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)


# In[37]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train,y_train)


# In[40]:


ypred=knn.predict(x_test)
ypred


# In[41]:


ypred2=knn.predict(x_train)
ypred2


# In[44]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,ypred)
print(cm)
accuracy_score(y_train,ypred2)


# In[49]:


df=pd.read_csv("Social_Network_Ads.csv")


# In[50]:


df


# In[52]:


x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
x


# In[53]:


y


# In[56]:


x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)


# In[57]:


fig=px.scatter(df,x="Age",y="EstimatedSalary",color='Purchased')
fig


# In[59]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)


# In[61]:


ypred3=lr.predict(x_test)
ypred3


# In[ ]:




