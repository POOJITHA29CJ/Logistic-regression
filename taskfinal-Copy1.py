#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv("HR.csv")

df.columns


# In[2]:


print(df)


# In[29]:


df


# In[3]:


left=df[df['left']==1]
left.shape


# In[31]:


print(left.to_string())


# In[35]:


df['promotion_last_5years'].value_counts()


# In[27]:


df.info()


# In[8]:


retain=df[df['left']==0]
retain.shape


# In[32]:


print(retain.to_string())


# In[9]:


newdf=df.drop(columns=['Department','salary'])
newdf.groupby('left').mean()


# In[10]:


pd.crosstab(df['salary'],df['left'])


# In[11]:


pd.crosstab(df['salary'],df['left']).plot(kind='bar')


# In[12]:


pd.crosstab(df['Department'],df['left']).plot(kind='bar')


# In[13]:


x=df.drop(columns=['last_evaluation','number_project','Work_accident','left','Department','salary'])
print(x.to_string())
print(x.columns)


# In[14]:


y=df['left']
print(y.to_string())


# In[15]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[16]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)


# In[17]:


x_train.shape


# In[18]:


x_test.shape


# In[19]:


df.shape


# In[20]:


model.fit(x_train,y_train)


# In[21]:


ans=model.predict(x_test)
print(ans)


# In[22]:


model.score(x_train,y_train)


# In[23]:


x_train.columns


# In[33]:


n=int(input("Enter no of details: "))
lists=[]
for i in range(n):
    x=float(input("satisfaction_level : "))
    y=int(input("average_montly_hours: "))
    z=int(input("time_spend_company : "))
    p=int(input("promotion_last_5years 0 if not and 1 if promoted:"))
    nested=list([x,y,z,p])
    lists.append(nested)
print(lists)
ans=model.predict(lists)
for i in ans:
    print(i)


# In[ ]:




