
# coding: utf-8

# In[13]:


import numpy
from matplotlib import pyplot as plt
import pandas


# In[4]:


dataset=pandas.read_csv('C:\\Users\\mishr\\Desktop\\Salary_Data.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values
print(X,Y)


# In[5]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2,random_state=0)


# In[6]:


from sklearn import linear_model
alg = linear_model.LinearRegression
alg.fit(xtrain,ytrain)


# In[15]:


ypred=alg.predict(xtest)
print(ypred)


# In[20]:


plt.scatter(xtest,ytest,color='g')
plt.plot(xtest,alg.predict(xtest),color='r')
plt.title('Test set')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

