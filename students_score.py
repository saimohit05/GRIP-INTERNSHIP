#!/usr/bin/env python
# coding: utf-8

# # THE SPARKS FOUNDATION
# 
# ## TASK 1: Predict the  percentage of a student based on the number of study hours.
# 
# ## NAME : V. SAI MOHIT

# #### we use linear Regression inorder to predict the percentage of students based on the no.of study hours.Linear Regression is a machine learning algorithm based on supervised learning and it provides the relationship between the target variable and predicted variable . 
# #### mathematically , it is represented as 
# ### y = mx +c  
# #### where ;
# #### y= dependent variable
# #### x= independent variable
# #### m= coefficient 
# #### c= intercept of the line

# In[1]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns


# In[27]:


df1=pd.read_excel('C:\\Users\\hp\\OneDrive\\Desktop\\score.xls')


# In[28]:


df1


# In[58]:


df1['scores']=df1['scores'].fillna(0)
df1.head()


# ##  Data Visualization 

# In[31]:


df.plot(x='Hours',y='scores',style='o')
plt.title('Hours vs Percentage Scores')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[32]:


plt.hist(x="Hours",data=df,color="green")
plt.show()


# ## Training the algorithm
#  we use Sckikit-learn library which contain a lot of effcient tools for machine  learning 

# In[39]:


x=df1.iloc[:,0:1].values
y=df1.iloc[:,1].values


# In[37]:


x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)


# In[38]:


from sklearn.linear_model import LinearRegression  
reg = LinearRegression()  
reg.fit(x_train, y_train) 


# ### Accuracy

# In[44]:


reg.score(x_test,y_test)


# In[50]:


print(reg.coef_)


# In[49]:


print(reg.intercept_)


# In[55]:


fit_line = reg.coef_*x+reg.intercept_
plt.scatter(x, y,color='red')
plt.plot(x, fit_line);
plt.title('Hours vs Percentage Score')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[59]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))


# ### PREDICTING THE SCORE FOR 9.5 HOURS

# In[56]:


Hours=[[9.25]]
predict_value=reg.predict(Hours)
predict_value


# ### CONCLUSION :
# ##### PREDICTED PRECENTAGE SCORE = 93.69173249
