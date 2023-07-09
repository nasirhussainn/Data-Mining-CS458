#!/usr/bin/env python
# coding: utf-8

# <h4 style="color:purple">Nasir Hussain 04072013040</h4>

# <center>
# <h3 style="color:purple">Introduction to Data Mining</h3>
# <h4 style="color:purple">Assignment : 01</h4>
# </center>

# <p style="color:red">Pandas Library is not used for any task of assignment. However it is used for some elaboration</p>

# In[44]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# <h3 style="color:purple">Task 1:Load the dataset</h3>

# In[45]:


dataset=np.genfromtxt("Bank Marketing Dataset.csv",delimiter=",",dtype=str)
dataset = dataset[:1001]
dataset


# In[46]:


data=pd.read_csv("Bank Marketing Dataset.csv")
data=data.iloc[:1000]
data.head()


# In[47]:


len(data)


# <h3 style="color:purple">Task 2:Find the average, mean, median and standard deviation of one numerical attribute</h3>

# I have select the Balance column of Dataset which is 5th column of dataset

# In[48]:


Customer_Balance = dataset[1:, 5].astype(float)


# In[49]:


Mean_Balance=np.mean(Customer_Balance)
Median_Balance=np.median(Customer_Balance)
Average_Balance=np.average(Customer_Balance)
Std_Balance=np.std(Customer_Balance)


# In[50]:


print("Mean of Customer Balance is : ",Mean_Balance)
print("Median of Customer Balance is : ",Median_Balance)
print("Average of Customer Balance is : ",Average_Balance)
print("Standard Deviation of Customer Balance is : ",Std_Balance)


# <p style="color:green">Here data is dataframe and we can verify our result by using below Pandas method</p>

# In[51]:


data.describe()


# <h3 style="color:purple">Task 3:Visualize one or more attributes using scatter plots</h3>

# In[52]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[53]:


Customer_Ages = dataset[1:, 0].astype(float)
Campaign = dataset[1:, 12].astype(float)


# In[54]:


plt.scatter(Customer_Ages,(Customer_Balance/100),marker="D",s=5)
plt.xlabel("Ages")
plt.ylabel("Balance (scaled down by a factor of 100)")
plt.title("Customer Age vs. Balance")

plt.grid(color="red", linestyle=":", alpha=0.5)


# <p style="color:maroon"I have used only 100 data from dataset and we can we see that there is no relationship between balance and ages</p>

# <p style="color:blue">We can see that there is no relationship between Ages and Balance which mean Balance is not depend on Age.</p>

# In[55]:


plt.scatter((Customer_Balance/1000),Campaign,marker='D',s=5)

plt.xlabel("Balance (scaled down by a factor of 1000)")
plt.ylabel("Campaign")
plt.title("Balance vs. Campaign")

plt.grid(color="red", linestyle=":", alpha=0.5)


# <h3 style="color:purple">Task 4:Find the similarity using an appropriate measure between any two categorical attributes</h3>

# <center> <h3 style="color:green"> simple matching method -> dij = (p - m) / p </h3></center>

# In[56]:


Customer_Marital = dataset[1:, 2]
np.unique(Customer_Marital)


# In[57]:


Customer_Education = dataset[1:, 3]
np.unique(Customer_Education)


# In[58]:


summ=0
count=0
dsim_matrix_nom = np.zeros((1000, 1000))
for i in range(1000):
    for j in range(i):
        p=2
        m=0
        if(Customer_Marital[i]==Customer_Marital[j]):
            m+=1
        if(Customer_Education[i]==Customer_Education[j]):
            m+=1
        r=(p-m)/p
        dsim_matrix_nom[i][j]=r
        summ+=dsim_matrix_nom[i][j]
        count+=1
        


# In[59]:


dsim_matrix_nom


# In[60]:


mean_matrix=summ/count
mean_matrix


# In[61]:


simmilarity_nom = 1 - mean_matrix
simmilarity_nom


# <p style="color:magenta"> The average simmilarity between two categorical attribute (Marital and Education) of all data-object is <b>0.4068968968968969</b></p>

# <h3 style="color:purple">Task 5:Find the similarity using an appropriate measure between any two numeric attributes</h3>

# <center> <h3 style="color:green"> normalization formula -> dij = |xif - xjf| / max - min</h3></center>

# In[62]:


max_age=np.max(Customer_Ages)
min_age=np.min(Customer_Ages)
max_bal=np.max(Customer_Balance)
min_bal=np.min(Customer_Balance)


# In[63]:


n_summ=0
n_count=0

norm_Customer_Ages = np.zeros((1000, 1000))
norm_Customer_Balance = np.zeros((1000, 1000))
norm_Age_Bal=np.zeros((1000,1000))

for i in range(1000):
    for j in range(i):
        norm_Customer_Ages[i][j]=(abs(Customer_Ages[i]-Customer_Ages[j]))/(max_age-min_age)
        
        norm_Customer_Balance[i][j]=(abs(Customer_Balance[i]-Customer_Balance[j]))/(max_bal-min_bal)
        norm_Age_Bal[i][j]=norm_Customer_Ages[i][j]+norm_Customer_Balance[i][j]
        
        n_summ+=(norm_Customer_Ages[i][j]+norm_Customer_Balance[i][j])
        n_count+=1


# In[64]:


norm_Age_Bal


# In[65]:


n_dis_mean=n_summ/n_count
n_dis_mean


# In[66]:


simmilarity_numeric=1-n_dis_mean
simmilarity_numeric


# <p style="color:magenta"> The average simmilarity between two numeric attribute (Age and Balance) of all data-object is <b>0.7389533947700335</b></p>

# <h4 style="color:red">-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------</h4>

# <center><h2 style="color:purple">END OF ASSIGNMENT</h2></center>

# <h4 style="color:red">-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------</h4>

# <p style="color:red">Tried with Euclidean. Not part of assignment</p>

# In[67]:


age_bal_simm=np.stack((Customer_Ages,Customer_Balance),axis=1)


# In[68]:


summ2=0
count2=0
dsimm_matrix = np.zeros((1000, 1000))
for i in range(1000):
    for j in range(i):
        dsimm_matrix[i][j] = np.linalg.norm(age_bal_simm[i] - age_bal_simm[j])
        summ2+=dsimm_matrix[i][j]
        count2+=1


# In[69]:


dsimm_matrix


# In[70]:


min_val = np.min(dsimm_matrix)
max_val = np.max(dsimm_matrix)

mean_matrix_num=summ2/count2
normalized_disim = (mean_matrix_num - min_val) / (max_val-min_val)

normalized_disim


# In[71]:


1-normalized_disim


# In[ ]:




