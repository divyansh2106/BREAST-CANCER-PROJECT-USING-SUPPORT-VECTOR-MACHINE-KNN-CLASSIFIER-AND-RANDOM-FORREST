#!/usr/bin/env python
# coding: utf-8

# # BREAST CANCER PROJECT USING SUPPORT VECTOR MACHINE, KNN CLASSIFIER AND RANDOM FORREST

# In[13]:


#Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[14]:


df = pd.read_csv('Breast_Cancer_Prediction.csv')


# In[15]:


df.head


# In[16]:


df.shape


# In[19]:


df.info()


# In[20]:


df.isnull().sum()


# In[23]:


df['diagnosis'].value_counts()


# In[24]:


# mapping categorical values to numerical values
df['diagnosis']=df['diagnosis'].map({'B':0,'M':1})


# In[25]:


df['diagnosis'].value_counts()


# In[34]:


from sklearn.model_selection import train_test_split

# splitting data
X_train, X_test, y_train, y_test = train_test_split(
                df.drop('diagnosis', axis=1),
                df['diagnosis'],
                test_size=0.2,
                random_state=42)


# In[35]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)


# In[36]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report


# In[37]:


knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)


# In[38]:


print(confusion_matrix(y_test, predictions))
print("\n")
print(classification_report(y_test, predictions))


# In[40]:


knn_model_acc = accuracy_score(y_test, predictions)
print("Accuracy of K Neighbors Classifier Model is: ", knn_model_acc)


# In[43]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(X_train, y_train)
predictions2 = rfc.predict(X_test)


# In[44]:


print("Confusion Matrix: \n", confusion_matrix(y_test, predictions2))
print("\n")
print(classification_report(y_test, predictions2))


# In[45]:


rfc_acc = accuracy_score(y_test, predictions2)
print("Accuracy of Random Forests Model is: ", rfc_acc)


# In[46]:


from sklearn.svm import SVC

svc_model = SVC(kernel="rbf")
svc_model.fit(X_train, y_train)
predictions3 = svc_model.predict(X_test)


# In[47]:


print("Confusion Matrix: \n", confusion_matrix(y_test, predictions3))
print("\n")
print(classification_report(y_test, predictions3))


# In[48]:


svm_acc = accuracy_score(y_test, predictions3)
print("Accuracy of SVM model is: ", svm_acc)


# In[50]:


plt.figure(figsize=(12,6))
model_acc = [knn_model_acc, rfc_acc, svm_acc]
model_name = ['KNN', 'RandomForests', 'SVM']
sns.barplot(x= model_acc, y=model_name, palette='magma')

