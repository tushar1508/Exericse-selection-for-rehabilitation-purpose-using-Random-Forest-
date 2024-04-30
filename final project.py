#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv("synthetic_rehab_data.csv")


# In[ ]:


df.shape()


# In[15]:


df


# In[18]:


min_improvement_score = 3
filtered_df = df[df['Improvement Score'] > min_improvement_score]


# In[19]:


filtered_df


# In[20]:


filtered_df.shape


# In[71]:


importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.title("Feature Importances")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()


# In[72]:


X = filtered_df.drop(columns=["Patient ID"]) 
y = filtered_df[["Exercise 1", "Exercise 2", "Exercise 3"]]


# In[73]:


numerical_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
imputer = SimpleImputer(strategy="mean")
X[numerical_cols] = imputer.fit_transform(X[numerical_cols])


# In[74]:


categorical_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
for col in categorical_cols:
    X[col] = LabelEncoder().fit_transform(X[col])


# In[75]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[76]:


model = RandomForestClassifier(n_estimators=120)
model.fit(X_train, y_train)


# In[77]:


y_pred = model.predict(X_test)


# In[78]:


accuracies = {}
for i, exercise in enumerate(["Exercise 1", "Exercise 2", "Exercise 3"]):
    accuracies[exercise] = accuracy_score(y_test[exercise], y_pred[:, i])


# In[79]:


overall_accuracy = sum(accuracies.values()) / len(accuracies)

print("Overall Accuracy:", overall_accuracy)
print("Individual Accuracies:")
for exercise, acc in accuracies.items():
    print(f"{exercise}: {acc}")


# In[80]:


tree_to_visualize = model.estimators_[0]

# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(tree_to_visualize, feature_names=X_train.columns, class_names=model.classes_, filled=True)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




