#!/usr/bin/env python
# coding: utf-8

# # Loan Default Prediction

# In[1]:


#importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# loading the dataset 
df = pd.read_csv("Dataset (2).csv")
df.head()


# # Data Preprocessing 1

# In[3]:


df.info()


# In[4]:


#checking the shape of the dataset
df.shape


# <b> checking for null and missing values </b>

# In[5]:


df.isnull().sum()


# <b> checking datatype of coumns</b>

# In[6]:


# columns datatype
df.dtypes


# <b>checking duplicate values</b>

# In[7]:


# duplicate values
df.duplicated().sum()


# <b>Descriptive Statistics</b>

# In[8]:


df.describe()


# In[9]:


for column in df.columns:
    print(column,":",df[column].unique())
    print("*"*20)


# <b>Outlier Analysising</b>

# In[10]:


df_list = df.drop(columns=['credit.policy','purpose','inq.last.6mths','delinq.2yrs','pub.rec','not.fully.paid'])


# In[11]:


df_list.head()


# In[12]:


df.head()


# In[13]:


for column in df_list.columns:
    sns.boxplot(column,data=df_list)
    plt.show()


# In[14]:


# There are outliers in 'int.rate','installment','log.annual.inc','fico','days.with.cr.line',and 'revol.bal'. 
# As of now, lets proceed the data without outlier treatment.


# # Explorative Data Analysis 
# 
# In the exploratory data analysis, I will be looking at the distribution of the data, the
# coorelation between features and the target variable and the relationship between the
# features and the target variable. I will start by looking at the distribution of the data,
# followed by the relationship between the features and the target variable.

# In[15]:


df.head()


# In[16]:


df.columns


# In[17]:


cat_cols = ['credit.policy','purpose','inq.last.6mths','delinq.2yrs','pub.rec','not.fully.paid']
con_cols = list(set(df.columns)-set(cat_cols)) 
print("continuous:", con_cols)
print("categorical:",cat_cols) 


# <h3>Univariate Anlaysis</h3>

# <b> Continuous Data - Histogram or KDE </b>

# In[18]:


for col in con_cols:
    sns.kdeplot(x=col,data=df)
    plt.show()


# In[19]:


for col in con_cols:
    print(col,":", df[col].skew())


# In[20]:


# revol.bal,installment, and days.with.cr.line  are right skewed 


# <b> Categorical Data - Bar graphs</b>

# In[21]:



for col in cat_cols:
    plt.figure(figsize=(10,5))
    sns.countplot(x=col,data=df)
    plt.show()


# In[22]:


for col in cat_cols:
    print(col,":")
    print( df[col].value_counts(normalize=True).round(2))
    print("*"*50)


# In[23]:


# Inferences

# 1) mostly credit.policy 1 is applied for user.
# 2) inq.last.6mths is mostly has between 0 to 3.
# 3) purpose of loan is 'debt_consolidation', 'credit_card' and 'all_other'.
# 4) delinq.2yrs, pub.rec, and not.fully.paid are mostly has 0.


# <h3>Bivariate Analysis</h3>

# In[24]:


# Target Variable Vs Independent variable

# Default vs Other variable


# In[25]:


# Target vs Continuous data

# Categorical vs Continuous data

# box plot


# In[26]:


for col in con_cols:
    sns.boxplot(x='not.fully.paid', y=col, data=df)
    plt.show()


# In[27]:


# 'fico' of 'not.fully.paid' is lower than 'fully.paid'
# 'revol.util', 'int.rate' of 'not.fully.paid' is higher than 'fully.paid'
# Other has no impact on 'not.fully.paid'


# In[28]:


df.columns


# In[29]:


# Target vs Categorical data

# Categorical vs Categorical data

# Stacked bar plot


# In[30]:


for col in cat_cols:
    pd.crosstab(df[col],df['not.fully.paid']).plot(kind='bar')
    plt.show()


# In[31]:


# 'credit.policy': 1, 'purpose':'debt_consolidation', and 'all_other', 'inq.last.6mths': 0 - 3, 'delinq.2yrs': 0, 
# 'pub.rec': 0 are having more defaulter count.


# <h1> Data Preprocessing-2 </h1>

# # Coorelation Matrix Heatmap

# In[32]:


plt.figure(figsize=(12,12))
sns.heatmap(df.corr(),annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[33]:


# 'credit.policy', 'log.annual.inc',and 'days.with.cr.line' are negatively correlated with 'not.fully.paid'.
# 'int.rate', 'revol.util' and 'inq.last.6mths' are positively correlated with 'not.fully.paid'.
# All other have no significant coorelation with  not.fully.paid'. So, I will proceed to model building.


# In[34]:


# Correlation between all the columns of DataFrame.
#df2 = df.corr()
#print(df2)


# ## Scaling

# In[35]:


# Scaling  - Continuous data


# In[36]:


from sklearn.preprocessing import StandardScaler


# In[37]:


ss = StandardScaler()
x_con_scaled = pd.DataFrame(ss.fit_transform(df[con_cols]), columns=con_cols, index = df.index)
x_con_scaled.head()


# ## Encoding

# In[38]:


# Categorical data - Numerical Data

# One hot encoding


# In[39]:


cat_cols.remove('not.fully.paid')


# In[40]:


cat_cols


# In[41]:


# function to execute one hot encoding 
x_col_enc = pd.get_dummies(df[cat_cols], drop_first=True)


# ## Marge Cat and Con Data

# In[42]:


x_final = pd.concat([x_con_scaled, x_col_enc], axis=1)
x_final


# # Train Test Split

# In[43]:


from sklearn.model_selection import train_test_split


# In[44]:


y = df['not.fully.paid']


# In[45]:


# Training = 80%, Testing = 20% ( Random Selection of train_test_split)


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(x_final, y, test_size=0.2, random_state=42)


# # Implementation of Decision Tree Classifiere

# In[47]:


from sklearn.tree import DecisionTreeClassifier


# In[48]:


dt_classifire = DecisionTreeClassifier(criterion='gini')
dt_classifire.fit(X_train,y_train)


# In[49]:


from sklearn import tree


# In[50]:


plt.figure(figsize=(20,20))
tree.plot_tree(dt_classifire)
plt.show()


# ## Train & Test Score

# In[51]:


from sklearn.metrics import confusion_matrix, classification_report


# In[52]:


y_train_pred = dt_classifire.predict(X_train)
y_test_pred = dt_classifire.predict(X_test)


# In[53]:


print('Train Confusion Matrix')
sns.heatmap(confusion_matrix(y_train,y_train_pred), annot=True, fmt='.0f')
plt.show()

print('Test Confusion Matrix')
sns.heatmap(confusion_matrix(y_test,y_test_pred), annot=True, fmt='.0f')
plt.show()


# In[54]:


print('Train Classification Report:')
print(classification_report(y_train,y_train_pred))
print('-'*100)

print('Test Classification Report:')
print(classification_report(y_test,y_test_pred))
print('-'*100)


# In[55]:


# Full grown decision tree will always overfit.


# ### Distribution Plot

# In[56]:


ax = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
sns.distplot(y_test_pred, hist=False, color="b", label="Fitted Values" , ax=ax)


# # Cross Validation Score

# In[57]:


from sklearn.model_selection import cross_val_score


# In[58]:


scores = cross_val_score(dt_classifire, X_train, y_train, scoring='recall', cv=5)
print('Score:', scores)
print('Avg. Score', np.mean(scores))
print('Std. Score', np.std(scores))


# # Hyperparameter Tuning for Decision Tree Classifier

# In[59]:


from sklearn.model_selection import GridSearchCV


# In[60]:


#defining parameter range
grid = {'max_depth':range(1,10), 'min_samples_split': range(4,8,1), 
       'max_leaf_nodes':range(3,10,1)}

#Creating grid search object
grid_src = GridSearchCV(DecisionTreeClassifier(class_weight='balanced'), param_grid=grid, cv=5,scoring= 'recall')

#Fitting the grid search object to the training data
grid_src.fit(X_train,y_train)


# In[61]:


#Printing the best parameters
print('Best parameters found: ', grid_src.best_params_)

print('Best cv result found: ',grid_src.cv_results_)


# In[62]:


pd.DataFrame(grid_src.cv_results_)


# In[63]:


grid_src.best_estimator_


# # Performance Matrics

# In[64]:


dt_tunned = DecisionTreeClassifier(criterion='gini', max_depth=3, max_leaf_nodes=8, 
                                   min_samples_split=5, class_weight='balanced')
dt_tunned.fit(X_train, y_train)


# In[ ]:





# ### Train & Test Score

# In[65]:


y_train_pred = dt_tunned.predict(X_train)
y_test_pred = dt_tunned.predict(X_test)


# In[66]:


print('Train Confusion Matrix')
sns.heatmap(confusion_matrix(y_train,y_train_pred), annot=True, fmt='.0f')
plt.show()

print('Test Confusion Matrix')
sns.heatmap(confusion_matrix(y_test,y_test_pred), annot=True, fmt='.0f')
plt.show()


# In[67]:


print('Train Classification Report:')
print(classification_report(y_train,y_train_pred))
print('-'*100)

print('Test Classification Report:')
print(classification_report(y_test,y_test_pred))
print('-'*100)


# In[68]:


# Before HT:

# Train Recall - 100%
# Test Recall - 21%

# After HT:

# Train Recall - 45%
# Test Recall - 44%

# Handled the overfitting and built and generalized model


# ### Distribution Plot

# In[69]:


ax = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
sns.distplot(y_test_pred, hist=False, color="b", label="Fitted Values" , ax=ax)


# In[ ]:





# ### Cross Validation Score

# In[70]:


scores = cross_val_score(dt_tunned, X_train, y_train, scoring='recall', cv=5)
print('Score:', scores)
print('Avg. Score', np.mean(scores))
print('Std. Score', np.std(scores))


# In[71]:


# Hyperparameter tuning is mandatory in Decision Tree. 
# If we go with default values, we will get a full grown 
# Decision Tree which will overfit.


# # Ensemble Methods

# ## Bagging Classifier

# In[72]:


from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression


# In[73]:


bag_class_log = BaggingClassifier(base_estimator=LogisticRegression(),
                                 n_estimators=20, max_samples=0.8, max_features=0.5,
                                 bootstrap=True, bootstrap_features=False)
bag_class_log.fit(X_train, y_train)


# In[74]:


# To Find the best parameters, perform Grid Search


# ### Train & Test Score

# In[75]:


y_train_pred = bag_class_log.predict(X_train)
y_test_pred = bag_class_log.predict(X_test)


# In[76]:


print('Train Confusion Matrix')
sns.heatmap(confusion_matrix(y_train,y_train_pred), annot=True, fmt='.0f')
plt.show()

print('Test Confusion Matrix')
sns.heatmap(confusion_matrix(y_test,y_test_pred), annot=True, fmt='.0f')
plt.show()


# In[77]:


print('Train Classification Report:')
print(classification_report(y_train,y_train_pred))
print('-'*100)

print('Test Classification Report:')
print(classification_report(y_test,y_test_pred))
print('-'*100)


# In[78]:


# After Bagging Classifier:

# Train Recall - 0%
# Test Recall - 0%

# Handled the overfitting and built and generalized model


# ### Distribution Plot

# In[79]:


ax = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
sns.distplot(y_test_pred, hist=False, color="b", label="Fitted Values" , ax=ax)


# In[ ]:





# ### Cross Validation Score

# In[80]:


scores = cross_val_score(bag_class_log, X_train, y_train, scoring='recall', cv=5)
print('Score:', scores)
print('Avg. Score', np.mean(scores))
print('Std. Score', np.std(scores))


# # Random Forest Classifier

# In[81]:


from sklearn.ensemble import RandomForestClassifier


# In[82]:


#creating Random Forest Classifer object
rf = RandomForestClassifier(n_estimators=150, criterion='gini', max_depth=5,
                           min_samples_split=4, max_leaf_nodes=8, max_samples=0.7,
                           max_features=0.5, bootstrap=True, class_weight='balanced')
rf.fit(X_train, y_train)


# ### Train & Test Score

# In[83]:


y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)


# In[84]:


print('Train Confusion Matrix')
sns.heatmap(confusion_matrix(y_train,y_train_pred), annot=True, fmt='.0f')
plt.show()

print('Test Confusion Matrix')
sns.heatmap(confusion_matrix(y_test,y_test_pred), annot=True, fmt='.0f')
plt.show()


# In[85]:


print('Train Classification Report:')
print(classification_report(y_train,y_train_pred))
print('-'*100)

print('Test Classification Report:')
print(classification_report(y_test,y_test_pred))
print('-'*100)


# In[86]:


# After Bagging Classifier:

# Train Recall - 58%
# Test Recall - 56%

# Handled the overfitting and built and generalized model


# ### Distribution Plot

# In[87]:


ax = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
sns.distplot(y_test_pred, hist=False, color="b", label="Fitted Values" , ax=ax)


# ### Cross Validation Score

# In[88]:


scores = cross_val_score(rf, X_train, y_train, scoring='recall', cv=5)
print('Score:', scores)
print('Avg. Score', np.mean(scores))
print('Std. Score', np.std(scores))


# In[ ]:





# In[ ]:





# In[ ]:




