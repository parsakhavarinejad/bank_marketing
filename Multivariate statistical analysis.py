#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis (EDA)
# 
# Firstly, let's load the bank-additional-full.csv data and perform some exploratory data analysis on it. In Python, we can use the Pandas library to load and manipulate datasets. Here's  code:

# In[20]:


import pandas as pd

# Load the dataset
data = pd.read_csv('bank-additional-full.csv')

# Print the first 5 rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Get descriptive statistics of the numerical variables
print(data.describe())


# The above code will load the dataset, print the first 5 rows, check for missing values, and get descriptive statistics of the numerical variables.
# 
# 

# # Label Encoding
# 
# Next, we need to encode the categorical variables using label encoding. Label encoding is a process of converting categorical variables into numerical form so that they can be used in machine learning models. In Python, we can use Scikit-learn's LabelEncoder class to perform label encoding. Here's the code:

# In[21]:


from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder object
le = LabelEncoder()

# Encode the categorical variables
data['job'] = le.fit_transform(data['job'])
data['marital'] = le.fit_transform(data['marital'])
data['education'] = le.fit_transform(data['education'])
data['default'] = le.fit_transform(data['default'])
data['housing'] = le.fit_transform(data['housing'])
data['loan'] = le.fit_transform(data['loan'])
data['contact'] = le.fit_transform(data['contact'])
data['month'] = le.fit_transform(data['month'])
data['day_of_week'] = le.fit_transform(data['day_of_week'])
data['poutcome'] = le.fit_transform(data['poutcome'])
data['y'] = le.fit_transform(data['y'])

# Print the first 5 rows of the dataset after encoding
print(data.head())


# The above code will create a LabelEncoder object, encode the categorical variables, and print the first 5 rows of the dataset after encoding.

# # Train Test Split
# 
# After encoding the dataset, the next step is to split the data into training and testing sets. In Python, we can use Scikit-learn's train_test_split function to perform this operation. Here's the code:

# In[22]:


from sklearn.model_selection import train_test_split

# Split the data into X (features) and y (target)
X = data.drop('y', axis=1)
y = data['y']

# Split the data into training and testing sets with a 65:35 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Print the shape of the training and testing sets
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)


# ## randomization  👆
# 
# `random_state = 42` is a parameter commonly used in machine learning algorithms, including scikit-learn, to ensure reproducibility of the results. When this parameter is set to a specific value, such as 42, it will initialize the random number generator used by the algorithm with this seed value. This means that every time the algorithm is run with the same seed value, it will produce the same sequence of random numbers, which in turn will result in the same set of outputs.
# 
# The way the randomization occurs depends on the specific algorithm being used. However, in general, the algorithm will use a pseudo-random number generator (PRNG) to generate a sequence of apparently random numbers based on a predetermined algorithm. The PRNG generates these numbers deterministically, based on an initial seed value and a mathematical formula, so they are not truly random. However, the resulting sequence appears to be random for practical purposes.
# 
# By setting the `random_state` parameter to a fixed value, we can ensure that the same sequence of "random" numbers is generated each time we run the algorithm. This can be useful for testing and debugging, as well as ensuring that results are consistent across different runs.

# # Logistic Regression Model
# 
# Now that we have our training and testing sets, we can build a logistic regression model. Logistic regression is a method for analyzing a dataset in which there are one or more independent variables that determine an outcome. It is commonly used for binary classification problems. In Python, we can use Scikit-learn's LogisticRegression class to build the model. Here's The code:

# In[23]:


from sklearn.linear_model import LogisticRegression

# Create a Logistic Regression object
lr = LogisticRegression()

# Fit the model using the training data
lr.fit(X_train, y_train)

# Find the coefficients
coefs = lr.coef_

# Predict the target variable using the testing data
y_pred = lr.predict(X_test)

# Print the accuracy score of the model
print("Accuracy score:", lr.score(X_test, y_test))
print("The coefficients are: {}".format(coefs))


# The above code will create a Logistic Regression object, fit the model using the training data, predict the target variable using the testing data, and print the accuracy score of the model.

# # Summary of the Model
# 
# We can get a summary of the logistic regression model using the statsmodels library. Here's The example code:

# In[24]:


import statsmodels.api as sm


# In[25]:


# Add a constant column to the X_train dataset
X_train_sm = sm.add_constant(X_train)

# Create a Logistic Regression object using statsmodels
lr_sm = sm.Logit(y_train, X_train_sm)

# Fit the model using the training data
lr_sm_fit = lr_sm.fit()

# Print the summary of the model
print(lr_sm_fit.summary())


# # Discriminant Analysis
# 
# Another classification method we can use is discriminant analysis. Discriminant analysis is a statistical technique used to identify the underlying factors that differentiate between two or more groups. In Python, we can use Scikit-learn's LinearDiscriminantAnalysis class to perform discriminant analysis. Here's theexample code:

# In[26]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Create a Linear Discriminant Analysis object
lda = LinearDiscriminantAnalysis()

# Fit the model using the training data
lda.fit(X_train, y_train)

# Predict the target variable using the testing data
y_pred_lda = lda.predict(X_test)

# Print the accuracy score of the model
print("Accuracy score:", lda.score(X_test, y_test))


# The above code will create a Linear Discriminant Analysis object, fit the model using the training data, predict the target variable using the testing data, and print the accuracy score of the model.

# # Evaluation of the Model
# 
# To evaluate the performance of the logistic regression and discriminant analysis models, we can use metrics such as accuracy score, precision, recall, and F1-score. In Python, we can use Scikit-learn's classification_report function to get these metrics. Here's an example code:

# In[27]:


from sklearn.metrics import classification_report

# Print the classification report for logistic regression model
print("Logistic Regression")
print(classification_report(y_test, y_pred))

# Print the classification report for LDA model
print("Linear Discriminant Analysis")
print(classification_report(y_test, y_pred_lda))


# The above code will print the classification report for both the logistic regression and LDA models. The classification report includes information such as precision, recall, F1-score, and support for each class.
