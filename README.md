# Bank Marketing Analysis

This repository contains code and data for a machine learning project on bank marketing analysis. The goal of this project is to predict whether clients will subscribe to a term deposit based on a variety of demographic, economic, and social factors.

## Data

The data used in this project comes from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing). It consists of 45,211 records with 20 input features and one target variable (whether the client subscribed to the term deposit or not).

## Exploratory Data Analysis (EDA)

Before building any models, we explored the data to gain insights into the relationships between the input features and the target variable. We visualized the distributions of each feature and the correlations between them using various statistical plots and graphs.

## Label Encoding

Since some of the input features are categorical variables, we used label encoding to convert them into numerical values that could be used in our models.

## Train-Test Split

To evaluate the performance of our models, we split the data into training and testing sets using a randomization technique. We used a 65-35 split, where 65% of the data was used for training and 35% for testing.

## Logistic Regression Model

We built a logistic regression model to predict the probability of a client subscribing to a term deposit. We used the scikit-learn library to fit the model to the training data and evaluated its performance on the testing data using various metrics, such as accuracy, precision, recall, and F1 score.

## Discriminant Analysis

In addition to the logistic regression model, we also built a discriminant analysis model to predict the probability of a client subscribing to a term deposit. We used the same evaluation metrics as the logistic regression model to compare their performance.

## Evaluation

We evaluated the performance of both models on the testing data and compared their results. We also visualized the results using various plots and graphs to gain insights into the strengths and weaknesses of each model.

### Conclusion

Overall, this project demonstrates how machine learning can be used to predict the likelihood of clients subscribing to a term deposit based on various demographic, economic, and social factors. By exploring the data, building models, and evaluating their performance, we gained valuable insights into the relationships between the input features and the target variable.

