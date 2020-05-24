# Spam_filter
This is simple Naive based classifier spam filter.

Naive Bayes is a supervised classification technique based on Bayes' Theorem with an assumption of independence among predictors. That is, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.

It is a popular technique for text categorization, judging documents as belonging to one category or the other (such as spam or legitimate, sports or politics, etc.) with word frequencies as features.

Goal: Previously unseen records should be assigned a class as accurately as possible

We have a bunch of emails classified as 'spam' and a bunch of emails classified as 'ham' (not spam)
The emails are first read and stored in a dataframe. They are then parsed using CountVectorizer
This information is used to train the model and its prediction is then tested with a sample input
Python Libraries used: pandas, numpy, io, os, CountVectorizer and MultinomialNB from sklearn

The Spam classifier classifies the given input as a spam/ham. A sample output is found here.

## Result
<img width="957" alt="result" src="https://user-images.githubusercontent.com/57071101/82764469-782a2900-9e2c-11ea-9812-0360ca0869da.PNG">

## APP LINK :
https://spam-filter-001.herokuapp.com/
