import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib
import pickle
messages = pd.read_csv('SMSSpamCollection', sep='\t',
                       names=["label", "message"])

# Data cleaning and preprocessing

nltk.download('stopwords')



ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()
pickle.dump(cv, open('transform.pkl','wb'))
y = pd.get_dummies(messages['label'])
y = y.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


clf = MultinomialNB()

clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print(score)
filename = 'spam_model.pkl'
pickle.dump(clf,open(filename, 'wb'))