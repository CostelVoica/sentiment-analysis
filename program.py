import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.model_selection as ms
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Load the data with pd.read_pickle() and turn the texts into a document-term matrix

amazon = pd.read_pickle("Amazon_reviews.pkl")
vectorizer = CountVectorizer()
dtm = vectorizer.fit_transform(amazon['text'])
df_dtm = pd.DataFrame(data = dtm.toarray(),
                      columns = vectorizer.get_feature_names())

# Analyzing the data:

# 1. Across all texts, these are the five most common words, and this is how many times they occur:

wordcount = df_dtm.sum()
wordcount.sort_values(ascending=False)

#the           15704
#and            9558
#it             8725
#to             8170
#is             4908

# Performing a 60/20 train/test split

X = df_dtm
y = amazon['label']

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.4, random_state=12345)


# Training a logistic regression classifier on the training set to predict whether a review is positive or negative.

log_reg = LogisticRegression(solver='newton-cg')
print(log_reg.fit(X_train, y_train))
y_predict = log_reg.predict(X_test)
print(metrics.accuracy_score(y_test, y_predict))

# Accuracy is 0.700625

# Finding the word most strongly associated with a positive review

import numpy as np
#find largest of this and see what its index is. then plug this index into the dataframe 
print(log_reg.coef_)
np.max(log_reg.coef_)
result = np.where(log_reg.coef_ == np.max(log_reg.coef_))
#index is 3695
df_dtm.iloc[:, 3695]

# "easier" is most strongly associated with a positive review

# Naive Bayes Model

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

X = df_dtm
y = amazon['label']

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.4, random_state=12345)


text_clf = Pipeline([
                     ('clf', MultinomialNB()),
])


text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)
predicted
print(metrics.accuracy_score(y_test, predicted))
#accuracy is 0.6975 slightly lower than the original logistic reg model
print(metrics.classification_report(y_test, predicted))


#instead of having a 60/40 train/test split, I changed it to a 80/20 split

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=12345)
text_clf = Pipeline([
                     ('clf', MultinomialNB()),
])


text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)
predicted
print(metrics.accuracy_score(y_test, predicted))
print(metrics.classification_report(y_test, predicted))

# notice that the accuracy,precision,recall and f1-score are higher in the 80/20 split
# overall, the 80/20 split is a better choice than the 60/40 one because of improvement in all the clas_report metrics
