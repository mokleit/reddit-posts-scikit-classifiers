import inline as inline
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import nltk
from sklearn.naive_bayes import ComplementNB

#best alphas=0.22,0.29

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

train_labels = pd.read_csv('resources/train_labels.csv', names=['label'], header=None)
train_examples = pd.read_csv('resources/train_examples.csv', names=['example'], engine='python', header=None, delimiter='\t\n')
test_examples = pd.read_csv('resources/test_examples.csv', names=['example'], engine='python', header=None, delimiter='\t\n')

# print("WORDS BEFORE CLEANING:", train_examples['example'].apply(lambda x: len(x.split(' '))).sum())
#
#
# def clean_text(text):
#     text = BeautifulSoup(text, "html.parser").text
#     text = text.lower()
#     text = REPLACE_BY_SPACE_RE.sub(' ', text)
#     text = BAD_SYMBOLS_RE.sub('', text)
#     text = ' '.join(word for word in text.split() if word not in STOPWORDS)
#     return text
#
#
# start_cleaning = time.process_time()
# print("CLEANING TRAINING...")
# train_examples['example'] = train_examples['example'].apply(clean_text)
# end_cleaning = time.process_time()
# print("TOOK", end_cleaning-start_cleaning, 'SECONDS')
#
# print("WORDS AFTER CLEANING TRAINING DATA:", train_examples['example'].apply(lambda x: len(x.split(' '))).sum())
# np.savetxt("resources/clean_train_examples.csv", train_examples, fmt="%s", delimiter='\t\n')

X = train_examples.example
y = train_labels.label
labels = np.unique(train_labels.label)

#######################################################################################################################################################
#--------------------GRIDSEARCH------------------------------------
# pipeline = Pipeline([('vectorizer', CountVectorizer()),
#                      ('tfidf', TfidfTransformer()),
#                      ('classifier', AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=1000)())
#                      ])
# param_grid = {'classifier__n_estimators': [500, 1000, 1500, 2000]}
# print("PARAM SEARCH...")
# start_search = time.process_time()
# search = GridSearchCV(pipeline, param_grid=param_grid, iid=True, cv=5,  n_jobs=-1,refit=True)
# search.fit(X, y)
# end_search = time.process_time()
# print("TOOK", end_search-start_search, 'SECONDS')
#
# print("\nBEST SCORE", search.best_score_)
# print("BEST PARAMETER", search.best_params_)

#######################################################################################################################################################

print("SPLITTING...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

pipeline = Pipeline([('vectorizer', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('classifier', AdaBoostClassifier(ComplementNB(alpha=1.2400000000000002), n_estimators=500))
                     ])

print("TRAINING...")
start_training = time.process_time()
pipeline.fit(X_train, y_train)
end_training = time.process_time()
print("TOOK", end_training-start_training, 'SECONDS')

print("PREDICTING TEST...")
start_test_predicting = time.process_time()
test_predictions = pipeline.predict(X_test)
end_test_predicting = time.process_time()
print("TOOK", end_test_predicting- start_test_predicting, 'SECONDS')

print('ACCURACY %s' % accuracy_score(test_predictions, y_test))
report = classification_report(y_test, test_predictions, target_names=labels, output_dict=True)
print(classification_report(y_test, test_predictions, target_names=labels))
# classification_report = pd.DataFrame(report).transpose()
# classification_report.to_csv('adaboost_classification_report.csv')


