import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

train_labels = pd.read_csv('resources/train_labels.csv', names=['label'], header=None)
train_examples = pd.read_csv('resources/train_examples.csv', names=['example'], engine='python', header=None, delimiter='\t\n')
test_examples = pd.read_csv('resources/test_examples.csv', names=['example'], engine='python', header=None, delimiter='\t\n')

print("WORDS BEFORE CLEANING:", train_examples['example'].apply(lambda x: len(x.split(' '))).sum())


def clean_text(text):
    text = BeautifulSoup(text, "html.parser").text
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text


start_cleaning = time.process_time()
print("CLEANING...")
train_examples['example'] = train_examples['example'].apply(clean_text)
end_cleaning = time.process_time()
print("TOOK", end_cleaning-start_cleaning, 'SECONDS')

print("WORDS AFTER CLEANING:", train_examples['example'].apply(lambda x: len(x.split(' '))).sum())
np.savetxt("resources/clean_train_examples.csv", train_examples, fmt="%s", delimiter='\t\n')

X = train_examples.example
y = train_labels.label
labels = np.unique(train_labels.label)


#######################################################################################################################################################
#--------------------GRIDSEARCH------------------------------------
pipeline = Pipeline([('vectorizer', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     # ('classifier', SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, random_state=42, max_iter=5, tol=None))
                     ('classifier', SVC())
                     ])
param_grid = {'classifier__C': [1, 10, 100, 1000]}
print("PARAM SEARCH...")
start_search = time.process_time()
search = GridSearchCV(pipeline, param_grid=param_grid, iid=True, cv=5,  n_jobs=-1,refit=True)
search.fit(X, y)
end_search = time.process_time()
print("TOOK", end_search-start_search, 'SECONDS')

print("\nBEST SCORE", search.best_score_)
print("BEST PARAMETER", search.best_params_)

#######################################################################################################################################################
#--------------------TRAIN/VALIDATION------------------------------------
#alpha = 0.96
# pipeline = Pipeline([('vectorizer', CountVectorizer()),
#                      ('tfidf', TfidfTransformer()),
#                      # ('classifier', SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, random_state=42, max_iter=5, tol=None))
#                      ('classifier', SVC())
#                      ])

# print("SPLITTING...")
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
#
# print("TRAINING...")
# start_training = time.process_time()
# pipeline.fit(X_train, y_train)
# end_training = time.process_time()
# print("TOOK", end_training-start_training, 'SECONDS')
#
# print("PREDICTING VAL...")
# start_val_predicting = time.process_time()
# val_predictions = pipeline.predict(X_test)
# end_val_predicting = time.process_time()
# print("TOOK", end_val_predicting - start_val_predicting, 'SECONDS')
#
# print('ACCURACY %s' % accuracy_score(val_predictions, y_test))
# report = classification_report(y_test, val_predictions, target_names=labels, output_dict=True)
# print(classification_report(y_test, val_predictions, target_names=labels))
# classification_report = pd.DataFrame(report).transpose()
# classification_report.to_csv('naive_bayes_classification_report.csv')

# preds = pd.DataFrame(data=val_predictions, columns=['Prediction'])
# preds.index.name = 'Id'
# actual = pd.DataFrame({'Actual': y_test})
# actual.index = preds.index
# print(actual)
# preds = preds.join(actual)
# print(preds)
# preds.to_csv('naive_bayes_val_predictions.csv')


#######################################################################################################################################################
#--------------------GET PREDICTIONS------------------------------------
#alpha = 0.96
#X_test = test_examples.example

# pipeline = Pipeline([('vectorizer', CountVectorizer()),
#                      ('tfidf', TfidfTransformer()),
#                      # ('classifier', SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, random_state=42, max_iter=5, tol=None))
#                      ('classifier', SVC())
#                      ])

# start_cleaning = time.process_time()
# print("CLEANING TESTING...")
# test_examples['example'] = test_examples['example'].apply(clean_text)
# end_cleaning = time.process_time()
# print("TOOK", end_cleaning-start_cleaning, 'SECONDS')
#
# print("WORDS AFTER CLEANING TESTING DATA:", test_examples['example'].apply(lambda x: len(x.split(' '))).sum())
# np.savetxt("resources/clean_test_examples.csv", test_examples, fmt="%s", delimiter='\t\n')

# print("TRAINING...")
# start_training = time.process_time()
# pipeline.fit(X, y)
# end_training = time.process_time()
# print("TOOK", end_training-start_training, 'SECONDS')

#print("PREDICTING TEST...")
# start_test_predicting = time.process_time()
# test_predictions = pipeline.predict(X_test)
# end_test_predicting = time.process_time()
# print("TOOK", end_test_predicting- start_test_predicting, 'SECONDS')
#
# test_preds = pd.DataFrame(data=test_predictions, columns=['Category'])
# test_preds.index.name = 'Id'
# test_preds.to_csv('naive_bayes_test_predictions.csv')
