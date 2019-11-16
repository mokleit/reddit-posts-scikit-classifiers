import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline

n_estimators = 1000

train_labels = pd.read_csv('../resources/train_labels.csv', names=['label'], header=None)
train_examples = pd.read_csv('../resources/train_examples.csv', names=['example'], engine='python', header=None, delimiter='\t\n')
test_examples = pd.read_csv('../resources/test_examples.csv', names=['example'], engine='python', header=None, delimiter='\t\n')

train = np.array(train_examples.example)
test = np.array(test_examples.example)
test_index = np.where(np.in1d(test, train))[0]

X = train_examples.example
y = train_labels.label
X_test = test_examples.example

pipeline = Pipeline([('vectorizer', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('classifier', AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=n_estimators))                     ])

pipeline.fit(X, y)
test_predictions = pipeline.predict(X_test)

for (i,x) in enumerate(test_index):
    train_index = np.where(np.in1d(train, test[x]))[0][0]
    test_predictions[x] = train_labels.label[train_index]

test_preds = pd.DataFrame(data=test_predictions, columns=['Category'])
test_preds.index.name = 'Id'
test_preds.to_csv('adaboost_test_predictions.csv')
