import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


train_labels = pd.read_csv('../resources/train_labels.csv', names=['label'], header=None)
train_examples = pd.read_csv('../resources/train_examples.csv', names=['example'], engine='python', header=None, delimiter='\t\n')
test_examples = pd.read_csv('../resources/test_examples.csv', names=['example'], engine='python', header=None, delimiter='\t\n')

X = train_examples.example
y = train_labels.label
labels = np.unique(train_labels.label)

pipeline = Pipeline([('vectorizer', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('classifier', AdaBoostClassifier(DecisionTreeClassifier(max_depth=1)))
                     ])
param_grid = {'classifier__n_estimators': [500, 1000, 1500, 2000]}

search = GridSearchCV(pipeline, param_grid=param_grid, iid=True, cv=5,  n_jobs=-1,refit=True)
search.fit(X, y)

print("\nBEST SCORE", search.best_score_)
print("BEST PARAMETER", search.best_params_)
