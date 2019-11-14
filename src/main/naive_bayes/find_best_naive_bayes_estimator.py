import pandas as pd
from sklearn.naive_bayes import  ComplementNB
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


train_labels = pd.read_csv('resources/train_labels.csv', names=['label'], header=None)
train_examples = pd.read_csv('resources/train_examples.csv', names=['example'], engine='python', header=None, delimiter='\t\n')
test_examples = pd.read_csv('resources/test_examples.csv', names=['example'], engine='python', header=None, delimiter='\t\n')

X = train_examples.example
y = train_labels.label

pipeline = Pipeline([('vectorizer', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('classifier', ComplementNB())
                     ])

param_grid = {'classifier__alpha': [0.89, 0.96, 1.5, 1.75, 2.51, 2.049, 1.93, 1.2400000000000002]}
search = GridSearchCV(pipeline, param_grid=param_grid, iid=True, cv=10,  n_jobs=-1,refit=True)
search.fit(X, y)

print("\nBEST SCORE", search.best_score_)
print("BEST PARAMETER", search.best_params_)
