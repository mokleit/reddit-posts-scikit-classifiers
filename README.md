Mo Kleit

This repository includes different machine learning classifiers implemented using sci-kit functionnalities for text classification in the context of a Kaggle competition.

We train our models on 70000 sub-reddit posts and compute predictions (topic of sub-reddit post: nba, nfl, soccer, anime, etc) for 30,000 test examples. 

# HOW TO RUN
In order to obtain the predictions, simply run the x_classifier.py script under each classifier module. The predictions will be saved
in a csv file with the name of the classifier under the main folder.

# STRUCTURE
The project is divided into 5 sub modules.

The resources module has the data stored as csv.
The import.py file is used to save the data as csv in the resources folder.
We have trained and tested 4 different classifiers.
Each classifier has a module:
    - adaboost
    - logistic_regression
    - naive_bayes
    - linear svm

In each classifier module, there are two python scripts:
    - find_best_x_estimator.py: used for tuning hyper-parameters using gridsearch
    - x_classifier.py: used after estimating best hyper-parameters to find predictions

# CODE ORGANISATION (same for all classifiers)
    - find_best_x_estimator.py
            1. We import all train examples, test examples and train labels
            2. Create a pipeline with a Vectorizer, Tfid Transformer and the Classifier we want to train
            3. Define a param grid for hyper-parameter tuning
            4. Fit the classifier under different parameters
            5. Return best parameter
    - x_classifier.py
            1. We import all train examples, test examples and train labels
            2. Get indices of test examples that are in common with train examples
            3. Create a pipeline with a Vectorizer, Tfid Transformer and the Classifier we want to train
            4. Fit the classifier using best parameter found in find_best_x_estimator.py
            5. Get predictions and correct mistakes done using common elements from step 2

# RESULTS

The classifier that gave us the best accuracy is Naive Bayes. In order of accuracy, we can class the classifiers
as follows:

    1. Naive Bayes
    2. Linear SVM
    3. Logistic Regression
    4. Adaboost





