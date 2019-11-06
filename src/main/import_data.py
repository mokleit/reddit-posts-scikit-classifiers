import numpy as np
import os


class ImportData:

    path = os.path.dirname(os.path.abspath(__file__))
    data_train = os.path.join(path, "resources/data_train.pkl")
    data_test = os.path.join(path, "resources/data_test.pkl")

    train_data = np.load(data_train, allow_pickle=True)
    train_examples = [ex.replace('\n', ' ').replace('\r', ' ') for ex in train_data[0]]
    np.savetxt("resources/train_examples.csv", train_examples, fmt="%s")
    np.savetxt("resources/train_labels.csv", train_data[1], fmt="%s")

    test_data = [test.replace('\n',' ').replace('\r', ' ') for test in np.load(data_test, allow_pickle=True)]
    np.savetxt("resources/test_data.csv", test_data, fmt="%s")
