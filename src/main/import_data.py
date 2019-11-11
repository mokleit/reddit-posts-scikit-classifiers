import numpy as np
import os


class ImportData:

    path = os.path.dirname(os.path.abspath(__file__))
    data_train = os.path.join(path, "resources/data_train.pkl")
    data_test = os.path.join(path, "resources/data_test.pkl")

    train_data = np.load(data_train, allow_pickle=True)
    train_examples = [ex.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ') for ex in train_data[0]]
    train_labels = train_data[1]
    np.savetxt("resources/train_examples.csv", train_examples, fmt="%s", delimiter='\t\n')
    np.savetxt("resources/train_labels.csv", train_labels, fmt="%s")
    np.savetxt("resources/train_data.csv", np.column_stack((train_examples, train_labels)), fmt="%s", delimiter='\t\t')

    test_examples = [test.replace('\n',' ').replace('\r', ' ').replace('\t', ' ') for test in np.load(data_test, allow_pickle=True)]
    np.savetxt("resources/test_examples.csv", test_examples, fmt="%s", delimiter='\t\n')
