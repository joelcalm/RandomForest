import numpy as np
import matplotlib.pyplot as plt
from random_forest_isolation import RandomForestIsolation
import pickle

def load_MNIST():
    with open("C:/Users/joelc/OneDrive - UAB/UNI/1r CURS/2n Semestre/POO/RandomForest/mnist.pkl", "rb") as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


def test_MNIST(digit=8):
    X_train, y_train, X_test, y_test = load_MNIST()
    X = np.vstack([X_train,X_test]) # join train and test samples
    y = np.concatenate([y_train,y_test])
    idx_digit = np.where(y==digit)[0]
    X = X[idx_digit]
    downsample = 2 # reduce the number of features = pixels
    X2 = np.reshape(X,(len(X),28,28))[:,::downsample,::downsample]
    X2 = np.reshape(X2,(len(X2),28*28//downsample**2))
    num_samples = len(X)
    
    ratio_samples, num_trees, num_random_features, do_multiprocessing, extra_trees = 0.5, 2000, 1, False, 'none'
    iso = RandomForestIsolation(ratio_samples, num_trees, num_random_features, do_multiprocessing, extra_trees)

    iso.fit(X)
    scores = iso.predict(X)
    plt.figure(), plt.hist(scores, bins=100)
    plt.title('histogram of scores')
    percent_anomalies = 0.5
    num_anomalies = int(percent_anomalies * num_samples / 100.)
    idx = np.argsort(scores)
    idx_predicted_anomalies = idx[-num_anomalies:]
    precision = y[idx_predicted_anomalies].sum()/num_anomalies
    print('precision for {} % anomalies : {} %' \
    .format(percent_anomalies,100*precision))
    recall = y[idx_predicted_anomalies].sum()/y.sum()
    print('recall for {} % anomalies : {} %' \
    .format(percent_anomalies,100*recall))
    plt.show(block=True)
    
if __name__ == '__main__':
    test_MNIST()