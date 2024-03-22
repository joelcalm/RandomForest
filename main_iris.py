import numpy as np
import sklearn.datasets
from random_forest_classifier import RandomForestClassifier
from mylogg import mylogger
import logging
import time

logger = mylogger("main", logging.INFO)
def main():
    try:
        t1 = time.time()
        iris = sklearn.datasets.load_iris()
        X, y = iris.data, iris.target # type: ignore
        logger.info("dataset downloaded")
        ratio_train, ratio_test = 0.7, 0.3
        num_samples, num_features = X.shape
        idx = np.random.permutation(range(num_samples))
        num_samples_train = int(num_samples*ratio_train)
        num_samples_test = int(num_samples*ratio_test)
        idx_train = idx[:num_samples_train]
        idx_test = idx[num_samples_train : num_samples_train+num_samples_test]
        X_train, y_train = X[idx_train], y[idx_train]
        X_test, y_test = X[idx_test], y[idx_test]
        max_depth = 10
        min_size_split = 5
        num_trees = 10
        num_random_features = int(np.sqrt(num_features))
        criterion = 'entropy'
        randomforest = RandomForestClassifier(max_depth, min_size_split, ratio_train, num_trees, num_random_features, criterion, do_multiprocessing=True,extra_trees='extra_trees')
        randomforest.fit(X_train,y_train)
        logger.info("randomforest fit done")
        ypred = randomforest.predict(X_test)
        num_samples_test = len(y_test)
        num_correct_predictions = np.sum(ypred == y_test)
        accuracy = num_correct_predictions/float(num_samples_test)
        print('accuracy {} %'.format(100*np.round(accuracy,decimals=2)))
        t2 = time.time()
        print("Total time: ", t2-t1)

        occurrences = randomforest.feature_importance()
        print('Iris occurrences for {} trees'.format(randomforest.num_trees))
        print(occurrences)

        #randomforest.print_trees()

    except Exception as msg:
        logger.critical("Exception occured: ", exc_info=True)
    

if __name__ == '__main__':
    main()
