import numpy as np
import sklearn.datasets
from random_forest_classifier import RandomForestClassifier
from mylogg import mylogger
import logging
import pickle
import matplotlib.pyplot as plt

def load_MNIST():
            with open("C:/Users/hecto/OneDrive/Escritorio/matcad/primer/quatri 2/Programació Orientada als Objectes/practiques/RandomForest/RandomForest/RandomForest/mnist.pkl", 'rb') as f:
                mnist = pickle.load(f)
            return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

logger = mylogger("main", logging.INFO)

def main():
    try:
        X_train, y_train, X_test, y_test = load_MNIST()
        logger.info("dataset downloaded")
        plt.close("all")
        plt.figure()
        for i in range(10):
            for j in range(20):
                n_sample = 20*i +j
                plt.subplot(10,20, n_sample+1)
                plt.imshow(np.reshape(X_train[n_sample], (28,28)), interpolation="nearest",cmap=plt.cm.gray) # type: ignore
                plt.title(str(y_train[n_sample]),fontsize=8)
                plt.axis("off")            
        
        
        max_depth = 20
        min_size_split = 20
        ratio_samples = 0.25
        num_trees = 80
        num_random_features = 28
        criterion = 'gini'
        randomforest = RandomForestClassifier(max_depth, min_size_split, ratio_samples,
                                    num_trees, num_random_features, criterion,do_multiprocessing=True,extra_trees='extra_trees')
        randomforest.fit(X_train,y_train)
        logger.info("randomforest fit done")
        ypred = randomforest.predict(X_test)
        num_samples_test = len(y_test)
        num_correct_predictions = np.sum(ypred == y_test)
        accuracy = num_correct_predictions/float(num_samples_test)
        print('accuracy {} %'.format(100*np.round(accuracy,decimals=2)))


        
        occurrences = randomforest.feature_importance()
        ima = np.zeros(28*28)
        for k in occurrences.keys():
            ima[k] = occurrences[k]
        plt.figure()
        plt.imshow(np.reshape(ima,(28,28)))
        plt.colorbar()
        plt.title('Feature importance MNIST')
        plt.show()

    except Exception as msg:
        logger.critical("Exception occured: ", exc_info=True)
    

if __name__ == '__main__':
    main()
