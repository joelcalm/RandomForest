from random_forest import RandomForest
import numpy as np
from Node import Leaf
from Criterion import SumSquareError

class RandomForestRegressor(RandomForest):
    # creates a leaf with the mean value of the given dataset
    def _make_leaf(self, dataset: 'Dataset', depth: int) -> 'Leaf':
        return Leaf(dataset.mean_value())

    # returns the mean value of the predictions
    def _combine_predictions(self, predictions: 'List[float]') -> float:
        return np.mean(predictions)
    
    # impurity measure used in the regressor
    def _make_impurity(self, name: str) -> 'SumSquareError':
        return SumSquareError()
