from random_forest import RandomForest
import numpy as np
from Node import Leaf
from Criterion import Gini, Entropy

class RandomForestClassifier(RandomForest):
    def _make_leaf(self, dataset: 'Dataset', depth: int) -> 'Leaf':
        # creates a leaf node with the most frequent class in the dataset.
        return Leaf(dataset.most_frequent_label())
        
    def _combine_predictions(self, predictions: 'List[Any]') -> 'Any':
        # returns the most probable prediction
        return np.argmax(np.bincount(predictions))

    def _make_impurity(self, name: str) -> 'Criterion':
        # returns the criterion that we want
        if name == 'gini':
            return Gini()
        elif name == 'entropy':
            return Entropy()
