from random_forest import RandomForest
import numpy as np
from Node import Leaf


class RandomForestIsolation(RandomForest):
    def __init__(self, ratio_samples: float, num_trees: int, num_random_features: int,
                 do_multiprocessing: bool, extra_trees: bool):
        self.max_depth: int = None
        self.min_size: int = 1
        self.criterion = None
        super().__init__(self.max_depth, self.min_size, ratio_samples, num_trees, self.criterion, num_random_features, do_multiprocessing,
                         extra_trees)

    def predict(self, X: 'np.ndarray') -> 'np.ndarray':
        self.test_size: int = len(X)
        return super().predict(X)
   
    def fit(self, X: 'np.ndarray') -> None:
        self.train_size: int = len(X)
        self.max_depth: int = int(np.log2(len(X)))
        y: 'np.ndarray' = np.zeros(self.train_size)
        super().fit(X, y)
    
    # creates a leaf with the mean value of the given dataset
    def _make_leaf(self, dataset: 'Dataset', depth: int) -> 'Leaf':
        return Leaf(depth)

    # returns the mean value of the predictions
    def _combine_predictions(self, predictions: 'List[Any]') -> float:
        # predictions are the depths h(x) in the decision trees for a given x
        Ehx: float = np.mean(predictions)  # mean depth
        cn: float = 2 * (np.log(self.train_size - 1) + 0.57721) - 2 * (self.train_size - 1) / float(self.test_size)
        return 2 ** (-Ehx / cn)

    # impurity measure used in the regressor
    def _make_impurity(self, name: str) -> None:
        return None

    def _best_split(self, idx_features: int, dataset: 'Dataset') -> tuple[int, float, float, list['Dataset']]:
        # find the best pair (feature, threshold) by exploring all possible pairs
        best_feature_index: int = np.Inf
        best_threshold: float = np.Inf
        minimum_cost: float = np.Inf
        best_split: list['Dataset'] = None
        
        val: float = np.random.uniform(np.min(dataset.X[:, idx_features]), np.max(dataset.X[:, idx_features]))
        # for val in values:
        left_dataset, right_dataset = dataset.split(idx_features, val)
        best_split, best_threshold, best_feature_index = [left_dataset, right_dataset], val, idx_features
        return best_feature_index, best_threshold, minimum_cost, best_split
