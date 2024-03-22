import numpy as np
from abc import ABC, abstractmethod
from Node import Parent, Node
from dataset import Dataset
from visitor import FeatureImportance, PrinterTree
import time
import multiprocessing
from mylogg import mylogger
import logging
from typing import Tuple,List, Dict, Any

logger = mylogger("random_forest_classifier", logging.INFO)


class RandomForest(ABC):
    def __init__(self, max_depth: int, min_size: int, ratio_samples: float, num_trees: int,
                 num_random_features: int, criterion: str, do_multiprocessing: bool,
                 extra_trees: str):
        self.max_depth = max_depth
        self.min_size = min_size
        self.ratio_samples = ratio_samples
        self.num_trees = num_trees
        self.num_random_features = num_random_features
        self.criterion = self._make_impurity(criterion)
        self.do_multiproessing = do_multiprocessing
        self.extra_trees = extra_trees
        self.num_processes = multiprocessing.cpu_count()
        self.ocurrences = {}
        self.depth = 0
        self.actual_tree = 1
    
    def predict(self, X: np.ndarray[float]) -> np.ndarray:
        ypred = []
        for x in X:
            predictions = [root.predict(x) for root in self.decision_trees]
            # majority voting
            ypred.append(self._combine_predictions(predictions))
        return np.array(ypred)

    def fit(self, X: np.ndarray[float], y: np.ndarray) -> None:
        try:
            # a pair (X,y) is a dataset, with its own responsibilities
            dataset = Dataset(X, y)
            if self.do_multiproessing is False:
                self._make_decision_trees(dataset)
            elif self.do_multiproessing is True:
                self._make_decision_trees_multiprocessing(dataset)
        except Exception as msg:
            logger.critical("Exception occurred (while setting multiprocessing) ", exc_info=True)

    def _worker(self, subset: 'Dataset') -> 'Node':
        # takes in a subset and returns a decision tree.
        return self._make_node(subset, 1)

    def _make_decision_trees_multiprocessing(self, dataset: 'Dataset') -> None:
        try:
            t1 = time.time()
            with multiprocessing.Pool(processes=self.num_processes) as pool:
                self.decision_trees = pool.starmap(self._worker, [(dataset.random_sampling(self.ratio_samples),) for _ in range(self.num_trees)])
            t2 = time.time()
            logger.info('{} seconds per tree'.format((t2-t1)/self.num_trees))
        except Exception as msg:
            logger.critical("Exception occurred: ", exc_info=True)

    def _make_decision_trees(self, dataset: 'Dataset') -> None:
        self.decision_trees = []
        t1 = time.time()
        for i in range(self.num_trees):
            # sample a subset of the dataset with replacement using
            # np.random.choice() to get the indices of rows in X and y
            subset = dataset.random_sampling(self.ratio_samples)
            tree = self._make_node(subset, 1)  # the root of the decision tree
            self.decision_trees.append(tree)
        t2 = time.time()
        print("{} seconds per tree".format((t2-t1)/self.num_trees))

    def _make_node(self, dataset: 'Dataset', depth: int) -> 'Node':
        logger.debug("Making node at depth {}".format(depth))
        if depth == self.max_depth or dataset.num_samples <= self.min_size:
            # last condition is true if all samples belong to the same class
            node = self._make_leaf(dataset, depth)
        else:
            node = self._make_parent_or_leaf(dataset, depth)
        return node


    def _make_parent_or_leaf(self, dataset: 'Dataset', depth: int) -> 'Node':
        try:
            # select a random subset of features, to make trees more diverse
            idx_features = np.random.choice(range(dataset.num_features), self.num_random_features, replace=False)
            if self.extra_trees == 'extra_trees':
                best_feature_index, best_threshold, minimum_cost, best_split = self._best_split_extra_trees(idx_features, dataset)
            elif self.extra_trees == 'none':
                best_feature_index, best_threshold, minimum_cost, best_split = self._best_split(idx_features, dataset)

            left_dataset, right_dataset = best_split
            assert left_dataset.num_samples > 0 or right_dataset.num_samples > 0
            if left_dataset.num_samples == 0 or right_dataset.num_samples == 0:
                # this is a special case: the dataset has samples of at least two
                # classes, but the best split is moving all samples to the left or right
                # dataset and none to the other, so we make a leaf instead of a parent
                return self._make_leaf(dataset, depth)
            else:
                node = Parent(best_feature_index, best_threshold)
                node.left_child = self._make_node(left_dataset, depth + 1)
                node.right_child = self._make_node(right_dataset, depth + 1)
                return node

        except Exception as msg:
            logger.critical("Exception occurred (while setting optimization) ", exc_info=True)


    def _best_split_extra_trees(self, idx_features: np.ndarray, dataset: 'Dataset') -> Tuple[int, float, float, List['Dataset']]:
        # find the best pair (feature, threshold) by exploring all possible pairs
        best_feature_index, best_threshold, minimum_cost, best_split = np.Inf, np.Inf, np.Inf, None
        for idx in idx_features:
            val = np.random.uniform(np.min(dataset.X[:, idx]), np.max(dataset.X[:, idx]))
            left_dataset, right_dataset = dataset.split(idx, val)
            cost = self._CART_cost(left_dataset, right_dataset)  # J(k,v)
            if cost < minimum_cost:
                best_feature_index, best_threshold, minimum_cost, best_split = idx, val, cost, [left_dataset, right_dataset]

        return best_feature_index, best_threshold, minimum_cost, best_split


    def _best_split(self, idx_features: np.ndarray, dataset: 'Dataset') -> Tuple[int, float, float, List['Dataset']]:
        # find the best pair (feature, threshold) by exploring all possible pairs
        best_feature_index, best_threshold, minimum_cost, best_split = np.Inf, np.Inf, np.Inf, None
        for idx in idx_features:
            values = np.unique(dataset.X[:, idx])
            for val in values:
                left_dataset, right_dataset = dataset.split(idx, val)
                cost = self._CART_cost(left_dataset, right_dataset)  # J(k,v)
                if cost < minimum_cost:
                    best_feature_index, best_threshold, minimum_cost, best_split = idx, val, cost, [left_dataset, right_dataset]
        return best_feature_index, best_threshold, minimum_cost, best_split


    def _CART_cost(self, left_dataset: 'Dataset', right_dataset: 'Dataset') -> float:
        logger.debug("criterion {} setted".format(self.criterion))
        mleft = left_dataset.X.shape[0]
        mright = right_dataset.X.shape[0]
        # the J(k,v) equation in the slides, using Gini or Entropy criterion
        left_impurity = self.criterion.calculate_impurity(left_dataset)
        right_impurity = self.criterion.calculate_impurity(right_dataset)
        cost = (mleft / (mleft + mright)) * left_impurity + (mright / (mleft + mright)) * right_impurity
        return cost


    def feature_importance(self) -> Dict[int, int]:
        feat_imp_visitor = FeatureImportance(self.ocurrences)
        for tree in self.decision_trees:
            tree.acceptVisitor(feat_imp_visitor)
        return feat_imp_visitor.ocurrences


    def print_trees(self) -> None:
        for tree in self.decision_trees:
            print("decision tree", self.actual_tree)
            tree_printer = PrinterTree(self.depth)
            tree.acceptVisitor(tree_printer)
            self.actual_tree += 1


    @abstractmethod
    def _make_impurity(self, name: str) -> 'Criterion':
        pass

    @abstractmethod
    def _combine_predictions(self, predictions: List) -> Any:
        pass

    @abstractmethod
    def _make_leaf(self, dataset: 'Dataset', depth: int) -> 'Node':
        pass
