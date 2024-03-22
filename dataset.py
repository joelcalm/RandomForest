import numpy as np
from mylogg import mylogger
import logging

logger = mylogger("dataset", logging.INFO)


class Dataset:
    def __init__(self, X: np.ndarray[float], y: np.ndarray[int])-> None:
        self.X = X
        self.y = y
        self.num_samples: int = X.shape[0]
        self.num_features: int = X.shape[1]

    def split(self, feature_index: int, threshold: float) -> tuple['Dataset', 'Dataset']:
        """Splits the dataset into two datasets using the given feature
        and threshold."""
        left_mask: np.ndarray[bool] = self.X[:, feature_index] < threshold
        left_dataset: 'Dataset' = Dataset(self.X[left_mask], self.y[left_mask])
        right_dataset: 'Dataset' = Dataset(self.X[~left_mask], self.y[~left_mask])
        logger.debug("left and right dataset have been created")
        return left_dataset, right_dataset

    def random_sampling(self, ratio_samples: float) -> 'Dataset':
        """Returns a random subset of the dataset with replacement."""
        num_samples_subset: int = int(self.num_samples * ratio_samples)
        indices: np.ndarray[int] = np.random.choice(self.num_samples, num_samples_subset,
                                                    replace=True)
        subset: 'Dataset' = Dataset(self.X[indices], self.y[indices])
        return subset

    def most_frequent_label(self) -> np.ndarray[int]:
        """Returns the most frequent label in the dataset."""
        unique_labels, counts = np.unique(self.y, return_counts=True)
        most_frequent_label: np.ndarray[int] = unique_labels[np.argmax(counts)]
        logger.debug("The most frequent label is {}"
                     .format(most_frequent_label))
        return most_frequent_label

    def mean_value(self) -> float:
        mean_value: float = np.mean(self.y)
        logger.debug("The mean value is {}".format(mean_value))
        return mean_value       

    def relative_frequency(self) -> np.ndarray[float]:
        try:
            return np.bincount(self.y) / self.num_samples
        except Exception as msg:
            logger.critical("An exception has occurred:", exc_info=True)
