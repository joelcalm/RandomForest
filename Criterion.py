from abc import ABC, abstractmethod
import numpy as np

class Criterion(ABC):
    @abstractmethod
    def calculate_impurity(self, dataset: 'Dataset') -> float:
        pass
    
class Gini(Criterion):
    def calculate_impurity(self, dataset: 'Dataset') -> float:
        frequency_array: np.ndarray[float] = dataset.relative_frequency()
        squared_sum: float = sum(np.square(frequency_array))
        gini_impurity: float = 1.0 - squared_sum
        return gini_impurity

class Entropy(Criterion):
    def calculate_impurity(self, dataset: 'Dataset') -> float:
        frequency_array: np.ndarray[float] = dataset.relative_frequency()
        f_positive: np.ndarray[float] = frequency_array[frequency_array > 0]
        entropy_impurity: float = -sum(f_positive * np.log(f_positive))
        return entropy_impurity

class SumSquareError(Criterion):
    def calculate_impurity(self, dataset: 'Dataset') -> float:
        mean: float = dataset.mean_value()
        sum_sq_err_impurity: float = np.sum(np.square(dataset.y - mean))
        return sum_sq_err_impurity

class Isolation(Criterion):
    def calculate_impurity(self, dataset: 'Dataset') -> None:
        return None
