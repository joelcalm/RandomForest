from abc import ABC, abstractmethod

class Node(ABC):
    @abstractmethod
    def predict(self, x: 'np.ndarray') -> 'Any':
        pass

    @abstractmethod
    def acceptVisitor(self, visitor: 'Visitor') -> None:
        pass


class Parent(Node):
    def __init__(self, feature_index: int, threshold: float, left_child: 'Node' = None, right_child: 'Node' = None):
        self.feature_index: int = feature_index
        self.threshold: float = threshold
        self.left_child: 'Node' = left_child
        self.right_child: 'Node' = right_child

    def predict(self, x: 'np.ndarray') -> 'Any':
        if x[self.feature_index] < self.threshold:
            return self.left_child.predict(x)  # type: ignore
        else:
            return self.right_child.predict(x)  # type: ignore

    def acceptVisitor(self, visitor: 'Visitor') -> None:
        visitor.visitParent(self)


class Leaf(Node):
    def __init__(self, label: 'Any'):
        self.label: 'Any' = label

    def predict(self, x: 'np.ndarray') -> 'Any':
        return self.label

    def acceptVisitor(self, visitor: 'Visitor') -> None:
        visitor.visitLeaf(self)
