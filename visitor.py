from abc import ABC, abstractmethod
from Node import Node

class Visitor(ABC):
    @abstractmethod
    def visitParent(self, parent: 'Node') -> None:
        pass
    
    @abstractmethod
    def visitLeaf(self, leaf: 'Node') -> None:
        pass
    

class FeatureImportance(Visitor):
    
    def __init__(self, ocurrences: dict):
        self.ocurrences = ocurrences
    
    def visitParent(self, parent: 'Node') -> None:
        k = parent.feature_index
        if k in self.ocurrences.keys():
            self.ocurrences[k] += 1
        else:
            self.ocurrences[k] = 1
        parent.left_child.acceptVisitor(self)
        parent.right_child.acceptVisitor(self)
    
    def visitLeaf(self, leaf: 'Node') -> None:
        pass
        

class PrinterTree(Visitor):
    def __init__(self, depth: int):
        self.depth = depth

    def visitParent(self, parent: 'Node') -> None:
        print("     " * self.depth + "parent, {}, {}".format(parent.feature_index, parent.threshold))
        self.depth += 1
        parent.left_child.acceptVisitor(self)
        parent.right_child.acceptVisitor(self)
        self.depth -= 1

    def visitLeaf(self, leaf: 'Node') -> None:
        print("     " * self.depth + "leaf, {}".format(leaf.label))
