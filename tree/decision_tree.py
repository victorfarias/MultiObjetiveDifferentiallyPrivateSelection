import uuid
from typing import Optional

import numpy as np
from bigtree import BinaryNode

from data.loader import NOMINAL_ATT, NUMERICAL_ATT, Dataset


TERMINAL_NODE = 0
DECISION_NODE = 1


class DecisionTree():

    def __init__(self, dataset:Dataset, initial_depth:Optional[int]=None, root:Optional[BinaryNode]=None):
        if(initial_depth is None and root is None):
            raise Exception("You must specify either depth or root")
                
        self.initial_depth = initial_depth
        self.root = root

        self.n_classes = dataset.n_classes
        self.n_attributes = dataset.n_attributes   
        self.attr_types = dataset.attr_types 
        self.n_nominal_values = dataset.n_nominal_values
        self.dataset = dataset
        
    # def __str__(self):
    #     if self.root is not None:
    #         print_tree(self.root, attr_list=["type", "att_type", "decision_value", "threshold", "value", "attribute", "depth", "subtree_size"])
    #     return ""
    
    def fit(self):
        self.root = self.create_random_tree(self.initial_depth)
        self.assign_tree_size()
        return self
    
    
    def _predict(self, x:np.ndarray)->float:
        if self.root is None:  
            raise Exception("You must fit the model first")
        node = self.root
        while 1:
            if(node.type == TERMINAL_NODE):
                return node.value
            elif(node.att_type == NUMERICAL_ATT):                
                if(x[node.attribute] < node.threshold):
                    node = node.left
                else:
                    node = node.right
            else:
                if((node.decision_value - x[node.attribute])<1e-6 and (node.decision_value - x[node.attribute])>-1e-6):
                    node = node.left
                else:
                    node = node.right

    

    def predict(self, X:np.ndarray)->np.ndarray|float:
        if len(X.shape)==1:
            return self._predict(X)
        else:
            return np.array([self._predict(x) for x in X])

    def _clone(self, tree) -> BinaryNode:            
        if tree.type==TERMINAL_NODE:
            return BinaryNode(name=uuid.uuid4(), 
                              type=TERMINAL_NODE, 
                              value=tree.value,
                              subtree_size=tree.subtree_size)
        
        elif tree.att_type==NOMINAL_ATT:
            dn = BinaryNode(name=uuid.uuid4(), 
                            type=DECISION_NODE,
                            decision_value=tree.decision_value, 
                            attribute=tree.attribute,
                            att_type=NOMINAL_ATT,
                            subtree_size=tree.subtree_size)
        else:
            dn = BinaryNode(name=uuid.uuid4(), 
                            type=DECISION_NODE,
                            threshold=tree.threshold, 
                            attribute=tree.attribute,
                            att_type=NUMERICAL_ATT,
                            subtree_size=tree.subtree_size)

    
        # dn = BinaryNode(name=uuid.uuid4(), 
        #                 type=DECISION_NODE,
        #                 threshold=tree.threshold, 
        #                 attribute=tree.attribute,
        #                 subtree_size=tree.subtree_size)
        dn.left = self._clone(tree.left)
        dn.right = self._clone(tree.right)
        return dn

    
    def clone(self) -> 'DecisionTree':
        return DecisionTree(self.dataset, root=self._clone(self.root))
    
    def create_random_tree(self, depth) -> BinaryNode:
        if depth==1:
            return BinaryNode(name=uuid.uuid4(), 
                              type=TERMINAL_NODE, 
                              value=np.random.choice(self.n_classes), 
                              subtree_size=None)

        attribute_i = np.random.choice(self.n_attributes)

        if self.attr_types[attribute_i] == NOMINAL_ATT:
            dn = BinaryNode(name=uuid.uuid4(), 
                            type=DECISION_NODE,
                            att_type=NOMINAL_ATT,
                            attribute=attribute_i,
                            decision_value=np.random.choice(self.n_nominal_values[attribute_i]),
                            subtree_size=None)
        else:
            dn = BinaryNode(name=uuid.uuid4(), 
                            type=DECISION_NODE,
                            att_type=NUMERICAL_ATT,
                            attribute=attribute_i,
                            threshold=np.random.rand(), 
                            subtree_size=None)
        
        dn.left = self.create_random_tree(depth-1)
        dn.right = self.create_random_tree(depth-1)
        
        return dn


    def _assign_subtree_size(self, tree):
        if tree.type==TERMINAL_NODE:
            tree.subtree_size = 1
            return 1
        
        tree.subtree_size = 1 + self._assign_subtree_size(tree.left) + self._assign_subtree_size(tree.right)
        return tree.subtree_size

    def assign_tree_size(self):
        self._assign_subtree_size(self.root)
    
    def choose_random_node_from_tree(self, root) -> BinaryNode:
        if root.type==TERMINAL_NODE or (root.left is None and root.right is None):
            return root
            
        subtree_size = root.subtree_size    
        random = np.random.rand()

        if root.left is not None:
            left_size = root.left.subtree_size
            if(random < (left_size/subtree_size)):
                return self.choose_random_node_from_tree(root.left)

        if root.right is not None:
            right_size = root.right.subtree_size    
            if(random > 1-(right_size/subtree_size)):
                return self.choose_random_node_from_tree(root.right)
        
        return root
        
    def choose_random_node(self) -> BinaryNode:
        return self.choose_random_node_from_tree(self.root)

    
    def _prune_tree(self, tree, depth, n_classes):
        if depth==2:
            if tree.left is not None and tree.left.type == DECISION_NODE:
                tree.left = BinaryNode(name=uuid.uuid4(), type=TERMINAL_NODE, value=np.random.choice(n_classes))
            if tree.right is not None and tree.right.type == DECISION_NODE:
                tree.right = BinaryNode(name=uuid.uuid4(), type=TERMINAL_NODE, value=np.random.choice(n_classes))
            return
        
        if tree.left is not None:
            self._prune_tree(tree.left, depth-1, n_classes)

        if tree.right is not None:
           self._prune_tree(tree.right, depth-1, n_classes)

    def prune_tree(self, depth, n_classes):
        return self._prune_tree(self.root, depth, n_classes)        

if __name__ == '__main__':

    from data.loader import load_pimas

    dataset = load_pimas()

    dt = DecisionTree(dataset, 4)
    dt.fit()
    print(dt)
    # print(dt.predict(np.array([0.5, 0.2])))
    