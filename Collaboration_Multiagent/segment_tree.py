import numpy as np

### Implements Segment tree
class SegmentTree:
    
    def __init__(self, total_elements):
        
        ## leaf nodes contains the elements to be stored in segment tree
        self.leaf_nodes = total_elements
        
        ## assuming the number of leaf nodes is a power of 2, 
        ## if number of leaf nodes are n then total nodes in a tree are 2*n
        self.tree_size = 2 * self.leaf_nodes
        
        ## Initializing tree as an array
        self.tree = [0 for _ in range(self.tree_size)]
        
        ## keeping track of the minimum value of priority
        self.min_priority = np.inf
        
    def get_interval_sum(self, intv_start, intv_end, current_node,subtree_start, subtree_end):
        if intv_start == subtree_start and intv_end == subtree_end:
            return self.tree[current_node]
        
        mid = (subtree_start + subtree_end) // 2
        
        ## if the interval is in left of the current subtree
        if intv_end <= mid:
            return self.get_interval_sum(intv_start, intv_end, 2*current_node,
                                   subtree_start, mid)
        else:
            ## if the interval is on the right of current subtree
            if mid+1 <= intv_start:
                return self.get_interval_sum(intv_start, intv_end, 2*current_node+1,
                                       mid+1, subtree_end)
            else: ## if the interval overlaps left and right side of current subtree
                left_sum = self.get_interval_sum(intv_start, mid, 2*current_node,
                                                 subtree_start, mid)
                right_sum = self.get_interval_sum(mid+1,intv_end, 2*current_node+1,
                                                  mid+1, subtree_end)
                return left_sum + right_sum
            
    def interval_sum(self, start=0, end=None):
        if end == None:
            end = self.leaf_nodes
        return self.get_interval_sum(start, end-1,1,0,self.leaf_nodes-1)
    
    def set_value(self, ind, val):
        
        ## index of the leaf node
        ind = self.leaf_nodes + ind
        self.tree[ind] = val
        
        ## updating minimum priority
        self.min_priority = min(self.min_priority, val)
        
        ## index of the parent node
        ind = ind // 2
        
        while ind >= 1:
            ##modifying the value of parent
            self.tree[ind] = self.tree[2*ind] + self.tree[2*ind +1]
            ## changing index to parent of the current node
            ind = ind // 2
    
    def get_value(self, ind):
        assert 0 <= ind < self.leaf_nodes
        return self.tree[self.leaf_nodes + ind]
    
    def find_prefixsum_idx(self, prefixsum):
        
        assert 0 <= prefixsum <= self.interval_sum() + 1e-5
        idx = 1
        while idx < self.leaf_nodes:  # while non-leaf
            if self.tree[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self.tree[2 * idx]
                idx = 2 * idx + 1
        return idx - self.leaf_nodes
    
    def get_minimum(self):
        return self.min_priority
    
    def get_inner_value(self,ind):
        assert 0 <= ind < self.leaf_nodes
        return self.tree[ind]
