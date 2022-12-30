import numpy


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.n_entries = 0
        self.write = 0

    # update to the root node

    def reset(self):
        self.tree = numpy.zeros(2 * self.capacity - 1)
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p):
        # print(self.write,p)
        # print(self.tree)
        idx = self.write + self.capacity - 1

        # self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        # print(idx)
        dataIdx = idx - self.capacity + 1

        # return (idx, self.tree[idx], self.data[dataIdx])
        return (idx, self.tree[idx], dataIdx)

class MinTree:

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.ones(2 * capacity - 1) * numpy.inf
        self.n_entries = 0
        self.write = 0

    # update to the root node

    def reset(self):
        self.tree = numpy.ones(2 * self.capacity - 1) * numpy.inf
        self.n_entries = 0
        self.write = 0

    def min(self):
        return self.tree[0]

    def add(self , p):
        idx = self.write + self.capacity - 1

        # self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):

        self.tree[idx] = p
        self._propagate(idx)

    def _propagate(self, idx):
        parent = (idx - 1) // 2

        self.tree[parent] = min(self.tree[2 * parent + 1] , self.tree[2 * parent + 2])

        if parent != 0:
            self._propagate(parent)






