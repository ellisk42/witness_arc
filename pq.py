import heapq

class PQ(object):
    """why the fuck does Python not wrap this in a class"""

    def __init__(self):
        self.h = []
        self.index2value = {}
        self.nextIndex = 0

    def push(self, priority, v):
        self.index2value[self.nextIndex] = v
        heapq.heappush(self.h, (-priority, self.nextIndex))
        self.nextIndex += 1

    def popMaximum(self):
        p, i = heapq.heappop(self.h)
        v = self.index2value[i]
        del self.index2value[i]
        return -p, v

    def __iter__(self):
        for _, v in self.h:
            yield self.index2value[v]

    def __len__(self): return len(self.h)
