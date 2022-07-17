import numpy as np

class Object():
    def __init__(self, name, position, **parameters):
        self.name, self.parameters, self.position = name, parameters, position

    def translate(self, displacement):
        dx, dy=displacement
        return Object(self.name, (self.position[0]+dx, self.position[1]+dy), **self.parameters)

    def __str__(self):
        return f"{self.name}(p={self.position}, {', '.join(k+'='+str(v) for k, v in self.parameters.items() )})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self)==str(other)

    def __ne__(self, other):
        return not (self==other)

    def render(self, i):
        if self.name=="rectangle":
            i[self.position[0]:self.position[0]+self.parameters["shape"][0],
              self.position[1]:self.position[1]+self.parameters["shape"][1]] = self.parameters["color"]
        if self.name=="diagonal":
            w=self.parameters["length"]
            view=i[self.position[0]:self.position[0]+w,
                   self.position[1]:self.position[1]+w]
            xs, ys={(1,1): (np.arange(w), np.arange(w)),
                    (1,-1): (np.arange(w), np.arange(w-1,-1,-1)),
                    (-1,1): (np.arange(w-1,-1,-1), np.arange(w)),
                    (-1,-1): (np.arange(w-1,-1,-1), np.arange(w-1,-1,-1))}[self.parameters["dir"]]
            view[xs, ys] = self.parameters["color"]
    
        if self.name=="sprite":
            pixels = self.parameters["pixels"]
            view=i[self.position[0]:self.position[0]+pixels.shape[0],
                   self.position[1]:self.position[1]+pixels.shape[1]]
            view[pixels>0]=pixels[pixels>0]

    def cost(self):
        # how many factors of ten
        if self.name=="rectangle":
            return 5
        if self.name=="diagonal":
            return 4
    
        if self.name=="sprite":
            pixels = self.parameters["pixels"]
            return 5+pixels.shape[0]*pixels.shape[1]
    

def translate(d, object_or_iterator):

    if isinstance(object_or_iterator, frozenset):
        return frozenset([ translate(d, x) for x in object_or_iterator ])

    if isinstance(object_or_iterator, tuple):
        return tuple([ translate(d, x) for x in object_or_iterator ])

    if isinstance(object_or_iterator, list):
        return tuple([ translate(d, x) for x in object_or_iterator ])

    if isinstance(object_or_iterator, Object):
        return object_or_iterator.translate(d)

    if object_or_iterator is None:
        return None

    import pdb; pdb.set_trace()
    
    assert False

def render(z, i=None):
    if isinstance(z, frozenset) or isinstance(z, tuple) or isinstance(z, list):
        for x in z:
            i = render(x, i)

    if isinstance(z, Object):
        z.render(i)
    return i

def flatten_z(z):
    if isinstance(z, frozenset) or isinstance(z, tuple) or isinstance(z, list):
        return [y for x in z for y in flatten_z(x) ]
    if isinstance(z, Object):
        return [z]
    if z is None:
        return []
    
    assert False
        
def animate(z, base):

    def flatten(z):
        if isinstance(z, frozenset) or isinstance(z, tuple) or isinstance(z, list):
            return [y for x in z for y in flatten(x) ]
        if isinstance(z, Object):
            return [z]
        assert False

    l=flatten(z)
    return [ render(l[:n], np.zeros_like(base)) for n in range(len(l)+1) ]
