import matplotlib.pyplot as plt
import cProfile
import numpy as np
import json
from collections import namedtuple
from skimage.morphology import flood_fill
import sys
import os
import time
from pq import PQ

np.set_printoptions(threshold=sys.maxsize)



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

def animate(z, base):

    def flatten(z):
        if isinstance(z, frozenset) or isinstance(z, tuple) or isinstance(z, list):
            return [y for x in z for y in flatten(x) ]
        if isinstance(z, Object):
            return [z]
        assert False

    l=flatten(z)
    return [ render(l[:n], np.zeros_like(base)) for n in range(len(l)+1) ]

class Parser():
    def __init__(self, *arguments):
        self.arguments = arguments

    def __str__(self):
        return self.__class__.__name__+f"({', '.join(str(a) for a in self.arguments)})"

    def __repr__(self):
        return str(self)

class Rectangle(Parser):
    def __init__(self, color=None, height=None, width=None):
        super().__init__(color, height, width)
        self.color = color
        self.height, self.width = height, width

    def parse(self, i):
        
        if self.height and i.shape[1]!=self.height:
            return 
        if self.width and i.shape[0]!=self.width:
            return 

        if np.all(i<=0):
            return 
        if self.color is None:
            c = i[i>0][0]
        else:
            c = self.color

        if np.all(np.logical_or(i==c, i==-1)):
            yield Object("rectangle", (0,0), color=c, shape=i.shape), np.zeros(i.shape)-1
        
        return

    def cost(self):
        return 1 + (color_cost if self.color else 0) + (height_cost if self.height else 0) + (width_cost if self.width else 0)

class Sprite(Parser):
    def __init__(self, color=None, height=None, width=None, contiguous=False, diffuse=False):
        super().__init__(color, height, width, contiguous, diffuse)
        self.color = color
        self.height, self.width, self.contiguous, self.diffuse = height, width, contiguous, diffuse

    def parse(self, i):
        
        if self.height and i.shape[1]!=self.height:
            return 
        if self.width and i.shape[0]!=self.width:
            return 

        if self.color is None:
            if np.all(i<=0):
                return 
            c = i[i>0][0]
        else:
            c = self.color

        if not np.all(np.logical_or(i==c, i<=0)):
            return

        if self.diffuse:
            # must have color
            if np.all(i<=0): return
        else:
            # must have color around the border
            if np.all(i[:,0]<=0) or np.all(i[:,-1]<=0) or np.all(i[0,:]<=0) or np.all(i[-1,:]<=0):
                return 

        if not self.contiguous:
            pixels = np.copy(i)
            pixels[pixels<=0]=0
            yield Object("sprite", (0, 0), color=c, pixels=pixels), np.zeros(i.shape)-1
        else:
            nz=np.nonzero(i>0)
            try:
                ff = flood_fill(i, (nz[0][0], nz[1][0]), -2, connectivity=1)
            except:
                import pdb; pdb.set_trace()
                

            residual=np.copy(i)

            residual[ff==-2]=-1
            
            pixels = ff
            ff[ff!=-2]=0
            ff[ff==-2]=c
            
            yield Object("sprite", (0, 0), color=c, pixels=pixels), residual

    def cost(self):
        return 20 + (color_cost if self.color else 0)
            
            

class Diagonal(Parser):
    def __init__(self, color=None):
        super().__init__(color)
        self.color = color

    def parse(self, i):
        if i.shape[0]!=i.shape[1]:
            return
        
        w = i.shape[0]

        def get_color(slice):
            if np.all(slice<=0):
                return None
            valid_subset=slice[slice>=0]
            c=valid_subset[0]
            if np.all(valid_subset==c):
                return c
            return None

        for xslice, yslice, dir in [(np.arange(w), np.arange(w), (1,1)),
                                    (np.arange(w), np.arange(w-1,-1,-1), (1,-1)),
                                    (np.arange(w-1,-1,-1), np.arange(w), (-1,1)),
                                    (np.arange(w-1,-1,-1), np.arange(w-1,-1,-1), (-1,-1))]:
            
            
            a = i[xslice, yslice]
            c = get_color(a)
            if c is not None:
                if self.color is None or self.color == c:
                    residual = np.copy(i)
                    residual[xslice, yslice] = -1
                    yield Object("diagonal", (0,0), color=c, length=w, dir=dir), residual

    def cost(self):
        return 1 + (color_cost if self.color else 0)
                    

class Floating(Parser):
    def __init__(self, child):
        super().__init__(child)
        self.child = child

    def cost(self):
        return 0.1 + self.child.cost()
    

    def parse(self, i):
        if i.shape[0]<1 or i.shape[1]<1 or np.all(i<=0):
            return
        
        nz = np.nonzero(i>0)
        try:
            _i = i[nz[0].min():nz[0].max()+1,
                   nz[1].min():nz[1].max()+1]
        except:
            import pdb; pdb.set_trace()
            
            
        for p, r in self.child.parse(_i):
            residual=np.zeros(i.shape)
            residual[nz[0].min():nz[0].max()+1, nz[1].min():nz[1].max()+1] = r
            yield translate((nz[0].min(), nz[1].min()), p), residual

class Horizontal(Parser):
    def __init__(self, left, right):
        super().__init__(left, right)
        self.left, self.right = left, right

    def cost(self):
        return 0.1 + self.left.cost() + self.right.cost()

    def parse(self, i):
        for dx in range(i.shape[0]//2):
            for x in {i.shape[0]//2-dx, i.shape[0]//2+dx}:
                for l_p, l_r in self.left.parse(i[:x]):
                    for r_p, r_r in self.right.parse(i[x:]):
                        yield (l_p, translate((x, 0), r_p)), np.concatenate((l_r,r_r), 0)

class Vertical(Parser):
    def __init__(self, left, right):
        super().__init__(left, right)
        self.left, self.right = left, right

    def cost(self):
        return 0.1 + self.left.cost() + self.right.cost()

    def parse(self, i):
        for dx in range(i.shape[1]//2):
            for x in {i.shape[1]//2-dx, i.shape[1]//2+dx}:
                for l_p, l_r in self.left.parse(i[:, :x]):
                    for r_p, r_r in self.right.parse(i[:, x:]):
                        yield (l_p,translate((0, x), r_p)), np.concatenate((l_r,r_r), 1)
                        
class Nothing(Parser):
    def __init__(self):
        super().__init__()
        pass

    def parse(self, i):
        yield None, i
    
def _subregions(w,h,size_bound=9999999999, aligned=False):
    if aligned:
        regions=[(0,ux,0,h)
                 for ux in range(1, w+1)]+\
                [(lx,w,0,h)
                 for lx in range(0, w)]+\
                [(0,w,0,uy)
                 for uy in range(1, h+1)]+\
                [(0,w,ly,h)
                 for ly in range(0, h)]
    else:
        regions=[(lx,ux,ly,uy)
             for lx in range(w)
             for ly in range(h)
             for ux in range(lx+1, w+1)
             for uy in range(ly+1, h+1)
             if (ux-lx)*(uy-ly)<size_bound
        ]
    regions.sort(key=lambda z: -(z[1]-z[0])*(z[3]-z[2]))
    return regions

class Union(Parser):
    def __init__(self, *children):
        super().__init__(children)
        self.children = children

    def cost(self):
        return 0.1 + sum(c.cost() for c in self.children)

    def parse(self, i, size_bound=9999999999):

        def f(j, still_need_to_parse):
            
            if len(still_need_to_parse)==0:
                yield [], j
                return 

            for lx,ux,ly,uy in _subregions(*j.shape):
                for prefix, r in still_need_to_parse[0].parse(j[lx:ux,ly:uy]):
                    prefix = translate((lx, ly), prefix)
                    residual = np.copy(j)
                    residual[lx:ux,ly:uy] = r
                    for suffix, final_residual in f(residual, still_need_to_parse[1:]):
                        yield [prefix]+suffix, final_residual
                        
        yield from f(i, self.children)

class UnionAligned(Parser):
    def __init__(self, *children):
        super().__init__(children)
        self.children = children

    def cost(self):
        return 0.1 + sum(c.cost() for c in self.children)

    def parse(self, i, size_bound=9999999999):

        def f(j, still_need_to_parse):
            
            if len(still_need_to_parse)==0:
                yield [], j
                return 

            for lx,ux,ly,uy in _subregions(*j.shape, aligned=True):
                for prefix, r in Floating(still_need_to_parse[0]).parse(j[lx:ux,ly:uy]):
                    prefix = translate((lx, ly), prefix)
                    residual = np.copy(j)
                    residual[lx:ux,ly:uy] = r
                    for suffix, final_residual in f(residual, still_need_to_parse[1:]):
                        yield [prefix]+suffix, final_residual
                        
        yield from f(i, self.children)

class Repeat(Parser):
    def __init__(self, child):
        super().__init__(child)
        self.child = child

    def cost(self):
        return 0.1 + self.child.cost()

    def parse(self, i, size_bound=9999999999):

        def f(j):
            
            could_be_just_one=False
            for z1, residual in Floating(self.child).parse(j):
                yield frozenset({z1}), residual
                could_be_just_one=True

            # if could_be_just_one:
            #     return
            
            for lx,ux,ly,uy in _subregions(*j.shape):
                if np.all(j[lx:ux,ly:uy]<=0): continue
                for prefix, r in self.child.parse(j[lx:ux,ly:uy]):
                    prefix = translate((lx, ly), prefix)
                    residual = np.copy(j)
                    residual[lx:ux,ly:uy] = r
                    for suffix, final_residual in f(residual):
                        yield frozenset({prefix})|suffix, final_residual
                            
        yield from f(i)

class RepeatAligned(Parser):
    """like repeat but axes aligned: repetitions have to be along rows or columns"""
    def __init__(self, child):
        super().__init__(child)
        self.child = child

    def cost(self):
        return 0.1 + self.child.cost()

    def parse(self, i, size_bound=9999999999):

        def f(j):
            could_be_just_one=False
            for z1, residual in Floating(self.child).parse(j):
                yield frozenset({z1}), residual
                could_be_just_one=True

            # if could_be_just_one:
            #     return
            
            for lx,ux,ly,uy in _subregions(*j.shape, aligned=True):
                if np.all(j[lx:ux,ly:uy]<=0): continue
                for prefix, r in Floating(self.child).parse(j[lx:ux,ly:uy]):
                    prefix = translate((lx, ly), prefix)
                    residual = np.copy(j)
                    residual[lx:ux,ly:uy] = r
                    for suffix, final_residual in f(residual):
                        yield frozenset({prefix})|suffix, final_residual
                            
        yield from f(i)


testcases = [
    ("5c0a986e", Union(Rectangle(color=1), Rectangle(color=2))),
    ("ded97339", RepeatAligned(Rectangle(color=8, height=1, width=1))),
    ("d631b094", Floating(Sprite())),
    ("a87f7484", RepeatAligned(Sprite())),
    
    ("025d127b", RepeatAligned(Vertical(Floating(Rectangle()),
                                        Vertical(Horizontal(Floating(Diagonal()), Diagonal()),
                                                 Floating(Rectangle()))))), 
    
    ("1caeab9d", Horizontal(Horizontal(Floating(Rectangle()), Floating(Rectangle())),
                            Floating(Rectangle()))),
    ("b8cdaf2b", Horizontal(Floating(Rectangle()),
                            Horizontal(Floating(Vertical(Rectangle(), Rectangle())),
                                       Floating(Rectangle())))), 
    ("99b1bc43", Vertical(Sprite(diffuse=True),
                          Vertical(Rectangle(),
                                   Sprite(diffuse=True)))),
    
    ("7ddcd7ec", Union(Rectangle(height=2, width=2),
                       Repeat(Rectangle(height=1, width=1)))),
    ("97999447", RepeatAligned(Rectangle(height=1, width=1))),
    ("a3325580", RepeatAligned(Sprite())),
    ("a78176bb", Union(Diagonal(),
                       Repeat(Sprite(color=5, contiguous=True)))),   

             # requires overlapping sprites
    # ("e5062a87", Union(Sprite(color=2), Sprite(color=5))), 
             
]

def test_manual_parsers():
    parses={}
    errors=[]
    times={}
    for code, parser in testcases:

        parses[code]=[]

        print()
        print("STARTING ", code)
        data = json.load(open(f'../ARC/data/training/{code}.json'))
        inputs = [ np.array(input_output["input"]).T
                   for input_output in data["train"]]
        
        import time
        t0=time.time()
        for n, x in enumerate(inputs):
            print("Parsing input #", n)
            print(x)

            found_parse=False
            for parse, residual in parser.parse(x):
                if np.all(residual<=0):
                    print(parse)
                    print(render(parse, np.zeros_like(x)))
                    if not np.all(render(parse, np.zeros_like(x))==x):
                        print("rendering failure")
                    parses[code].append((parse, x))
                    found_parse=True
                    # print(residual)
                    break
                else:
                    ...
                    print("Incomplete parse, skipping")
                    print(residual)
            if not found_parse:
                errors.append((code, n))
        times[code]=time.time()-t0


    if errors:
        print("ERRORS!")
        for problem_code, example_index in errors:
            print(problem_code, "example", example_index)
    else:
        print("no errors")

    print("Parsing times:")
    for code, time in sorted(times.items(), key=lambda ct: ct[1]):
        print(code, time)
    print("Total time:", sum(times.values()))

    from plotting import plot_arc_array
    for code, parses in parses.items():
        os.system(f"rm -r parses/{code}; mkdir -p parses/{code}/")
        for n, (parse, x) in enumerate(parses):
            #lt.figure()
            plot_arc_array([animate(parse, x)])
            plt.savefig(f"parses/{code}/{n}.png")
            plt.close()


# specifying these in the program reduces your cost
color_cost=-0.5
height_cost=-0.5
width_cost=-0.5

def infer_parses(images):

    common_colors = set(range(1, 10))
    for i in images:
        common_colors &= { c for r in i for c in r if c>0 }

    atomic = [a
              for c in common_colors | {None}
              for a in [Diagonal(color=c), Rectangle(color=c), Sprite(color=c)] ]
    repeats = [RepeatAligned(a) for a in atomic ]
    combinators = [UnionAligned, Vertical, Horizontal]

    def priority(program):
        matches=[]
        for i in images:
            match=None
            for p, r in Union(program, Nothing()).parse(i):
                match=(p, r)
                break
            if match is None:
                return float("-inf"), float("inf")
            matches.append(match)

        pixel_cost = sum(np.sum(r>0) for _, r in matches)

        program_cost = program.cost()

        total_cost = pixel_cost + program_cost

        return -total_cost, pixel_cost

    def successors(e):
        for a in atomic+repeats:
            for c in combinators:
                yield c(e, a)
        if not isinstance(e, RepeatAligned):
            yield RepeatAligned(e)
        if not isinstance(e, Floating):
            yield Floating(e)

    pq = PQ()

    for a in atomic:
        f, _ = priority(a)
        
        if f>float("-inf"):
            pq.push(f, a)
            print(a, f)

    while len(pq):
        p, e = pq.popMaximum()
        print("popping", p, e)
        
        if priority(e)[1]==0:
            print("explains all the pixels")
            return e
            
        #     break
        for ep in successors(e):
            pp, residual = priority(ep)
            if pp>float("-inf"):
                pq.push(pp, ep)
                        
                    
    

    #Union(Floating(a), Sprite())
    

def test_parse_inference():
    for code, parser in testcases:
        print()
        print("STARTING ", code)
        data = json.load(open(f'../ARC/data/training/{code}.json'))
        inputs = [ np.array(input_output["input"]).T
                   for input_output in data["train"]]

        infer_parses(inputs)
        
#test_manual_parsers()
test_parse_inference()
