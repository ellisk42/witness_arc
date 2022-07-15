import matplotlib.pyplot as plt
import cProfile
import numpy as np
import json
from collections import namedtuple
from skimage.morphology import flood_fill
import sys
import os

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

    if isinstance(object_or_iterator, Object):
        return object_or_iterator.translate(d)

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
        

class Rectangle():
    def __init__(self, color=None, height=None, width=None):
        self.color = color
        self.height, self.width = height, width

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

        if np.all(np.logical_or(i==c, i==-1)):
            yield Object("rectangle", (0,0), color=c, shape=i.shape), np.zeros(i.shape)-1
        
        return

class Sprite():
    def __init__(self, color=None, height=None, width=None, contiguous=False, diffuse=False):
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
            
            

class Diagonal():
    def __init__(self, color=None):
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
                    

class Floating():
    def __init__(self, child):
        self.child = child

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

class Horizontal():
    def __init__(self, left, right):
        self.left, self.right = left, right

    def parse(self, i):
        for dx in range(i.shape[0]//2):
            for x in {i.shape[0]//2-dx, i.shape[0]//2+dx}:
                for l_p, l_r in self.left.parse(i[:x]):
                    for r_p, r_r in self.right.parse(i[x:]):
                        yield (l_p, translate((x, 0), r_p)), np.concatenate((l_r,r_r), 0)

class Vertical():
    def __init__(self, left, right):
        self.left, self.right = left, right

    def parse(self, i):
        for dx in range(i.shape[1]//2):
            for x in {i.shape[1]//2-dx, i.shape[1]//2+dx}:
                for l_p, l_r in self.left.parse(i[:, :x]):
                    for r_p, r_r in self.right.parse(i[:, x:]):
                        yield (l_p,translate((0, x), r_p)), np.concatenate((l_r,r_r), 1)
                        

    
def _subregions(w,h,size_bound=9999999999):
    regions=[(lx,ux,ly,uy)
             for lx in range(w)
             for ly in range(h)
             for ux in range(lx+1, w+1)
             for uy in range(ly+1, h+1)
             if (ux-lx)*(uy-ly)<size_bound
    ]
    regions.sort(key=lambda z: -(z[1]-z[0])*(z[3]-z[2]))
    return regions

class Union():
    def __init__(self, *children):
        self.children = children

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

class Repeat():
    def __init__(self, child):
        self.child = child

    def parse(self, i, size_bound=9999999999):

        def f(j):
            

            could_be_just_one=False
            for z1, residual in Floating(self.child).parse(j):
                yield frozenset({z1}), residual
                could_be_just_one=True

            # if could_be_just_one:
            #     return
            
            for lx,ux,ly,uy in _subregions(*j.shape):
                for prefix, r in self.child.parse(j[lx:ux,ly:uy]):
                    prefix = translate((lx, ly), prefix)
                    residual = np.copy(j)
                    residual[lx:ux,ly:uy] = r
                    for suffix, final_residual in f(residual):
                        yield frozenset({prefix})|suffix, final_residual
                            
        yield from f(i)


testcases = [("1caeab9d", Horizontal(Horizontal(Floating(Rectangle()), Floating(Rectangle())),
                                     Floating(Rectangle()))),
             ("b8cdaf2b", Horizontal(Floating(Rectangle()),
                                    Horizontal(Floating(Vertical(Rectangle(), Rectangle())),
                                               Floating(Rectangle())))), 
             ("5c0a986e", Union(Rectangle(color=2), Rectangle(color=1))),
             ("99b1bc43", Vertical(Sprite(diffuse=True),
                                   Vertical(Rectangle(),
                                            Sprite(diffuse=True)))),
             ("d631b094", Floating(Sprite())),
             ("7ddcd7ec", Union(Rectangle(height=2, width=2),
                                Repeat(Rectangle(height=1, width=1)))),
             ("97999447", Repeat(Rectangle(height=1, width=1))),
             ("a3325580", Repeat(Sprite())),
             ("a78176bb", Union(Diagonal(),
                                Repeat(Sprite(color=5, contiguous=True)))),
             ("a87f7484", Repeat(Sprite())),
             ("e5062a87", Union(Sprite(color=5), Sprite(color=2))), 
             #("ded97339", Repeat(Rectangle(color=8))),
             
]

parses={}
errors=[]
for code, parser in testcases:
    
    parses[code]=[]
    
    print()
    print("STARTING ", code)
    data = json.load(open(f'../ARC/data/training/{code}.json'))
    inputs = [ np.array(input_output["input"]).T
               for input_output in data["train"]]
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
                #print("Incomplete parse, skipping")
        if not found_parse:
            errors.append((code, n))
            
                
if errors:
    print("ERRORS!")
    for problem_code, example_index in errors:
        print(problem_code, "example", example_index)
else:
    print("no errors")

from plotting import plot_arc_array
for code, parses in parses.items():
    os.system(f"rm -r parses/{code}; mkdir -p parses/{code}/")
    for n, (parse, x) in enumerate(parses):
        #lt.figure()
        plot_arc_array([animate(parse, x)])
        plt.savefig(f"parses/{code}/{n}.png")
        plt.close()
