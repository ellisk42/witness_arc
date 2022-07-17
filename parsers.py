import numpy as np
from objects import *


# specifying these in the program reduces your cost
color_cost=-0.5
height_cost=-0.5
width_cost=-0.5


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
        return 1 + (color_cost if self.color else 0)
            
            

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
            _i = i[nz[0].min():nz[0].max()+1, nz[1].min():nz[1].max()+1]
        except Exception as e:
            print(i)
            print(nz)
            print(e)
            print()
            assert False
            
            
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
            
            if len(still_need_to_parse)==0 or still_need_to_parse[0].__class__ is Nothing:
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
            # could_be_just_one=False
            # for z1, residual in Floating(self.child).parse(j):
            #     yield frozenset({z1}), residual
            #     could_be_just_one=True

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

            yield frozenset({}), j
                            
        yield from f(i)
