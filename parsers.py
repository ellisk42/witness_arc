import numpy as np
from objects import *
from skimage.morphology import flood_fill
import itertools

color_cost=1
height_cost=1
width_cost=1

_parser_caches = {}
def clear_parser_caches():
    global _parser_caches
    # for k in _parser_caches:
    #     _parser_caches[k].clear()
    _parser_caches = {}

class Parser():
    def __init__(self, *arguments):
        self.arguments = arguments
        self.cache = _parser_caches.get(str(self), {})
        _parser_caches[str(self)] = self.cache

    def __str__(self):
        return self.__class__.__name__+f"({', '.join(str(a) for a in self.arguments)})"

    def __repr__(self):
        return str(self)

    def __getstate__(self):
        return {k:v for k, v in self.__dict__.items() if k != "cache" }
    def __setstate__(self, d):
        self.__dict__ = d
        self.cache = _parser_caches.get(str(self), {})
        _parser_caches[str(self)] = self.cache

    def extent(self): return None, None

    def parse(self, i):
        """
        i: image as 2d array
        yields: different interpretations of the image
        each interpretation constitutes objects, latent, residual
        """
        hi = (i.shape, i.tostring())
        if hi not in self.cache:
            self.cache[hi] = ([], self._parse(i))
        the_list, the_generator = self.cache[hi]
        n = 0
        while True:
            if n < len(the_list):
                n+=1
                yield the_list[n-1]                
            else:
                assert n == len(the_list)
                sentinel = object()
                next_one = next(the_generator, sentinel)
                assert n == len(the_list)
                if next_one is sentinel:
                    return
                the_list.append(next_one)
                n+=1
                yield next_one
                
            
        
        #assert False, "implement in child"

    def render(self, z):
        """
        z: latent
        returns: image
        """
        assert False, "implement in child"

        

class Rectangle(Parser):
    def __init__(self, color=None, height=None, width=None):
        super().__init__(color, height, width)
        self.color = color
        self.height, self.width = height, width

    

    def render(self, z):
        c = self.color or z["c"]
        w = self.width or z["w"]
        h = self.height or z["h"]

        i = np.zeros((w,h))
        i[:, :] = c
        return i

    def extent(self): return self.width, self.height

    @staticmethod
    def invert(images):
        colors = []
        ws = []
        hs = []
        for i in images:
            if np.all(i<=0):
                return None
            c = i[i>0][0]
            colors.append(c)
            if not np.all(np.logical_or(i==c, i==-1)):
                return None
            ws.append(i.shape[0])
            hs.append(i.shape[1])

        if len(set(colors)) == 1:
            c = colors[0]
        else:
            c = None

        if len(set(ws)) == 1:
            w = ws[0]
        else:
            w = None

        if len(set(hs)) == 1:
            h = hs[0]
        else:
            h = None

        program = Rectangle(color=c, height=h, width=w)
        for n, i in enumerate(images):
            zs.append({})
            if c is None: zs[-1]["c"] = colors[n]
            if h is None: zs[-1]["h"] = hs[n]
            if w is None: zs[-1]["w"] = ws[n]

        return program, zs        

    def _parse(self, i):
        
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
            
            z = {}
            if self.color is None: z["c"]=c
            if self.width is None: z["w"]=i.shape[0]
            if self.height is None: z["h"]=i.shape[1]
            
            yield Object("rectangle", (0,0), color=c, shape=i.shape), z, np.zeros(i.shape)-1
        
        

    def cost(self):
        return 1 + (color_cost if self.color else 0) + (height_cost if self.height and self.height>1 else 0) + (width_cost if self.width and self.width>1 else 0)

class Sprite(Parser):
    def __init__(self, color=None, height=None, width=None, contiguous=False, diffuse=False):
        super().__init__(color, height, width, contiguous, diffuse)
        self.color = color
        self.height, self.width, self.contiguous, self.diffuse = height, width, contiguous, diffuse

    def extent(self): return self.width, self.height

    def render(self, z):
        c = self.color or z["c"]
        w = self.width or z.get("_w", None) or z["w"]
        h = self.height or z.get("_h", None) or z["h"]
        x, y = z.get("_x", 0), z.get("_y", 0)

        i = np.zeros((w,h), dtype=np.int64)-1
        i[x:x+z["w"],y:y+z["h"]][z["mask"]] = c
        
        return i

    def _parse(self, i):
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

        if not self.contiguous and not np.all(np.logical_or(i==c, i<=0)):
            return

        if self.diffuse:
            # must have color
            if np.all(i<=0): return
        else:
            # must have color around the border
            if np.all(i[:,0]<=0) or np.all(i[:,-1]<=0) or np.all(i[0,:]<=0) or np.all(i[-1,:]<=0):
                return

        z = {"_w": i.shape[0], "_h": i.shape[1]}
        if self.color is None: z["c"]=c
        

        if not self.contiguous:
            if self.width is None: z["w"]=i.shape[0]
            if self.height is None: z["h"]=i.shape[1]
            
            pixels = np.copy(i)
            pixels[pixels<=0]=0

            z["mask"]=i>0

            # needs to occupy at least 25% of the space
            if not self.diffuse and np.sum(z["mask"])<0.25*i.shape[0]*i.shape[1]:
                return 
            
            yield Object("sprite", (0, 0), color=c, pixels=pixels), z, np.zeros_like(i)-1
        else:
            if self.color is None:
                nz=np.nonzero(i>0)
            else:
                nz=np.nonzero(i == self.color)

            try:
                ff = flood_fill(i, (nz[0][0], nz[1][0]), -2, connectivity=1)
            except: return 
                
                

            residual=np.copy(i)

            residual[ff==-2]=-1

            nz=np.nonzero(ff==-2)
            
            pixels = ff[nz[0].min():nz[0].max()+1,
                        nz[1].min():nz[1].max()+1]
            pixels[pixels!=-2]=0
            pixels[pixels==-2]=c

            z["mask"]=pixels>0
            if self.width is None: z["w"]=z["mask"].shape[0]
            if self.height is None: z["h"]=z["mask"].shape[1]

            x, y = nz[0].min(), nz[1].min()
            z["_x"], z["_y"] = x, y
            
            yield Object("sprite", (x, y), color=c, pixels=pixels), z, residual

    def cost(self):
        return 1 + (color_cost if self.color else 0)
            
            

class Diagonal(Parser):
    def __init__(self, color=None):
        super().__init__(color)
        self.color = color
        self.width, self.height = None, None

    def extent(self): return self.width, self.height

    def render(self, z):
        c = self.color or z["c"]
        w = self.width or z["w"]
        i = np.zeros((w, w), dtype=np.int64)-1
        xs, ys={(1,1): (np.arange(w), np.arange(w)),
                    (1,-1): (np.arange(w), np.arange(w-1,-1,-1)),
                    (-1,1): (np.arange(w-1,-1,-1), np.arange(w)),
                    (-1,-1): (np.arange(w-1,-1,-1), np.arange(w-1,-1,-1))}[z["dir"]]
        i[xs, ys] = c
        return i

    def _parse(self, i):
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

                    z={"w":w, "dir":dir}
                    if self.color is None: z["c"]=c
                    
                    yield Object("diagonal", (0,0), color=c, length=w, dir=dir), z, residual

    def cost(self):
        return 1 + (color_cost if self.color else 0)
                    

class Floating(Parser):
    def __init__(self, child):
        super().__init__(child)
        self.child = child

    def cost(self):
        return 0.1 + self.child.cost()

    def render(self, z):
        i = self.child.render(z["child"])

        j = np.zeros((z["_w"], z["_h"]), dtype=np.int64)-1
        j[z["x"]:i.shape[0]+z["x"],
          z["y"]:i.shape[1]+z["y"]] = i

        return j

    @staticmethod
    def invert(images):
        zs = []
        residuals = []
        for i in images:
            if i.shape[0]<1 or i.shape[1]<1 or np.all(i<=0):
                return

            nz = np.nonzero(i>0)
            _i = i[nz[0].min():nz[0].max()+1, nz[1].min():nz[1].max()+1]
            
            zs.append({"x": nz[0].min(),
                       "y": nz[1].min(),
                       "_w": i.shape[0],
                       "_h": i.shape[1],
                       "child": _i})

        return Floating(), zs

    def _parse(self, i):
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
            
        for p, zc, r in self.child.parse(_i):
            residual=np.zeros_like(i)
            residual[nz[0].min():nz[0].max()+1, nz[1].min():nz[1].max()+1] = r
            

            z = {"child": zc,
                 "x": nz[0].min(),
                 "y": nz[1].min(),
                 "_w": i.shape[0],
                 "_h": i.shape[1]}
            
            yield translate((nz[0].min(), nz[1].min()), p), z, residual

class Horizontal(Parser):
    def __init__(self, left, right):
        super().__init__(left, right)
        self.left, self.right = left, right

    def render(self, z):
        return np.concatenate((self.left.render(z["child1"]),
                               self.right.render(z["child2"])), 0)

    def cost(self):
        return 0.1 + self.left.cost() + self.right.cost()

    def _parse(self, i):
        for dx in range(i.shape[0]//2):
            for x in {i.shape[0]//2-dx, i.shape[0]//2+dx}:
                for l_p, l_z, l_r in self.left.parse(i[:x]):
                    for r_p, r_z, r_r in self.right.parse(i[x:]):
                        yield (l_p, translate((x, 0), r_p)), {"child1": l_z, "child2": r_z, "x": x}, np.concatenate((l_r,r_r), 0)

class Vertical(Parser):
    def __init__(self, left, right):
        super().__init__(left, right)
        self.left, self.right = left, right

    def render(self, z):
        return np.concatenate((self.left.render(z["child1"]),
                               self.right.render(z["child2"])), 1)

    def cost(self):
        return 0.1 + self.left.cost() + self.right.cost()

    def _parse(self, i):
        for dx in range(i.shape[1]//2):
            for x in {i.shape[1]//2-dx, i.shape[1]//2+dx}:
                for l_p, l_z, l_r in self.left.parse(i[:, :x]):
                    for r_p, r_z, r_r in self.right.parse(i[:, x:]):
                        yield (l_p,translate((0, x), r_p)), {"child1": l_z, "child2": r_z, "x": x}, np.concatenate((l_r,r_r), 1)
                        
class Nothing(Parser):
    def __init__(self):
        super().__init__()
        pass

    def render(self, z):
        return None

    def _parse(self, i):
        yield None, None, i
    
def _subregions(i, size_bound=9999999999, aligned=False, extent=None):
    if extent: ew,eh = extent
    else: ew,eh = None,None
        
    w,h = i.shape
    
        
        
    if aligned:
        if ew is not None and eh is not None:
            regions=[(lx,lx+ew,ly,ly+eh)
                 for lx in range(0, w-ew+1)
                 for ly in range(0, h-eh+1)]
        elif ew is not None and eh is None:
            regions=[(lx,lx+ew,ly,uy)
                     for lx in range(w)
                     for ly in range(h)
                     for uy in range(ly+1, h+1)]
        elif ew is None and eh is not None:
            regions=[(lx,ux,ly,ly+eh)
                     for lx in range(w)
                     for ux in range(lx+1, w+1)
                     for ly in range(0, h-eh+1)]
        else:
            
            regions=[(lx,ux,0,h)
                 for lx in range(0, w)
                 for ux in range(lx+1, w+1)]+\
                [(0,w,ly,uy)
                 for ly in range(h) 
                 for uy in range(ly+1, h+1)]+\
                     [(0,ux,0,uy)
                     for ux in range(1, w+1)
                     for uy in range(1, h+1)]+\
                     [(0,ux,ly,h)
                     for ux in range(1, w+1)
                     for ly in range(0, h)]+\
                     [(lx,w,0,uy)
                     for uy in range(1, h+1)
                     for lx in range(0, w)]+\
                     [(lx,w,ly,h)
                     for ly in range(0, h)
                     for lx in range(0, w)]
        regions = list(set(regions))

        cropped_regions = []
        for lx,ux,ly,uy in regions:
            region = i[lx:ux, ly:uy]
            if np.all(region<=0): continue

            nz = np.nonzero(region>0)
            lx,ux,ly,uy = nz[0].min()+lx, nz[0].max()+1+lx, nz[1].min()+ly, nz[1].max()+1+ly
                
            cropped_regions.append((lx,ux,ly,uy))
        regions = list(set(cropped_regions))
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
    def __init__(self, *children, aligned=True, backtrack=False):
        super().__init__(children)
        self.children = children
        self.aligned = aligned
        self.backtrack = backtrack

    def cost(self):
        return 0.1 + sum(c.cost() for c in self.children)

    def render(self, z):
        return render_stack([Floating(child).render(child_z)
                             for child, child_z in reversed(list(zip(self.children, z))) ],
                            transparent=-1)

    def _parse(self, i, size_bound=9999999999):

        def f(j, still_need_to_parse):
            
            if len(still_need_to_parse)==0 or still_need_to_parse[0].__class__ is Nothing:
                yield [], (), j
                return 

            
            for lx,ux,ly,uy in _subregions(j, aligned=self.aligned,extent=still_need_to_parse[0].extent()):                
                for prefix, z0, r in Floating(still_need_to_parse[0]).parse(j[lx:ux,ly:uy]):
                    

                    z0 = dict(z0)
                    z0["_w"], z0["_h"] = j.shape[0], j.shape[1]
                    z0["x"] += lx
                    z0["y"] += ly
                    
                    prefix = translate((lx, ly), prefix)
                    residual = np.copy(j)
                    residual[lx:ux,ly:uy] = r
                    for suffix, z1, final_residual in f(residual, still_need_to_parse[1:]):
                        yield [prefix]+suffix, tuple([z0]+list(z1)), final_residual
                        if not self.backtrack:
                            return

                    if not self.backtrack:
                        return
                    
        yield from f(i, self.children)

class Repeat(Parser):
    def __init__(self, child, aligned=True, backtrack=False):
        super().__init__(child)
        self.child = child
        self.aligned = aligned
        self.backtrack = backtrack

    def cost(self):
        return 0.1 + self.child.cost()

    

    def render(self, z):
        return render_stack([Floating(self.child).render(c) for c in reversed(z) ], transparent=-1)

    def _parse(self, i, size_bound=9999999999):

        regions = _subregions(i, aligned=self.aligned, extent=self.child.extent())
        
        def f(j):
            nonlocal regions
            for lx,ux,ly,uy in regions:
                if np.all(j[lx:ux,ly:uy]<=0): continue
                for prefix, z0, r in Floating(self.child).parse(j[lx:ux,ly:uy]):

                    z0 = dict(z0)
                    z0["_w"], z0["_h"] = j.shape[0], j.shape[1]
                    z0["x"] += lx
                    z0["y"] += ly
                    
                    prefix = translate((lx, ly), prefix)
                    residual = np.copy(j)
                    residual[lx:ux,ly:uy] = r
                    for suffix, z1, final_residual in f(residual):
                        yield frozenset({prefix})|suffix, [z0]+z1, final_residual
                        if not self.backtrack: return 
                        

            yield frozenset({}), [], j
                            
        yield from f(i)

def render_stack(stack, transparent=-1):
    if len(stack)==1: return stack[0]

    x = render_stack(stack[1:], transparent=transparent)
    y = np.copy(stack[0])

    y[x!=transparent]=x[x!=transparent]

    return y
    

    
