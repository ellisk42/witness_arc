x=np.zeros((12, 12))

x[1:3,1:4]=3
x[5:6,6:8]=3

print(x)

parser=Horizontal(Floating(Rectangle()), Rectangle())
#parser=Floating(Rectangle())

for parse, residual in parser.parse(x):
    if np.all(residual<=0):
        print(parse)
        print(residual)
    else:
        print("Incomplete parse, skipping")
        

def rectangle(i):
    if i[0,0]>0 and np.min(i)==np.max(i):
        return (i[0,0], i.shape)
    return None

def _trim(i):
    while np.all(i[:,0]<=0) and i.shape[1]>1: i=i[:,1:]
    while np.all(i[0,:]<=0) and i.shape[0]>1: i=i[1:,:]
    while np.all(i[:,-1]<=0) and i.shape[1]>1: i=i[:,:-1]
    while np.all(i[-1,:]<=0) and i.shape[0]>1: i=i[:-1,:]
    return i

def _subregions(w,h,size_bound=9999999999):
    regions=[(lx,ux,ly,uy)
     for lx in range(w)
     for ly in range(h)
     for ux in range(lx+1, w)
     for uy in range(ly+1, h)
             if (ux-lx)*(uy-ly)<size_bound
    ]
    regions.sort(key=lambda z: -(z[1]-z[0])*(z[3]-z[2]))
    return regions
    
def floating(p):
    def f(i):
        return p(_trim(i))
    return f

def concatenate_vertical(p, q):
    def f(i):
        possibilities = {(x,y)
                         for y in range(1,i.shape[1])
                         for x in [p(i[:,:y])]
                         if x
                         for y in [q(i[:,y:])]
                         if y}
        if len(possibilities)==1:
            for z in possibilities: return z
        return None
    return f

def repeat(p):
    
    def f(i, size_bound=9999999999):
        just_one = floating(p)(i)
        if just_one:
            print('just')
            return frozenset({just_one})

        for lx,ux,ly,uy in _subregions(*i.shape, size_bound=size_bound):
            
            sl=_trim(i[lx:ux,ly:uy])            
            just_one = p(sl)
            if just_one is None:
                continue

            size_bound=(lx-ux)*(ly-uy)+1
            print("size_bound", size_bound)

            ip=np.copy(i)
            ip[lx:ux,ly:uy]=0
            attempt=f(ip, size_bound=size_bound)
            if attempt:
                return attempt|frozenset({just_one})
            return None
        return None
    return f
def test():
    for _ in range(10):
        print(concatenate_vertical(floating(rectangle), floating(rectangle))(x))
        #print(repeat(rectangle)(x))
cProfile.run("test()")

            

