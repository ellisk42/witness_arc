from objects import *
from parsers import *
from pq import PQ
import time
import numpy as np
from func_timeout import func_timeout, FunctionTimedOut

def partial_parse(i, program):
    for p, z, r in Union(program, Nothing()).parse(i):
        return p, z, r
    return None


def infer_parses(images, time_out, reference_solution):

    common_colors = set(range(1, 10))
    for i in images:
        common_colors &= { c for r in i for c in r if c>0 }

    atomic = [a
              for c in common_colors | {None}
              for a in [Diagonal(color=c), Rectangle(color=c), Rectangle(color=c, width=1, height=1), Sprite(color=c)] ]
    repeats = [Repeat(a) for a in atomic ]
    combinators = [Union, Vertical, Horizontal]

    times=[]

    def parse_cost(z):
        if isinstance(z, list):
            return sum(parse_cost(x) for x in z)+1
        if isinstance(z, tuple):
            return sum(parse_cost(x) for x in z)
        assert isinstance(z, dict)

        c=0
        for k, v in z.items():

            if k in ["c", "x", "y", "w", "h"]:
                c+=1
            elif "child" in k:
                c+=parse_cost(v)
            elif k=="mask":
                c += v.shape[0]*v.shape[1] * 0.3

            elif k=="dir": c+=0.6
                
            elif k.startswith("_"):
                continue
            else:
                assert False, k
        return c               
    
    def priority(program, verbose=False):
        
        matches=[]
        for i in images:
            t0=time.time()
            
            try:
                match = func_timeout(0.05, partial_parse, (i, program))
            except FunctionTimedOut:
                match = None
            
            times.append(time.time()-t0)
            if match is None:
                return float("-inf"), float("inf")
            matches.append(match)

        pixel_cost = sum(np.sum(r>0) for _, _, r in matches)
        
        z_cost = sum( parse_cost(z) for _, z, _ in matches)

        program_cost = program.cost()

        total_cost = 0.5*pixel_cost + program_cost + z_cost

        return -total_cost, pixel_cost, program_cost, z_cost, [z for _, z, _ in matches]

    def successors(e):
        for a in atomic+repeats:
            for c in combinators:
                yield c(e, a)
        if not isinstance(e, (Repeat, Floating)):
            yield Floating(e)
            yield Repeat(e)

    pq = PQ()
    visited=set()

    best=None

    start_time = time.time()

    def push(program, priority, residual):
        nonlocal best
        if priority>float("-inf"):
            pq.push(priority, program)
            visited.add(a)
            if residual==0:
                if best is None or priority>best[0]:
                    best=(priority, program)
                    print("Best parse so far:", program)

    for a in atomic:
        f, residual_size = priority(a)[:2]
        push(a, f, residual_size)


    while len(pq) and time.time()-start_time<time_out:
        p, e = pq.popMaximum()
        #print("popping", p, e, priority(e)[1])

        for ep in successors(e):
            if time.time()-start_time>=time_out:
                break
                
            if ep in visited:
                continue
            pp, residual = priority(ep)[:2]
            if pp>float("-inf"):
                push(ep, pp, residual)
                #print("\t pushing", pp, ep, residual)
                

    if best is None:
        print("Could not find any parse that explained all pixels")
        return None

    best_priority, program = best
    print("\t", program)
    print("\t", "explains all the pixels")
    print("\t", "various costs", priority(program)[1:-1])
    print("\t", "vs reference solution costs", priority(reference_solution)[1:-1])
    times.sort()
    #print(times, sum(times)/len(times))
    return program
