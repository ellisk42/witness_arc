import random
from objects import *
from parsers import *
from parsers import _subregions
from pq import PQ
import time
import numpy as np
from func_timeout import func_timeout, FunctionTimedOut
import math
import traceback

def partial_parse(i, program):
    for p, z, r in Union(program, Nothing()).parse(i):
        return p, z, r
    return None

def complete_parse(i, program):
    for p, z, r in program.parse(i):
        return p, z, r
    return None

def signature(images, regionss, program):
    s = []
    for i, regions in zip(images, regionss):
        for lx,ux,ly,uy in reversed(regions):
            try:
                match = func_timeout(0.1, complete_parse, (i[lx:ux,ly:uy], program))
            except FunctionTimedOut:
                match = None
            s.append(match is None)
    return tuple(s)



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

            occupied_probability = 0.6
            occupied_cost = -math.log10(occupied_probability)
            unoccupied_cost = -math.log10(1-occupied_probability)
            n_occupied = np.sum(v*1)
            n_unoccupied = v.shape[0]*v.shape[1] - n_occupied
            c += n_occupied * occupied_cost + n_unoccupied * unoccupied_cost

        elif k=="dir": c+=0.6

        elif k.startswith("_"):
            continue
        else:
            assert False, k
    return c


    
def infer_parses(images, time_out, reference_solution):
    clear_parser_caches()
    

    common_colors = set(range(1, 10))
    for i in images:
        common_colors &= { c for r in i for c in r if c>0 }

    atomic = [a
              for c in common_colors | {None}
              for a in [Diagonal(color=c),
                        Rectangle(color=c), Rectangle(color=c, width=1, height=1),
                        Rectangle(color=c, width=1),
                        Rectangle(color=c, height=1),
                        Sprite(color=c), Sprite(color=c, contiguous=True)] ]
    
    repeats = [Repeat(a) for a in atomic ]
    combinators = [Union, Vertical, Horizontal]

    times=[]

                   
    collisions, collision_opportunities, visited_signatures = 0, 0, {}
    def priority(program, verbose=False):
        nonlocal collisions, collision_opportunities, visited_signatures
        matches=[]
        for i in images:
            t0=time.time()
            
            try:
                match = func_timeout(0.1, partial_parse, (i, program))
            except FunctionTimedOut:
                match = None
            except Exception as e:
                match = None
                print("Some kind of error in parser")
                traceback.print_exc()
                print()
                
                
            times.append(time.time()-t0)
            if match is None:
                return float("-inf"), float("inf"), []
            matches.append(match)

        pixel_cost = sum(np.sum(r>0) for _, _, r in matches)
        
        
        
        
        z_cost = sum( parse_cost(z) for _, z, _ in matches)

        program_cost = program.cost()

        total_cost = 0.5*pixel_cost + program_cost + z_cost

        residual_key = tuple(r.tostring() for _, _, r in matches)
        decomposition = tuple([tuple(flatten_z(p)) for p, _, _ in matches])
        collision_key = (decomposition, residual_key)
        collision_opportunities+=1
        if collision_key in visited_signatures:
            collisions+=1
            visited_signatures[collision_key].append(program)
        else:
            visited_signatures[collision_key] = [program]

        return -total_cost, pixel_cost, decomposition, program_cost, z_cost, [z for _, z, _ in matches]

    def successors(e):
        for a in atomic+repeats:
            for c in combinators:
                yield c(e, a)
        if not isinstance(e, (Repeat, Floating)):
            #yield Floating(e)
            yield Repeat(e)

    pq = PQ()
    visited=set()

    # for each unique decomposition of the input, what is the best scoring program which yields that decomposition?
    best_per_decomposition = {}

    best=None

    start_time = time.time()

    def push(program, priority, residual, decomposition):
        nonlocal best, best_per_decomposition
        if priority>float("-inf"):
            pq.push(priority, program)
            visited.add(a)
            if residual==0:
                if best is None or priority>best[0]:
                    best=(priority, program)
                    print("Best parse so far:", program, priority)

                decomposition = frozenset(decomposition)
                if decomposition not in best_per_decomposition or \
                   best_per_decomposition[decomposition][0]<priority:
                    best_per_decomposition[decomposition] = (priority, program, time.time())
                    print("\t(best decomposition)", program, priority)
                    

    for a in atomic:
        f, residual_size, decomposition = priority(a)[:3]
        push(a, f, residual_size, decomposition)


    while len(pq) and time.time()-start_time<time_out:
        p, e = pq.popMaximum()
        print("popping", p, e)

        for ep in successors(e):
            if time.time()-start_time>=time_out:
                break
                
            if ep in visited:
                continue
            pp, residual, decomposition = priority(ep)[:3]
            if pp>float("-inf"):
                push(ep, pp, residual, decomposition)
                
                #print("\t pushing", pp, ep, residual)
                

    if best is None:
        print("Could not find any parse that explained all pixels")
        return None

    best_priority, program = best
    
    discovered_priority, _, _, discovered_program_cost, discovered_z_cost, _ = priority(program)
    print("DISCOVERED:\t", program)
    print("\t", f"explains all the pixels:\n\tpriority = {discovered_priority}\n\tprogram cost = {discovered_program_cost}\n\tlatent cost = {discovered_z_cost}\n")
    

    my_priority, _, _, my_program_cost, my_z_cost, _ = priority(reference_solution)
    print("HARD CODED:\t", reference_solution)
    print("\t", f"explains all the pixels:\n\tpriority = {my_priority}\n\tprogram cost = {my_program_cost}\n\tlatent cost = {my_z_cost}\n")
    
    print("Hall of fame:")
    best_per_decomposition = list(sorted(best_per_decomposition.values(), key=lambda sp: sp[0]))
    best_per_decomposition = best_per_decomposition[-10:]
    for score, program, t1 in best_per_decomposition:
        print(score, program, t1-start_time)
    times.sort()
    print("collisions", collisions, collision_opportunities, collisions/ collision_opportunities)

    print("collision analysis")
    visited_signatures = [(signature, matches) for signature, matches in visited_signatures.items() if len(matches) > 1]
    random.shuffle(visited_signatures)
    for _, collisions in visited_signatures[:20]:
        print(len(collisions), "\t".join(map(str, collisions[:10])))
    print()
        
    
    #print(times, sum(times)/len(times))
    return [program for score, program, t1 in reversed(best_per_decomposition)]
