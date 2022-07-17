import matplotlib.pyplot as plt
import cProfile
import numpy as np
import json
from collections import namedtuple
import sys
import os
import time


from objects import *
from parsers import *
from inference import *
np.set_printoptions(threshold=sys.maxsize)








testcases = [
    ("ded97339", Repeat(Rectangle(color=8, height=1, width=1))),
    ("5c0a986e", Union(Rectangle(color=1), Rectangle(color=2))),
    
    ("d631b094", Floating(Sprite())),
    ("a87f7484", Repeat(Sprite())),
    
    ("025d127b", Repeat(Vertical(Floating(Rectangle()),
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
                       Repeat(Rectangle(height=1, width=1)),
                       aligned=False)),
    ("97999447", Repeat(Rectangle(height=1, width=1))),
    ("a3325580", Repeat(Sprite())),
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
            for parse, z, residual in parser.parse(x):
                if np.all(residual<=0):
                    print(parse)
                    if not np.all(render(parse, np.zeros_like(x))==x):
                        print("rendering failure")
                    print(z)
                    r = parser.render(z)
                    r[r<0]=0
                    if not np.all(r==x):
                        print("rendering failure")
                        import pdb; pdb.set_trace()
                    
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


def test_parse_inference():
    for code, parser in testcases:
        print()
        print("STARTING ", code)
        data = json.load(open(f'../ARC/data/training/{code}.json'))
        inputs = [ np.array(input_output["input"]).T
                   for input_output in data["train"]]

        infer_parses(inputs, 10, parser)
        print("Human written solution:")
        print(parser)
        
#test_manual_parsers()
test_parse_inference()
