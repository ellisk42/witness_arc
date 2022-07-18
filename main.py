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
    ("6c434453", Repeat(Sprite(color=1, contiguous=True))),
    
    ("7c008303",  Union(Rectangle(height=1, color=8),
                        Union(Rectangle(width=1, color=8),
                              Sprite(color=3),
                              Horizontal(Vertical(Rectangle(height=1, width=1),Rectangle(height=1, width=1)),
                                         Vertical(Rectangle(height=1, width=1),Rectangle(height=1, width=1)))))),
    
    
    ("6e82a1ae", Repeat(Sprite(color=5, contiguous=True))),
    
    ("3af2c5a8", Floating(Sprite())),
    ("7ddcd7ec", Union(Rectangle(height=None, width=None),
                       Repeat(Rectangle(height=1, width=1)))),
    
    
    
    

    ("2281f1f4", Repeat(Rectangle(color=5, height=1, width=1))), 
    ("ded97339", Repeat(Rectangle(color=8, height=1, width=1))),
    ("5c0a986e", Union(Rectangle(color=1), Rectangle(color=2))),
    
    
    
    ("d631b094", Repeat(Rectangle(height=1, width=1))),
    ("ac0a08a4", Repeat(Rectangle(height=1, width=1))),
    ("6150a2bd", Repeat(Rectangle(height=1, width=1))),
    ("62c24649", Repeat(Rectangle(height=1, width=1))),
    
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
    
    
    ("97999447", Repeat(Rectangle(height=1, width=1))),
    ("a3325580", Repeat(Sprite())),
    ("a78176bb", Union(Diagonal(),
                       Repeat(Sprite(color=5, contiguous=True)))),

    ("47c1f68c",
     Union(Union(Rectangle(height=1), Rectangle(width=1)),
                       Sprite())),
    

    #          requires overlapping sprites
    # ("e5062a87", Union(Sprite(color=2), Sprite(color=5))),

    

    # requires extensive backtracking
    # ("b94a9452", Floating(Union(Rectangle(), Rectangle(), aligned=False))),              
]

correct_numbers_of_objects = {'7c008303': [7, 7, 7], '6c434453': [5, 5], '6e82a1ae': [6, 5, 4], '3af2c5a8': [1, 1, 1], '7ddcd7ec': [2, 3, 3], '2281f1f4': [5, 7, 9], 'ded97339': [4, 5, 6], '5c0a986e': [2, 2, 2], 'd631b094': [2, 3, 1, 4], 'ac0a08a4': [2, 3, 5], '6150a2bd': [6, 4], '62c24649': [7, 7, 5], 'a87f7484': [3, 4, 5, 4], '025d127b': [8, 4], '1caeab9d': [3, 3, 3], 'b8cdaf2b': [4, 4, 4, 4], '99b1bc43': [3, 3, 3, 3], '97999447': [2, 3, 1], 'a3325580': [4, 3, 4, 3, 2, 3], 'a78176bb': [2, 2, 3], '47c1f68c': [3, 3, 3]}


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
    print("Total problems:", len(times))

    
    print("# objects:",
          {code: [len(flatten_z(p[0])) for p in ps ]
           for code, ps in parses.items() })
    for code, correct_numbers in correct_numbers_of_objects.items():
        actual_numbers = [len(flatten_z(p[0])) for p in parses[code] ]
        for n, (actual, correct) in enumerate(zip(actual_numbers, correct_numbers)):
            if actual != correct:
                print("Possibly bad inference for", code, "#", n)
        

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

        infer_parses(inputs, 30, parser)
        print("Human written solution:")
        print(parser)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("task", default="inference", choices=["inference", "test"])
    
    arguments = parser.parse_args()
    
    if arguments.task=="test": test_manual_parsers()
    if arguments.task=="inference": test_parse_inference()
