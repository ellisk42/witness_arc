import pickle
import matplotlib.pyplot as plt
import cProfile
import numpy as np
import json
from collections import namedtuple
import sys
import os
import time


from synthesizer import synthesize
from objects import *
from parsers import *
from inference import *
np.set_printoptions(threshold=sys.maxsize)








testcases = [
    #("444801d8", Repeat(Union(Sprite(color=1, contiguous=True), Rectangle(height=1, width=1)))),
    
    ("c8f0f002", Union(Repeat(Rectangle(color=1, height=1, width=1)),
                       Repeat(Rectangle(color=7, height=1, width=1)),
                       Repeat(Rectangle(color=8, height=1, width=1)))), 
    ("b8cdaf2b", Horizontal(Floating(Rectangle()),
                            Horizontal(Floating(Vertical(Rectangle(), Rectangle())),
                                       Floating(Rectangle())))),
    
    ("5c0a986e", Union(Rectangle(color=1), Rectangle(color=2))),
        ("a78176bb", Union(Diagonal(),
                       Repeat(Sprite(color=5, contiguous=True)))),
    
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
    
    ("99b1bc43", Vertical(Sprite(diffuse=True),
                          Vertical(Rectangle(),
                                   Sprite(diffuse=True)))),
    
    
    ("97999447", Repeat(Rectangle(height=1, width=1))),
    ("a3325580", Repeat(Sprite())),
    

    ("47c1f68c",
     Union(Union(Rectangle(height=1), Rectangle(width=1)),
                       Sprite())),
    

    #          requires overlapping sprites
    # ("e5062a87", Union(Sprite(color=2), Sprite(color=5))),

    

    # requires extensive backtracking
    # ("b94a9452", Floating(Union(Rectangle(), Rectangle(), aligned=False))),

    # requires better region proposal
    # ("6773b310", Union(Repeat(Rectangle(color=8)),
    #                            Repeat(Sprite(color=6, height=3, width=3, diffuse=True)),
    #                            aligned=True)), 
]

correct_numbers_of_objects = {'7c008303': [7, 7, 7], '6c434453': [5, 5], '6e82a1ae': [6, 5, 4], '3af2c5a8': [1, 1, 1], '7ddcd7ec': [2, 3, 3], '2281f1f4': [5, 7, 9], 'ded97339': [4, 5, 6], '5c0a986e': [2, 2, 2], 'd631b094': [2, 3, 1, 4], 'ac0a08a4': [2, 3, 5], '6150a2bd': [6, 4], '62c24649': [7, 7, 5], 'a87f7484': [3, 4, 5, 4], '025d127b': [8, 4], '1caeab9d': [3, 3, 3], 'b8cdaf2b': [4, 4, 4, 4], '99b1bc43': [3, 3, 3, 3], '97999447': [2, 3, 1], 'a3325580': [4, 3, 4, 3, 2, 3], 'a78176bb': [2, 2, 3], '47c1f68c': [3, 3, 3]}


def test_manual_parsers(visualize=True):
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
        times[code]=(time.time()-t0)/len(inputs)


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
          {code: [len(flatten_decomposition(p[0])) for p in ps ]
           for code, ps in parses.items() })
    for code, correct_numbers in correct_numbers_of_objects.items():
        if code not in parses: continue
        actual_numbers = [len(flatten_decomposition(p[0])) for p in parses[code] ]
        for n, (actual, correct) in enumerate(zip(actual_numbers, correct_numbers)):
            if actual != correct:
                print("Possibly bad inference for", code, "#", n)

    if not visualize: return 
        

    from plotting import plot_arc_array
    for code, parses in parses.items():
        os.system(f"rm -r parses/{code}; mkdir -p parses/{code}/")
        for n, (parse, x) in enumerate(parses):
            #lt.figure()
            plot_arc_array([animate(parse, x)])
            plt.savefig(f"parses/{code}/{n}.png")
            plt.close()


def test_parse_inference(timeout, bottom=False, visualize=True):
    for code, parser in testcases:
        print()
        print("STARTING ", code)
        data = json.load(open(f'../ARC/data/training/{code}.json'))
        inputs = [ np.array(input_output["input"]).T
                   for input_output in data["train"]]

        if bottom:
            programs = bottom_up_enumeration(inputs, timeout, parser)
        else:
            programs = infer_parses(inputs, timeout, parser)
        print("Human written solution:")
        print(parser)

        clear_parser_caches()
        
        if visualize:
            os.system(f"rm parses/{code}/*_predicted.png")
            from plotting import plot_arc_array
            for n, x in enumerate(inputs):
                for pi, (program, _, _) in enumerate(programs):
                    parse = next(Union(program, Nothing()).parse(x))
                    os.system(f"mkdir -p parses/{code}/")

                    plot_arc_array([animate(parse[0], x)])
                    plt.savefig(f"parses/{code}/{n}_predicted_{pi}.png")
                    plt.close()
            try:
                with open(f"parses/{code}/inferred.pickle", "wb") as handle:
                    pickle.dump(programs, handle)
            except:
                import pdb; pdb.set_trace()
            
def analyze_parsing_performance():

    time_to_find_ground_truth_decomposition = []
    ranking_of_ground_truth_decomposition = []
    cost_difference = []
    
    for code, parser in testcases:
        print()
        print("STARTING ", code)
        
        data = json.load(open(f'../ARC/data/training/{code}.json'))
        inputs = [ np.array(input_output["input"]).T
                   for input_output in data["train"]]
        
        with open(f"parses/{code}/inferred.pickle", "rb") as handle:
            programs = pickle.load(handle)

        def analyze_program(e):
            """Returns the overall cost and decomposition"""
            try:
                outputs = [ next(Union(e, Nothing()).parse(x))[:2] for x in inputs ]
            except: import pdb; pdb.set_trace()
            
            z_cost = sum( parse_cost(z) for _, z in outputs)
            objects = tuple([ frozenset(flatten_decomposition(os)) for os, z in outputs ])
            return e.cost()+z_cost, objects

        ground_truth_cost, ground_truth_objects = analyze_program(parser)

        analysis = [(t, analyze_program(e)) for e, _, t in programs ]

        
        cost_difference.append(ground_truth_cost - min(c for _, (c, _) in analysis))

        try: ranking = [ os for _, (_, os) in analysis].index(ground_truth_objects)
        except ValueError: ranking = 11
        ranking_of_ground_truth_decomposition.append(ranking)

        if ranking != 11:
            time_to_find_ground_truth_decomposition.append(analysis[ranking][0])

    fig, axs = plt.subplots(3,1)
    axs[0].set_xlabel("time to gt decomposition")
    axs[0].hist(time_to_find_ground_truth_decomposition)
    axs[1].set_xlabel("ranking of gt decomposition")
    axs[1].hist(ranking_of_ground_truth_decomposition)
    axs[2].set_xlabel("cost gt  - cost predicted")
    axs[2].hist(cost_difference)
    plt.tight_layout()
    plt.savefig(f"analysis.png")
    #import pdb; pdb.set_trace()
    
        
        
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("task", default="inference", choices=["inference", "test", "analyze", "synthesize"])
    parser.add_argument("--timeout", "-t", default=30, type=int)
    parser.add_argument("--profile", "-p", default=False, action="store_true")
    parser.add_argument("--bottom", "-b", default=False, action="store_true")
    
    arguments = parser.parse_args()
    if arguments.task=="synthesize":
        
        problem = "d631b094"
        #problem = "e179c5f4"
        synthesize(problem)
        
    if arguments.task=="analyze":
        analyze_parsing_performance()
    if arguments.task=="test":
        if arguments.profile:
            cProfile.run("test_manual_parsers(visualize=False)")
        else:
            test_manual_parsers()
    if arguments.task=="inference":
        if arguments.profile:
            cProfile.run("test_parse_inference(arguments.timeout, visualize=False, bottom=arguments.bottom)")
        else:
            test_parse_inference(arguments.timeout, bottom=arguments.bottom)
