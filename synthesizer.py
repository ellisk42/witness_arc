from enumerator import *
from inference import *
from parsers import *

import json

def synthesize(problem_code):
    data = json.load(open(f'../ARC/data/training/{problem_code}.json'))
    inputs = [ np.array(input_output["input"]).T
               for input_output in data["train"] ]
    outputs = [ np.array(input_output["output"]).T
               for input_output in data["train"] ]
    
    input_parses = infer_parses(inputs, 10)
    output_parses = infer_parses(outputs, 10)

    input_parses = [input_parse for input_parse, _, _ in input_parses ]
    output_parses = [output_parse for output_parse, _, _ in output_parses ]
    
    

    synthesize_given_parses(list(zip(inputs, outputs)),
                            input_parses,
                            output_parses)

def create_synthesis_inputs(input_outputs,
                            input_parses):
    inputs = {}
    for parser_index, input_parse in enumerate(input_parses):
        decompositions = []
        for x, _ in input_outputs:
            decomposition, z, _ = next(Union(input_parse, Nothing()).parse(x))
            decomposition = flatten_decomposition(decomposition)
            decompositions.append(frozenset(decomposition))

        inputs[f"decomposition{parser_index}"] = ("set(object)", tuple(decompositions))

    inputs["canvas_width"] = ("number", tuple(x.shape[0] for x, _ in input_outputs ))
    inputs["canvas_height"] = ("number", tuple(x.shape[1] for x, _ in input_outputs ))

    return inputs

def create_synthesis_output_targets(input_outputs, output_parses):

    targets = {"set(object)": []} # maps from tp to valuations which correspond to different ways of constructing the output

    def z_to_targets(zs, field=None):
        if field is not None and field.startswith("_"): field = field[1:]
        
        if all( isinstance(z, (int, np.int64)) for z in zs ):
            
            if field == "c": yield "color", tuple(zs)
            elif field in ["x", "y", "w", "h"]: yield "number", tuple(zs)
            elif field in "xy": yield "number", tuple(zs)
            else: assert False, f"field equals {field}"
                
            
        elif all( isinstance(z, tuple) for z in zs ):
            for i in range(len(zs[0])):
                yield from z_to_targets([z[i] for z in zs ], field=field)
                
        elif all( isinstance(z, dict) for z in zs ):
            for k in zs[0]:
                yield from z_to_targets([z[k] for z in zs ], field=k)
        elif field == "mask":
            return
        elif all( isinstance(z, list) for z in zs ):
            # todo lifted operators
            return 
        else:
            import pdb; pdb.set_trace()
            
            assert False
    
    for parser_index, output_parse in enumerate(output_parses):
        
        decompositions = []
        latents = []
        for _, y in input_outputs:
            decomposition, z, _ = next(Union(output_parse, Nothing()).parse(y))

            latents.append(z)
            
            decomposition = flatten_decomposition(decomposition)
            decompositions.append(frozenset(decomposition))

        targets["set(object)"] = targets.get("set(object)", []) + [tuple(decompositions)]
        if all(len(os) == 1 for os in decompositions ):
            targets["object"] = targets.get("object", []) + [tuple(list(os)[0]
                                                                   for os in decompositions)]

        for tp, vs in z_to_targets(latents):
            targets[tp] = targets.get(tp, []) + [tuple(vs)]
            
    targets = {tp: list(set(valuation_list))
               for tp, valuation_list in targets.items() }
    return targets
        
        

def synthesize_given_parses(input_outputs,
                            input_parses,
                            output_parses):

    inputs = create_synthesis_inputs(input_outputs, input_parses)
    
    output_targets = create_synthesis_output_targets(input_outputs, output_parses)

    common_colors = set(range(1, 10))
    for _, y in input_outputs:
        common_colors &= { c for r in y for c in r if c>0 }

    primitives = [p for p in allPrimitives
                  if p.return_type != "color" or p.implementation in common_colors ]
    print(primitives)

    number_of_programs = 0
    for e, t, v in enumerate_expressions_bottom_up(primitives, inputs):
        number_of_programs+=1
        if str(number_of_programs)[0]=="1" and all(c =="0" for c in str(number_of_programs)[1:]):
            print(number_of_programs, e, t, v)

        if t in output_targets:
            if v in output_targets[t]:
                print("looks useful", e)

                

    

    
