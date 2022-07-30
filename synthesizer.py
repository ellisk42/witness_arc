from enumerator import *
from inference import *
from parsers import *

import json
import time

def synthesize(problem_code, parsing_timeout):
    data = json.load(open(f'arc_problems/{problem_code}.json'))
    inputs = [ np.array(input_output["input"]).T
               for input_output in data["train"] ]
    test_inputs = [ np.array(input_output["input"]).T
                    for input_output in data["test"] ]
    test_outputs = [ np.array(input_output["output"]).T
                    for input_output in data["test"] ]
    
    outputs = [ np.array(input_output["output"]).T
               for input_output in data["train"] ]
    
    input_parses = infer_parses(inputs+test_inputs, parsing_timeout)
    output_parses = infer_parses(outputs, parsing_timeout)

    input_parses = [input_parse for input_parse, _, _ in input_parses[:1] ]
    output_parses = [output_parse for output_parse, _, _ in output_parses[:1] ]

    start_time = time.time()
    for prediction_index, test_predictions in enumerate(synthesize_given_parses(inputs+test_inputs, 
                                                                                input_parses,
                                                                                outputs,
                                                                                output_parses)):
        print(f"synthesized program #{prediction_index} after {time.time()-start_time} sec")
        print("The predictions for this program are:")
        print(test_predictions)

        fraction_correct = sum(np.all(yh == y) for yh, y in zip(test_predictions, test_outputs) ) /\
                           len(test_outputs)
        print(f"{fraction_correct*100}% correct")
        if fraction_correct >= 0.99:
            print("SUCCESS")
            return 


def create_synthesis_inputs(inputs,
                            input_parses):
    synthesis_inputs = {}
    for parser_index, input_parse in enumerate(input_parses):
        decompositions = []
        for x in inputs:
            decomposition, z, _ = next(Union(input_parse, Nothing()).parse(x))
            decomposition = flatten_decomposition(decomposition)
            decompositions.append(frozenset(decomposition))

        if any(len(d) > 1 for d in decompositions ):
            synthesis_inputs[f"decomposition{parser_index}"] = ("set(object)", tuple(decompositions))
        else:
            decompositions = [list(d)[0] for d in decompositions ]
            synthesis_inputs[f"decomposition{parser_index}"] = ("object", tuple(decompositions))
            

    synthesis_inputs["canvas_width"] = ("number", tuple(x.shape[0] for x in inputs ))
    synthesis_inputs["canvas_height"] = ("number", tuple(x.shape[1] for x in inputs ))

    return synthesis_inputs

def create_synthesis_output_targets(outputs, output_parses):

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

    def z_to_targets(zs, field=None):

        """zs: list of z structures, one for each output
        returns: tuple of 1. list of (tp, valuation) that the enumerator wants to get
                          2. a function which, given useful_programs, tries to produce the z's mapping for the test instances"""

        original_field = field
        if field is not None and field.startswith("_"): field = field[1:]
        
        if all( isinstance(z, (int, np.int64)) for z in zs ):

            
            if field == "c": tp = "color"
            elif field in ["x", "y", "w", "h"]: tp = "number"
            else: assert False, f"field equals {field}"

            def reconstructor(useful_programs):
                return useful_programs[(tp, tuple(zs))][1]
            
            return [(tp, tuple(zs))], reconstructor
                
            
        elif all( isinstance(z, tuple) for z in zs ):
            tuple_length = len(zs[0])
            
            children = [z_to_targets([z[i] for z in zs ], field=field)
                        for i in range(tuple_length)]
            all_targets = [ t for ts, _ in children for t in ts ]
            all_reconstructors = [ r for _, r in children ]

            def tuple_reconstructor(useful_programs):
                # return a list of tuples

                
                reconstructed = [ r(useful_programs) for r in all_reconstructors ]
                I = len(reconstructed)
                J = len(reconstructed[0])
                return [ tuple(reconstructed[i][j] for i in range(I)) for j in range(J) ]

            return all_targets, tuple_reconstructor
                
                
        elif all( isinstance(z, dict) for z in zs ):
            reconstructors = {}
            all_targets = []
            for k in zs[0]:
                these_targets, reconstructors[k] = z_to_targets([z[k] for z in zs ], field=k)
                all_targets.extend(these_targets)

                
            def dictionary_reconstructor(useful_programs):
                # return a list of dictionaries
                reconstructions = {field: r(useful_programs)
                                   for field, r in reconstructors.items()}
                
                n_test = len(list(reconstructions.values())[0])
                return [ {field: values[n] for field, values in reconstructions.items() }
                         for n in range(n_test) ]
                
            return all_targets, dictionary_reconstructor
        elif field == "mask":
            return
        elif all( isinstance(z, list) for z in zs ):
            # todo lifted operators
            return 
        else:
            import pdb; pdb.set_trace()
            
            assert False
    reconstructors = []
    for parser_index, output_parse in enumerate(output_parses):
        
        decompositions = []
        latents = []
        wrapped_parser = Union(output_parse, Nothing())
        for y in outputs:
            decomposition, z, _ = next(wrapped_parser.parse(y))

            latents.append(z)
            
            decomposition = flatten_decomposition(decomposition)
            decompositions.append(frozenset(decomposition))

        targets["set(object)"] = targets.get("set(object)", []) + [tuple(decompositions)]
        if all(len(os) == 1 for os in decompositions ):
            targets["object"] = targets.get("object", []) + [tuple(list(os)[0]
                                                                   for os in decompositions)]

        these_targets, this_reconstructor = z_to_targets(latents)
        for tp, vs in these_targets:
            targets[tp] = targets.get(tp, []) + [tuple(vs)]
        reconstructors.append((this_reconstructor, wrapped_parser))

            
            
    targets = {tp: list(set(valuation_list))
               for tp, valuation_list in targets.items() }
    return targets, reconstructors
        
        

def synthesize_given_parses(inputs,
                            input_parses,
                            outputs, 
                            output_parses):

    inputs = create_synthesis_inputs(inputs, input_parses)
    
    output_targets, reconstructors = create_synthesis_output_targets(outputs, output_parses)

    n_training = len(outputs)

    common_colors = set(range(1, 10))
    for y in outputs:
        common_colors &= { c for r in y for c in r if c>0 }

    primitives = [p for p in allPrimitives
                  if p.return_type != "color" or p.implementation in common_colors ]
    print(primitives)

    useful_programs = {} # maps from (tp, train valuation) to (expression, test valuation)
    
    number_of_programs = 0
    for e, t, v in enumerate_expressions_bottom_up(primitives, inputs):
        number_of_programs+=1
        if str(number_of_programs)[0]=="1" and all(c =="0" for c in str(number_of_programs)[1:]):
            print("We have explored", number_of_programs)
            # print("\tTo get a sense of what we are seeing",
            #       "this is the latest program\n\t", e, t, v)
            print("Checking to see if synthesis is successful...")

            for r, output_generator in reconstructors:
                try: test_zs = r(useful_programs)
                except KeyError: continue
                

                yield [output_generator.render(test_z) for test_z in test_zs]
                
                    
            

        if t in output_targets:
            
            if v[:n_training] in output_targets[t]:
                if (t, v[:n_training]) not in useful_programs:
                    print("looks like a useful expression for solving the problem:", e)
                    useful_programs[(t, v[:n_training])] = (e, v[n_training:])

        
