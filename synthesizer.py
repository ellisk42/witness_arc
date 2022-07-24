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
def synthesize_given_parses(input_outputs,
                            input_parses,
                            output_parses):

    inputs = create_synthesis_inputs(input_outputs, input_parses)
    print(inputs)

    number_of_programs = 0
    for e, t, v in enumerate_expressions_bottom_up(allPrimitives, inputs):
        number_of_programs+=1
        if str(number_of_programs)[0]=="1" and all(c =="0" for c in str(number_of_programs)[1:]):
            print(number_of_programs, e, t, v)

    

    
