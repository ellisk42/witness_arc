import itertools
from dsl import *

def enumerate_expressions_bottom_up(primitives, inputs):
    """
    inputs: dictionary mapping variable name to (type, values)
    """

    n_examples = { len(values) for _, (_, values) in inputs.items()}
    if len(n_examples) != 1:
        assert False, "each variable should have the same number of valuations"

    for x in n_examples:
        n_examples = x
        break

    
    
    types = {tp for tp, _ in inputs.values()  } |\
            {tp for primitive in primitives for tp in [primitive.return_type]+ primitive.argument_types }

    input_variable_names = set(inputs.keys())

    # frontier maps from a type to a list of expression sizes whose elements are map from observational behavior to expression
    frontier = {tp: [None for sz in range(100) ]
                for tp in types }

    def register(expression, tp, valuation, sz):
        valuation_dictionary = frontier[tp][sz]
        if valuation_dictionary is None:
            valuation_dictionary = {}
            frontier[tp][sz] = valuation_dictionary
        valuation_dictionary[valuation] = expression

    # populate with constants
    for component in primitives:
        if len(component.argument_types) > 0: continue

        tp = component.return_type
        register(component, tp, tuple([component.implementation]*n_examples), 1)

    # populate with variables
    for nm, (tp, values) in inputs.items():        
        component = Variable(nm, tp)
        register(component, tp, tuple(values), 1)    
    
    print("Initial frontier:")
    for tp,behaviors in frontier.items():
        for sz,bs in enumerate(behaviors):
            if frontier[tp][sz] is None: continue
            for behavior, expr in frontier[tp][sz].items():
                print(f"{expr} (sz={sz}) has valuation {behavior}")
    print("This is the entire initial frontier.")

    
    components = [component for component in allPrimitives if len(component.argument_types) > 0]
    
    # first give the expressions which initially seeded the search
    for tp,sizedToBehaviors in frontier.items():
        for sz,behaviors in enumerate(sizedToBehaviors):
            if behaviors is None: continue
            for valuation, expression in behaviors.items():
                yield expression, tp, valuation
    desiredSize = 2
    while True:
        for component in components:
            arguments = component.argument_types
            if any( at not in frontier for at in arguments ): continue

            def argument_lists(childTypes, size):
                assert len(childTypes) > 0, "should have already handled primitives and variables"

                if len(childTypes) == 1:
                    a1 = frontier[childTypes[0]][size]
                    if a1 is None: return
                    for behavior, expr in a1.items():
                        yield [behavior], [expr]

                if len(childTypes) == 2:
                    for s1 in range(1, size):
                        a1 = frontier[childTypes[0]][s1]
                        if a1 is None: continue
                        s2 = size  - s1
                        a2 = frontier[childTypes[1]][s2]
                        if a2 is None: continue
                        for behavior1, expr1 in a1.items():
                            for behavior2, expr2 in a2.items():
                                yield [behavior1, behavior2], [expr1, expr2]

                if len(childTypes) == 3:
                    for s1 in range(1, size-1):
                        a1 = frontier[childTypes[0]][s1]
                        if a1 is None: continue
                        for s2 in range(1, size - s1):
                            s3 = size - s1 - s2
                            assert s3>0
                            a2 = frontier[childTypes[1]][s2]
                            a3 = frontier[childTypes[2]][s3]
                            if a2 is None or a3 is None: continue
                            for behavior1, expr1 in a1.items():
                                for behavior2, expr2 in a2.items():
                                    for behavior3, expr3 in a3.items():
                                        yield [behavior1, behavior2, behavior3], [expr1, expr2, expr3]
                if len(childTypes) > 3: assert False
                    
                


            for behaviors, expressions in argument_lists(arguments, desiredSize - 1):
                newValuation = []
                bad_expression = False
                for n in range(n_examples):
                    try: new_value = component.implementation(*[ v[n] for v in behaviors ])
                    except Exception as exn:
                        print(exn, "when executing", Application(component, *expressions), "with inputs", [ v[n] for v in behaviors ])
                        assert False
                        
                    if new_value is None:
                        bad_expression = True
                        break
                    newValuation.append(new_value)
                    
                if bad_expression: continue

                newValuation = tuple(newValuation)
                
                newExpression = Application(component, *expressions)

                valuation_dictionary = frontier[component.return_type][desiredSize]
                if valuation_dictionary is None:
                    valuation_dictionary = {}
                    frontier[component.return_type][desiredSize] = valuation_dictionary
                # does the behavior already exist
                if any( newValuation in otherBehaviors
                        for otherBehaviors in frontier[component.return_type][:desiredSize] if otherBehaviors is not None):
                    continue
                frontier[component.return_type][desiredSize][newValuation] = newExpression
                yield newExpression, component.return_type, newValuation
        desiredSize += 1
        print("enumerated all programs of size",desiredSize - 1)

if __name__ == '__main__':
    inputs = {"W": ("number", (5,3)),
              "H": ("number", (4, 3))}
    number_of_programs = 0
    for e, t, v in enumerate_expressions_bottom_up(allPrimitives, inputs):
        number_of_programs+=1
        #print(number_of_programs, e, t, v)
        if str(number_of_programs)[0]=="1" and all(c =="0" for c in str(number_of_programs)[1:]):
            print(number_of_programs, e, t, v)
