import numpy as np
from immutable import Immutable

class Expression():
    def __repr__(self): return str(self)
    def __eq__(self,o): return str(self) == str(o)
    def __ne__(self,o): return str(self) != str(o)
    def __hash__(self): return hash(str(self))

    @property
    def isApplication(self): return False
    @property
    def isPrimitive(self): return False
    @property
    def isVariable(self): return False


class Application(Expression):
    def __init__(self, function, *arguments):
        self.f = function
        self.xs = arguments
        self.children = [function] + list(arguments)

    @property
    def isApplication(self): return True

    def __str__(self):
        return f"{self.f}({', '.join(map(str,self.xs))})"

    def execute(self, environment):
        return self.f.execute(environment)(*[x.execute(environment) for x in self.xs ])

    def expressionSize(self): return 1 + sum(k.expressionSize() for k in self.xs )

class Primitive(Expression):
    def __init__(self, name, tp, argument_types, implementation):
        self.name = name
        self.implementation = implementation
        self.children = []
        self.return_type = tp
        self.argument_types = argument_types

    def __str__(self):
        return self.name

    def execute(self, environment):
        return self.implementation

    def __call__(self, *xs):
        return Application(self, *xs)

    def expressionSize(self): return 1

    @property
    def isPrimitive(self): return True

class Variable(Expression):
    def __init__(self, name, tp):
        self.name = name
        self.tp = tp

    def __str__(self):
        return self.name

    def execute(self, environment):
        return environment[self.name]

    def expressionSize(self): return 1

    @property
    def isVariable(self): return True

# _rectangle = Primitive("rectangle", "object", ["vector","vector","color"], Rectangle)
# _line = Primitive("line", "object", ["vector","vector","depth","color"], Line)
# _pixel = Primitive("pixel", "object", ["vector","depth","color"], Pixel)
# drawingPrimitives = [
#     _rectangle,
#     _line,
#     _pixel
# ]

getters = [
    Primitive(".color", "color", ["object"], lambda o: o.color),
    Primitive(".mass", "number", ["object"], lambda o: o.mass),
    Primitive(".width", "number", ["object"], lambda o: o.width),
    Primitive(".height", "number", ["object"], lambda o: o.height),
    Primitive(".northwest", "vector", ["object"], lambda o: o.northwest),
    Primitive(".northeast", "vector", ["object"], lambda o: o.northeast),
    Primitive(".southwest", "vector", ["object"], lambda o: o.southwest),
    Primitive(".southeast", "vector", ["object"], lambda o: o.southeast),
    Primitive(".x", "number", ["vector"], lambda v: v[0]),
    Primitive(".y", "number", ["vector"], lambda v: v[1])
]

arithmetic = [
    Primitive("min", "number", ["number","number"], lambda a,b: min(a,b)),
    Primitive("max", "number", ["number","number"], lambda a,b: max(a,b)),
    Primitive("+", "number", ["number","number"], lambda a,b: a+b),
    Primitive("-", "number", ["number","number"], lambda a,b: a-b),
    # Primitive("H-1", "number", [], None),
    # Primitive("W-1", "number", [], None),
]

_vector = Primitive("vector", "vector", ["number","number"], lambda x,y: Immutable([x,y]))

vector_algebra = [
    Primitive("-v", "vector", ["vector","vector"], lambda a,b: a-b),
    Primitive("+v", "vector", ["vector","vector"], lambda a,b: a+b),
    Primitive("*v", "vector", ["vector","number"], lambda a,b: a*b),
    Primitive("north", "vector", [], Immutable([0,1])),
    Primitive("north-west", "vector", [], Immutable([-1,1])),
    Primitive("north-east", "vector", [], Immutable([1,1])),
    Primitive("south", "vector", [], Immutable([0,-1])),
    Primitive("south-west", "vector", [], Immutable([-1,-1])),
    Primitive("south-east", "vector", [], Immutable([1,-1])),
    Primitive("east", "vector", [], Immutable([1,0])),
    Primitive("west", "vector", [], Immutable([-1,0])),
    _vector
]

set_manipulators = [
    Primitive("singleton", "object", ["set(object)"], lambda s: None if len(s) != 1 else list(s)[0])
]

_numbers = [
    Primitive(str(n),"number",[],n)
    for n in [0,1]
]

COLORS = ["black","blue",
          "red","green",
          "yellow","grey",
          "magenta","orange",
          "cyan","brown"]
_colors = [
    Primitive(c, "color", [], i)
    for i,c in enumerate(COLORS)
]
_bool = {True: Primitive("true", "bool", [], True),
         False: Primitive("false", "bool", [], False)}

allPrimitives = getters + arithmetic + vector_algebra + _numbers + _colors + list(_bool.values()) + set_manipulators
