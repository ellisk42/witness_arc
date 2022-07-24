import itertools

itertools.tee

def my_generator(i):
    for j in range(i):
        print("computing and yielding", j, 'from', i )
        yield j

i1, i2 = itertools.tee(my_generator(5), 2)
import pdb; pdb.set_trace()


cache = {}
def smart_generator(i):
    if i not in cache: cache[i] = ([], my_generator(i))

    the_list, the_generator = cache[i]

    yield from the_list
    for rv in the_generator:
        the_list.append(rv)
        yield rv
        

