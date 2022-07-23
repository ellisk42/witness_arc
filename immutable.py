
# ISC License (ISC)
#
# Copyright 2021 Christopher Fuller
#
# Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted,
# provided that the above copyright notice and this permission notice appear in all copies.

# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER
# IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.

from numpy import ndarray, asarray


class Immutable(ndarray):
    """https://numpy.org/doc/stable/user/basics.subclassing.html

    >>> import numpy as np
    >>> a=np.arange(4)
    >>> ia=ImmutArray(a)
    >>> (ia==[0,1,2,3]).all().item()
    True
    >>> (ia==(0,1,2,3)).all().item()
    True
    >>> ((0,1,2,3)==ia).all().item()
    True
    >>> ([0,1,2,3]==ia).all().item()
    True
    >>> (ia==range(4)).all().item()
    True
    >>> (range(4)==ia).all().item()
    True
    >>> a[1]=-1
    >>> a
    array([ 0, -1,  2,  3])
    >>> (a==ia).all().item()
    False
    >>> (ia==a).all().item()
    False
    >>> ia[1]=-1
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: assignment destination is read-only
    >>> ia
    ImmutArray([0, 1, 2, 3])
    >>> (np.arange(4)==ia).all().item()
    True
    >>> (ia==np.arange(4)).all().item()
    True
    >>> ib=ia[1:3]
    >>> ib
    ImmutArray([1, 2])
    >>> ib[0]=9
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: assignment destination is read-only
    >>> ia=np.arange(4).view(ImmutArray)
    >>> ia[0]=9
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: assignment destination is read-only
    """
    def __new__(cls, arr):
        obj = asarray(arr).copy().view(cls)
        obj.flags.writeable = False
        obj._hash = cls._gethash(obj)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self._hash = getattr(obj, '_hash', None) or self._gethash(obj)
        self.flags.writeable = False


    @staticmethod
    def _gethash(arr):
        return hash((arr.shape, tuple(arr.flat)))

    def __hash__(self):
        return self._hash

    #__eq__ = ndarray.__eq__
    def __eq__(self, other):
        return ndarray.__eq__(self, other).all().item()


if __name__ == '__main__':
    import doctest
    results = doctest.testmod()

    if results.attempted:
        if results.failed:
            print('Failure!')
        else:
            print('Success!')
    else:
        print('No tests found!')
