# Python Tricks

Here are some tricks/usages of Python3 built-in functions and standard libraries that I found useful and cool. It will be keep updating.

### Table of Content

- [map(), filter(), and reduce()](#map_filter_reduce)
- [bisect()](#bisect)
- [itertools.product()](#product)
- [partial()](#partial)
- [iter()](#iter)

<a name='map_filter_reduce'></a>

### map(), filter() and reduce()

https://book.pythontips.com/en/latest/map_filter.html

`map`(function, iteratbles): Excecute a specified function for each iterm in an iterable.

```python
word_list = ['apple', 'banana', 'cherry']
word_len = list(map(len, word_len))
print(word_len)  # [5, 6, 6]
```

`filter` (function, iterable): Filter given iterable by the function defined.

```python
number_list = range(-5, 5)
less_than_zero = list(filter(lambda x: x < 0, number_list))
print(less_than_zero)  # [-5, -4, -3, -2, -1]
```

`reduce`(function, iterable): Applies rolling computation and return the result.

```python
from functools import reduce
product = reduce((lambda x, y: x * y), [1, 2, 3, 4])
print(product)  # 24
```

<a name='bisect'></a>

### bisect

Python3 standard library: https://docs.python.org/3/library/bisect.html

The module can help us find the bisection index given a sorted list and a number. There are two parts in this module: `bisect` and `insort`.

##### bisect

`bisect.bisect_left`(*a*, *x*, *lo=0*, *hi=len(a)*): 

- If the number x is not in a, then it returns the index of the element at where it would be in the list (same as `bisect` or `bisect_right`)
- If the number x is in a, then it returns the index of the first instance of the number (leftmost of this number after insertion)

```python
import bisect

l = [1, 2, 3, 3, 3, 5]
bisect.bisect_left(l, 4)  # Not in list - output: 5
bisect.bisect_left(l, 3)  # In list - output: 2
```

`bisect.bisect`(*a*, *x*, *lo=0*, *hi=len(a)*) (Equivalent to `bisect.bisect_right` below)

`bisect.bisect_right`(*a*, *x*, *lo=0*, *hi=len(a)*): 

- If the number x is not in a, then it returns the index of the element at where it would be in the list.
- If the number x is in a, then it returns the index of the last instance + 1 of the number (rightmost of this number after insertion)

```python
import bisect

l = [1, 2, 3, 3, 3, 5]
bisect.bisect_right(l, 4)  # Not in list - output: 5
bisect.bisect_right(l, 3)  # In list - output: 5
```

**Note**: The `lo` and `hi` arguments can used to specify the subset of the list that we want to bisect.

##### insort

Nothing special, but to insert the value in place to the list at the index returned by the corresponding `bisect` method. Eg.  `bisect.insort_left(a, x, lo, hi)` is equivalent to`a.insert(bisect.bisect_left(a, x, lo, hi), x)`

<a name='product'></a>

### itertools.product()

`itertools.``product`(**iterables*, *repeat=1*)

Python 3 itertools: https://docs.python.org/3/library/itertools.html#itertools.product

Cartesian product of input iterables.

```python
from itertools import product

l = [[1, 2, 3], ['a', 'b', 'c']]
for i in product(*l):
    print(i)
# (1, 'a')
# (1, 'b')
# (1, 'c')
# (2, 'a')
# (2, 'b')
# (2, 'c')
# (3, 'a')
# (3, 'b')
# (3, 'c')
```

<a name='partial'></a>

### partial()

`functools.partial` (*func*, */*, **args*, ***keywords*)

Python 3 functools: https://docs.python.org/3.8/library/functools.html

It makes functions with partial functionalities of other functions, but specifying arguments.

```python
from functools import partial

basetwo = partial(int, base=2)
basetwo.__doc__ = 'Convert base 2 string to an int.'
basetwo('10010')  # 18
```

<a name='iter'></a>

### iter()

`iter`(*object*[, *sentinel*])

Python 3 standard library: https://docs.python.org/3/library/functions.html#iter

Reference for the examples: https://www.programiz.com/python-programming/methods/built-in/iter

Lists can be made into iterator objects, which can be used with more flexibility than for/while loops.

**Example 1**: Basic usage of having list into iterator

```python
l = ['a', 'b', 'c']
l_iter = iter(l)

print(next(l_iter))  # 'a'
print(next(l_iter))  # 'b'
print(next(l_iter))  # 'c'
print(next(l_iter))  #  StopIteration
```

**Note**: If the action goes beyond the length of the iterator, StopIteration will be raised. Therefore, it might be good idea to couple with try-except block if there's no plan to pay attention to the length of the iterator. It can be modified like below:

```python
l = ['a', 'b', 'c']
l_iter = iter(l)

try:
    while True:
        print(next(l_iter))
except StopIteration:
    pass
```

It will then print out the same results without havign the StopIteration raised.

**Example 2**: For custom objects

```python
class PrintNumber:
    def __init__(self, max):
        self.max = max

    def __iter__(self):
        self.num = 0
        return self

    def __next__(self):
        if(self.num >= self.max):
            raise StopIteration
        self.num += 1
        return self.num

print_num = PrintNumber(3)

print_num_iter = iter(print_num)
print(next(print_num_iter))  # 1
print(next(print_num_iter))  # 2
print(next(print_num_iter))  # 3

# raises StopIteration
print(next(print_num_iter))
```

**Note**: The object needs to have `__iter__` and `__next__` methods in order to use the `iter()` function.

**Example 3**: Using sentinel parameter

```python
with open('mydata.txt') as fp:
    for line in iter(fp.readline, ''):
        processLine(line)
```

**Note**: The sentinel can be used to specify where to stop.

