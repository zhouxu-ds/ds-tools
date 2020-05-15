# Numpy

This part of study follows the book [Python Data Science Handbook: Essential Tools for Working with Data](http://shop.oreilly.com/product/0636920034919.do).

The details of this page can be found in the [jupyter notebook](numpy.ipynb), including examples and code excutions.

Here are some tricks that I feel useful:

- [Array Creation](#array_creation)
- [Array Attributes](#array_attributes)
- [Broadcasting](#broadcasting)

<a name='array_creatioin'></a>

## Array Creation

#### Basic

`np.array([1, 2, 3, 4])` - Create array from list

`np.zeros((3, 5))` - Create array with 0's

`np.ones((3, 5))` - Create array with 1's

`np.full((3, 5), 3.14)` - Create array with specified value

**Note**: For `np.zeros`, `np.ones` and `np.full`, instead of specifying the dimensions on our own, we can create an array with the same shape as another, such as `np.zeros_like(another_arr)`, `np.ones_like(another_arr)` and `np.full_like(another_arr)`.

#### Linear Sequences

`np.arange(0, 20, 2)` - Create linear sequence using start, end and step size.

`np.linspace(0, 1, 5` - Create linear sequence using start, end and number of elements.

#### Random Values

`np.random.random((3, 3))` - Random number between 0 and 1

`np.random.normal(0, 1, (3, 3))` - Random number generated using normal distribution with mean of 0 and standard deviation of 1, size specified.

#### Others

`np.eye(3)` - A 3 x 3 identity matrix

`np.empty(3)` - Create an empty array with values that already exist at that memory location.

<a name='array_attributes'></a>

## Array Attributes

`ndim` - Number of dimensions

`shape` - Size of each dimension

`size` - Total size of the array

`dtype` - Data type

`itemsize` - Bytes for each item in the array

`nbytes` - Total size in bytes of the array

<a name='concatenate_and_split'></a>

## Concatenate and Split

`np.concatenate(x, y, axis=0)` - Concatenate array x and array y on axis 0

`np.vstack([x, y])` - Stack the arrays vertically, equivalent to concatination along axis 0

`np.hstack([x, y])` - Stack the arrays horizontally, equivalent to concatenation along axis 1

`x1, x2, x3 = np.split(x, [3, 5])` - Split the array by indices, in this example, at indices of 3 and 5

`upper, lower = np.vsplit(grid, 2)` - Split the array vertically

`left, right = np.hsplit(grid, 2)` - Split the array horizontally

<a name='broadcasting'></a>

## Broadcasting

The broadcasting in Numpy follows a strict set of rules to determine the interaction between two arrays: 
- **Rule 1**: If the two arrays differ in their number of dimensions, the shape of one with fewer dimension is padded with one on its leading (left) side.
- **Rule 2**: If the shape of the two arrays does not match in any dimension, the array with shape equal to 1 in that dimension is stretched to match the other shape.
- **Rule 3**: If in any dimension the size disagree and neither is equal to 1, an error is raised.

**Example 1**:

 ```R
# Original shape of M and a
M.shape = (2, 3)
a.shape = (3,)

# According to rule 1, the shape of a is padded on the left side
M.shape -> (2, 3)
a.shape -> (1, 3)

# According to rule 2, a is stretched to match the other shape
M.shape -> (2, 3)
a.shape -> (2, 3)
 ```

**Example 2**:

```R
# Original shape of a and b
a.shape = (3, 1)
b.shape = (3,)

# According to rule 1, the shape of b is padded on the left side
M.shape -> (3, 1)
b.shape -> (1, 3)

# According to rule 2, both a and b are stretched to match the other shape
M.shape -> (3, 3)
a.shape -> (3, 3)
```

**Example 3**:

```R
# Original shape of M and a
M.shape = (3, 2)
a.shape = (3,)

# According to rule 1, the shape of a is padded on the left side
M.shape -> (3, 2)
a.shape -> (1, 3)

# According to rule 2, a is stretched to match the other shape
M.shape -> (3, 2)
a.shape -> (3, 3)

# However, no more change will be maded after.
# According to rule 3, the error will be raised.
```

For mor details of examples, see the [jupyter notebook](numpy.ipynb).

