# Algorithm Problems

Here are some important/tricky algorithm problems that I noted from Leetcode.

### Table of Content

- [Climbing Stairs (Fibonacci Number)](#climbing_stairs)
- [Maxinum Depth of N-ary Tree (DFS)](#max_depth)
- [Range Sum of BST](#range_sum_of_bst)

<a name='climbing_stairs'></a>

## Climbing Stairs (Fibonacci Number)

### Problem Statement

https://leetcode.com/problems/climbing-stairs/

You are climbing a stair case. It takes *n* steps to reach to the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

**Note:** Given *n* will be a positive integer.

**Example 1:**

```
Input: 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps
```

**Example 2:**

```
Input: 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step
```

### Python Implementation

This is essentially a [Fibonacci number](https://en.wikipedia.org/wiki/Fibonacci_number) problem - The next number equals to the sum of previous two numbers, such that the sequence is like: 0, 1, 1, 2, 5, 8, 13, 21, 34, 55, 89, 144...

When n = 1, the answer should be $F_2=1$​;

When n = 2, the answer should be $F_3=2$;

When n = 3, the answer should be $F_4=3$;

...

Therefore, we can implement in such way:

```python
def climb_stairs(n):
    a, b = 0, 1
    for i in range(n):
        a, b = b, a+b
    return b
```

<a name='max_depth'></a>

## Maxinum Depth of N-ary Tree (DFS)

### Problem Statement

https://leetcode.com/problems/maximum-depth-of-n-ary-tree/

Given a n-ary tree, find its maximum depth.

The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

![max_depth_tree](max_depth_tree.png)

For example, in the tree above, the depth is 3.

### Python Implementation

The Node for the N-ary tree can be implemented as:

```python
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
```

Then there can be two ways to find the maximum depth using DFS:

```python
def maxDepth(root):
    """Find the max depth of the tree using recursion"""
    if root is None: 
        return 0 
    elif root.children == []:
        return 1
    else: 
        height = [self.maxDepth(c) for c in root.children]
        return max(height) + 1 
```

```python
def maxDepth(root):
	"""
	Find the max depth of the n-ary tree using iteration.
	It store all nodes at all levels in the stack and then use variable 'detph' to record the max depth ever seen.
	"""
    stack = []
    if root is not None:
        stack.append((1, root))

    depth = 0
    while stack != []:
        current_depth, root = stack.pop()
        if root is not None:
            depth = max(depth, current_depth)
            for c in root.children:
                stack.append((current_depth + 1, c))

    return depth
```

I personally prefer the recuirison method because it is more concise and clear. It can also be extended to use with the depth of binary trees:

```python
def max_depth(root):
    """Find the max depth of binary tree using recursion"""
    if root is None:
        return 0
    if root.left is None and root.right is None:
        return 1
    else:
        return max(max_depth(root.left), max_depth(root.right)) + 1
```

<a name='range_sum_of_bst'></a>

## Range Sum of BST

### Problem Statement

https://leetcode.com/problems/range-sum-of-bst/

Given the `root` node of a binary search tree, return the sum of values of all nodes with value between `L` and `R` (inclusive).

The binary search tree is guaranteed to have unique values.

### Python Implementation

The TreeNode for BST is:

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
```

There are two ways of doing it using DFS:

```python
def rangeSumBST(root, L, R):
    """Standard way of using recursion"""
    if root is None:
        return 0
    elif root.val < L:
        return self.rangeSumBST(root.right, L, R)
    elif root.val > R:
        return self.rangeSumBST(root.left, L, R)
    return root.val + self.rangeSumBST(root.left, L, R) + self.rangeSumBST(root.right, L, R)
```

```python
def rangeSumBST(root, L, R):
    """Using iteration"""
    stack = [root]
    sum = 0
    while stack:
        node = stack.pop()
        if node:
            if L <= node.val <= R:
                sum += node.val
            if node.val > L:
                stack.append(node.left)
            if node.val < R:
                stack.append(node.right)
    return sum
```

