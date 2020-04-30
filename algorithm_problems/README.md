# Algorithm Problems

Here are some important/tricky algorithm problems that I noted from Leetcode.

### Table of Content

- [Climbing Stairs (Fibonacci Number)](#climbing_stairs)
- [Reverse Linked List](#reverse_linked_list)
- [Maxinum Depth of N-ary Tree](#max_depth)
- [N-ary Tree Preorder Traversal](#preorder_traversal)
- [N-ary Tree Postorder Traversal](#postorder_traversal)
- [Range Sum of BST](#range_sum_of_bst)
- [Merge Two Binary Trees](#merge_two_binary_trees)

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

<a name='reverse_linked_list'></a>

## Reverse Linked List

### Problem Statement

https://leetcode.com/problems/reverse-linked-list/

Reverse a singly linked list.

**Example:**

```python
Input: 1->2->3->4->5->NULL
Output: 5->4->3->2->1->NULL
```

### Python Implementation

It can be solved both recursively and iteratively:

```python
def reverseList(head):
   	"""Recursive solution"""
    def recur(cur, prev=None):
        if cur is None:
            return prev
       	nex = cur.next
        cur.next = prev
        return recur(nex, cur)
    return recur(head)
```

```python
def reverseList(head):
    """Iterative solution, intuitive version using temporary values"""
    cur, prev = head, None
    while cur:
        nex = cur.next
        cur.next = prev
        prev = cur
        cur = nex
    return prev
```

This can be more concise by using multiple assignment:

```python
def reverseList(head):
    """Iterative solution, concise version by using multiple assignment"""
    cur, prev = head, None
    while cur:
		cur.next, prev, cur = prev, cur, cur.next
        # Note the order matters here, it wont work if it is like:
        # cur, prev, cur.next = cur.next, cur, prev
    return prev
```

<a name='max_depth'></a>

## Maxinum Depth of N-ary Tree

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

<a name='preorder_traversal'></a>

## N-ary Tree Preorder Traversal

### Problem Statement

https://leetcode.com/problems/n-ary-tree-preorder-traversal/

Given an n-ary tree, return the *preorder* traversal of its nodes' values.

**Note**: See [difference of tree traversals (inorder, preorder and postorder)](https://www.geeksforgeeks.org/tree-traversals-inorder-preorder-and-postorder/)

**Example**: 

![preorder_traversal_tree](preorder_traversal_tree.png)

In such tree, we need to return the list `[1, 3, 5, 6, 2, 4]`.

### Python Implementation

The tree node is the same as above for N-ary trees:

```python
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
```

There are two ways to implement using DFS, with recursion or iteration:

```python
def preorder(root):
    """Return the preorder traversal using recursion"""
    if root is None:
        return []
    res = [root.val]
    for child in root.children:
        res.extend(preorder(child))
    return res
```

```python
def preorder(root):
    """Return the preorder traversal using iteration"""
    if root is None:
        return []
    res = []
    stack = [root]
    while stack:
        node = stack.pop()
        res.append(node.val)
        stack.extend(node.children[::-1])
    return res
```

<a name='postorder_traversal'></a>

## N-ary Tree Postorder Traversal

### Problem Statement

https://leetcode.com/problems/n-ary-tree-postorder-traversal/

Given an n-ary tree, return the *postorder* traversal of its nodes' values.

**Note**: See [difference of tree traversals (inorder, preorder and postorder)](https://www.geeksforgeeks.org/tree-traversals-inorder-preorder-and-postorder/)

**Example**: 

![preorder_traversal_tree](preorder_traversal_tree.png)

In such tree, we need to return the list `[5, 6, 3, 2, 4, 1]`.

### Python Implementation

The tree node is the same for N-ary trees:

```python
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
```

We can implement algorithms of either recursion or iteration:

```python
def postorder(root):
    """
    Return postorder traversal of the tree.
    Compared to the recursive method in preorder traversal, this one append the value after doing recursion.
    """
    if root is None:
        return []
    res = []
    def recursion(root, res):
        for child in root.children:
            recursion(root, res)
		res.append(root.val)
    recursion(root, res)
    return res
```

```python
def postorder(root):
    """
    Return post order traversal of the tree.
    It uses a reverse hack: It does preorder traversal, but from right to left.
    At the very end, reverse the order so that it becomes a postorder traversal.
    """
    if root is None:
        return []
    res, stack = [], [root]
    while stack:
        node = stack.pop()
        res.append(node.val)
        stack.extend(node.children)
    return res[::-1]
```

```python
def postorder(root):
    """
    Return post order traversal of the tree.
    Instead of using the reverse hack, this one is the traditional way of implementing
    postorder traversal, where we keep track of if we have visited the node before 
    actually appending the value.
    """
    if root is None:
        return []
    res, stack = [], [(root, 0)]
    while stack:
        node, visited = stack.pop()
        if node is None:
            continue
        if visited: # Won't append the value unless it's visited before
            res.append(root.val)
        else: # Mark it as visited, and then append the children into the stack
            stack.append((node, 1))
            stack.extend([(child, 0) for child in node.children[::-1]])
    return res
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

<a name='merge_two_binary_trees'></a>

## Merge Two Binary Trees

### Problem Statement

https://leetcode.com/problems/merge-two-binary-trees/

Given two binary trees and imagine that when you put one of them to cover the other, some nodes of the two trees are overlapped while the others are not.

You need to merge them into a new binary tree. The merge rule is that if two nodes overlap, then sum node values up as the new value of the merged node. Otherwise, the NOT null node will be used as the node of new tree.

**Example 1:**

```
Input: 
	Tree 1                     Tree 2                  
          1                         2                             
         / \                       / \                            
        3   2                     1   3                        
       /                           \   \                      
      5                             4   7                  
Output: 
Merged tree:
	     3
	    / \
	   4   5
	  / \   \ 
	 5   4   7
```

### Python Implementation

The TreeNode for the binary tree is:

```python
class TreeNode:
    def __init__(self, x=0):
        self.val = x
        self.left = None
        self.right = None
```

The recursive solution for it can be:

```python
def mergeTrees(t1, t2):
    if t1 and t2:
        root = TreeNode(t1.val + t2.val)
        root.left = self.mergeTrees(t1.left, t2.left)
        root.right = self.mergeTrees(t1.right, t2.right)
        return root
    else:
        return t1 or t2
```

