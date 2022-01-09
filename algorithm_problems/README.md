# Algorithm Problems

Here are some important/tricky algorithm problems that I noted from Leetcode.

### Table of Content

- [Climbing Stairs (Fibonacci Number)](#climbing_stairs)
- [DI String Match](#di_string_match)
- [Move Zeroes](#move_zeroes)
- [Reverse Linked List](#reverse_linked_list)
- [Maxinum Depth of N-ary Tree](#max_depth)
- [N-ary Tree Preorder Traversal](#preorder_traversal)
- [N-ary Tree Postorder Traversal](#postorder_traversal)
- [★ BST Inorder Traversal](#bst_inorder_traversal)
- [★ BST Preorder Traversal](#bst_preorder_traversal)
- [★ BST Postorder Traversal](#bst_postorder_traversal)
- [Increasing Order BST(Inorder Traversal)](#inorder_traversal)
- [Range Sum of BST](#range_sum_of_bst)
- [Trim a BST](#trim_bst)
- [Merge Two Binary Trees](#merge_two_binary_trees)
- [Min Cost Climbing Stairs (Dynamic Programming)](#min_cost_climbing_stairs)
- [Paint House (Memoization and Dynamic Programming)](#paint_house)

### Basics

- [Binary Search](#binary_search)

- [Greatest Common Divisor](#gcd)

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

<a name='di_string_match'></a>

## DI String Match

### Problem Statement

https://leetcode.com/problems/di-string-match/

Given a string `S` that **only** contains "I" (increase) or "D" (decrease), let `N = S.length`.

Return **any** permutation `A` of `[0, 1, ..., N]` such that for all `i = 0, ..., N-1`:

- If `S[i] == "I"`, then `A[i] < A[i+1]`
- If `S[i] == "D"`, then `A[i] > A[i+1]`

**Example 1:**

```
Input: "IDID"
Output: [0,4,1,3,2]
```

**Example 2:**

```
Input: "III"
Output: [0,1,2,3]
```

**Example 3:**

```
Input: "DDI"
Output: [3,2,0,1]
```

### Hint

It seems difficult to come up with a solution. However, by observation, we can find some rules behind:

- Starting from the first element:
  - If it is "I", we can always go from min, which is 0.
  - If it is "D", we can always go from the max, which is N.
- Then the number used should be crossed out in the available options, so the min becomes 1, and max becomes N-1, etc.

### Python Implementation

With this trick, we can implement it in Python easily:

```python
def diStringMatch(S):
    lo, hi = 0, len(S)
    res = []
    for ch in S:
        if ch == 'I':
            res.append(lo)
            lo += 1
        else:
            res.append(hi)
            hi -= 1
    return res + [hi] # or [lo]. It doesn't matter because they are same right now
```

<a name='move_zeroes'></a>

## Move Zeroes

### Problem Statement

https://leetcode.com/problems/move-zeroes/

Given an array `nums`, write a function to move all `0`'s to the end of it while maintaining the relative order of the non-zero elements.

**Example:**

```
Input: [0,1,0,3,12]
Output: [1,3,12,0,0]
```

**Note**:

1. You must do this **in-place** without making a copy of the array.
2. Minimize the total number of operations.

### Python Implementation

Here is a a tricky/clever way of implementing it:

```python
def moveZeros(nums):
    i = 0
    for j in range(len(nums)):
        if nums[j] != 0:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
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

<a name='BST_inorder_traversal'></a>

## ★ BST Inorder Traversal

**Note**: See [difference of tree traversals (inorder, preorder and postorder)](https://www.geeksforgeeks.org/tree-traversals-inorder-preorder-and-postorder/)

### Problem Statement

https://leetcode.com/problems/binary-tree-inorder-traversal/

Given the `root` of a binary tree, return *the inorder traversal of its nodes' values*.

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/09/15/inorder_1.jpg)

```
Input: root = [1,null,2,3]
Output: [1,3,2]
```

**Example 2:**

```
Input: root = []
Output: []
```

**Example 3:**

```
Input: root = [1]
Output: [1]
```

**Example 4:**

![img](https://assets.leetcode.com/uploads/2020/09/15/inorder_5.jpg)

```
Input: root = [1,2]
Output: [2,1]
```

**Example 5:**

![img](https://assets.leetcode.com/uploads/2020/09/15/inorder_4.jpg)

```
Input: root = [1,null,2]
Output: [1,2]
```

### Python Implementation

The `TreeNode` class for BST is defined as below:

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

It can be implemented either recursively or iteratively:

```python
def inorderTraversal(root):
    """Recursive solution"""
    if root is None:
        return []
    res = inorderTraversal(root.left) + [root.val] + inorderTraversal(root.right)
    return res
```

```python
def inorderTraversal(root):
    """Iterative solution"""
    if root is None:
        return []
    res, stack = [], [(root, 0)]
    while stack:
        node, visited = stack.pop()
        if node:
            if visited:
                res.append(node.val)
            else:
                stack.extend([(node.right, 0), (node, 1), (node.left, 0)])
    return res
```

<a name='bst_pretorder_traversal'></a>

## ★BST Pretorder Traversal

**Note**: See [difference of tree traversals (inorder, preorder and postorder)](https://www.geeksforgeeks.org/tree-traversals-inorder-preorder-and-postorder/)

### Problem Statement

https://leetcode.com/problems/binary-tree-preorder-traversal/

Given the `root` of a binary tree, return *the preorder traversal of its nodes' values*.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/09/15/inorder_1.jpg)

```
Input: root = [1,null,2,3]
Output: [1,2,3]
```

### Python Implementation

The `TreeNode` class for BST is defined as below:

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

It can be implemented both recursively and iteratively:

```python
def preorderTraversal(root):
    """Recursive solution"""
    def recursion(node):
        if node:
            res.append(node.val)
            recursion(node.left)
            recursion(node.right)

    res = []
    recursion(root)
    return res
```

```python
def preorderTraversal(root):
    """Iterative solution"""
    res, stack = [], [root]
    while stack:
        node = stack.pop()
        if node:
            res.append(node.val)
            stack.extend([node.right, node.left])
    return res
```

<a name='bst_postorder_traversal'></a>

## ★BST Postorder Traversal

**Note**: See [difference of tree traversals (inorder, preorder and postorder)](https://www.geeksforgeeks.org/tree-traversals-inorder-preorder-and-postorder/)

### Problem Statement

https://leetcode.com/problems/binary-tree-postorder-traversal/

Given the `root` of a binary tree, return *the postorder traversal of its nodes' values*.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/08/28/pre1.jpg)

```
Input: root = [1,null,2,3]
Output: [3,2,1]
```

### Python Implementation

The `TreeNode` class for BST is defined as below:

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

It can be implemented both recursively and iteratively:

```python
def postorderTraversal(root):
    """Recursive solution"""
    def recursion(node):
        if node:
            recursion(node.left)
            recursion(node.right)
            res.append(node.val)

    res = []
    recursion(root)
    return res
```

```python
def postorderTraversal(root):
    """Iterative solution"""
    res, stack = [], [(root, 0)]
    while stack:
        node, visited = stack.pop()
        if node:
            if visited:
                res.append(node.val)
            else:
                stack.extend([(node, 1), (node.right, 0), (node.left, 0)])
    return res
```

<a name='inorder_traversal'></a>

## Increasing Order BST (Inorder Traversal)

### Problem Statement 

https://leetcode.com/problems/increasing-order-search-tree/

Given a binary search tree, rearrange the tree in **in-order** so that the leftmost node in the tree is now the root of the tree, and every node has no left child and only 1 right child.

```
Example 1:
Input: [5,3,6,2,4,null,8,1,null,null,null,7,9]

       5
      / \
    3    6
   / \    \
  2   4    8
 /        / \ 
1        7   9

Output: [1,null,2,null,3,null,4,null,5,null,6,null,7,null,8,null,9]

 1
  \
   2
    \
     3
      \
       4
        \
         5
          \
           6
            \
             7
              \
               8
                \
                 9  
```

### Python Implementation

The `TreeNode` class for BST is defined as below:

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

This can be seen as an in-order traversal, which can be solved using either recursion or iteration:

```python
def increasingBST(root):
    """Recursive solution"""
    def inorder(node):
        if node:
            yield from inorder(node.left)
            yield node.val
            yield from inorder(node.right)       
    res = cur = TreeNode(None)
    for num in inorder(root):
        cur.right = TreeNode(num)
        cur = cur.right
    return res.right
```

```python
def increasingBST(root):
    """Iterative solution"""
    def inorder(node):
        stack = [(root, 0)]
        while stack:
            head, lv = stack.pop()
            if lv = 1: # the left node is visited
                yield head.val
                if head.right is not None:
                    stack.append((head.right, 0))
            else: # the left node has not been visited
                stack.append((head, 1))
                if head.left is not None:
                    stack.append((head.left, 0))
    res = cur = TreeNode(None)
    for num in inorder(root):
        cur.right = TreeNode(num)
        cur = cur.right
    return res.right
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

<a name='trim_bst'></a>

## Trim a Binary Search Tree

### Problem Statement

https://leetcode.com/problems/trim-a-binary-search-tree/

Given a binary search tree and the lowest and highest boundaries as `L` and `R`, trim the tree so that all its elements lies in `[L, R]` (R >= L). You might need to change the root of the tree, so the result should return the new root of the trimmed binary search tree.

**Example 1:**

```
Input: 
    1
   / \
  0   2

  L = 1
  R = 2

Output: 
    1
      \
       2
```



**Example 2:**

```
Input: 
    3
   / \
  0   4
   \
    2
   /
  1

  L = 1
  R = 3

Output: 
      3
     / 
   2   
  /
 1
```

### Python Implementation

The `TreeNode` of BST is defined as below:

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

It can be solved both recursively or iteratively:

```python
def trimBST(root, L, R):
    """Recursive solution"""
    if root is None:
        return None
	if root.val > R: # Find the right boundary node
        return trimBST(root.left, L, R)
    elif root.val < L: # Find the left boundary node
        return trimBST(root.left, L, R)
    # For nodes that in the range, trim both branches
    root.left = trimBST(root.left. L, R)
    root.right = trimBST(root.right, L, R)
    return root
```

```python
def trimBST(root, L, R):
    """Iterative solution"""
    if root is None:
        return None
    # Adjust the root if it is not in range
	while root and (root.val < L or root.val > R):
        if root.val < L:
            root = root.right
        else:
            root = root.left
    # Go through left ang right branches and trim
    lnode = rnode = root
    while lnode.left:
        if lnode.left.val < L:
            lnode.left = lnode.left.right
        else:
            lnode = lnode.left
    while rnode.right:
        if rnode.right.val > R:
            rnode.right = rnode.right.left
        else:
            rnode = rnode.right
    return root
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

<a name='min_cost_climbing_stairs'></a>

## Min Cost Climbing Stairs (Dynamic Programming)

### Problem Statement

https://leetcode.com/problems/min-cost-climbing-stairs/

On a staircase, the `i`-th step has some non-negative cost `cost[i]` assigned (0 indexed).

Once you pay the cost, you can either climb one or two steps. You need to find minimum cost to reach the top of the floor, and you can either start from the step with index 0, or the step with index 1.

**Example 1:**

```
Input: cost = [10, 15, 20]
Output: 15
Explanation: Cheapest is start on cost[1], pay that cost and go to the top.
```

**Example 2:**

```
Input: cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
Output: 6
Explanation: Cheapest is start on cost[0], and only step on 1s, skipping cost[3].
```

### Dynamic Programming Thinking

- For any step(index) in stairs(cost), we can jump into that step(index) from step - 2 or step -1.
- Therefore, we accumulate costs
- return cost[-1] or cost[-2] because both steps(indexes) can complete climbing.

### Python Implementation

```python
def minCostClimbingStairs(cost):
    for i in range(2, len(cost)): 
        cost[i] += min(cost[i - 1], cost[i - 2])
        return min(cost[-1], cost[-2])
```

<a name='paint_house'></a>

## Paint House (Memoization and Dynamic Programming)

### Problem Statement

https://leetcode.com/problems/paint-house/

There are a row of *n* houses, each house can be painted with one of the three colors: red, blue or green. The cost of painting each house with a certain color is different. You have to paint all the houses such that no two adjacent houses have the same color.

The cost of painting each house with a certain color is represented by a `*n* x *3*` cost matrix. For example, `costs[0][0]` is the cost of painting house 0 with color red; `costs[1][2]` is the cost of painting house 1 with color green, and so on... Find the minimum cost to paint all houses.

**Note:**
All costs are positive integers.

**Example:**

```
Input: [[17,2,17],[16,16,5],[14,3,19]]
Output: 10
Explanation: Paint house 0 into blue, paint house 1 into green, paint house 2 into blue. 
             Minimum cost: 2 + 5 + 3 = 10.
```

### Python Implementation

There are two ways to implement: memoization and dynamic programming.

Both **Memoization** and **Dynamic Programming** solves individual subproblem only once. **Memoization** uses recursion and works top-down, whereas **Dynamic programming** (iterative) moves in opposite direction solving the problem bottom-up.

The detailed explanation of this problem can be found in solution of this problem: https://leetcode.com/problems/paint-house/

**Memoization**: It uses recursion (to mimic bottom up tree) with a dictionary to memoize the calculated results.

```python
def minCost(self, costs):
    def paint_cost(n, color):
        if (n, color) in self.memo:
            return self.memo[(n, color)]
        total_cost = costs[n][color]
        if n == len(costs) - 1:
            pass
        elif color == 0:
            total_cost += min(paint_cost(n + 1, 1), paint_cost(n + 1, 2))
        elif color == 1:
            total_cost += min(paint_cost(n + 1, 0), paint_cost(n + 1, 2))
        else:
            total_cost += min(paint_cost(n + 1, 0), paint_cost(n + 1, 1))
        self.memo[(n, color)] = total_cost
        return total_cost

    if costs == []:
        return 0

    self.memo = {}
    return min(paint_cost(0, 0), paint_cost(0, 1), paint_cost(0, 2))
```

**Dynamic Programming**: It is a bottom up approach that calculates the minimum value of each value below until we reach the top level.

```python
def minCost(self, costs: List[List[int]]) -> int:    
    for n in reversed(range(len(costs) - 1)):
        # Total cost of painting nth house red.
        costs[n][0] += min(costs[n + 1][1], costs[n + 1][2])
        # Total cost of painting nth house green.
        costs[n][1] += min(costs[n + 1][0], costs[n + 1][2])
        # Total cost of painting nth house blue.
        costs[n][2] += min(costs[n + 1][0], costs[n + 1][1])

    if len(costs) == 0: return 0
    return min(costs[0]) # Return the minimum in the first row.
```

## Basics

<a name='binary_search'></a>

### Binary Search

There can be two ways of implementation: recursive and iterative.

```python
def binary_search(arr, lo, hi, x):
    """Recursive implementation"""
    if hi >= lo:
        mid = (hi + lo) // 2
        if x == arr[mid]:
            return mid
        elif x < arr[mid]:
            return binary_search(arr, lo, mid - 1, x)
        else:
            return binary_search(arr, mid + 1, hi, x)
    else:
        return -1  
```

```python
def binary_search(arr, x):
    """Iterative implementation"""
    lo, hi = 0, len(x) - 1
    while lo <= hi:
    	mid = (lo + hi) // 2
        if x == arr[mid]:
            return mid
        elif x < arr[mid]:
            hi = mid - 1
        else:
            lo = mid + 1
    return -1
```

<a name='gcd'></a>

### Greatest Common Divisor

Here is the implementation using Euclidean algorithm:

```python
def cal_gcd(a, b):
    """Note: a should be smaller or equal to b"""
    while a:
        a, b = a % b, a
    return a
```



