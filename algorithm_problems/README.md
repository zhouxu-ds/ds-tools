# Algorithm Problems

Here are some important/tricky algorithm problems that I noted from Leetcode.

### Table of Content

- [Climbing Stairs (Fibonacci Number)](#climbing_stairs)

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

