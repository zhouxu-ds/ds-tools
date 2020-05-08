def square(a):
    """Return the square of a."""
    return a ** 2

for n in range(4):
    print('{0} squared is {1}'.format(n, square(n)))