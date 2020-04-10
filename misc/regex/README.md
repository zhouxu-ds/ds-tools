# Regular Expression

### Table of content

- [Expression Cheat Sheet](#cheat_sheet)
- [`re` module in Python](#re_module)
- [Examples](#examples)
- [References](#references)

<a name='cheat_sheet'></a>


### Expression Cheat Sheet

`.`        Any Character Except New Line
`\d`      Digit (0-9)
`\D`      Not a Digit (0-9)
`\w`      Word Character (a-z, A-Z, 0-9, _)
`\W`      Not a Word Character
`\s`      Whitespace (space, tab, newline)
`\S`      Not Whitespace (space, tab, newline)

`\b`      Word Boundary
`\B`      Not a Word Boundary
`^`        Beginning of a String
`$`        End of a String

`[]`      Matches Characters in brackets
`[^ ]`  Matches Characters NOT in brackets
`|`        Either Or
`()`      Group

**Quantifiers:**

`*`         0 or More

`+` 	    1 or More
`?`         0 or One
`{3}`     Exact Number
`{3,4}` Range of Numbers (Minimum, Maximum)

<a name='re_module'></a>

### `re` Module in Python

Here is a typical use of regular expression in Python:

```python
import re

# Compile the pattern and find all matches in the text
pattern = re.compile(r'\w')
matches = patter.finditer(text_to_search)

# Print out the object including the span index and the actual match
for match in matches:
    print(match)
```

There are a couple of methods in `re` module that we can use beside `re.finditer`. Their usage can be found below:

`finditer`: Returen an iterator yielding match objects over all non-overlapping matches, including span index and matches.

`search`: Single version of `finditer`. Return the match obeject of the first match.

`findall`: Returns all matches in a list. However, if groups are used in the patterns, then it will be list of groups.

`match`: Only matches the beginning of the string. Not quite useful in many cases.

<a name='examples'></a>

### Examples

See [jupyter notebook](examples.ipynb).

<a name='references'></a>

### References

YouTube Tutorial by Corey Schafer: https://www.youtube.com/watch?v=K8L6KVGG-7o

GitHub page for materials in the video: https://github.com/CoreyMSchafer/code_snippets/tree/master/Python-Regular-Expressions

Python3 re library: https://docs.python.org/3/library/re.html