# Regular Expression

### Table of content

- Expression Cheat Sheet
- re module in Python
- Use of group
- Examples
- References

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


#### Sample Regexs ####

[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+

### References

YouTube Tutorial by Corey Schafer: https://www.youtube.com/watch?v=K8L6KVGG-7o

GitHub page for materials in the video: https://github.com/CoreyMSchafer/code_snippets/tree/master/Python-Regular-Expressions