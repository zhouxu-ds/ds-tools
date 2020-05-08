# IPython (Jupyter Notebook)

The content is from [Python Data Science Handbook: Essential Tools for Working with Data](http://shop.oreilly.com/product/0636920034919.do).

The details of this page can be found in the [jupyter notebook](ipython.pynb), including examples and code excutions.

Here are some tricks that I feel useful:

- [Misc Tricks](#misc_tricks)
- [Magic Commands](#magic_commands)
- [Shell Commands in IPython](#shell_commands)
- [Errors and Debugging](#errors_and_debugging)

<a name='misc_tricks'></a>

### Misc Tricks

`?` -  Can be used to access documentations. eg. `len?`, `str.insert?`, 

`??` - Can be used to access source code. **Note**: It won't display the source code sometimes because many functions are not written in python. They are written in C, or else. In this case, `??` only functions as `?`.

`*` `?` - They can together be used for wildcard matching beyond tab completions. eg. `*Warning?`, `str.*find*?`

<a name='magic_commands'></a>

### Magic Commands

`%lsmagic` - Shows available line magics.

`%run` - Run external code.

`%timeit` - Loop through the command and gives mean and standard deviation of the time spent.

`%time` - Time without looping.

<a name='shell_commands'></a>

### ! - Shell Commands in IPython

eg. `!ls`, `!pwd`, etc

<a name='errors_and_debugging'></a>

### Errors and Debugging

`%xmode Plain`, `%xmode Verbose`, etc. can be used to change how much info to show for the exception handlers.

`%pdb on` can be used to automatically start pdb when there is any error raised.







