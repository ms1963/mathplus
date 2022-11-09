# matrix_vector_for_python
Just a trivial implementation of matrices and vectors done just for fun.

The demo code in demo_matrix.py shows how to use the trivial library. Given the fact that libraries like pandas, numpy exist, it is obviously unnecessary to come up with a matrix and vector implementation. Nonetheless, I did it for the fun of experimentation.


Pros and Cons

Pros:

+ minimalistic library that constrains itself to essential functionality
+ call by value: only the setters __setitem__ manipulate an object directly. All other functionality tries to stick with a functional approach
+ small code size
+ easy to use

Cons:

+ requires more memory that the memory-efficient numpy arrays
+ uses efficient algorithms whenever possible, but is not optimized for runtime efficiency
+ does not provide any means for multithreading

Usage: see demo code
