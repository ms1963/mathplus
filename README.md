# mathplus
Just my own implementation of matrices, vectors, polynomials and rational numbers and more,  done just for fun.

The demo code in demo_matrix.py shows how to use the library. Given the fact that libraries like scipy, numpy exist, it is obviously unnecessary to come up with yet another math library. Nonetheless, I did it for the fun of experimentation and to have a library that does not depend on code written in other languages, in particular C and C++.

graph.py implements a naive graph structure that uses a matrix for visualizing the edges between nodes.

Note: Rational cannot be used as a dtype for Matrices as this does not make a sense. You may use the Rational.float() conversion to bridge the gap.


Pros and Cons

Pros:

+ minimalistic library that constrains itself to essential functionality
+ call by value: only the setters __setitem__ manipulate an object directly. All other functionality tries to stick to a functional approach. Occasionally methods have an in_situ argument so that users can choose to change the source object directly
+ small code size
+ easy to use (yes, this is very subjective)
+ pure Python

Cons:

+ requires more memory than the runtime- and memory-efficient numpy arrays
+ uses efficient algorithms whenever possible, but is not optimized for runtime efficiency
+ does not provide any means for multithreading

Usage: see demo code
