# mathplus
This is my own implementation of matrices, vectors, polynomials and rational numbers and more,  done just for fun. This is my experimentation platform to deal with mathematical ideas and concepts.

The demo code in demo_matrix.py shows how to use the library. Given the fact that libraries like scipy, numpy exist, it may appear unnecessary to come up with yet another math library. Nonetheless, I did it for the fun of experimentation and to have a library that does not depend on code written in other languages, in particular C and C++. mathplus is useful for problems where significant performance boosts are less an issue than programming safety. If however, performance is the most significant requirement, it is strongly advised to use numpy (and scipy) instead.

graph.py implements a naive graph structure that uses a matrix for visualizing the edges between nodes.

Note: Rational cannot be used as a dtype for Matrices as this does not make sense. You may use the Rational.float() conversion to bridge the gap.


Pros and Cons

Pros:

+ minimalistic library that constrains itself to essential functionality
+ call by value: only the setters __setitem__ manipulate an object directly. All other functionality tries to stick to a functional approach. Occasionally methods have an in_situ argument so that users can choose to change the source object directly
+ small code size
+ easy to use (yes, this is very subjective)
+ brings its own style, but tries to mimic numpy where possible so that many things should appear familar
+ pure Python

Cons:

+ requires more memory than the runtime- and memory-efficient numpy arrays
+ uses efficient algorithms whenever possible, but is not optimized for runtime efficiency. With other words, numpy is significantly faster.
+ does not provide any means for multithreading

Usage: see demo code
