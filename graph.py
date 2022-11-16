

"""
graph.py is licensed under the
GNU General Public License v3.0

Permissions of this strong copyleft license are conditioned on making available complete source code of licensed works and modifications, which include larger works using a licensed work, under the same license. Copyright and license notices must be preserved. Contributors provide an express grant of patent rights.
"""

from mathplus import Matrix

"""
Sample Implementation of a Graph with
nodes and weighted edges illustrating
the use of Matrix
"""
class Graph:
    def __init__(self):
        self.count = 0   # used to get the next free integer
        self.dict = {}   # stores all edges that exit the node
                         # together with a weight
        self.proj = {}   # stores the association between the node name
                         # and the integer assigned to it
        self.reverse_proj = {} # stores the association between an integer
                               # and the node name
    
    # an extry is a tuple (target_node, content) associated with edge)
    # node is the source node
    def new_edge(self, node, entry): 
        key = node
        if key in self.dict.keys():
            list = self.dict[key]
            list.append(entry)
        else: 
            self.dict[key] = [entry]
            self.reverse_proj[self.count]= key
            self.proj[key] = self.count
            self.count += 1
        key2 = entry[0]
        if not key2 in self.dict.keys():
            self.dict[key2] = []
            self.reverse_proj[self.count] = key2
            self.proj[key2] = self.count
            self.count += 1
            
    def __getitem__(self, key):
        return self.dict[key]
        
    def __str__(self):
        res = "Entries of dictionary: \n"
        for key in self.dict.keys():
            for entry in self.dict[key]:
                (target, weight) = entry
                res += "from " + str(key) + " to "  + str(target) + " with associated weight = " + str(weight) + "\n"
        return res

        
    def compute_reachability(self, iterations = 1):
        matrix = Matrix.identity(len(self.reverse_proj), dtype = int)
        for key in self.dict.keys():
            list = self.dict[key]
            for entry in list:
                (target_node, weight) = entry
                matrix[self.proj[key]][self.proj[target_node]] = weight
        return pow(matrix,iterations)
        
    def travel(self, start):
        matrix = self.compute_reachability(1)
        current_location = self.proj[start]
        itinery = []
        if not start in self.proj.keys():
            print("Sorry, this city is not available")
        else: 
            print("Travel begins in " +str(start))
        while True:
            if len(itinery) == matrix.dim1:
                break
            itinery.append(current_location)
            found = False
            for c in range(0, matrix.dim2):
                if (matrix[current_location][c] != 0) and (c not in itinery):
                    print("I am enjoying my trip from " + str(self.reverse_proj[current_location]) + " to " + str(self.reverse_proj[c]))
                    current_location = c
                    found = True
            if not found:
                break            
        print("Travel done")
        
        
# create graph       
g = Graph()
# add edges to g
g.new_edge('munich',    ('hamburg',   1))
g.new_edge('munich',    ('frankfurt', 1))
g.new_edge('frankfurt', ('cologne',   1))
g.new_edge('cologne',   ('berlin',    1))
g.new_edge('berlin',    ('munich',    1))
g.new_edge('berlin',    ('frankfurt', 1))
g.new_edge('hamburg',   ('munich',    1))
g.new_edge('hamburg',   ('berlin',    1))
g.new_edge('frankfurt', ('berlin',    1))
g.travel("munich")

# calculate reachability matrix 
print(g.compute_reachability(1))
# show how indices are mapped to node names
print(g.reverse_proj)
print()
# print the internal structure used by Graph
print(g)
g.travel('munich')



