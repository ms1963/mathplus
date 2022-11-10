
from matrix import Matrix

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

        
    def compute_reachability_matrix(self, iterations = 1):
        matrix = Matrix.identity(len(self.reverse_proj), dtype = int)
        for key in self.dict.keys():
            list = self.dict[key]
            for entry in list:
                (target_node, weight) = entry
                matrix[self.proj[key]][self.proj[target_node]] = weight
        return pow(matrix,iterations)
        
        
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

# calculate reachability matrix 
print(g.compute_reachability_matrix(2))
# show how indices are mapped to node names
print(g.reverse_proj)
print()
# print the internal structure used by Graph
print(g)


