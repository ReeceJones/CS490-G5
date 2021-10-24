import numpy as np
from numpy.core import numeric
from lib import parser


class Point(object):
    def __init__(self, data=[], neighbors=set()) -> None:
        """
        Initialize a new point object
        """
        self.data = data
        self.neighbors = neighbors

    def set_data(self, data):
        """
        Set data stored at a point
        """
        self.data = data

    def set_neighbor(self, neighbors):
        """
        Set neighbors of a point
        """
        self.neighbors = neighbors

    def add_neighbor(self, neighbor):
        """
        Add a neighbor to a point (creates an edge between points)
        """
        self.neighbors.add(neighbor)

    def get_neighbors(self):
        return self.neighbors

    def get_data(self):
        return self.data


class Graph(object):
    def __init__(self, graph_dictionary={}):
        """
        Initialize a graph object
        """
        self.graph_dict = graph_dictionary

    def get_edges(self, vertice):
        """
        Returns a list of all the edges (adjacent vertices) of a vertice
        """
        return self.graph_dict[vertice]

    def get_vertices(self):
        """
        Returns a list of all of the vertices in the graph
        """
        return list(self.graph_dict.keys())

    def add_vertice(self, vertice):
        """
        Adds a vertice to the graph
        """
        if vertice not in self.graph_dict:
            self.graph_dict[vertice] = vertice.get_neighbors()

    def add_edge(self, vertice1, vertice2):
        """
        Adds an edge between two vertices
        """
        self.graph_dict[vertice1].add(vertice2)
        self.graph_dict[vertice2].add(vertice1)


def dist(p1, p2):
    """
    Calculate euclidean distance between two points

    Inputs
    p1: a point in P
    p2: a point in P
    """
    return np.linalg.norm(p1.data - p2.data)


def robustPrune(p, V, alpha, R):
    """
    Inputs
    p: a point p from P
    V: candidate set
    alpha: distance threshold that is >= 1
    R: a degree bound

    Output
    G is modified by setting at most R new out-neighbors for p
    """
    V = (V.union(p)) - p
    p.neighbors = None
    p_star = None

    while len(V) != 0:
        distance = float('inf')
        for p_prime in V:
            if dist(p, p_prime) < distance:
                p_star = p_prime
                distance = dist(p, p_prime)
        p.neighbors = p.neighbors.union(p_star)
        if len(p.neighbors) == R:
            break
        for p_prime in V:
            if alpha * dist(p_star, p_prime) <= dist(p, p_prime):
                V = V - p_prime
    return


# if __name__ == "__main__":
#     f = open(r"C:\Users\aidan\OneDrive\Desktop\School\Fall 2021\CS 490\Final Project\siftsmall\siftsmall_base.fvecs", "rb")
#     # p1 = Point([1, 0, 0], set())
#     # p2 = Point([0, 1, 0], {p1})

#     # print(p1.data)
#     # p1.set_data([1, 2, 3, 4])
#     # print(p1.data)
#     # print(type(p2.neighbors))
#     # p1.add_neighbor(p2)
#     # print(p1.neighbors)

#     g = Graph()
#     for val in parser.read_vectors(f):
#         p = Point(list(val))
#         g.add_vertice(p)
#         print(g.get_vertices()[0].get_data())
#         print(g.get_vertices()[0].get_neighbors())

#         break
