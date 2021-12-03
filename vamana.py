import numpy
from lib import parser, fsgraph
from sklearn.metrics import pairwise_distances
import parse_test
import pandas as pd
import RobustPrune
import numpy as np
import random 

#Vamana constructs G in an iterative manner

#Graph G initialized where each vertex has R randomly chosen out-neighbors.
#   If R > log n, the graph will be well connected, but does not ensure best performance for GreedySearch

#Variable s denotes the medoid of the dataset P, which is the starting node for the algorithm
#   Medoid - Point with the smallest average dissimilarity (distance?) with all other nodes in the graph.

def GreedySearch(s, x, k, L, G):
    A = set()
    A.add(s)
    B = set()

    while len(A - B) > 0:
        diff = list(A - B)
        norms = [numpy.linalg.norm(np.array(G.get_data(y)) - x) for y in diff]
        p_prime = diff[numpy.argmin(norms)]
        A = A.union(G.get_neighbors(p_prime))
        B.add(p_prime)
        if len(A) > L:
            A = set(sorted(list(A), key=lambda y: numpy.linalg.norm(x - numpy.array(G.get_data(y))))[:L])

    return set(sorted(list(A), key=lambda y: numpy.linalg.norm(numpy.array(G.get_data(y)) - x))[:k]),B

def LightRobustPrune(p, V, alpha, R, G):
    V = V.union(G[p]) - {p}
    G[p] = set()
    while len(V) > 0:
        _V = list(V)
        norms = [numpy.linalg.norm(numpy.array(G.get_data(y)) - numpy.array(G.get_data(p))) for y in _V]
        p_prime = _V[numpy.argmin(norms)]
        G[p] = G[p].union({p_prime})
        if len(G[p]) == R:
            break
        for i in range(len(_V)):
            if alpha*numpy.linalg.norm(numpy.array(G.get_data(p_prime)) - G.get_data(_V[i])) <= norms[i]:
                V.remove(_V[i])
    return G

def medoid(file, fast=False):
    if fast:
        distMatrix = pairwise_distances(parse_test.parse_to_df('data/siftsmall/siftsmall_base.fvecs'))
        return numpy.argmin(distMatrix.sum(axis=0))
    else:
        with open(file, 'rb') as f1:
            with open(file, 'rb') as f2:
                mindist = 100000000000
                minindex = 0
                index = 0
                for y in parser.read_vectors(f1):
                    print(index)
                    dist = 0
                    for x in parser.read_vectors(f2):
                        dist += np.linalg.norm(np.array(x)-y)
                    if dist < mindist:
                        mindist = dist
                        minindex = index
                    index += 1
                return minindex
        return None

def VamanaAlgo(a:float, L, R, N, file):
    s = medoid(file)
    lens = set()

    # Create new FSGraph
    G = fsgraph.FSGraph('test.fsg')
    G.new(R,128,'fvec',N)

    #Initializing G to an adjacency matrix where each point has R random out-neighbors MUST OPTIMIZE
    with open('data/siftsmall/siftsmall_base.fvecs', 'rb') as f:
        index = 0
        for point in parser.read_vectors(f):
            ### New FSGraph Logic
            npsig = list(range(N))
            npsig.pop(index)
            G.set_data(index, point)
            G.set_neighbors(index, set(i for i in random.sample(npsig, R)))
            index = index + 1

    print(f'Adjacency lens: {lens}')
    print('finished building adjacency matrix')
    for index in range(N):
        point = G.get_data(index)
        L_, V = GreedySearch(s, point, 1, L, G)
        print(index)
        G = LightRobustPrune(index, V, a, R, G)
        for j in G[index]:
            z = G[j].union({index})
            if len(z) > R:
                G = LightRobustPrune(j, z, a, R, G)
            else:
                G[j] = z
    return s, G

s, G = VamanaAlgo(a=1, L=1, R=1, N=10000, file='data/siftsmall/siftsmall_base.fvecs')

# test
correct = 0
total = 0
print('Testing')
with open('data/siftsmall/siftsmall_base.fvecs', 'rb') as f:
    for vec in parser.read_vectors(f):
        A, B = GreedySearch(s, vec, 1, 1, G)
        C = sorted(list(A), key=lambda y: numpy.linalg.norm(numpy.array(G.get_data(y)) - numpy.array(vec)))
        if G.get_data(C[0])==vec:
            correct += 1
        total += 1

print(f'Accuracy: {correct/total}')