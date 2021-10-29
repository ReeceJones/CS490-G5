import numpy
from lib import dist
from sklearn.metrics import pairwise_distances
import parse_test
import pandas as pd
import RobustPrune
#import greedysearch
#import robustprune

#Vamana constructs G in an iterative manner

#Graph G initialized where each vertex has R randomly chosen out-neighbors.
#   If R > log n, the graph will be well connected, but does not ensure best performance for GreedySearch

#Variable s denotes the medoid of the dataset P, which is the starting node for the algorithm
#   Medoid - Point with the smallest average dissimilarity (distance?) with all other nodes in the graph.

df = parse_test.parse_to_df('data/siftsmall/siftsmall_base.fvecs')

def GreedySearch(s, x, k, L, G):
    A = {tuple(s)}
    B = set()

    while len(A - B) > 0:
        diff = list(A - B)
        norms = [numpy.linalg.norm(numpy.array(y) - x) for y in diff]
        p_prime = diff[numpy.argmin(norms)]
        A = A.union(G[tuple(p_prime)])
        B.add(tuple(p_prime))
        if len(A) > L:
            A = set(sorted(list(A), key=lambda y: numpy.linalg.norm(numpy.array(x) - numpy.array(y)))[:L])

    return set(sorted(list(A), key=lambda y: numpy.linalg.norm(numpy.array(y) - numpy.array(x)))[:k]),B

def LightRobustPrune(p, V, alpha, R, G):
    V = V.union(G[tuple(p)]) - {tuple(p)}
    G[tuple(p)] = set()
    while len(V) > 0:
        _V = list(V)
        norms = [numpy.linalg.norm(numpy.array(y) - p) for y in _V]
        p_prime = _V[numpy.argmin(norms)]
        G[tuple(p)] = G[tuple(p)].union({p_prime})
        if len(G[tuple(p)]) == R:
            break
        for i in range(len(_V)):
            if alpha*numpy.linalg.norm(numpy.array(p_prime) - _V[i]) <= norms[i]:
                V.remove(_V[i])
    return G

def medoid(df):
    distMatrix = pairwise_distances(df)
    return numpy.argmin(distMatrix.sum(axis=0))

def VamanaAlgo(P:pd.DataFrame, a:float, L, R):
    G = {}
    s = medoid(P) 
    sigma = P.sample(frac=1)
    lens = set()
    #Initializing G to an adjacency matrix where each point has R random out-neighbors MUST OPTIMIZE
    print('building adjacency matrix')
    for index in range(len(sigma)):
        point = sigma.iloc[index]
        tuppoint = tuple(sigma.iloc[index])
        npsig = sigma.drop(index=index, axis=0, inplace=False)
        G[tuppoint] = set(tuple(i) for i in npsig.sample(n=R, axis=0).to_numpy())
        lens = lens.union({len(i) for i in G[tuppoint]})
    print(f'Adjacency lens: {lens}')
    print('finished building adjacency matrix')
    for index in range(len(sigma)):
        point = tuple(sigma.iloc[index])
        L_, V = GreedySearch(sigma.iloc[s], point, 20, L, G) # must implement GreedySearch, this is placeholder
        print(index)
        G = LightRobustPrune(point, V, a, R, G)
        for j in G[point]:
            z = G[j].union({point})
            if len(z) > R:
                G = LightRobustPrune(j, z, a, R, G)
            else:
                G[j] = z

VamanaAlgo(df, 1, 10, 10)
