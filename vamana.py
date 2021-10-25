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

df = parse_test.parse_to_df(r".\data\siftsmall\siftsmall_base.fvecs")

def GreedySearch(s, x, k, L):
    pass

def medoid(df):
    distMatrix = pairwise_distances(df)
    return numpy.argmin(distMatrix.sum(axis=0))

def VamanaAlgo(P:pd.DataFrame, a:float, L, R):
    G = {}
    s = medoid(P) 
    sigma = P.sample(frac=1)
    #Initializing G to an adjacency matrix where each point has R random out-neighbors MUST OPTIMIZE
    for index in range(len(sigma)):
        point = sigma.iloc[index]
        tuppoint = tuple(sigma.iloc[index])
        npsig = sigma.drop(index=index, axis=0, inplace=False)
        G[tuppoint] = set(tuple(i) for i in npsig.sample(n=R, axis=0).to_numpy())
    
    for index in range(len(sigma)):
        point = tuple(sigma.iloc[index])
        L_, V = GreedySearch(s, point, 20, L) # must implement GreedySearch, this is placeholder
        RobustPrune.robustPrune(point, V, a, R)
        for j in G[point]:
            z = G[j].union(point)
            if len(z) > R:
                RobustPrune.robustPrune(j, z, a, R)
            else:
                G[j] = z

VamanaAlgo(df, 1, 10, 10)