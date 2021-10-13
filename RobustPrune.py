import numpy as np


def dist(p1, p2):
    # Inputs
    # p1: a point in P
    # p2: a point in P
    return np.linalg.norm(p1.vector_data - p2.vector_data)


def robustPrune(p, V, alpha, R):
    # Inputs
    # p: a point p from P
    # V: candidate set
    # alpha: distance threshold that is >= 1
    # R: a degree bound

    # Output
    # G is modified by setting at most R new out-neighbors for p
    V = (V.union(p.out_edges)) - p
    p.out_edges = None
    p_star = None

    while len(V) != 0:
        distance = float('inf')
        for p_prime in V:
            if dist(p, p_prime) < distance:
                p_star = p_prime
                distance = dist(p, p_prime)
        p.out_edges = p.out_edges.union(p_star)
        if len(p.out_edges) == R:
            break
        for p_prime in V:
            if alpha * dist(p_star, p_prime) <= dist(p, p_prime):
                V = V - p_prime

    return
