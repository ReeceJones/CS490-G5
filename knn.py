from lib.dist import L2

def knn(data, target, n):
    results = list()
    for x in data:
        d = L2(x,target)
        if len(results) < n:
            results.append(x)
        elif d < sorted(results, key=lambda x: x[1])[len(results)-1][1]:
            results[len(results)-1] = (x, d)

    return results


if __name__=='__main__':
    results = knn(
        data=[(1,1,1), (1,2,1), (2,3,1), (5,5,5)],
        target=(1,1,1),
        n=2
    )
    print(results)
