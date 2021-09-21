
def L2(v1, v2):
    if len(v1) != len(v2):
        return None
    s = 0
    for i in range(len(v1)):
        s += (v1[i] - v2[i]) ** 2

    return s ** 0.5
