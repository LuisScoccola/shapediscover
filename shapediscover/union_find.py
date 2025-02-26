import numba as nb
import numpy as np


def union_find_create(n_elements):
    res = np.empty((4,n_elements), dtype=int)
    # sizes
    res[0,:] = np.full(n_elements, 1)
    # parents
    res[1,:] = np.array(list(range(n_elements)))
    # neighbors
    res[2,:] = np.array(list(range(n_elements)))
    return res

@nb.njit
def union_find_find(uf, x):
    parents = uf[1]
    while x != parents[x]:
        parents[x] = parents[parents[x]]
        x = parents[x]
    return x

@nb.njit
def union_find_union(uf, x, y):
    sizes = uf[0]
    parents = uf[1]
    nbrs = uf[2]

    xr = union_find_find(uf, x)
    yr = union_find_find(uf, y)
    if xr == yr:
        return False

    if (sizes[xr], yr) < (sizes[yr], xr):
        xr, yr = yr, xr
    parents[yr] = xr
    sizes[xr] += sizes[yr]
    nbrs[xr], nbrs[yr] = nbrs[yr], nbrs[xr]
    return True

@nb.njit
def union_find_subset(uf, x):
    nbrs = uf[2]

    result = [x]
    nxt = nbrs[x]
    while nxt != x:
        result.append(nxt)
        nxt = nbrs[nxt]
    return np.array(result)

@nb.njit
def subsets(uf):
    result = []
    visited = set()
    for x in range(uf.shape[1]):
        if x not in visited:
            xset = union_find_subset(uf,x)
            visited.update(xset)
            result.append(xset)
    return result
