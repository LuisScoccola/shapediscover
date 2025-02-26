import numpy as np
import numba as nb

from .union_find import (
    union_find_create,
    union_find_find,
    union_find_subset,
    union_find_union,
)


def persistence_based_flattening(adjacency_list, vertex_filtration, threshold):
    flat_neighbors, neighbors_start_end = adjacency_list
    n_points = len(vertex_filtration)
    appearances = np.argsort(vertex_filtration)[::-1]
    ranks = np.argsort(appearances)
    # contains the current clusters
    uf = union_find_create(n_points)
    # contains the birth time of clusters that are alive
    clusters_birth = np.full(n_points, -1, dtype=float)
    clusters_died = np.full(n_points, False, dtype=bool)
    # contains the flat clusters
    births = np.full(n_points, -1, dtype=float)
    deaths = np.full(n_points, -1, dtype=float)
    sizes = np.full(n_points, -1, dtype=int)
    clusters = np.full(n_points, -1, dtype=int)

    current_max, current_cluster = _main_loop_persistence_based_flattening(
        n_points,
        flat_neighbors,
        neighbors_start_end,
        vertex_filtration,
        appearances,
        ranks,
        clusters_birth,
        clusters_died,
        births,
        deaths,
        sizes,
        clusters,
        uf,
        threshold,
    )

    res_clust = []
    res_deaths = []

    for i in range(current_cluster):
        if i != current_max:
            cluster_as_list_of_lists = np.argwhere(clusters == i)
            cluster_points = cluster_as_list_of_lists.flatten()
            ## debug
            #print(cluster_points)
            res_clust.append(cluster_points)
            res_deaths.append(deaths[i])
        # debug
        #else:
        #    cluster_as_list_of_lists = np.argwhere(clusters == i)
        #    cluster_points = cluster_as_list_of_lists.flatten()
        #    #print(cluster_points)
            
    ## debug
    #print("number of clusters", len(res_clust))
    #for cl in res_clust:
    #   print(len(cl))
    return res_clust, res_deaths


@nb.njit
def _main_loop_persistence_based_flattening(
    n_points: int,
    flat_neighbors: np.ndarray,
    neighbors_start_end: np.ndarray,
    vertex_filtration: np.ndarray,
    appearances: np.ndarray,
    ranks: np.ndarray,
    clusters_birth: np.ndarray,
    clusters_died: np.ndarray,
    births: np.ndarray,
    deaths: np.ndarray,
    sizes: np.ndarray,
    clusters: np.ndarray,
    uf: np.ndarray,
    threshold: float,
):
    current_cluster = 0
    for hind in nb.prange(n_points):
        x = appearances[hind]
        clusters_birth[appearances[hind]] = vertex_filtration[appearances[hind]]
        for y in flat_neighbors[neighbors_start_end[x][0] : neighbors_start_end[x][1]]:
            if ranks[y] < hind:
                rx = union_find_find(uf, x)
                ry = union_find_find(uf, y)
                # if already together, there is nothing to do
                if rx == ry:
                    continue
                # otherwise, we have a merge
                merge_height = vertex_filtration[x]
                # if both clusters are alive
                if not clusters_died[rx] and not clusters_died[ry]:
                    bx = clusters_birth[rx]
                    by = clusters_birth[ry]

                    # if both have lived for more than the threshold, have them as flat clusters
                    if bx - threshold > merge_height and by - threshold > merge_height:
                        c = union_find_subset(uf, x)
                        clusters[c] = current_cluster
                        births[current_cluster] = bx
                        deaths[current_cluster] = merge_height
                        sizes[current_cluster] = len(c)
                        current_cluster += 1

                        c = union_find_subset(uf, y)
                        clusters[c] = current_cluster
                        births[current_cluster] = by
                        deaths[current_cluster] = merge_height
                        sizes[current_cluster] = len(c)
                        current_cluster += 1

                        union_find_union(uf, x, y)
                        rxy = union_find_find(uf, x)
                        clusters_died[rxy] = True

                    # otherwise, merge them
                    else:
                        union_find_union(uf, x, y)
                        rxy = union_find_find(uf, x)
                        clusters_birth[rxy] = max(bx, by)

                # if both clusters are already dead, just merge them into a dead cluster
                elif clusters_died[rx] and clusters_died[ry]:
                    union_find_union(uf, x, y)
                    rxy = union_find_find(uf, x)
                    clusters_died[rxy] = True
                # if only one of them is dead
                else:
                    # we make it so that ry already died and rx just died
                    if clusters_died[rx]:
                        x, y = y, x
                        rx, ry = ry, rx
                    # if x has lived for longer than the threshold, have it as a flat cluster
                    if clusters_birth[rx] - threshold > merge_height:
                        c = union_find_subset(uf, x)
                        clusters[c] = current_cluster
                        births[current_cluster] = clusters_birth[rx]
                        deaths[current_cluster] = merge_height
                        sizes[current_cluster] = len(c)
                        current_cluster += 1

                    # then merge the clusters into a dead cluster
                    union_find_union(uf, x, y)
                    rxy = union_find_find(uf, x)
                    clusters_died[rxy] = True

    # go through all clusters that have been born but haven't been merged
    for x in range(n_points):
        rx = union_find_find(uf, x)
        if not clusters_died[rx]:
            c = union_find_subset(uf, x)
            clusters[c] = current_cluster
            births[current_cluster] = clusters_birth[rx]
            deaths[current_cluster] = 0
            sizes[current_cluster] = len(c)
            current_cluster += 1

            clusters_died[rx] = True

    max_height = -1
    max_size = -1
    current_max = -1
    for i, hs in enumerate(zip(births, sizes)):
        h, s = hs
        if i < current_cluster:
            if (np.abs(h - max_height) < threshold and s > max_size) or (
                h > max_height
            ):
                max_height = h
                max_size = s
                current_max = i

    return current_max, current_cluster
