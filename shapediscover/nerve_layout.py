import warnings
import scipy as sp
import numpy as np
import networkx as nx
from sklearn.manifold import MDS

from .fuzzy_cover import (
    fuzzy_cover_to_weighted_edges,
    threshold_fuzzy_cover,
)


def nerve_layout(fuzzy_cover, threshold, method=None, seed=0):
    # turn fuzzy cover to cover
    thresholded_fuzzy_cover, non_zero_indices = threshold_fuzzy_cover(
        fuzzy_cover, threshold
    )

    n_vertices = thresholded_fuzzy_cover.shape[0]

    # compute edge weights
    at_least_one_point = 1
    edges, intersection_sizes = fuzzy_cover_to_weighted_edges(
        thresholded_fuzzy_cover, min_weight=at_least_one_point
    )
    adjacency_matrix = np.zeros((n_vertices, n_vertices), dtype=float)
    for (i, j), intersection_size in zip(edges, intersection_sizes):
        #adjacency_matrix[i, j] = intersection_size
        adjacency_matrix[i, j] = 1

    # take log of intersection sizes
    #adjacency_matrix[adjacency_matrix > 0] = np.log(
    #    adjacency_matrix[adjacency_matrix > 0]
    #)
    #adjacency_matrix = adjacency_matrix / np.where(
    #    np.max(adjacency_matrix, axis=1) > 0, np.max(adjacency_matrix, axis=1), 1
    #)

    # build networkx graph
    weights = np.full_like(intersection_sizes,1)
    edges_weights = [
        (i, j, {"weight": str(weight)}) for (i, j), weight in zip(edges, weights)
    ]
    graph = nx.Graph()
    vertices = list(range(n_vertices))

    graph.add_nodes_from(vertices)
    graph.add_edges_from(edges_weights)

    
    initial_positions_dict = nx.spectral_layout(graph)
    #initial_positions = np.array([initial_positions_dict[i] for i in range(n_vertices)])
    #final_positions = fruchterman_reingold_networkx(adjacency_matrix,initial_positions,repulsion,n_iterations,)

    final_positions_dict = nx.spring_layout(
        graph, pos=initial_positions_dict, iterations=100, seed=seed
        #graph, iterations=50, seed=seed
    )

    final_positions_array = np.array(
        [final_positions_dict[i] for i in range(len(final_positions_dict))]
    )

    return final_positions_array, thresholded_fuzzy_cover


def fruchterman_reingold_networkx(
    adjacency_matrix, initial_positions, repulsion=1.0, n_iterations=50, threshold=1e-4, seed=0
):
    n_vertices = adjacency_matrix.shape[0]
    k = np.sqrt(1.0 / n_vertices)


    np.random.seed(seed)
    pos = initial_positions + np.random.random_sample(initial_positions.shape) * 1e-10

    # the initial "temperature"  is about .1 of domain area (=1x1)
    temperature = (
        max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    )
    # simple cooling scheme: linearly step down by dt on each iteration so last iteration is size dt.
    delta_temperature = temperature / (n_iterations + 1)
    delta = np.zeros(
        (pos.shape[0], pos.shape[0], pos.shape[1]), dtype=adjacency_matrix.dtype
    )
    for _ in range(n_iterations):
        # matrix of difference between points
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        # distance between points
        distance = np.linalg.norm(delta, axis=-1)
        # enforce minimum distance of 0.01
        np.clip(distance, 0.01, None, out=distance)
        # compute displacement
        # first summand accounts for repulsion, and second for attraction
        displacement = np.einsum(
            "ijk,ij->ik",
            delta,
            (repulsion * (k * k / distance**2) - adjacency_matrix * distance / k),
            #(repulsion * (k * k / np.exp(distance)) - adjacency_matrix * distance / k),
        )
        # update positions
        length = np.linalg.norm(displacement, axis=-1)
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = np.einsum("ij,i->ij", displacement, temperature / length)
        pos += delta_pos
        temperature -= delta_temperature
        if (np.linalg.norm(delta_pos) / n_vertices) < threshold:
            break
    return pos
