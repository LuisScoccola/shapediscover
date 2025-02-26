from sklearn.neighbors import BallTree
import scipy as sp
import numpy as np
import numba as nb
import warnings
from warnings import warn
from sklearn.decomposition import TruncatedSVD


class WeightedGraph:
    def __init__(self, adjacency_matrix):
        """
        Assumptions:
            - Simple, undirected, weighted graph, without loops, encoded as weighted symmetric adjacency matrix.
            - Input is scipy sparse matrix.
        """
        self._adjacency_matrix = adjacency_matrix

    def adjacency_matrix(self):
        return self._adjacency_matrix

    def n_vertices(self):
        return self.adjacency_matrix().shape[0]

    # def total_edge_weight(self):
    #    return self.adjacency_matrix().sum() / 2

    def coboundary_matrix_and_edge_weights(self):

        @nb.jit(nopython=True)
        def _weighted_coboundary_matrix_main_loop(
            entries,
            rows,
            columns,
            coboundary_matrix_row,
            coboundary_matrix_col,
            coboundary_matrix_data,
            edge_weights,
        ):
            i = 0
            for entry, row, column in zip(entries, rows, columns):
                if row < column:
                    coboundary_matrix_row[2 * i] = i
                    coboundary_matrix_col[2 * i] = row
                    coboundary_matrix_data[2 * i] = 1
                    coboundary_matrix_row[2 * i + 1] = i
                    coboundary_matrix_col[2 * i + 1] = column
                    coboundary_matrix_data[2 * i + 1] = -1
                    edge_weights[i] = entry
                    i += 1

        adjacency_matrix = sp.sparse.coo_array(self.adjacency_matrix())
        entries = adjacency_matrix.data

        # entries = np.sqrt(entries)

        rows = adjacency_matrix.row
        columns = adjacency_matrix.col

        n_edges = entries.shape[0] // 2

        # coboundary_matrix = sp.sparse.lil_array((n_edges, n_points))
        coboundary_matrix_row = np.zeros(2 * n_edges, dtype=int)
        coboundary_matrix_col = np.zeros(2 * n_edges, dtype=int)
        coboundary_matrix_data = np.zeros(2 * n_edges)
        edge_weights = np.zeros(n_edges)

        _weighted_coboundary_matrix_main_loop(
            entries,
            rows,
            columns,
            coboundary_matrix_row,
            coboundary_matrix_col,
            coboundary_matrix_data,
            edge_weights,
        )

        return (
            sp.sparse.coo_matrix(
                (coboundary_matrix_data, (coboundary_matrix_row, coboundary_matrix_col))
            ),
            edge_weights,
        )

    def efficient_adjacency_list(self):
        if self.flat_neighbors_ is not None:
            return self.flat_neighbors_, self.flat_neighbors_start_end_
        else:
            n_points = self.n_vertices()
            adjacency_matrix = self.adjacency_matrix()
            neighbors = [adjacency_matrix[:, [i]].nonzero()[0] for i in range(n_points)]
            flat_neighbors = np.array([n for ns in neighbors for n in ns], dtype=int)
            neighbors_lenghts = [len(ns) for ns in neighbors]
            neighbors_start_end = []
            current_start = 0
            for i in range(n_points):
                current_end = current_start + neighbors_lenghts[i]
                neighbors_start_end.append([current_start, current_end])
                current_start = current_end
            self.flat_neighbors_ = flat_neighbors
            self.flat_neighbors_start_end_ = np.array(neighbors_start_end, dtype=int)
            return self.flat_neighbors_, self.flat_neighbors_start_end_

    # Function laplacian_eigenfunctions taken from UMAP's codebase: https://github.com/lmcinnes/umap
    # Authors: McInnes, Leland and Healy, John and Saul, Nathaniel and Grossberger, Lukas
    # Commit and line: https://github.com/lmcinnes/umap/blob/d4d4c4aeb96e0d2296b5098d9dc9736de79e4e96/umap/spectral.py#L395
    def laplacian_eigenfunctions(
        self,
        dim,
        init="random",
        random_state=None,
        method=None,
        tol=0.0,
        maxiter=0,
    ):
        graph = self.adjacency_matrix()

        sqrt_deg = np.sqrt(np.asarray(graph.sum(axis=0)).squeeze())
        I = sp.sparse.identity(graph.shape[0], dtype=np.float64)
        D = sp.sparse.spdiags(1.0 / sqrt_deg, 0, graph.shape[0], graph.shape[0])
        L = I - D * graph * D
        if not sp.sparse.issparse(L):
            L = np.asarray(L)

        k = dim + 1
        num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(graph.shape[0])))
        gen = (
            random_state
            if isinstance(random_state, (np.random.Generator, np.random.RandomState))
            else np.random.default_rng(seed=random_state)
        )
        if not method:
            method = "eigsh" if L.shape[0] < 2000000 else "lobpcg"

        try:
            if init == "random":
                X = gen.normal(size=(L.shape[0], k))
            elif init == "tsvd":
                X = TruncatedSVD(
                    n_components=k,
                    random_state=random_state,
                ).fit_transform(L)
            else:
                raise ValueError(
                    "The init parameter must be either 'random' or 'tsvd': "
                    f"{init} is invalid."
                )
            # For such a normalized Laplacian, the first eigenvector is always
            # proportional to sqrt(degrees). We thus replace the first t-SVD guess
            # with the exact value.
            X[:, 0] = sqrt_deg / np.linalg.norm(sqrt_deg)

            if method == "eigsh":
                eigenvalues, eigenvectors = sp.sparse.linalg.eigsh(
                    L,
                    k,
                    which="SM",
                    ncv=num_lanczos_vectors,
                    tol=tol or 1e-4,
                    v0=np.ones(L.shape[0]),
                    maxiter=maxiter or graph.shape[0] * 5,
                )
            elif method == "lobpcg":
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        category=UserWarning,
                        message=r"(?ms).*not reaching the requested tolerance",
                        action="error",
                    )
                    eigenvalues, eigenvectors = sp.sparse.linalg.lobpcg(
                        L,
                        np.asarray(X),
                        largest=False,
                        tol=tol or 1e-4,
                        maxiter=maxiter or 5 * graph.shape[0],
                    )
            else:
                raise ValueError("Method should either be None, 'eigsh' or 'lobpcg'")

            order = np.argsort(eigenvalues)[0:k]
            return eigenvectors[:, order]
        except:
            warn(
                "Spectral initialisation failed! The eigenvector solver\n"
                "failed. This is likely due to too small an eigengap. Consider\n"
                "adding some noise or jitter to your data.\n\n"
                "Falling back to random initialisation!"
            )
            return gen.uniform(low=-10.0, high=10.0, size=(graph.shape[0], dim))


def graph_from_pointcloud(pointcloud, n_neighbors, algorithm="knn", metric="euclidean"):
    # TODO: pass metric to knn
    n_points = pointcloud.shape[0]
    if algorithm == "knn":
        ball_tree = BallTree(pointcloud)
        _, neighbor_indices = ball_tree.query(pointcloud, n_neighbors)
        adjacency_matrix = sp.sparse.lil_array((n_points, n_points))
        for i in range(n_points):
            adjacency_matrix[i, neighbor_indices[i]] = 1
            adjacency_matrix[i, i] = 0
        adjacency_matrix = sp.sparse.coo_array(adjacency_matrix)
        symmetric_adjacency_matrix = adjacency_matrix.maximum(adjacency_matrix.T)
        out = WeightedGraph(symmetric_adjacency_matrix)

        out.flat_neighbors_ = neighbor_indices.flatten()

    elif algorithm == "umap":
        import umap

        adjacency_matrix, _, _ = umap.umap_.fuzzy_simplicial_set(
            pointcloud, n_neighbors, random_state=None, metric=metric
        )
        out = WeightedGraph(adjacency_matrix)

        out.flat_neighbors_ = np.array(umap.umap_.nearest_neighbors(
            pointcloud,
            n_neighbors,
            metric=metric,
            metric_kwds={},
            angular=False,
            random_state=None,
        )[0], dtype=int).flatten()

    else:
        raise Exception("Algorithm not recognized", algorithm)

    flat_neighbors_start_end_col0 = np.arange(
        0, n_points * n_neighbors, n_neighbors, dtype=int
    )
    flat_neighbors_start_end_col1 = (
        np.arange(0, n_points * n_neighbors, n_neighbors, dtype=int) + n_neighbors - 1
    )
    flat_neighbors_start_end = np.zeros((n_points, 2), dtype=int)
    flat_neighbors_start_end[:, 0] = flat_neighbors_start_end_col0
    flat_neighbors_start_end[:, 1] = flat_neighbors_start_end_col1
    out.flat_neighbors_start_end_ = flat_neighbors_start_end

    return out

