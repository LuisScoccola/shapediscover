import numpy as np
import networkx as nx


class FilteredComplex:
    """
    Implements a simplicial complex filtered by the real line.
    """
    def __init__(self, simplices, births):
        self._simplices = simplices
        self._births = births

    def cut(self, threshold):
        #assert threshold >= 0 and threshold <= 1
        return [
            simplices_of_dimension[births_of_dimension >= threshold]
            for simplices_of_dimension, births_of_dimension in zip(self._simplices, self._births)
        ]


    def to_networkx_graph(self, threshold):
        simplicial_complex = self.cut(threshold)

        if len(simplicial_complex) >= 1:
            vertices = simplicial_complex[0]
        else:
            vertices = []
        if len(simplicial_complex) >= 2:
            edges = simplicial_complex[1]
        else:
            edges = []

        output_graph = nx.Graph()
        output_graph.add_nodes_from(vertices)
        output_graph.add_edges_from(edges)

        return output_graph

    def to_simplex_tree(self, log_normalization=True):
        # TODO: check for gudhi, and if not installed fail and warn user
        import gudhi
        simplex_tree = gudhi.SimplexTree()

        for simplices_of_dimension, births_of_dimension in zip(self._simplices, self._births):
            births = -births_of_dimension if not log_normalization else -np.log(births_of_dimension)
            simplex_tree.insert_batch(simplices_of_dimension.T, births)
        
        return simplex_tree
