import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as gnn
import numpy as np


class PartitionOfUnity(torch.nn.Module):
    def __init__(
        self,
        model,
    ):
        """
            Implements an optimizable partition of unity built on top of an optimizable function.

            Parameters
            ----------

            model : an optimizable function

        """
        super().__init__()
        self._model = model

    def forward(self):
        return torch.softmax(self._model(), dim=0)


class PointCloudFunction(torch.nn.Module):
    def __init__(self, pointcloud, n_dimensions, inner_layer_widths):
        super().__init__()

        data_dimension = pointcloud.shape[1]

        self._pointcloud = torch.tensor(
            pointcloud, dtype=torch.float32, requires_grad=False
        )

        if len(inner_layer_widths) == 0:
            layers = [nn.Linear(data_dimension, n_dimensions)]
        else:
            first_width = inner_layer_widths[0]
            last_width = inner_layer_widths[-1]
            layers = [nn.Linear(data_dimension, first_width), nn.Sigmoid()]
            for i in range(len(inner_layer_widths) - 1):
                layers.append(
                    nn.Linear(inner_layer_widths[i], inner_layer_widths[i + 1])
                )
                layers.append(nn.Sigmoid())
                # layers.append(nn.Relu())
            layers.append(nn.Linear(last_width, n_dimensions))

        self._model = nn.Sequential(*layers)

    def forward(self):
        return self._model(self._pointcloud).T


class GraphFunction(nn.Module):
    def __init__(self, graph, node_attributes, n_dimensions, inner_layer_widths):
        super().__init__()

        edge_indices, edge_attributes = torch_geometric.utils.from_scipy_sparse_matrix(
            graph.adjacency_matrix()
        )
        edge_attributes = edge_attributes.to(torch.float32)
        edge_attributes.requires_grad = False

        self._node_attributes = torch.tensor(
            node_attributes, dtype=torch.float32, requires_grad=False
        )
        self._edge_index = edge_indices
        self._edge_weight = edge_attributes

        if len(inner_layer_widths) == 0:
            layers = [
                (
                    gnn.GCNConv(node_attributes.shape[1], n_dimensions),
                    "x, edge_index, edge_weight -> x",
                )
            ]
        else:
            first_width = inner_layer_widths[0]
            last_width = inner_layer_widths[-1]
            layers = [
                (
                    gnn.GCNConv(node_attributes.shape[1], first_width),
                    "x, edge_index, edge_weight -> x",
                ),
                (nn.Sigmoid(), "x -> x"),
            ]
            for i in range(len(inner_layer_widths) - 1):
                layers.append(
                    (
                        gnn.GCNConv(inner_layer_widths[i], inner_layer_widths[i + 1]),
                        "x, edge_index, edge_weight -> x",
                    )
                )
                layers.append((nn.Sigmoid(), "x -> x"))
            layers.append(
                (
                    gnn.GCNConv(last_width, n_dimensions),
                    "x, edge_index, edge_weight -> x",
                )
            )

        self._model = gnn.Sequential("x, edge_index, edge_weight", layers)

    def forward(self):
        return self._model(self._node_attributes, self._edge_index, self._edge_weight).T


class SetFunction(nn.Module):
    def __init__(self, set_size, n_dimensions, initialization=None):
        """
            Implements an optimizable function defined on the set {1, ..., set_size} and taking values in R to the n_dimensions.

            Parameters
            ----------
                set_size : int

                n_dimensions : int

                initialization : None or numpy array of shape (n_dimensions, set_size)
        """
        super().__init__()

        if initialization is None:
            initialization = np.random.random((n_dimensions, set_size))

        self._param = nn.Parameter(
            torch.tensor(initialization, dtype=torch.float32, requires_grad=True)
        )

    def forward(self):
        return self._param
