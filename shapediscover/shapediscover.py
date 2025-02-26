import torch
import numpy as np
import time

from .parametric_fuzzy_cover import (
    PointCloudFunction,
    SetFunction,
    GraphFunction,
    PartitionOfUnity,
)
from .fuzzy_cover import (
    FuzzyCoverLossFunction,
    fuzzy_cover_from_kmeans,
    fuzzy_cover_to_filtered_complex,
)
from .weighted_graph import graph_from_pointcloud
from .shapediscover_plot import plot_losses


class ShapeDiscover:
    def __init__(
        self,
        n_cover=10,
        knn=15,
        loss_weights=[1, 10, 1, 10],
        graph_algorithm="umap",
        # either random, kmeans, or spectral_clustering
        initialization_algorithm="spectral_clustering",
        # either set_function, pointcloud_nn, or graph_nn
        model="set_function",
        simplex_p=5,
        inner_layer_widths=None,
        n_eigenfunctions=None,
        learning_rate=1e-1,
        n_max_iter=250,
        early_stop=True,
        early_stop_tolerance=1e-4,
    ):
        if initialization_algorithm not in ["random", "kmeans", "spectral_clustering"]:
            raise Exception(
                "Initialization method not recognized", initialization_algorithm
            )
        if model not in ["set_function", "pointcloud_nn", "graph_nn"]:
            raise Exception("Model not recognized", model)

        self._n_cover = n_cover
        self._knn = knn
        self._loss_weights = loss_weights
        self._loss_probabilities = [1, 1, 1, 1]
        self._graph_algorithm = graph_algorithm
        self._initialization_algorithm = initialization_algorithm
        self._model = model
        self._simplex_p = simplex_p

        self._inner_layer_widths = inner_layer_widths
        if not n_eigenfunctions:
            n_eigenfunctions = n_cover
        self._n_eigenfunctions = n_eigenfunctions

        self._optimization_algorithm = "adam"
        self._learning_rate = learning_rate
        self._n_max_iter = n_max_iter
        self._early_stop = early_stop
        self._early_stop_tolerance = early_stop_tolerance

        self.graph_ = None
        self.initialization_precover_ = None
        self.model_ = None
        self.initialization_losses_ = None
        self.historical_outputs_ = None
        self.main_optimization_losses_ = None
        self.loss_names_ = None
        self.precover_ = None
        self.cover_ = None

        self.simplex_tree_ = None
        self.persistence_diagram_ = None
        self.gudhi_persistence_diagram_ = None

    def fit(
        self,
        X,
        n_saved_iterations=0,
        verbose=True,
        plot_loss_curve=True,
        seed=0,
    ):

        torch.manual_seed(seed)

        if verbose:
            print("pointcloud shape:", X.shape)

        # 0. Preprocessing: knn graph
        time_start = time.time()
        graph = graph_from_pointcloud(
            X, n_neighbors=self._knn, algorithm=self._graph_algorithm
        )
        laplacian_eigenmaps = None
        self.graph_ = graph
        time_end = time.time()
        if verbose:
            print("time create graph", time_end - time_start)

        n_points = graph.n_vertices()

        if n_saved_iterations > 0:
            save_output_at_iterations = list(
                range(0, self._n_max_iter, int(self._n_max_iter / n_saved_iterations))
            )

        # 1. Compute initialization
        if self._initialization_algorithm != "random":
            time_start = time.time()
            if self._initialization_algorithm == "kmeans":
                clustering = fuzzy_cover_from_kmeans(
                    X, n_clusters=self._n_cover, seed=seed
                )
            elif self._initialization_algorithm == "spectral_clustering":
                if not laplacian_eigenmaps:
                    laplacian_eigenmaps = graph.laplacian_eigenfunctions(
                        self._n_eigenfunctions
                    )
                clustering = fuzzy_cover_from_kmeans(
                    laplacian_eigenmaps, n_clusters=self._n_cover, seed=seed
                )

            self.initialization_precover_ = clustering
            time_end = time.time()
            if verbose:
                print("time clustering", time_end - time_start)

        # 2. Construct optimizable partition of unity
        if self._model == "set_function":
            if self._initialization_algorithm == "random":
                vector_valued_function = SetFunction(n_points, self._n_cover)
            else:
                vector_valued_function = SetFunction(
                    n_points, self._n_cover, initialization=clustering
                )
        elif self._model == "pointcloud_nn":
            if not self._inner_layer_widths:
                self._inner_layer_widths = [self._n_cover]
            vector_valued_function = PointCloudFunction(
                X, self._n_cover, inner_layer_widths=self._inner_layer_widths
            )
        elif self._model == "graph_nn":
            if not self._inner_layer_widths:
                n_inner_layers = 2
                self._inner_layer_widths = [
                    self._n_cover for _ in range(n_inner_layers)
                ]
            if not laplacian_eigenmaps:
                laplacian_eigenmaps = graph.laplacian_eigenfunctions(
                    self._n_eigenfunctions
                )
            node_features = laplacian_eigenmaps
            vector_valued_function = GraphFunction(
                graph,
                node_features,
                self._n_cover,
                inner_layer_widths=self._inner_layer_widths,
            )
        partition_of_unity = PartitionOfUnity(vector_valued_function)
        self.model_ = partition_of_unity

        # 3. Initialize model on initialization
        if self._initialization_algorithm != "random" and self._model != "set_function":
            time_start = time.time()

            if self._optimization_algorithm == "adam":
                optimizer_initialization = torch.optim.Adam(
                    partition_of_unity.parameters(), lr=self._learning_rate
                )
            else:
                optimizer_initialization = torch.optim.SGD(
                    partition_of_unity.parameters(), lr=self._learning_rate
                )

            if self._early_stop:
                early_stopper = GradientEarlyStopper(
                    partition_of_unity, self._early_stop_tolerance
                )

            initialization_losses = []

            initialization_target = torch.tensor(
                clustering,
                dtype=torch.float32,
                requires_grad=False,
            )

            for iteration_number in range(self._n_max_iter):
                loss = torch.sum(
                    (partition_of_unity() - initialization_target) ** 2
                ) / (self._n_cover * n_points)
                initialization_losses.append([iteration_number, loss.detach().numpy()])
                optimizer_initialization.zero_grad()
                loss.backward()
                optimizer_initialization.step()

                if self._early_stop and early_stopper.early_stop():
                    break

            initialization_losses = np.array(initialization_losses)
            self.initialization_losses_ = initialization_losses

            time_end = time.time()
            if verbose:
                print("time initialization", time_end - time_start)
            if plot_loss_curve:
                plot_losses([initialization_losses], ["initialization loss"])

        # 4. Train model to minimize main loss function
        if self._optimization_algorithm == "adam":
            optimizer = torch.optim.Adam(
                partition_of_unity.parameters(), lr=self._learning_rate
            )
        else:
            optimizer = torch.optim.SGD(
                partition_of_unity.parameters(), lr=self._learning_rate
            )

        loss_function = FuzzyCoverLossFunction(
            graph, self._loss_weights, self._loss_probabilities, log=True, seed=seed
        )

        if self._early_stop:
            early_stopper = GradientEarlyStopper(
                partition_of_unity, self._early_stop_tolerance
            )

        historical_outputs = []
        self.historical_outputs_ = historical_outputs

        time_start = time.time()
        for iteration_number in range(self._n_max_iter):
            current_pfuzzy_cover = simplex_to_psimplex(
                partition_of_unity(), p=self._simplex_p
            )
            loss = loss_function(current_pfuzzy_cover, iteration_number)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if n_saved_iterations > 0:
                if iteration_number in save_output_at_iterations:
                    historical_outputs.append(current_pfuzzy_cover.detach().numpy())

            if self._early_stop and early_stopper.early_stop():
                break
        main_optimization_losses = loss_function._historical_losses
        self.main_optimization_losses_ = main_optimization_losses
        loss_names = loss_function.loss_names
        self.loss_names_ = loss_names
        time_end = time.time()
        if verbose:
            print("time optimization", time_end - time_start)
        if plot_loss_curve:
            plot_losses(main_optimization_losses, loss_names, from_onwards=0)

        last_pfuzzy_cover = simplex_to_psimplex(partition_of_unity(), p=self._simplex_p)
        self.precover_ = last_pfuzzy_cover.detach().numpy()
        output_cover = simplex_to_psimplex(last_pfuzzy_cover, p=float("inf"))
        self.cover_ = output_cover.detach().numpy()

    def fit_persistence(self, max_dimension=1, clique_complex=False, verbose=True):
        if self.cover_ is None:
            raise Exception("Must fit the ShapeDiscover object.")

        time_start = time.time()
        if clique_complex:
            simplex_tree = fuzzy_cover_to_filtered_complex(
                self.cover_, max_dimension=1
            ).to_simplex_tree()
            simplex_tree.expansion(max_dimension + 1)
        else:
            simplex_tree = fuzzy_cover_to_filtered_complex(
                self.cover_, max_dimension=max_dimension + 1
            ).to_simplex_tree()
        time_end = time.time()

        self.simplex_tree_ = simplex_tree

        if verbose:
            print("time create simplicial complex", time_end - time_start)

        time_start = time.time()
        self.gudhi_persistence_diagram_ = simplex_tree.persistence()
        self.persistence_diagram_ = [
            np.array(simplex_tree.persistence_intervals_in_dimension(i))
            for i in range(max_dimension + 1)
        ]
        time_end = time.time()
        if verbose:
            print("time compute persistence", time_end - time_start)

        # return self.persistence_diagram_


class GradientEarlyStopper:
    # https://stackoverflow.com/a/73704579
    def __init__(self, model, tolerance, patience=5):
        self._patience = patience
        self._tolerance = tolerance
        self._model = model
        self._counter = 0

    def early_stop(self):
        gradient_norm = model_gradient_norm(self._model)

        if gradient_norm < self._tolerance:
            self._counter += 1
            if self._counter >= self._patience:
                return True
        else:
            self._counter = 0
        return False


def model_gradient_norm(model):
    parameter_gradients = [
        parameter.grad.detach().flatten()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    return torch.cat(parameter_gradients).norm()


def simplex_to_psimplex(functions, p=2):
    return functions / torch.norm(functions, p=p, dim=0)
